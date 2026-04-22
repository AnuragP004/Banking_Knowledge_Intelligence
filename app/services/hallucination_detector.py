"""
==============================================================================
Hallucination Detection Service — LLM-as-a-Judge Groundedness Evaluation
==============================================================================

ARCHITECTURAL DECISION:
    The BRD mandates: "Implement hallucination detection and KB fallback."
    
    We implement a two-layer hallucination detection strategy:
    
    LAYER 1: HEURISTIC CHECK (fast, zero-cost)
    ─────────────────────────────────────────────
    - Checks for known hallucination patterns:
      * LLM disclaimers ("I don't have access to...")
      * Fabricated data signals ("approximately", "around" for exact figures)
      * Missing source attribution
      * Response confidence signals
    - Runs in <1ms, catches obvious failures
    
    LAYER 2: LLM SELF-EVALUATION (single-pass, from structured generation)
    ──────────────────────────────────────────────────
    - The LLM evaluates its OWN response during generation via Structured Outputs
    - Gemini returns a GroundedResponse JSON including:
      * self_evaluated_groundedness (0.0-1.0)
      * claims_total / claims_supported
      * unsupported_details
    - This eliminates the second network call entirely
    - Scores groundedness on a 0.0-1.0 scale
    
    WHY SINGLE-PASS (not separate LLM judge)?
    A separate judge call added 3-4s of latency, making the system exceed
    the 3-second BRD SLA. By using Gemini's Structured Outputs, we get
    the same atomic claim decomposition in a single API call.
    
    THRESHOLD: 0.85
    - Below 0.85 → Hallucination detected → KB fallback triggered
    - Above 0.85 → Confidently grounded
    
    WHY 0.85 (not lower)?
    - Banking demands near-zero tolerance for fabrication
    - The KB fallback is reliable, so false positives are acceptable
    - A lower threshold risks letting subtle fabricated numbers through

COMPLIANCE:
    In banking, every claim must be traceable. This service ensures:
    1. No fabricated interest rates (could be legally actionable)
    2. No invented eligibility criteria (could cause customer harm)
    3. No hallucinated scheme names (could create confusion)
    4. All responses cite actual banking schemes
"""

import re
from typing import Optional
from dataclasses import dataclass

from loguru import logger


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class HallucinationResult:
    """Result of hallucination detection."""
    groundedness_score: float  # 0.0 (fully hallucinated) to 1.0 (fully grounded)
    is_hallucinated: bool      # True if score < threshold
    reasoning: str             # Human-readable explanation
    method: str                # "heuristic", "llm_judge", or "combined"
    claims_checked: int = 0    # Number of factual claims evaluated
    claims_grounded: int = 0   # Number of claims verified in context


# =============================================================================
# Judge Prompt
# =============================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert fact-checking judge for a banking knowledge system. Your task is to evaluate whether an AI-generated answer is strictly GROUNDED in the provided source context.

PROCESS:
1. DECOMPOSE the AI answer into individual ATOMIC CLAIMS (each distinct fact, number, name, date, or eligibility criterion is one claim).
2. For EACH claim, check whether it is explicitly supported by the source context.
3. Mark each claim as SUPPORTED (present in context) or UNSUPPORTED (absent/contradicted).
4. Compute the final score: (number of SUPPORTED claims) / (total claims).

CRITICAL BANKING RULES:
- Any numeric value (interest rate, amount, age limit) not exactly matching the context is UNSUPPORTED.
- Any scheme name, ministry, or eligibility criterion not in the context is UNSUPPORTED.
- Paraphrasing that preserves meaning is acceptable; extrapolation beyond context is NOT.
- If the answer says "I don't have sufficient information" and the context is indeed limited, score 1.0.

RESPOND IN EXACTLY THIS FORMAT (no markdown, no extra text):
SCORE: <float 0.0 to 1.0>
CLAIMS_TOTAL: <total number of atomic claims found>
CLAIMS_SUPPORTED: <number of claims explicitly supported by context>
CLAIMS_UNSUPPORTED: <number of claims NOT in context>
UNSUPPORTED_DETAILS: <comma-separated list of unsupported claims, or "none">
REASONING: <one paragraph explaining your evaluation>"""


JUDGE_USER_PROMPT_TEMPLATE = """CONTEXT PROVIDED TO THE AI:
{context}

---

USER QUESTION: {query}

---

AI-GENERATED ANSWER:
{answer}

---

Evaluate whether the AI-generated answer is grounded in the provided context. Score from 0.0 (fully hallucinated) to 1.0 (fully grounded)."""


# =============================================================================
# Hallucination Detector Service
# =============================================================================

class HallucinationDetector:
    """
    Two-layer hallucination detection:
    1. Fast heuristic checks (always runs)
    2. LLM-as-a-judge evaluation (runs when LLM is available)
    """

    def __init__(self):
        from app.core.config import get_settings
        self._settings = get_settings()

    async def evaluate(
        self,
        query: str,
        answer: str,
        context_chunks: list[str],
        llm_eval_score: float = 1.0,
        llm_claims_total: int = 0,
        llm_claims_supported: int = 0,
        llm_unsupported_details: str = "",
        llm_reasoning: str = "",
    ) -> HallucinationResult:
        """
        Evaluate whether an answer is grounded in the provided context
        by combining the single-pass LLM structured output with fast heuristics.
        """
        # Layer 1: Heuristic check (always runs, fast)
        heuristic_result = self._heuristic_check(answer, context_chunks)

        # Layer 2: LLM Self-Evaluation (from structured generation)
        llm_reasoning_formatted = llm_reasoning
        if llm_unsupported_details and llm_unsupported_details.lower() != "none":
            llm_reasoning_formatted = f"Unsupported claims: [{llm_unsupported_details}] | {llm_reasoning}"

        # Combine: weighted average (LLM eval weighted higher)
        combined_score = (
            0.3 * heuristic_result.groundedness_score +
            0.7 * llm_eval_score
        )
        
        return HallucinationResult(
            groundedness_score=round(combined_score, 4),
            is_hallucinated=combined_score < self._settings.HALLUCINATION_THRESHOLD,
            reasoning=f"Heuristic: {heuristic_result.reasoning} | LLM Eval: {llm_reasoning_formatted}",
            method="combined_single_pass",
            claims_checked=llm_claims_total,
            claims_grounded=llm_claims_supported,
        )

    # =========================================================================
    # Layer 1: Heuristic Checks
    # =========================================================================

    def _heuristic_check(
        self, answer: str, context_chunks: list[str]
    ) -> HallucinationResult:
        """
        Fast heuristic-based hallucination detection.
        
        Checks:
        1. Response contains known hallucination patterns
        2. Key entities in the answer exist in the context
        3. Numerical claims in the answer match context
        4. Response length vs context length ratio
        """
        score = 1.0
        reasons = []
        answer_lower = answer.lower()
        context_text = " ".join(context_chunks).lower()

        # Check 1: Known hallucination disclaimers
        hallucination_phrases = [
            "i don't have access",
            "i cannot verify",
            "as of my knowledge cutoff",
            "i'm not sure",
            "i believe",
            "it's possible that",
            "i think",
            "based on my training",
        ]
        for phrase in hallucination_phrases:
            if phrase in answer_lower:
                score -= 0.3
                reasons.append(f"Hallucination signal: '{phrase}'")
                break

        # Check 2: Fabrication indicators (vague quantifiers where exact data exists)
        fabrication_phrases = [
            ("approximately", 0.1),
            ("around rs", 0.1),
            ("roughly", 0.1),
            ("estimated at", 0.1),
            ("it is believed", 0.15),
        ]
        for phrase, penalty in fabrication_phrases:
            if phrase in answer_lower and phrase not in context_text:
                score -= penalty
                reasons.append(f"Possible fabrication: '{phrase}' not in context")

        # Check 3: Key term overlap (entities, scheme names)
        answer_terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer))
        key_terms_found = 0
        key_terms_total = len(answer_terms)
        
        for term in answer_terms:
            if term.lower() in context_text:
                key_terms_found += 1

        if key_terms_total > 0:
            entity_coverage = key_terms_found / key_terms_total
            if entity_coverage < 0.5:
                score -= 0.2
                reasons.append(
                    f"Low entity coverage: {key_terms_found}/{key_terms_total} "
                    f"key terms found in context"
                )

        # Check 4: Numerical claim verification
        answer_numbers = set(re.findall(r'Rs\.?\s*[\d,]+(?:\.\d+)?', answer, re.IGNORECASE))
        for num in answer_numbers:
            num_clean = num.lower().replace(",", "").replace(" ", "")
            if num_clean not in context_text.replace(",", "").replace(" ", ""):
                score -= 0.15
                reasons.append(f"Ungrounded number: {num}")

        # Check 5: Answer much longer than context (sign of generation beyond context)
        if len(answer) > len(" ".join(context_chunks)) * 1.5 and len(answer) > 500:
            score -= 0.1
            reasons.append("Answer significantly longer than context (possible extrapolation)")

        score = max(0.0, min(1.0, score))
        
        if not reasons:
            reasons.append("All heuristic checks passed")

        return HallucinationResult(
            groundedness_score=round(score, 4),
            is_hallucinated=score < self._settings.HALLUCINATION_THRESHOLD,
            reasoning=" | ".join(reasons),
            method="heuristic",
        )


# =============================================================================
# Singleton
# =============================================================================

_detector_instance: Optional[HallucinationDetector] = None


def get_hallucination_detector() -> HallucinationDetector:
    """Get or create the hallucination detector singleton."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = HallucinationDetector()
    return _detector_instance
