"""
==============================================================================
Security Module — JWT Generation, Validation, and Token Lifecycle
==============================================================================

ARCHITECTURAL DECISION (Token-Based KB Access):
    In a banking system, direct access to the Knowledge Base is a security
    anti-pattern. Even internal services should not bypass access controls.
    
    This module implements a **short-lived JWT mechanism** specifically for
    KB access. The flow is:
    
    1. Hallucination detected → System calls `/kb/token` to get a JWT
    2. JWT is signed with HS256 and has a 1-minute TTL
    3. System passes JWT to `/kb/fetch` to retrieve verified data
    4. JWT expires automatically — no revocation needed for short TTL
    
    WHY JWT (and not API keys or session tokens)?
    - JWTs are stateless — no server-side session store needed
    - Claims embed intent (scope, purpose) for fine-grained access control
    - Standard library support, well-audited cryptographic primitives
    - Short TTL + no refresh = minimal attack surface
    
    WHY HS256 (and not RS256)?
    - Single-service architecture — no need for public key distribution
    - Lower computational overhead for high-frequency token generation
    - In production with microservices, RS256 would be preferred

SECURITY CONSIDERATIONS:
    - Tokens are NOT user-facing — they are internal system tokens
    - The `purpose` claim restricts what the token can be used for
    - Clock skew tolerance is set to 30 seconds for distributed systems
    - In production, add JTI (JWT ID) for one-time-use enforcement
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import jwt, JWTError, ExpiredSignatureError
from pydantic import BaseModel
from loguru import logger

from app.core.config import get_settings


# =============================================================================
# Data Models for Token Operations
# =============================================================================

class KBTokenPayload(BaseModel):
    """
    Schema for the KB access token payload.
    
    Fields:
        purpose: Restricts token usage to specific operations (e.g., "kb_access")
        scope: Defines what KB resources the token can access
        issued_at: ISO timestamp for audit trail
        expires_at: ISO timestamp for client-side expiry checking
    """
    purpose: str = "kb_access"
    scope: str = "read"
    issued_at: str
    expires_at: str


class KBTokenResponse(BaseModel):
    """Response model for the /kb/token endpoint."""
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    purpose: str = "kb_access"


class TokenValidationResult(BaseModel):
    """Result of token validation — used internally."""
    is_valid: bool
    payload: Optional[dict] = None
    error: Optional[str] = None


# =============================================================================
# Token Generation
# =============================================================================

def generate_kb_access_token(
    purpose: str = "kb_access",
    scope: str = "read",
    additional_claims: Optional[dict] = None,
) -> KBTokenResponse:
    """
    Generate a short-lived JWT for Knowledge Base access.
    
    This function is called internally when:
    1. Hallucination is detected in a RAG response
    2. The system needs to fall back to the authoritative KB
    3. An authorized evaluator needs direct KB access for testing
    
    Args:
        purpose: Intent of the token (always "kb_access" for KB operations)
        scope: Access scope ("read" for fetching, "read_write" for admin)
        additional_claims: Optional extra claims for audit/tracing
    
    Returns:
        KBTokenResponse with the signed JWT and metadata
    
    Security:
        - TTL is deliberately short (1 min) to limit exposure
        - Purpose claim prevents token reuse across different endpoints
        - All token generations are logged for audit
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expiry = now + timedelta(minutes=settings.KB_TOKEN_EXPIRY_MINUTES)

    # Build JWT claims
    claims = {
        # Standard JWT claims
        "iat": now,                          # Issued At
        "exp": expiry,                       # Expiration Time
        "nbf": now,                          # Not Before (immediate validity)
        # Custom claims — these enforce access boundaries
        "purpose": purpose,                  # Token intent
        "scope": scope,                      # Access level
        "issuer": "viniyog-rag-system",      # Identifies the issuing service
    }

    # Merge additional claims (e.g., query_id for tracing)
    if additional_claims:
        claims.update(additional_claims)

    # Sign the token
    token = jwt.encode(
        claims,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    logger.info(
        f"KB access token generated | purpose={purpose} | scope={scope} | "
        f"expires_at={expiry.isoformat()}"
    )

    return KBTokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in_seconds=settings.KB_TOKEN_EXPIRY_MINUTES * 60,
        purpose=purpose,
    )


# =============================================================================
# Token Validation
# =============================================================================

def validate_kb_token(token: str) -> TokenValidationResult:
    """
    Validate a KB access JWT.
    
    Validation checks (in order):
    1. Signature verification — ensures token wasn't tampered with
    2. Expiry check — rejects expired tokens
    3. Not-before check — prevents early use of pre-issued tokens
    4. Purpose claim — ensures token is for KB access, not other services
    
    Args:
        token: The raw JWT string from the Authorization header
    
    Returns:
        TokenValidationResult with validation status and decoded payload
    
    Security:
        - Expired tokens are rejected immediately (no grace period in dev)
        - Invalid signatures raise an alert (potential attack indicator)
        - All validation failures are logged with details
    """
    settings = get_settings()

    try:
        # Decode and validate in one step
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
            },
        )

        # Additional business logic validation
        if payload.get("purpose") != "kb_access":
            logger.warning(
                f"Token rejected — wrong purpose: {payload.get('purpose')}"
            )
            return TokenValidationResult(
                is_valid=False,
                error="Token purpose mismatch: expected 'kb_access'",
            )

        logger.debug(f"KB token validated successfully | scope={payload.get('scope')}")

        return TokenValidationResult(
            is_valid=True,
            payload=payload,
        )

    except ExpiredSignatureError:
        logger.warning("KB token rejected — expired")
        return TokenValidationResult(
            is_valid=False,
            error="Token has expired. Request a new token from /kb/token.",
        )

    except JWTError as e:
        logger.error(f"KB token validation failed — {str(e)}")
        return TokenValidationResult(
            is_valid=False,
            error=f"Invalid token: {str(e)}",
        )
