# Viniyog One: Dataset Overview

## Overview
The Viniyog One project runs atop a strictly pruned, authoritative JSON schema built to replicate a realistic internal Indian Banking Knowledge Base. 

### Dataset Artifact Details
- **Path:** `data/banking_schemes.json`
- **Format:** Structured JSON Array
- **Total Entities:** 15 Complete Financial Products

## Structural Fields
Every entry strictly conforms to a unified schema required by the ingestion layer to guarantee that variables like interest rates aren't lost to unstructured noise.

```json
{
  "scheme_id": "STAND-UP-001",
  "name": "Stand-Up India Scheme",
  "category": "MSME & Entrepreneurship",
  "description": "Facilitates bank loans between 10 lakh and 1 Crore...",
  "eligibility": [
    "SC/ST and/or woman entrepreneur",
    "Above 18 years of age"
  ],
  "benefits": [
    "Loan covers 75% of the project cost",
    "Composite loan"
  ],
  "interest_rate": "Details on repo rate + fixed elements",
  "documents_required": [
    "Identity proof",
    "Business address proof"
  ],
  "ministry": "Department of Financial Services",
  "launch_date": "2016-04-05",
  "url": "https://www.standupmitra.in/"
}
```

## System Interaction
1. **Idempotent Bootstrapping:** On startup, `ingestion_service.py` evaluates the hash signature of `banking_schemes.json`. If variations or updates exist, the ingestion loop automatically triggers context mapping.
2. **Text Generation Optimization:** Because standard NLP models are trained on prose, our ingest logic parses the JSON structural arrays and re-templates them into Markdown formatting (e.g., transforming `"eligibility": ["Over 18"]` into `## Eligibility \n - Over 18`) perfectly tuning it for BGE dense embedding alignment.
3. **Database Mirroring (KB Fallback):** Simultaneously, the JSON file imports into the isolated SQLite `/data/knowledge_base.db` powering the secure `/kb/token` lookup pathway.
