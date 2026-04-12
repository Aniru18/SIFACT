"""
config/settings.py
Central configuration loaded from environment variables.
All LLM calls go through Groq – one API key, three models.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")

NEWS_API_KEY        = os.getenv("NEWS_API_KEY", "")
GNEWS_API_KEY       = os.getenv("GNEWS_API_KEY", "")

# ── Model names (all via Groq) ────────────────────────────────────────────────
# Agent 1 – Extraction: strong instruction-following + JSON output
EXTRACTION_MODEL   = os.getenv("EXTRACTION_MODEL",   "llama-3.3-70b-versatile")

# Agent 2 – Verification: fast, runs in parallel for every claim
# VERIFICATION_MODEL = os.getenv("VERIFICATION_MODEL", "llama-3.1-8b-instant")
VERIFICATION_MODEL = os.getenv("VERIFICATION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# Agent 3 – Synthesis: deep reasoning for final verdict
SYNTHESIS_MODEL    = os.getenv("SYNTHESIS_MODEL",    "openai/gpt-oss-120b")
# Groq structured output: "function_calling" (default, wide support) or "json_schema" (e.g. openai/gpt-oss-*)
_raw_syn_method = os.getenv("SYNTHESIS_STRUCTURED_METHOD", "function_calling").strip().lower()
SYNTHESIS_STRUCTURED_METHOD: str = (
    _raw_syn_method if _raw_syn_method in ("function_calling", "json_schema") else "function_calling"
)

# ── Pipeline parameters ───────────────────────────────────────────────────────
MAX_SECONDARY_CLAIMS     = int(os.getenv("MAX_SECONDARY_CLAIMS", 5))
NEWS_ARTICLES_PER_CLAIM  = int(os.getenv("NEWS_ARTICLES_PER_CLAIM", 5))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")