# """
# agents/extraction_agent.py
# Agent 1 – Extraction (Groq: llama-3.3-70b-versatile)

# Reads the raw news article and produces:
#   • 1 central claim  (the core assertion of the article)
#   • up to 5 secondary claims (supporting sub-facts)
# """

# from __future__ import annotations

# import json
# import logging
# import re
# from typing import Any

# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage, SystemMessage

# from config.settings import EXTRACTION_MODEL, GROQ_API_KEY1, MAX_SECONDARY_CLAIMS
# from graph.state import Claim, SIFACTState

# logger = logging.getLogger(__name__)

# # ── Prompt ────────────────────────────────────────────────────────────────────

# SYSTEM_PROMPT = """You are a professional fact-checking analyst.
# Your job is to read a news article and extract the key factual claims that can be verified.

# Output ONLY valid JSON matching this exact schema (no markdown fences, no extra text):
# {
#   "central_claim": "<the single most important verifiable claim in the article>",
#   "secondary_claims": [
#     "<verifiable sub-claim 1>",
#     "<verifiable sub-claim 2>",
#     ...up to 5 items
#   ]
# }

# Rules:
# - Claims must be specific and verifiable (not opinions or predictions).
# - Each claim should be a single, self-contained sentence.
# - Do NOT include the article title as a claim.
# - Focus on facts: numbers, events, people, places, dates, statistics.
# - Preserve the article's exact SCOPE and STRENGTH in the central claim. If the text says
#   "all formats", "every country", "permanently", etc., the central claim must include that
#   wording — do not water it down to a weaker paraphrase (e.g. do not replace "all formats"
#   with a vague "announced retirement" if the article explicitly asserts all formats).
# - If the article bundles a strong assertion and a weaker true detail, put the strongest
#   checkable assertion in central_claim and use secondary_claims for narrower facts.
# """

# USER_TEMPLATE = """Extract the central claim and up to {max_secondary} secondary claims from the article below.

# ARTICLE:
# \"\"\"
# {article}
# \"\"\"
# """

# # ── Node function ─────────────────────────────────────────────────────────────

# def extraction_node(state: SIFACTState) -> dict[str, Any]:
#     logger.info("=== Phase 1: Extraction | Model: %s ===", EXTRACTION_MODEL)
#     article = state["article"]

#     llm = ChatGroq(
#         model=EXTRACTION_MODEL,
#         groq_api_key=GROQ_API_KEY1,
#         temperature=0.1,
#     )

#     messages = [
#         SystemMessage(content=SYSTEM_PROMPT),
#         HumanMessage(
#             content=USER_TEMPLATE.format(
#                 max_secondary=MAX_SECONDARY_CLAIMS,
#                 article=article[:6000],
#             )
#         ),
#     ]

#     response = llm.invoke(messages)
#     raw = response.content.strip()
#     raw = re.sub(r"^```(?:json)?\s*", "", raw)
#     raw = re.sub(r"\s*```$", "", raw)

#     # DeepSeek-style models wrap output in <think>…</think> — strip it
#     raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

#     try:
#         parsed = json.loads(raw)
#     except json.JSONDecodeError as exc:
#         logger.error("Extraction JSON parse error: %s\nRaw: %s", exc, raw)
#         return {
#             "claims": [Claim(id="central", text=article[:300], type="central")],
#             "error": f"Extraction parse error: {exc}",
#         }

#     claims: list[Claim] = []

#     central_text = parsed.get("central_claim", "").strip()
#     if central_text:
#         claims.append(Claim(id="central", text=central_text, type="central"))

#     for idx, sec_text in enumerate(parsed.get("secondary_claims", [])[:MAX_SECONDARY_CLAIMS]):
#         sec_text = sec_text.strip()
#         if sec_text:
#             claims.append(Claim(id=f"secondary_{idx + 1}", text=sec_text, type="secondary"))

#     logger.info("Extracted %d claims (1 central + %d secondary)", 1, len(claims) - 1)
#     return {"claims": claims}


# ................................Api Fall back code.........................


"""
agents/extraction_agent.py
Agent 1 – Extraction (Groq: llama-3.3-70b-versatile)

Reads the raw news article and produces:
  • 1 central claim  (the core assertion of the article)
  • up to 5 secondary claims (supporting sub-facts)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from groq import RateLimitError, AuthenticationError

from config.settings import EXTRACTION_MODEL, GROQ_API_KEY1, GROQ_API_KEY2, GROQ_API_KEY3, MAX_SECONDARY_CLAIMS
from graph.state import Claim, SIFACTState

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional fact-checking analyst.
Your job is to read a news article and extract the key factual claims that can be verified.

Output ONLY valid JSON matching this exact schema (no markdown fences, no extra text):
{
  "central_claim": "<the single most important verifiable claim in the article>",
  "secondary_claims": [
    "<verifiable sub-claim 1>",
    "<verifiable sub-claim 2>",
    ...up to 5 items
  ]
}

Rules:
- Claims must be specific and verifiable (not opinions or predictions).
- Each claim should be a single, self-contained sentence.
- Do NOT include the article title as a claim.
- Focus on facts: numbers, events, people, places, dates, statistics.
- Preserve the article's exact SCOPE and STRENGTH in the central claim. If the text says
  "all formats", "every country", "permanently", etc., the central claim must include that
  wording — do not water it down to a weaker paraphrase (e.g. do not replace "all formats"
  with a vague "announced retirement" if the article explicitly asserts all formats).
- If the article bundles a strong assertion and a weaker true detail, put the strongest
  checkable assertion in central_claim and use secondary_claims for narrower facts.
"""

USER_TEMPLATE = """Extract the central claim and up to {max_secondary} secondary claims from the article below.

ARTICLE:
\"\"\"
{article}
\"\"\"
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

# Keys tried in order; add/remove entries here as needed.
_GROQ_API_KEYS = [k for k in [GROQ_API_KEY1, GROQ_API_KEY2, GROQ_API_KEY3] if k]

# Exceptions that signal "this key is exhausted / invalid – try the next one".
_FALLBACK_EXCEPTIONS = (RateLimitError, AuthenticationError)


def _invoke_with_fallback(messages: list) -> str:
    """
    Try each Groq API key in order.
    Falls through to the next key on RateLimitError or AuthenticationError.
    Raises the last exception if every key fails.
    """
    last_exc: Exception | None = None

    for idx, api_key in enumerate(_GROQ_API_KEYS, start=1):
        try:
            logger.info("Extraction: trying API key slot %d/%d", idx, len(_GROQ_API_KEYS))
            llm = ChatGroq(
                model=EXTRACTION_MODEL,
                groq_api_key=api_key,
                temperature=0.1,
            )
            response = llm.invoke(messages)
            logger.info("Extraction: succeeded with API key slot %d", idx)
            return response.content.strip()

        except _FALLBACK_EXCEPTIONS as exc:
            logger.warning(
                "Extraction: API key slot %d failed (%s: %s). %s",
                idx,
                type(exc).__name__,
                exc,
                "Trying next key…" if idx < len(_GROQ_API_KEYS) else "No more keys.",
            )
            last_exc = exc

    raise last_exc  # all keys exhausted


# ── Node function ─────────────────────────────────────────────────────────────

def extraction_node(state: SIFACTState) -> dict[str, Any]:
    logger.info("=== Phase 1: Extraction | Model: %s ===", EXTRACTION_MODEL)
    article = state["article"]

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=USER_TEMPLATE.format(
                max_secondary=MAX_SECONDARY_CLAIMS,
                article=article[:6000],
            )
        ),
    ]

    # ── Call LLM with automatic key fallback ──────────────────────────────────
    try:
        raw = _invoke_with_fallback(messages)
    except Exception as exc:
        logger.error("Extraction: all API keys exhausted. Last error: %s", exc)
        return {
            "claims": [Claim(id="central", text=article[:300], type="central")],
            "error": f"All Groq API keys failed: {exc}",
        }

    # ── Post-process raw response ─────────────────────────────────────────────
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # DeepSeek-style models wrap output in <think>…</think> — strip it
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Extraction JSON parse error: %s\nRaw: %s", exc, raw)
        return {
            "claims": [Claim(id="central", text=article[:300], type="central")],
            "error": f"Extraction parse error: {exc}",
        }

    # ── Build claim list ──────────────────────────────────────────────────────
    claims: list[Claim] = []

    central_text = parsed.get("central_claim", "").strip()
    if central_text:
        claims.append(Claim(id="central", text=central_text, type="central"))

    for idx, sec_text in enumerate(parsed.get("secondary_claims", [])[:MAX_SECONDARY_CLAIMS]):
        sec_text = sec_text.strip()
        if sec_text:
            claims.append(Claim(id=f"secondary_{idx + 1}", text=sec_text, type="secondary"))

    logger.info("Extracted %d claims (1 central + %d secondary)", 1, len(claims) - 1)
    return {"claims": claims}



