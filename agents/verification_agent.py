# """
# agents/verification_agent.py
# Agent 2 – Verification (Groq: llama-3.1-8b-instant)

# For EACH claim:
#   1. Fetch evidence articles via the News API RAG tool
#   2. Pass claim + evidence to the verification LLM
#   3. Classify as: supported | baseless | inconclusive

# All claims are verified in parallel via ThreadPoolExecutor.
# """

# from __future__ import annotations

# import json
# import logging
# import re
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import Any

# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage, SystemMessage

# from config.settings import GROQ_API_KEY2, VERIFICATION_MODEL
# from graph.state import Claim, EvidenceArticle, SIFACTState, VerifiedStance
# from tools.news_api_rag import fetch_evidence

# logger = logging.getLogger(__name__)
# print(VERIFICATION_MODEL)
# # ── Prompt ────────────────────────────────────────────────────────────────────

# SYSTEM_PROMPT = """You are a rigorous fact-checking AI.
# You will be given:
#   1. A factual CLAIM from a news article (read it literally — every qualifier matters).
#   2. A set of EVIDENCE articles retrieved from recent news sources.

# Classify the claim as exactly one of:

#   - "supported"
#       Use ONLY if the evidence, taken together, clearly establishes the FULL claim as stated.
#       Every substantive restriction in the claim must be satisfied (scope, quantity, universals
#       like "all", "every", "only", time range, format/category). If the claim says e.g. "retired
#       from all formats" but headlines only support retirement from one format (e.g. Tests),
#       that is NOT supported — the claim is stronger than what the evidence proves.

#   - "baseless"
#       The claim is false or contradicted by evidence, OR no on-topic credible evidence exists,
#       OR the article clearly overstates relative to evidence (treating a narrower true fact as if
#       the broader false claim were true — call this out in evidence_summary).

#   - "inconclusive"
#       Mixed signals, weak sources, OR evidence supports a RELATED but STRICTLY WEAKER statement
#       than the claim (e.g. partial scope). Explain precisely what the evidence does and does not
#       establish versus the exact wording of the claim.

# Do NOT label "supported" just because a vague related event is true (e.g. "he retired" when
# the claim demands "all formats"). Match the claim's strength to what sources actually say.

# Output ONLY valid JSON (no markdown, no extra text):
# {
#   "stance": "<supported|baseless|inconclusive>",
#   "confidence": <float 0.0-1.0>,
#   "evidence_summary": "<2-3 sentence justification citing the evidence>"
# }
# """

# def _build_user_message(claim: Claim, articles: list[EvidenceArticle]) -> str:
#     evidence_text = ""
#     for i, art in enumerate(articles, 1):
#         evidence_text += (
#             f"\n[{i}] {art['source']} ({art['published_at'][:10]})\n"
#             f"    Title: {art['title']}\n"
#             f"    Summary: {art['description']}\n"
#         )
#     if not evidence_text.strip():
#         evidence_text = "\n  (No evidence articles retrieved)\n"
#     return (
#         f"CLAIM (verify this exact wording, including scope words like \"all\", \"every\", \"only\"):\n"
#         f"  \"{claim['text']}\"\n\n"
#         f"EVIDENCE ARTICLES:{evidence_text}"
#     )

# # ── Per-claim verification ────────────────────────────────────────────────────

# def _verify_single_claim(claim: Claim, llm) -> VerifiedStance:
#     logger.info("  Verifying [%s]: %s…", claim["id"], claim["text"][:60])

#     articles = fetch_evidence(claim["text"])
#     logger.debug("    Retrieved %d evidence articles", len(articles))

#     messages = [
#         SystemMessage(content=SYSTEM_PROMPT),
#         HumanMessage(content=_build_user_message(claim, articles)),
#     ]
#     response = llm.invoke(messages)
#     raw = response.content.strip()
#     raw = re.sub(r"^```(?:json)?\s*", "", raw)
#     raw = re.sub(r"\s*```$", "", raw)
#     raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

#     try:
#         parsed = json.loads(raw)
#         stance  = parsed.get("stance", "inconclusive")
#         conf    = float(parsed.get("confidence", 0.5))
#         summary = parsed.get("evidence_summary", "")
#     except (json.JSONDecodeError, ValueError) as exc:
#         logger.warning("  Verification parse error [%s]: %s", claim["id"], exc)
#         stance, conf, summary = "inconclusive", 0.3, "Parse error during verification."

#     logger.info("    → %s (conf=%.2f)", stance.upper(), conf)
#     return VerifiedStance(
#         claim_id=claim["id"],
#         claim_text=claim["text"],
#         stance=stance,
#         confidence=conf,
#         evidence_summary=summary,
#         evidence_articles=articles,
#     )

# # ── Node function ─────────────────────────────────────────────────────────────

# def verification_node(state: SIFACTState) -> dict[str, Any]:
#     logger.info("=== Phase 2: Verification (Parallel) | Model: %s ===", VERIFICATION_MODEL)
#     claims = state.get("claims", [])

#     if not claims:
#         return {"verified_stances": []}

#     # One shared LLM instance — ChatGroq is thread-safe
#     llm = ChatGroq(
#         model=VERIFICATION_MODEL,
#         groq_api_key=GROQ_API_KEY2,
#         temperature=0.0,
#     )

#     stances: list[VerifiedStance] = []
#     with ThreadPoolExecutor(max_workers=min(len(claims), 6)) as pool:
#         future_to_claim = {
#             pool.submit(_verify_single_claim, claim, llm): claim
#             for claim in claims
#         }
#         for future in as_completed(future_to_claim):
#             try:
#                 stances.append(future.result())
#             except Exception as exc:
#                 claim = future_to_claim[future]
#                 logger.error("Verification failed [%s]: %s", claim["id"], exc)
#                 stances.append(VerifiedStance(
#                     claim_id=claim["id"],
#                     claim_text=claim["text"],
#                     stance="inconclusive",
#                     confidence=0.0,
#                     evidence_summary="Verification error.",
#                     evidence_articles=[],
#                 ))

#     # Restore original claim order
#     id_order = {c["id"]: i for i, c in enumerate(claims)}
#     stances.sort(key=lambda s: id_order.get(s["claim_id"], 99))

#     logger.info("Verification complete: %d stances", len(stances))
#     return {"verified_stances": stances}


#........................api Fallback code.............................

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from groq import RateLimitError, AuthenticationError

from config.settings import GROQ_API_KEY1, GROQ_API_KEY2, GROQ_API_KEY3, VERIFICATION_MODEL
from graph.state import Claim, EvidenceArticle, SIFACTState, VerifiedStance
from tools.news_api_rag import fetch_evidence

logger = logging.getLogger(__name__)
print(VERIFICATION_MODEL)

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a rigorous fact-checking AI.
You will be given:
  1. A factual CLAIM from a news article (read it literally — every qualifier matters).
  2. A set of EVIDENCE articles retrieved from recent news sources.

Classify the claim as exactly one of:

  - "supported"
      Use ONLY if the evidence, taken together, clearly establishes the FULL claim as stated.
      Every substantive restriction in the claim must be satisfied (scope, quantity, universals
      like "all", "every", "only", time range, format/category). If the claim says e.g. "retired
      from all formats" but headlines only support retirement from one format (e.g. Tests),
      that is NOT supported — the claim is stronger than what the evidence proves.

  - "baseless"
      The claim is false or contradicted by evidence, OR no on-topic credible evidence exists,
      OR the article clearly overstates relative to evidence (treating a narrower true fact as if
      the broader false claim were true — call this out in evidence_summary).

  - "inconclusive"
      Mixed signals, weak sources, OR evidence supports a RELATED but STRICTLY WEAKER statement
      than the claim (e.g. partial scope). Explain precisely what the evidence does and does not
      establish versus the exact wording of the claim.

Do NOT label "supported" just because a vague related event is true (e.g. "he retired" when
the claim demands "all formats"). Match the claim's strength to what sources actually say.

Output ONLY valid JSON (no markdown, no extra text):
{
  "stance": "<supported|baseless|inconclusive>",
  "confidence": <float 0.0-1.0>,
  "evidence_summary": "<2-3 sentence justification citing the evidence>"
}
"""


def _build_user_message(claim: Claim, articles: list[EvidenceArticle]) -> str:
    evidence_text = ""
    for i, art in enumerate(articles, 1):
        evidence_text += (
            f"\n[{i}] {art['source']} ({art['published_at'][:10]})\n"
            f"    Title: {art['title']}\n"
            f"    Summary: {art['description']}\n"
        )
    if not evidence_text.strip():
        evidence_text = "\n  (No evidence articles retrieved)\n"
    return (
        f"CLAIM (verify this exact wording, including scope words like \"all\", \"every\", \"only\"):\n"
        f"  \"{claim['text']}\"\n\n"
        f"EVIDENCE ARTICLES:{evidence_text}"
    )


# ── Fallback helpers ──────────────────────────────────────────────────────────

# Order: Key 3 → Key 2 → Key 1
_GROQ_API_KEYS = [k for k in [GROQ_API_KEY3, GROQ_API_KEY2, GROQ_API_KEY1] if k]
_FALLBACK_EXCEPTIONS = (RateLimitError, AuthenticationError)


def _invoke_with_fallback(messages: list) -> str:
    """
    Try each Groq API key in order (3 → 2 → 1).
    Falls through to the next key on RateLimitError or AuthenticationError.
    Raises the last exception if every key fails.
    """
    last_exc: Exception | None = None

    for idx, api_key in enumerate(_GROQ_API_KEYS, start=1):
        try:
            logger.info("Verification: trying API key slot %d/%d", idx, len(_GROQ_API_KEYS))
            llm = ChatGroq(
                model=VERIFICATION_MODEL,
                groq_api_key=api_key,
                temperature=0.0,
            )
            response = llm.invoke(messages)
            logger.info("Verification: succeeded with API key slot %d", idx)
            return response.content.strip()

        except _FALLBACK_EXCEPTIONS as exc:
            logger.warning(
                "Verification: API key slot %d failed (%s: %s). %s",
                idx,
                type(exc).__name__,
                exc,
                "Trying next key…" if idx < len(_GROQ_API_KEYS) else "No more keys.",
            )
            last_exc = exc

    raise last_exc  # all keys exhausted


# ── Per-claim verification ────────────────────────────────────────────────────

def _verify_single_claim(claim: Claim) -> VerifiedStance:
    logger.info("  Verifying [%s]: %s…", claim["id"], claim["text"][:60])

    articles = fetch_evidence(claim["text"])
    logger.debug("    Retrieved %d evidence articles", len(articles))

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_user_message(claim, articles)),
    ]

    try:
        raw = _invoke_with_fallback(messages)
    except Exception as exc:
        logger.error("Verification: all API keys exhausted [%s]: %s", claim["id"], exc)
        return VerifiedStance(
            claim_id=claim["id"],
            claim_text=claim["text"],
            stance="inconclusive",
            confidence=0.0,
            evidence_summary="All Groq API keys exhausted during verification.",
            evidence_articles=articles,
        )

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    try:
        parsed = json.loads(raw)
        stance  = parsed.get("stance", "inconclusive")
        conf    = float(parsed.get("confidence", 0.5))
        summary = parsed.get("evidence_summary", "")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("  Verification parse error [%s]: %s", claim["id"], exc)
        stance, conf, summary = "inconclusive", 0.3, "Parse error during verification."

    logger.info("    → %s (conf=%.2f)", stance.upper(), conf)
    return VerifiedStance(
        claim_id=claim["id"],
        claim_text=claim["text"],
        stance=stance,
        confidence=conf,
        evidence_summary=summary,
        evidence_articles=articles,
    )


# ── Node function ─────────────────────────────────────────────────────────────

def verification_node(state: SIFACTState) -> dict[str, Any]:
    logger.info("=== Phase 2: Verification (Parallel) | Model: %s ===", VERIFICATION_MODEL)
    claims = state.get("claims", [])

    if not claims:
        return {"verified_stances": []}

    stances: list[VerifiedStance] = []
    with ThreadPoolExecutor(max_workers=min(len(claims), 6)) as pool:
        future_to_claim = {
            pool.submit(_verify_single_claim, claim): claim
            for claim in claims
        }
        for future in as_completed(future_to_claim):
            try:
                stances.append(future.result())
            except Exception as exc:
                claim = future_to_claim[future]
                logger.error("Verification failed [%s]: %s", claim["id"], exc)
                stances.append(VerifiedStance(
                    claim_id=claim["id"],
                    claim_text=claim["text"],
                    stance="inconclusive",
                    confidence=0.0,
                    evidence_summary="Verification error.",
                    evidence_articles=[],
                ))

    # Restore original claim order
    id_order = {c["id"]: i for i, c in enumerate(claims)}
    stances.sort(key=lambda s: id_order.get(s["claim_id"], 99))

    logger.info("Verification complete: %d stances", len(stances))
    return {"verified_stances": stances}


