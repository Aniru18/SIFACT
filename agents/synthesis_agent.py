"""
agents/synthesis_agent.py
Agent 3 – Synthesis (Groq)

Receives all VerifiedStances and produces the final verdict.
Uses LangChain + Groq structured output (tool calling or json_schema) so the model
returns a validated Pydantic object instead of free-form JSON text.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config.settings import GROQ_API_KEY, SYNTHESIS_MODEL, SYNTHESIS_STRUCTURED_METHOD
from graph.state import SIFACTState, VerifiedStance

logger = logging.getLogger(__name__)


class SynthesisVerdict(BaseModel):
    """Structured final verdict for the news article under review."""

    final_verdict: Literal["REAL", "FAKE", "UNCERTAIN"] = Field(
        description=(
            "REAL only if the article's main assertions match what evidence supports (no material "
            "overstatement). FAKE if the article is false OR misleading by exaggeration (e.g. "
            "'all X' when sources only support part of X). UNCERTAIN if evidence is too weak or "
            "mixed to decide — not for cases where overstatement is already clear from stances."
        )
    )
    is_fake: bool = Field(
        description=(
            "True for clear falsehoods AND for misleading articles where the core claim overstates "
            "what credible sources support, even if a weaker related fact might be true."
        )
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence in the verdict from 0.0 to 1.0.",
    )
    explanation: str = Field(
        description=(
            "Three to five sentences citing which claims were supported, baseless, or inconclusive."
        )
    )


# ── Few-Shot Examples (prose — model fills structured fields, not raw JSON blobs) ─

FEW_SHOT_EXAMPLES = """
=== EXAMPLE 1 ===
STANCES:
  [central]     stance=supported    conf=0.92  "Multiple outlets confirmed the event."
  [secondary_1] stance=supported    conf=0.88  "Witnesses corroborated the timeline."
  [secondary_2] stance=inconclusive conf=0.50  "Number disputed across sources."
Expected structured verdict: final_verdict REAL, is_fake false, confidence_score ~0.87,
explanation stresses strong central support and mostly verified secondaries.

=== EXAMPLE 2 ===
STANCES:
  [central]     stance=baseless     conf=0.85  "No credible source confirms this."
  [secondary_1] stance=baseless     conf=0.80  "Statistics contradict the claim."
  [secondary_2] stance=baseless     conf=0.75  "Event did not occur as described."
  [secondary_3] stance=supported    conf=0.60  "Minor background detail is accurate."
Expected: final_verdict FAKE, is_fake true, confidence_score ~0.82,
explanation cites baseless central and majority of secondaries.

=== EXAMPLE 3 ===
STANCES:
  [central]     stance=inconclusive conf=0.45  "Conflicting reports from different sources."
  [secondary_1] stance=supported    conf=0.70  "Date and location confirmed."
  [secondary_2] stance=baseless     conf=0.65  "Quoted figure is inaccurate."
Expected: final_verdict UNCERTAIN, is_fake false, confidence_score ~0.50,
explanation notes mixed evidence and undecided central claim.

=== EXAMPLE 4 (overstatement / scope mismatch) ===
STANCES:
  [central]     stance=inconclusive conf=0.78  "Evidence confirms retirement from Test cricket only; sources do not support retirement from all formats as claimed."
  [secondary_1] stance=supported    conf=0.85  "Career stats and identity of player confirmed."
Expected: final_verdict FAKE, is_fake true, confidence_score ~0.75–0.85,
explanation: the article's central assertion overstates verified facts (misleading even if a
narrower related fact is true). Do NOT output REAL.

=== EXAMPLE 5 (central explicitly baseless due to overstatement) ===
STANCES:
  [central]     stance=baseless     conf=0.82  "Sources report only a partial step; the claim of a complete nationwide rollout is false."
Expected: final_verdict FAKE, is_fake true, high confidence.
"""

SYSTEM_PROMPT = f"""You are the final arbitrator in a multi-agent fact-checking pipeline.

Weighting rules:
  • The "central" claim carries 50% of the total weight.
  • Each "secondary" claim shares the remaining 50% equally.
  • A "baseless" central claim alone is strong evidence of fake news.
  • An "inconclusive" central whose evidence_summary shows the claim is STRONGER than what
    sources support (scope/quantity/universal overstated) → treat as MISLEADING: prefer
    final_verdict FAKE and is_fake true, not REAL (see examples 4–5).
  • A plain "inconclusive" central (truly unclear sources) → UNCERTAIN unless secondaries
    are clearly baseless.

You must respond only through the required structured output fields (final_verdict, is_fake,
confidence_score, explanation). Do not put JSON or markdown code fences in plain assistant text.

Few-shot logic (apply when filling the structured fields):
{FEW_SHOT_EXAMPLES}
"""


def _format_stances(stances: list[VerifiedStance]) -> str:
    lines = []
    for s in stances:
        lines.append(
            f"  [{s['claim_id']:12s}] "
            f"stance={s['stance']:13s} "
            f"conf={s['confidence']:.2f}  "
            f"\"{s['evidence_summary'][:120]}\""
        )
    return "\n".join(lines) if lines else "  (no stances available)"


def synthesis_node(state: SIFACTState) -> dict[str, Any]:
    logger.info(
        "=== Phase 3: Synthesis | Model: %s | structured: %s ===",
        SYNTHESIS_MODEL,
        SYNTHESIS_STRUCTURED_METHOD,
    )
    stances = state.get("verified_stances", [])

    if not stances:
        return {
            "final_verdict": "UNCERTAIN",
            "is_fake": False,
            "confidence_score": 0.0,
            "explanation": "No claims were verified — cannot determine authenticity.",
        }

    user_message = (
        f"Here are the verified stances for the article's claims:\n\n"
        f"STANCES:\n{_format_stances(stances)}\n\n"
        "Return the structured final verdict for this article."
    )

    llm = ChatGroq(
        model=SYNTHESIS_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=1024,
    )

    structured = llm.with_structured_output(
        SynthesisVerdict,
        method=SYNTHESIS_STRUCTURED_METHOD,  # type: ignore[arg-type]
    )

    try:
        out: SynthesisVerdict = structured.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        )
    except Exception as exc:
        logger.exception(
            "Structured synthesis failed (method=%s): %s",
            SYNTHESIS_STRUCTURED_METHOD,
            exc,
        )
        return {
            "final_verdict": "UNCERTAIN",
            "is_fake": False,
            "confidence_score": 0.3,
            "explanation": "Synthesis step failed to produce a valid structured verdict.",
        }

    verdict = out.final_verdict
    is_fake = out.is_fake
    conf = float(out.confidence_score)
    explanation = out.explanation

    logger.info("Final verdict: %s (conf=%.2f, is_fake=%s)", verdict, conf, is_fake)
    return {
        "final_verdict": verdict,
        "is_fake": is_fake,
        "confidence_score": conf,
        "explanation": explanation,
    }
