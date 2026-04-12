"""
graph/state.py
Defines the shared state that flows through the LangGraph pipeline.
Every node reads from and writes to this TypedDict.
"""

from __future__ import annotations
from typing import Any, List, Literal, Optional
from typing_extensions import TypedDict


# ── Sub-types ─────────────────────────────────────────────────────────────────

class Claim(TypedDict):
    """A single factual claim extracted from the news article."""
    id: str                                   # e.g. "central", "secondary_1" …
    text: str                                 # the claim sentence
    type: Literal["central", "secondary"]


class EvidenceArticle(TypedDict):
    """One news article retrieved as evidence for a claim."""
    title: str
    description: str
    url: str
    source: str
    published_at: str


class VerifiedStance(TypedDict):
    """Result of verifying a single claim against retrieved evidence."""
    claim_id: str
    claim_text: str
    stance: Literal["supported", "baseless", "inconclusive"]
    confidence: float                         # 0.0 – 1.0
    evidence_summary: str                     # short justification
    evidence_articles: List[EvidenceArticle]


# ── Main pipeline state ───────────────────────────────────────────────────────

class SIFACTState(TypedDict):
    # ── Input
    article: str                              # raw news article text

    # ── Phase 1 output
    claims: List[Claim]                       # 1 central + up to 5 secondary

    # ── Phase 2 output (built incrementally by the parallel map)
    verified_stances: List[VerifiedStance]

    # ── Phase 3 output
    is_fake: bool
    confidence_score: float                   # overall 0.0 – 1.0
    final_verdict: Literal["REAL", "FAKE", "UNCERTAIN"]
    explanation: str                          # human-readable reasoning

    # ── Housekeeping
    error: Optional[str]                      # non-fatal error message if any