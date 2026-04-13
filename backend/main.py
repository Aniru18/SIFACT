"""
FastAPI backend for SIFACT — exposes the fact-checking graph over HTTP.
Run from repo root: uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph.state import SIFACTState
from graph.workflow import sifact_graph

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SIFACT API",
    description="Structured fact-checking pipeline (extraction → verification → synthesis)",
    version="0.1.0",
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://127.0.0.1:8501",
#         "http://localhost:8501",
#         "http://127.0.0.1:3000",
#         "http://localhost:3000",
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# .........Update for vercel.......
import os

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8501"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class AnalyzeRequest(BaseModel):
    article: str = Field(..., min_length=1, description="Raw news article text to analyze")


def _run_pipeline(article: str) -> SIFACTState:
    initial: SIFACTState = {
        "article": article,
        "claims": [],
        "verified_stances": [],
        "is_fake": False,
        "confidence_score": 0.0,
        "final_verdict": "UNCERTAIN",
        "explanation": "",
        "error": None,
    }
    return sifact_graph.invoke(initial)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok","version": "1.0"}


@app.post("/api/analyze")
def analyze(body: AnalyzeRequest) -> dict:
    """
    Run the full SIFACT pipeline and return structured results (JSON-serializable).
    """
    article = body.article.strip()
    if not article:
        raise HTTPException(status_code=400, detail="Article text is empty.")

    logger.info("Analyze request: %d chars", len(article))
    try:
        final = _run_pipeline(article)
    except Exception:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail="Fact-checking pipeline failed.") from None

    # TypedDict → plain dict for JSON response
    return {
        "article": final["article"],
        "claims": final.get("claims", []),
        "verified_stances": final.get("verified_stances", []),
        "final_verdict": final.get("final_verdict", "UNCERTAIN"),
        "is_fake": final.get("is_fake", False),
        "confidence_score": final.get("confidence_score", 0.0),
        "explanation": final.get("explanation", ""),
        "error": final.get("error"),
    }
