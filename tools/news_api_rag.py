# """
# tools/news_api_rag.py
# News API RAG Tool – fetches evidence articles for a given claim.

# Strategy:
#   1. Try newsapi.org (primary)
#   2. Fall back to GNews API
#   3. Fall back to Google News RSS (no key required)

# Returns a list of EvidenceArticle dicts.
# """

# from __future__ import annotations

# import logging
# import time
# from typing import List, Optional
# from urllib.parse import quote

# import requests
# from tenacity import retry, stop_after_attempt, wait_exponential

# from config.settings import (
#     GNEWS_API_KEY,
#     NEWS_API_KEY,
#     NEWS_ARTICLES_PER_CLAIM,
# )
# from graph.state import EvidenceArticle

# logger = logging.getLogger(__name__)


# # ── Helper ────────────────────────────────────────────────────────────────────

# def _truncate(text: str, max_chars: int = 500) -> str:
#     return text[:max_chars].strip() if text else ""


# # ── NewsAPI.org ───────────────────────────────────────────────────────────────

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
# def _fetch_newsapi(query: str, n: int) -> List[EvidenceArticle]:
#     """Fetch from newsapi.org"""
#     if not NEWS_API_KEY:
#         raise ValueError("NEWS_API_KEY not set")

#     url = "https://newsapi.org/v2/everything"
#     params = {
#         "q": query,
#         "pageSize": n,
#         "sortBy": "relevancy",
#         "language": "en",
#         "apiKey": NEWS_API_KEY,
#     }
#     resp = requests.get(url, params=params, timeout=10)
#     resp.raise_for_status()
#     data = resp.json()

#     articles: List[EvidenceArticle] = []
#     for art in data.get("articles", []):
#         articles.append(
#             EvidenceArticle(
#                 title=art.get("title") or "",
#                 description=_truncate(art.get("description") or art.get("content") or ""),
#                 url=art.get("url") or "",
#                 source=art.get("source", {}).get("name") or "Unknown",
#                 published_at=art.get("publishedAt") or "",
#             )
#         )
#     return articles


# # ── GNews.io ──────────────────────────────────────────────────────────────────

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
# def _fetch_gnews(query: str, n: int) -> List[EvidenceArticle]:
#     """Fetch from gnews.io"""
#     if not GNEWS_API_KEY:
#         raise ValueError("GNEWS_API_KEY not set")

#     url = "https://gnews.io/api/v4/search"
#     params = {
#         "q": query,
#         "max": n,
#         "lang": "en",
#         "token": GNEWS_API_KEY,
#     }
#     resp = requests.get(url, params=params, timeout=10)
#     resp.raise_for_status()
#     data = resp.json()

#     articles: List[EvidenceArticle] = []
#     for art in data.get("articles", []):
#         articles.append(
#             EvidenceArticle(
#                 title=art.get("title") or "",
#                 description=_truncate(art.get("description") or art.get("content") or ""),
#                 url=art.get("url") or "",
#                 source=art.get("source", {}).get("name") or "Unknown",
#                 published_at=art.get("publishedAt") or "",
#             )
#         )
#     return articles


# # ── Google News RSS (free fallback) ───────────────────────────────────────────

# def _fetch_google_rss(query: str, n: int) -> List[EvidenceArticle]:
#     """Fetch from Google News RSS – no API key needed."""
#     import xml.etree.ElementTree as ET

#     encoded = quote(query)
#     url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
#     headers = {"User-Agent": "Mozilla/5.0 (compatible; SIFACT/1.0)"}

#     resp = requests.get(url, headers=headers, timeout=10)
#     resp.raise_for_status()

#     root = ET.fromstring(resp.content)
#     channel = root.find("channel")
#     if channel is None:
#         return []

#     articles: List[EvidenceArticle] = []
#     for item in list(channel.findall("item"))[:n]:
#         title = item.findtext("title") or ""
#         link  = item.findtext("link") or ""
#         desc  = item.findtext("description") or ""
#         pub   = item.findtext("pubDate") or ""
#         src_el = item.find("source")
#         source = src_el.text if src_el is not None else "Google News"

#         articles.append(
#             EvidenceArticle(
#                 title=title,
#                 description=_truncate(desc, 400),
#                 url=link,
#                 source=source,
#                 published_at=pub,
#             )
#         )
#     return articles


# # ── Public API ────────────────────────────────────────────────────────────────

# def fetch_evidence(claim_text: str, n: int = NEWS_ARTICLES_PER_CLAIM) -> List[EvidenceArticle]:
#     """
#     Main entry-point used by the verification node.
#     Tries providers in order: NewsAPI → GNews → Google RSS.
#     """
#     # Build a concise keyword query from the claim
#     # (keep first 10 words to avoid query-too-long errors)
#     query_words = claim_text.split()[:10]
#     query = " ".join(query_words)

#     # 1. NewsAPI
#     if NEWS_API_KEY:
#         try:
#             articles = _fetch_newsapi(query, n)
#             if articles:
#                 logger.info("NewsAPI returned %d articles for claim", len(articles))
#                 return articles
#         except Exception as exc:
#             logger.warning("NewsAPI failed (%s), trying GNews…", exc)

#     # 2. GNews
#     if GNEWS_API_KEY:
#         try:
#             articles = _fetch_gnews(query, n)
#             if articles:
#                 logger.info("GNews returned %d articles for claim", len(articles))
#                 return articles
#         except Exception as exc:
#             logger.warning("GNews failed (%s), trying Google RSS…", exc)

#     # 3. Google RSS (always available)
#     try:
#         articles = _fetch_google_rss(query, n)
#         logger.info("Google RSS returned %d articles for claim", len(articles))
#         return articles
#     except Exception as exc:
#         logger.error("All news sources failed: %s", exc)
#         return []



# with only Google News RSS

"""
tools/news_api_rag.py
News API RAG Tool – fetches evidence articles for a given claim via Google News RSS.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import List
from urllib.parse import quote

import requests

from config.settings import NEWS_ARTICLES_PER_CLAIM
from graph.state import EvidenceArticle

logger = logging.getLogger(__name__)


def _truncate(text: str, max_chars: int = 500) -> str:
    return text[:max_chars].strip() if text else ""


def fetch_evidence(claim_text: str, n: int = NEWS_ARTICLES_PER_CLAIM) -> List[EvidenceArticle]:
    """
    Fetches evidence articles for a given claim via Google News RSS.
    No API key required.
    """
    query = " ".join(claim_text.split()[:10])
    encoded = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SIFACT/1.0)"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        channel = root.find("channel")
        if channel is None:
            logger.warning("Google RSS returned no channel element")
            return []

        articles: List[EvidenceArticle] = []
        for item in list(channel.findall("item"))[:n]:
            src_el = item.find("source")
            articles.append(
                EvidenceArticle(
                    title=item.findtext("title") or "",
                    description=_truncate(item.findtext("description") or "", 400),
                    url=item.findtext("link") or "",
                    source=src_el.text if src_el is not None else "Google News",
                    published_at=item.findtext("pubDate") or "",
                )
            )

        logger.info("Google RSS returned %d articles for claim", len(articles))
        return articles

    except Exception as exc:
        logger.error("Google RSS fetch failed: %s", exc)
        return []