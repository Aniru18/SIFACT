# """
# SIFACT Streamlit UI — calls the FastAPI backend.

# 1. Start API (repo root):
#    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# 2. Start UI (repo root):
#    streamlit run frontend/app.py

# Optional: set BACKEND_URL (default http://127.0.0.1:8000)
# """

# from __future__ import annotations

# import os

# import httpx
# import streamlit as st

# DEFAULT_BACKEND = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

# st.set_page_config(page_title="SIFACT — Fake News Detector", page_icon="📰", layout="wide")

# st.title("📰 SIFACT — Fake News Detector")
# st.caption("Paste an article below. The backend runs extraction, verification, and synthesis.")

# backend_url = st.sidebar.text_input("Backend URL", value=DEFAULT_BACKEND, help="FastAPI base URL")
# if st.sidebar.button("Check API health"):
#     try:
#         r = httpx.get(f"{backend_url.rstrip('/')}/health", timeout=10.0)
#         r.raise_for_status()
#         st.sidebar.success(r.json())
#     except Exception as e:
#         st.sidebar.error(f"Health check failed: {e}")

# article = st.text_area(
#     "Article text",
#     height=280,
#     placeholder="Paste the news article you want to fact-check…",
# )

# col1, col2 = st.columns([1, 4])
# with col1:
#     run = st.button("Analyze", type="primary", use_container_width=True)

# if run:
#     if not article or not article.strip():
#         st.warning("Please enter some article text.")
#     else:
#         with st.spinner("Running fact-check pipeline (this may take a minute)…"):
#             try:
#                 resp = httpx.post(
#                     f"{backend_url.rstrip('/')}/api/analyze",
#                     json={"article": article.strip()},
#                     timeout=600.0,
#                 )
#                 resp.raise_for_status()
#                 data = resp.json()
#             except httpx.HTTPStatusError as e:
#                 st.error(f"API error {e.response.status_code}: {e.response.text}")
#                 st.stop()
#             except Exception as e:
#                 st.error(f"Request failed: {e}")
#                 st.stop()

#         verdict = data.get("final_verdict", "UNCERTAIN")
#         is_fake = data.get("is_fake", False)
#         conf = float(data.get("confidence_score", 0.0))

#         if verdict == "REAL":
#             st.success(f"**Verdict: REAL** — confidence {conf:.0%}")
#         elif verdict == "FAKE":
#             st.error(f"**Verdict: FAKE** — confidence {conf:.0%}")
#         else:
#             st.warning(f"**Verdict: UNCERTAIN** — confidence {conf:.0%}")

#         st.markdown("### Explanation")
#         st.write(data.get("explanation") or "—")

#         st.markdown("### Phase 1 — Extracted claims")
#         claims = data.get("claims") or []
#         if claims:
#             for c in claims:
#                 st.markdown(f"- **{c.get('id', '')}** ({c.get('type', '')}): {c.get('text', '')}")
#         else:
#             st.info("No claims returned.")

#         st.markdown("### Phase 2 — Verification")
#         stances = data.get("verified_stances") or []
#         if stances:
#             for s in stances:
#                 with st.expander(f"{s.get('claim_id', '')} — **{str(s.get('stance', '')).upper()}** (conf {float(s.get('confidence', 0)):.2f})"):
#                     st.write(s.get("evidence_summary", ""))
#                     arts = s.get("evidence_articles") or []
#                     if arts:
#                         st.caption("Evidence headlines")
#                         for a in arts[:8]:
#                             title = a.get("title", "")
#                             src = a.get("source", "")
#                             url = a.get("url", "")
#                             st.markdown(f"- [{title}]({url}) — _{src}_")
#         else:
#             st.info("No verification results.")

#         with st.expander("Raw JSON response"):
#             st.json(data)

# Spinner update code
"""
SIFACT Streamlit UI — calls the FastAPI backend.

1. Start API (repo root):
   uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

2. Start UI (repo root):
   streamlit run frontend/app.py

Optional: set BACKEND_URL (default http://127.0.0.1:8000)
"""

from __future__ import annotations

import os
import time
import threading

import httpx
import streamlit as st

# DEFAULT_BACKEND = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
DEFAULT_BACKEND = st.secrets.get("BACKEND_URL", os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"))

# Approximate phase durations (seconds) — tweak to match your backend's real timing
PHASE_DELAYS = [
    (6,  "🔍 Phase 1 — Extracting claims from article…"),
    (20, "🌐 Phase 2 — Fetching evidence from news sources…"),
    (20, "⚖️  Phase 2 — Verifying claims against evidence…"),
    (10, "🧠 Phase 3 — Synthesising final verdict…"),
]

st.set_page_config(page_title="SIFACT — Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 SIFACT — Fake News Detector  v1.0")
st.caption("Paste an article below. The backend runs extraction, verification, and synthesis.")

backend_url = st.sidebar.text_input("Backend URL", value=DEFAULT_BACKEND, help="FastAPI base URL")
if st.sidebar.button("Check API health"):
    try:
        r = httpx.get(f"{backend_url.rstrip('/')}/health", timeout=10.0)
        r.raise_for_status()
        st.sidebar.success(r.json())
    except Exception as e:
        st.sidebar.error(f"Health check failed: {e}")

article = st.text_area(
    "Article text",
    height=280,
    placeholder="Paste the news article you want to fact-check…",
)

col1, col2 = st.columns([1, 4])
with col1:
    run = st.button("Analyze", type="primary", use_container_width=True)

if run:
    if not article or not article.strip():
        st.warning("Please enter some article text.")
    else:
        result: dict = {"data": None, "error": None}

        def _call_api() -> None:
            try:
                resp = httpx.post(
                    f"{backend_url.rstrip('/')}/api/analyze",
                    json={"article": article.strip()},
                    timeout=600.0,
                )
                resp.raise_for_status()
                result["data"] = resp.json()
            except httpx.HTTPStatusError as e:
                result["error"] = f"API error {e.response.status_code}: {e.response.text}"
            except Exception as e:
                result["error"] = str(e)

        thread = threading.Thread(target=_call_api, daemon=True)
        thread.start()

        with st.status("Starting fact-check pipeline…", expanded=True) as status:
            for delay, message in PHASE_DELAYS:
                status.update(label=message)
                # Poll in short ticks so we exit early if the backend finishes fast
                for _ in range(delay * 4):          # 4 ticks per second
                    if not thread.is_alive():
                        break
                    time.sleep(0.25)
                if not thread.is_alive():
                    break

            # Final join — handles any backend that takes longer than PHASE_DELAYS sum
            thread.join()

            if result["error"]:
                status.update(label=f"❌ Failed: {result['error']}", state="error")
                st.stop()

            status.update(label="✅ Fact-check complete!", state="complete")

        data = result["data"]

        verdict = data.get("final_verdict", "UNCERTAIN")
        is_fake = data.get("is_fake", False)
        conf = float(data.get("confidence_score", 0.0))

        if verdict == "REAL":
            st.success(f"**Verdict: REAL** — confidence {conf:.0%}")
        elif verdict == "FAKE":
            st.error(f"**Verdict: FAKE** — confidence {conf:.0%}")
        else:
            st.warning(f"**Verdict: UNCERTAIN** — confidence {conf:.0%}")

        st.markdown("### Explanation")
        st.write(data.get("explanation") or "—")

        st.markdown("### Phase 1 — Extracted claims")
        claims = data.get("claims") or []
        if claims:
            for c in claims:
                st.markdown(f"- **{c.get('id', '')}** ({c.get('type', '')}): {c.get('text', '')}")
        else:
            st.info("No claims returned.")

        st.markdown("### Phase 2 — Verification")
        stances = data.get("verified_stances") or []
        if stances:
            for s in stances:
                with st.expander(f"{s.get('claim_id', '')} — **{str(s.get('stance', '')).upper()}** (conf {float(s.get('confidence', 0)):.2f})"):
                    st.write(s.get("evidence_summary", ""))
                    arts = s.get("evidence_articles") or []
                    if arts:
                        st.caption("Evidence headlines")
                        for a in arts[:8]:
                            title = a.get("title", "")
                            src = a.get("source", "")
                            url = a.get("url", "")
                            st.markdown(f"- [{title}]({url}) — _{src}_")
        else:
            st.info("No verification results.")

        with st.expander("Raw JSON response"):
            st.json(data)