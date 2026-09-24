from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("FINANCE_RAG_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("FINANCE_RAG_API_KEY", "")

st.set_page_config(page_title="Finance RAG — Indian Financial Documents", page_icon="📊", layout="wide")

st.title("📊 Finance RAG")
st.caption("Ask questions about Indian annual reports, RBI circulars, SEBI regulations, and budget documents.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Settings")
    api_url = st.text_input("API URL", value=API_URL)
    st.caption(f"Backend health: checked on query")
    if st.button("Clear history"):
        st.session_state.history = []

query = st.chat_input("e.g. Compare HDFC Bank and ICICI Bank's net interest margins")


def _badge(score: float | None) -> str:
    if score is None:
        return "⚪ n/a"
    if score >= 0.8:
        return f"🟢 {score:.0%} faithful"
    if score >= 0.5:
        return f"🟡 {score:.0%} faithful"
    return f"🔴 {score:.0%} faithful"


def render_answer(turn: dict) -> None:
    st.write(turn["answer"])
    cols = st.columns(3)
    cols[0].metric("Query type", turn["query_type"])
    cols[1].markdown(f"**Faithfulness:** {_badge(turn.get('faithfulness_score'))}")
    if turn.get("latency_ms"):
        cols[2].metric("Latency", f"{turn['latency_ms']:.0f} ms")

    if turn.get("unsupported_claim_count", 0) > 0:
        st.warning(f"⚠️ {turn['unsupported_claim_count']} claim(s) were not clearly "
                   "supported by the cited source — treat those parts with caution.")

    citations = turn.get("citations") or []
    if citations:
        with st.expander(f"📎 Sources ({len(citations)} citation(s))"):
            for c in citations:
                icon = "✅" if c.get("supported") else "❓"
                page = f", p.{c['page_number']}" if c.get("page_number") else ""
                st.markdown(f"{icon} *{c['claim']}*  \n&nbsp;&nbsp;&nbsp;&nbsp;— {c['source_name']}{page}")


# Re-render full history each run (Streamlit's standard chat pattern).
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        render_answer(turn)

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating a grounded answer..."):
            try:
                headers = {"X-API-Key": API_KEY} if API_KEY else {}
                resp = requests.post(
                    f"{api_url}/query", json={"query": query}, headers=headers, timeout=120
                )
                resp.raise_for_status()
                turn = resp.json()
                turn["query"] = query
                render_answer(turn)
                st.session_state.history.append(turn)
            except requests.RequestException as exc:
                st.error(f"Request to backend failed: {exc}")
