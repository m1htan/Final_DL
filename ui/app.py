"""Streamlit interface entrypoint for the RAG-based question answering workflow."""
from __future__ import annotations
import sys, os, json, textwrap
from datetime import datetime
from typing import Any, Dict
import streamlit as st

# Add repo root for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import build_graph
from components.analytics import render_context_summary
from components.context_viewer import render_contexts
from components.ui_sidebar import render_sidebar
from components.ui_history import ensure_session_state, render_history, serialize_history
from components.ui_main_helper import render_samples, render_suggestion_buttons

@st.cache_resource(show_spinner=False)
def _load_query_graph():
    """Compile the LangGraph pipeline for query mode once per session."""
    return build_graph(force_mode="query")


def _invoke_query(
    question: str, top_k: int, context_limit: int, temperature: float
) -> Dict[str, Any]:
    try:
        graph = _load_query_graph()
        state = {
            "user_input": question,
            "mode": "query",
            "trace": [],
            "top_k": top_k,
            "max_context_chars": context_limit,
            "llm_temperature": temperature,
            "ui_request": True,
        }
        return graph.invoke(state)
    except Exception as e:
        return {"response": f"Lỗi pipeline: {e}", "trace": [str(e)]}


def main():
    st.set_page_config(page_title="RAG QA over Instruct2DS", page_icon="🤖", layout="wide")
    ensure_session_state()

    st.title("RAG-Based Question Answering for Online PDF Documents")
    st.caption(
        "Hệ thống hỏi-đáp tiếng Việt sử dụng kiến trúc Retrieval-Augmented Generation "
        "với LangGraph, ChromaDB, embedding `Alibaba-NLP/gte-Qwen2-1.5B-instruct` "
        "và mô hình điều phối/sinh trả lời `qwen2.5:7b` chạy qua Ollama."
    )

    # Sidebar
    with st.sidebar:
        top_k, context_limit, temperature, show_trace = render_sidebar()

    question = st.text_area(
        "Nhập câu hỏi của bạn",
        height=160,
        key="question_input",
        placeholder="Ví dụ: \"Mô hình RAG được giới thiệu ở hội nghị nào?\"",
    )
    submit = st.button("Truy vấn", type="primary", use_container_width=True)

    latest_result: Dict[str, Any] | None = None
    if submit:
        clean_q = question.strip()
        if not clean_q:
            st.warning("Vui lòng nhập câu hỏi trước khi truy vấn.")
        else:
            with st.spinner("Đang truy vấn và sinh câu trả lời..."):
                latest_result = _invoke_query(clean_q, top_k, context_limit, temperature)
            if latest_result:
                entry = {
                    "question": clean_q,
                    "response": latest_result.get("response"),
                    "sources": latest_result.get("sources_markdown"),
                    "latency": latest_result.get("latency_seconds"),
                    "retrieved_docs": list(latest_result.get("retrieved_docs", [])),
                    "trace": list(latest_result.get("trace", [])),
                    "answer_model": latest_result.get("answer_model"),
                    "query_args": dict(latest_result.get("query_args", {})),
                    "context_summary": latest_result.get("context_summary"),
                    "suggested_questions": latest_result.get("suggested_questions", []),
                    "llm_temperature": latest_result.get("llm_temperature"),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.session_state["history"].append(entry)
                st.session_state["history"] = st.session_state["history"][-50:]
                latest_result = entry

    if latest_result is None and st.session_state["history"]:
        latest_result = st.session_state["history"][-1]

    tabs = st.tabs(["Kết quả", "Ngữ cảnh", "Phân tích", "Lịch sử"])

    with tabs[0]:
        if latest_result:
            resp = latest_result.get("response", "")
            if resp:
                st.markdown(resp)
                if latest_result.get("retrieved_docs"):
                    with st.expander("Ngữ cảnh được dùng"):
                        for d in latest_result["retrieved_docs"]:
                            st.markdown(f"- **{d.metadata.get('source', '')}**: {d.page_content[:300]}...")
            cols = st.columns(4)
            latency = latest_result.get("latency") or latest_result.get("latency_seconds", 0.0)
        else:
            st.info("Chưa có câu trả lời nào. Hãy nhập câu hỏi để bắt đầu.")

    with tabs[1]:
        render_contexts(latest_result.get("retrieved_docs", []) if latest_result else [])

    with tabs[2]:
        render_context_summary(latest_result.get("context_summary") if latest_result else None)

    with tabs[3]:
        st.subheader("Lịch sử truy vấn")
        render_history(st.session_state["history"])


if __name__ == "__main__":
    main()