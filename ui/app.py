"""Streamlit interface for the RAG-based question answering workflow."""
from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from components.analytics import render_context_summary
from components.context_viewer import render_contexts

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import build_graph

SAMPLE_QUESTIONS = [
    "Mô hình RAG được giới thiệu ở hội nghị nào?",
    "Những hướng nghiên cứu tiêu biểu về dịch máy trong bộ dữ liệu này?",
    "Các tác giả Việt Nam có đóng góp gì nổi bật?",
]


@st.cache_resource(show_spinner=False)
def _load_query_graph():
    """Compile the LangGraph pipeline for query mode once per session."""

    return build_graph(force_mode="query")


def _invoke_query(
    question: str, top_k: int, context_limit: int, temperature: float
) -> Dict[str, Any]:
    graph = _load_query_graph()
    initial_state: Dict[str, Any] = {
        "user_input": question,
        "mode": "query",
        "trace": [],
        "top_k": top_k,
        "max_context_chars": context_limit,
        "llm_temperature": temperature,
        "ui_request": True,
    }
    return graph.invoke(initial_state)


def _render_history(history: List[Dict[str, Any]]) -> None:
    if not history:
        st.info("Chưa có lịch sử. Hãy đặt câu hỏi để bắt đầu.")
        return

    for item in reversed(history):
        title = item.get("question", "(không rõ câu hỏi)")
        ts = item.get("timestamp")
        if ts:
            title = f"{title} — {ts}"
        with st.expander(f"❓ {title}"):
            st.markdown(item.get("response", "(Không có phản hồi)"))
            sources = item.get("sources")
            if sources:
                st.markdown("**Nguồn:**\n" + sources)
            latency = item.get("latency")
            if latency is not None:
                st.caption(f"Thời gian xử lý: {latency:.2f}s")
            if item.get("suggested_questions"):
                st.caption(
                    "Gợi ý tiếp theo: "
                    + ", ".join(f"“{q}”" for q in item["suggested_questions"])
                )


def _serialize_history(history: List[Dict[str, Any]]) -> str:
    payload = []
    for item in history:
        payload.append(
            {
                "question": item.get("question"),
                "response": item.get("response"),
                "sources": item.get("sources"),
                "latency": item.get("latency"),
                "timestamp": item.get("timestamp"),
                "query_args": item.get("query_args"),
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _ensure_session_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("question_input", "")


def _render_suggestion_buttons(suggestions: List[str]) -> None:
    if not suggestions:
        return

    st.markdown("#### Gợi ý câu hỏi tiếp theo")
    cols = st.columns(len(suggestions))
    for idx, suggestion in enumerate(suggestions):
        if cols[idx].button(f"➡️ {suggestion}", key=f"suggestion_{idx}"):
            st.session_state["question_input"] = suggestion
            st.rerun()


def _render_samples() -> None:
    st.markdown("#### Gợi ý nhanh")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for idx, sample in enumerate(SAMPLE_QUESTIONS):
        if cols[idx].button(sample, key=f"sample_{idx}"):
            st.session_state["question_input"] = sample
            st.rerun()


def main():
    st.set_page_config(
        page_title="RAG QA over Instruct2DS",
        page_icon="🤖",
        layout="wide",
    )
    _ensure_session_state()

    st.title("RAG-Based Question Answering for Online PDF Documents")
    st.caption(
        "Hệ thống hỏi-đáp tiếng Việt sử dụng kiến trúc Retrieval-Augmented Generation "
        "với LangGraph, ChromaDB, embedding `Alibaba-NLP/gte-Qwen2-1.5B-instruct` "
        "và mô hình điều phối/sinh trả lời `qwen2.5:7b` chạy qua Ollama."
    )

    with st.sidebar:
        st.header("Thiết lập truy vấn")
        top_k = st.slider("Số đoạn context (top-k)", min_value=1, max_value=10, value=5, step=1)
        context_limit = st.slider(
            "Giới hạn độ dài ngữ cảnh (ký tự)", min_value=1000, max_value=8000, value=4000, step=500
        )
        temperature = st.slider(
            "Nhiệt độ trả lời (0 = bảo thủ, 1.0 = sáng tạo)",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            value=0.1,
        )
        show_trace = st.toggle("Hiển thị trace chi tiết", value=False)

        if st.session_state["history"]:
            if st.button("🗑️ Xoá lịch sử", use_container_width=True):
                st.session_state["history"] = []
                st.success("Đã xoá lịch sử truy vấn.")

            st.download_button(
                "💾 Tải lịch sử (.json)",
                data=_serialize_history(st.session_state["history"]),
                file_name="rag_query_history.json",
                mime="application/json",
                use_container_width=True,
            )

        st.divider()
        st.markdown(
            textwrap.dedent(
                """
                **Quy trình**

                1. Câu hỏi được nhúng bằng mô hình `Alibaba-NLP/gte-Qwen2-1.5B-instruct`.
                2. ChromaDB truy xuất top-k đoạn văn dựa trên cosine similarity.
                3. Các đoạn được nén để phù hợp giới hạn context và gửi tới `qwen2.5:7b` (Ollama).
                4. Câu trả lời được hậu xử lý với nguồn trích dẫn rõ ràng.
                """
            )
        )

    _render_samples()

    question = st.text_area(
        "Nhập câu hỏi của bạn",
        height=160,
        key="question_input",
        placeholder="Ví dụ: \"Mô hình RAG được giới thiệu ở hội nghị nào?\"",
    )
    submit = st.button("Truy vấn", type="primary", use_container_width=True)

    latest_result: Dict[str, Any] | None = None
    if submit:
        clean_question = question.strip()
        if not clean_question:
            st.warning("Vui lòng nhập câu hỏi trước khi truy vấn.")
        else:
            with st.spinner("Đang truy vấn và sinh câu trả lời..."):
                latest_result = _invoke_query(
                    clean_question,
                    top_k=top_k,
                    context_limit=context_limit,
                    temperature=temperature,
                )

            if latest_result:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                entry = {
                    "question": clean_question,
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
                    "timestamp": timestamp,
                }
                st.session_state["history"].append(entry)
                st.session_state["history"] = st.session_state["history"][-50:]
                latest_result = entry

    if latest_result is None and st.session_state["history"]:
        latest_result = st.session_state["history"][-1]

    tabs = st.tabs(["Kết quả", "Ngữ cảnh", "Phân tích", "Lịch sử"])

    with tabs[0]:
        if latest_result:
            response_md = latest_result.get("response", "")
            if response_md:
                st.markdown(response_md)

            meta_cols = st.columns(4)
            latency_value = latest_result.get("latency")
            if latency_value is None:
                latency_value = latest_result.get("latency_seconds", 0.0)
            meta_cols[0].metric("Thời gian", f"{latency_value:.2f}s")
            meta_cols[1].metric("Số đoạn", len(latest_result.get("retrieved_docs", [])))
            qa_args = latest_result.get("query_args", {})
            meta_cols[2].metric("Top-k", qa_args.get("top_k", top_k))

            _render_suggestion_buttons(latest_result.get("suggested_questions", []))

            if show_trace:
                st.markdown("### Trace LangGraph")
                for trace_line in latest_result.get("trace", []):
                    st.code(trace_line)
        else:
            st.info("Chưa có câu trả lời nào. Hãy nhập câu hỏi để bắt đầu.")

    with tabs[1]:
        if latest_result:
            render_contexts(latest_result.get("retrieved_docs", []))
        else:
            st.info("Chưa có ngữ cảnh để hiển thị.")

    with tabs[2]:
        render_context_summary(latest_result.get("context_summary") if latest_result else None)

    with tabs[3]:
        st.subheader("Lịch sử truy vấn")
        _render_history(st.session_state["history"])


if __name__ == "__main__":
    main()