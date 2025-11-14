"""Sidebar controls for the RAG Streamlit app."""
import textwrap
import streamlit as st
from .ui_history import serialize_history

def render_sidebar():
    st.header("Thiết lập truy vấn")

    top_k = st.slider("Top-k", 1, 10, 5, 1)
    context_limit = st.slider("Max token", 1000, 8000, 4000, 500)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    show_trace = st.toggle("Show detail trace", value=False)

    if st.session_state["history"]:
        if st.button("Xoá lịch sử", use_container_width=True):
            st.session_state["history"] = []
            st.success("Đã xoá lịch sử truy vấn.")
        st.download_button(
            "Tải lịch sử (.json)",
            data=serialize_history(st.session_state["history"]),
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
    return top_k, context_limit, temperature, show_trace
