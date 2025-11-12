"""Streamlit component helpers for displaying retrieved contexts."""
from __future__ import annotations

from typing import Iterable, Mapping, Any

import streamlit as st


def render_contexts(contexts: Iterable[Mapping[str, Any]]) -> None:
    """Render retrieved context chunks with expanders.

    Args:
        contexts: Iterable of dictionaries returned by the LangGraph query node.
    """
    contexts = list(contexts or [])
    if not contexts:
        st.info("Chưa có ngữ cảnh nào được truy xuất. Hãy đặt câu hỏi ở khung bên trên.")
        return

    for idx, ctx in enumerate(contexts, start=1):
        source = ctx.get("source", "unknown.pdf")
        chunk = ctx.get("chunk_id", "?")
        score = ctx.get("score")
        header = f"[{idx}] {source} • chunk {chunk}"
        if score is not None:
            header += f" • độ liên quan ~{score:.3f}"

        with st.expander(header, expanded=idx == 1):
            snippet = ctx.get("snippet") or ctx.get("content") or "(không có nội dung)"
            st.markdown(snippet)

            metadata = ctx.get("metadata") or {}
            if metadata:
                meta_lines = []
                for key, value in metadata.items():
                    if value:
                        meta_lines.append(f"- **{key}**: {value}")
                if meta_lines:
                    st.markdown("**Metadata:**\n" + "\n".join(meta_lines))

            paper_url = metadata.get("paper_url")
            if paper_url:
                st.markdown(f"[Mở paper gốc]({paper_url})")
