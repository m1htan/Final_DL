"""Analytics components for the Streamlit RAG dashboard."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import streamlit as st


def _render_table(title: str, rows: Sequence[Mapping[str, Any]], column_labels: Mapping[str, str]) -> None:
    if not rows:
        return

    st.markdown(f"#### {title}")
    normalized = []
    for row in rows:
        normalized.append({column_labels.get(k, k): v for k, v in row.items()})
    st.table(normalized)


def render_context_summary(summary: Mapping[str, Any] | None) -> None:
    """Display aggregated metadata about the retrieved documents."""

    if not summary:
        st.info("Chưa có thống kê nào vì bạn chưa chạy truy vấn.")
        return

    total_chunks = summary.get("total_chunks", 0)
    unique_sources = summary.get("unique_sources", 0)
    num_conferences = len(summary.get("top_conferences", []))

    m1, m2, m3 = st.columns(3)
    m1.metric("Đoạn context", total_chunks)
    m2.metric("Nguồn duy nhất", unique_sources)
    m3.metric("Hội nghị khác nhau", num_conferences)

    _render_table(
        "Hội nghị nổi bật",
        summary.get("top_conferences", []),
        {"conference": "Hội nghị", "count": "Số đoạn"},
    )

    _render_table(
        "Các năm được nhắc đến",
        summary.get("top_years", []),
        {"year": "Năm", "count": "Số đoạn"},
    )

    _render_table(
        "Tác giả xuất hiện thường xuyên",
        summary.get("top_authors", []),
        {"author": "Tác giả", "count": "Số lần"},
    )

    titles = summary.get("top_titles") or []
    if titles:
        st.markdown("#### Tên bài báo liên quan")
        for item in titles:
            conf = item.get("conference")
            suffix = f" — {conf}" if conf else ""
            st.markdown(f"- {item.get('title')} {suffix}")

    links = summary.get("paper_links") or []
    if links:
        st.markdown("#### Liên kết xem nhanh")
        for link in links:
            conf = link.get("conference")
            label = link.get("title") or link.get("conference")
            if conf:
                label = f"{label} ({conf})"
            st.markdown(f"- [{label}]({link.get('url')})")
