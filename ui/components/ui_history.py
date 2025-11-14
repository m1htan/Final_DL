"""History handling and serialization for RAG Streamlit UI."""
import json
import streamlit as st

def ensure_session_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("question_input", "")

def serialize_history(history):
    payload = [
        {
            "question": h.get("question"),
            "response": h.get("response"),
            "sources": h.get("sources"),
            "latency": h.get("latency"),
            "timestamp": h.get("timestamp"),
            "query_args": h.get("query_args"),
        }
        for h in history
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)

def render_history(history):
    if not history:
        st.info("Chưa có lịch sử. Hãy đặt câu hỏi để bắt đầu.")
        return
    for item in reversed(history):
        title = item.get("question", "(không rõ câu hỏi)")
        ts = item.get("timestamp")
        if ts:
            title = f"{title} — {ts}"
        with st.expander(f"{title}"):
            st.markdown(item.get("response", "(Không có phản hồi)"))
            sources = item.get("sources")
            if sources:
                st.markdown("**Nguồn:**\n" + sources)
            latency = item.get("latency")
            if latency is not None:
                st.caption(f"Thời gian xử lý: {latency:.2f}s")
            if item.get("suggested_questions"):
                st.caption("Gợi ý tiếp theo: " + ", ".join(f"“{q}”" for q in item["suggested_questions"]))
