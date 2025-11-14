"""Main UI helpers for Streamlit RAG app (samples and suggestion buttons)."""
import streamlit as st

def render_samples(samples):
    st.markdown("#### Gợi ý nhanh")
    cols = st.columns(len(samples))
    for i, sample in enumerate(samples):
        if cols[i].button(sample, key=f"sample_{i}"):
            st.session_state["question_input"] = sample
            st.rerun()

def render_suggestion_buttons(suggestions):
    if not suggestions:
        return
    st.markdown("#### Gợi ý câu hỏi tiếp theo")
    cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        if cols[i].button(f"{s}", key=f"suggestion_{i}"):
            st.session_state["question_input"] = s
            st.rerun()
