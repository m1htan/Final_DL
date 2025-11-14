"""Factories for LangChain LLM clients used across the project."""

from __future__ import annotations

from typing import Optional

from langchain_ollama import ChatOllama

from src.config import LLM_MODEL_OLLAMA, LLM_TEMPERATURE

def _coerce_temperature(value: Optional[float]) -> float:
    """Convert a user supplied temperature into a safe float range."""
    if value is None:
        return LLM_TEMPERATURE
    try:
        temp = float(value)
    except (TypeError, ValueError):
        return LLM_TEMPERATURE
    # Clamp to a reasonable range supported by Ollama to avoid errors.
    return max(0.0, min(1.5, temp))

def make_llm(temperature: Optional[float] = None):
    """Create a ChatOllama client configured for the project defaults."""

    model = LLM_MODEL_OLLAMA
    temp = _coerce_temperature(temperature)
    return ChatOllama(
        model=model,
        temperature=temp,
        num_predict=512,
        num_ctx=2048,
        num_gpu=1,
        low_vram=True
    )
