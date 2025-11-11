from langchain_ollama import ChatOllama
from src.config import LLM_MODEL_OLLAMA, LLM_TEMPERATURE

def make_llm(model_name=None):
    model_name = model_name or LLM_MODEL_OLLAMA
    return ChatOllama(
        model=model_name,
        temperature=LLM_TEMPERATURE,
        num_ctx=4096,
        num_predict=512,
    )
