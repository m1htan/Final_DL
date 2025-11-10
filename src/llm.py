from langchain_ollama import ChatOllama
from src.config import LLM_MODEL_OLLAMA, LLM_TEMPERATURE, EMBEDDING_MODEL, CHROMA_DIR

def make_llm(model_name=None):
    """Tạo Ollama model Qwen2.5."""
    model_name = model_name or LLM_MODEL_OLLAMA
    return ChatOllama(
        model=model_name,
        temperature=LLM_TEMPERATURE,
        num_ctx=4096,
        num_predict=512,
    )
