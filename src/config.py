import os
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv(dotenv_path='D:\Github\Final_DL\config\.env')

# OLLAMA BACKEND CONFIG
USE_BACKEND = os.getenv("USE_BACKEND")
LLM_MODEL_OLLAMA = os.getenv("LLM_MODEL_OLLAMA")
LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE")

# EMBEDDING CONFIG
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHROMA_DIR = os.getenv("CHROMA_DIR")
