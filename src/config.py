import os
from pathlib import Path

from dotenv import load_dotenv

# Attempt to load environment configuration from the legacy absolute path first
LEGACY_ENV = Path(r"D:\Github\Final_DL\config\.env")
if LEGACY_ENV.exists():
    load_dotenv(LEGACY_ENV, override=False)

# Fallback to the repository-local .env if available
ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_ENV = ROOT_DIR / "config" / ".env"
if LOCAL_ENV.exists():
    load_dotenv(LOCAL_ENV, override=False)

# Allow external environment variables to override the defaults above
load_dotenv(override=False)

# OLLAMA BACKEND CONFIG
LLM_MODEL_OLLAMA = os.getenv("LLM_MODEL_OLLAMA")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE"))

# EMBEDDING CONFIG
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHROMA_DIR = os.getenv("CHROMA_DIR", str(ROOT_DIR / "db" / "chroma_instruct2ds"))
