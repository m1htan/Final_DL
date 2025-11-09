import os
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv(dotenv_path='D:\Github\Final_DL_Codex\config\.env')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

if not GEMINI_API_KEY:
    raise RuntimeError("Thiếu GEMINI_API_KEY trong .env")
