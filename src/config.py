import os
from pathlib import Path
from datetime import datetime
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

LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG = LOG_DIR / "run.log"

def _log_to_file(message: str):
    """Ghi thông tin vào logs/run.log với timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

# OLLAMA BACKEND CONFIG
LLM_MODEL_OLLAMA = os.getenv("LLM_MODEL_OLLAMA")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE"))

# EMBEDDING CONFIG
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHROMA_DIR = os.getenv("CHROMA_DIR", str(ROOT_DIR / "db" / "chroma_instruct2ds"))

if EMBEDDING_MODEL and ("<timestamp>" in EMBEDDING_MODEL or EMBEDDING_MODEL.lower() == "latest"):
    base_dir = ROOT_DIR / "models" / "finetuned_embeddings"
    if base_dir.exists():
        # Chọn model có thời gian sửa mới nhất
        latest_run = max(base_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, default=None)
        if latest_run:
            EMBEDDING_MODEL = str(latest_run)
            print(f"[AUTO] Đang sử dụng model fine-tuned mới nhất: {EMBEDDING_MODEL}")
        else:
            print("[WARN] Không tìm thấy model fine-tuned nào trong models/finetuned_embeddings.")
    else:
        print("[WARN] Thư mục models/finetuned_embeddings không tồn tại, dùng model mặc định.")

chroma_default = ROOT_DIR / "db" / "chroma_instruct2ds"
chroma_finetuned = ROOT_DIR / "db" / "chroma_instruct2ds_finetuned"

if chroma_finetuned.exists() and any(chroma_finetuned.iterdir()):
    CHROMA_DIR = str(chroma_finetuned)
    print(f"[AUTO] Đang sử dụng Chroma fine-tuned: {CHROMA_DIR}")
else:
    CHROMA_DIR = str(chroma_default)
    print(f"[AUTO] Không tìm thấy Chroma fine-tuned → dùng mặc định: {CHROMA_DIR}")
