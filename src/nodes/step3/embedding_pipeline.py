import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import time
from pathlib import Path
from typing import List, Optional

import fitz
from tqdm import tqdm
from langchain_chroma import Chroma
from src.utils.logger import log
from src.nodes.step3.text_chunker import chunk_text
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import torch

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "papers_raw"
TEXT_DIR = DATA_DIR / "papers_text"
TEXT_DIR.mkdir(parents=True, exist_ok=True)

DB_DIR = Path("db/chroma_instruct2ds")
COLLECTION_NAME = "instruct2ds"


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Trích toàn bộ text từ PDF bằng PyMuPDF, trả về string."""
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text("text")
        return text.strip()
    except Exception as e:
        log(f"[ERROR] Không đọc được {pdf_path.name}: {e}")
        return ""


def _device_candidates() -> List[str]:
    """Xác định thứ tự ưu tiên các thiết bị có thể dùng cho embedding."""
    candidates: List[str] = []

    if torch is None:
        return ["cpu"]

    if torch.cuda.is_available():
        candidates.append("cuda")

    # Ưu tiên CPU trước MPS để tránh treo model khi backend MPS chưa ổn định
    candidates.append("cpu")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        candidates.append("mps")
    return candidates


def embedding_pipeline_node(state: dict) -> dict:
    """Thực hiện trích xuất text và embedding toàn bộ PDF vào Chroma."""
    start_time = time.time()
    log("=== BẮT ĐẦU BƯỚC 3: TRÍCH XUẤT & EMBEDDING ===")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    total_pdfs = len(pdf_files)
    if total_pdfs == 0:
        log("[WARN] Không tìm thấy PDF nào trong data/papers_raw/. Dừng pipeline.")
        state["response"] = "Không có file PDF nào để trích xuất."
        return state

    log(f"Phát hiện {total_pdfs} file PDF cần xử lý.")
    text_files_existing = list(TEXT_DIR.glob("*.txt"))
    log(f"Đã có {len(text_files_existing)} file text trích xuất sẵn trong papers_text/.")

    device_candidates = _device_candidates()
    model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    embeddings = None
    last_error: Optional[Exception] = None

    for device in device_candidates:
        log(f"Đang khởi tạo model {model_name} trên thiết bị '{device}'.")
        try:
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as err:
            last_error = err
            log(f"[WARN] Không khởi tạo được trên '{device}': {err}")
            continue

        log(f"Đã khởi tạo xong {model_name} trên '{device}'.")
        break

    if embeddings is None:
        raise RuntimeError(
            f"Không thể khởi tạo embedding model {model_name} trên các thiết bị {device_candidates}."
        ) from last_error
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
        embedding_function=embeddings
    )

    total_new_texts = 0
    total_chunks = 0

    for pdf_path in tqdm(pdf_files, desc="Xử lý PDF", unit="file"):
        log(f"Đang xử lý {pdf_path.name} ...")
        txt_path = TEXT_DIR / (pdf_path.stem + ".txt")

        if txt_path.exists():
            log(f"Bỏ qua {pdf_path.name} (đã có text).")
            continue

        try:
            text = extract_text_from_pdf(pdf_path)
        except Exception as e:
            log(f"[ERROR] Lỗi khi đọc {pdf_path.name}: {e}")
            continue

        if not text:
            log(f"[WARN] {pdf_path.name} trống hoặc không thể trích xuất.")
            continue

        txt_path.write_text(text, encoding="utf-8")
        total_new_texts += 1
        log(f"→ Đã trích {len(text)} ký tự từ {pdf_path.name}")

        chunks = chunk_text(text)
        if isinstance(chunks[0], dict):
            texts = [c.get("text", "") for c in chunks]
            metas = []
            for i, c in enumerate(chunks):
                metas.append({
                    "source": pdf_path.name,
                    "chunk_id": c.get("chunk_id", i + 1)
                })
        else:
            texts = chunks
            metas = [{"source": pdf_path.name, "chunk_id": i + 1} for i in range(len(chunks))]

        ids = [f"{pdf_path.stem}_{i}" for i in range(len(texts))]

        vectordb.add_texts(texts, ids=ids, metadatas=metas)
        total_chunks += len(texts)
        log(f"→ Đã lưu {len(texts)} đoạn từ {pdf_path.name} vào Chroma.")

    elapsed = time.time() - start_time
    log("=== HOÀN TẤT EMBEDDING PIPELINE ===")
    log(f"Tổng số PDF quét: {total_pdfs}")
    log(f"Số file text mới trích xuất: {total_new_texts}")
    log(f"Tổng số chunk đã embed: {total_chunks}")
    log(f"Thư mục DB: {DB_DIR}")
    log(f"Thời gian thực thi: {elapsed:.2f}s")

    state["response"] = (
        f"Đã trích xuất và embedding {total_new_texts} PDF ({total_chunks} chunks) "
        f"vào collection '{COLLECTION_NAME}'."
    )
    state.setdefault("trace", []).append(f"[embed] {total_new_texts} PDF, {total_chunks} chunks, lưu tại {DB_DIR}")
    return state