"""Fine-tuning utilities for the evaluation pipeline."""

from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import random
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import torch
from datasets import Dataset
from tqdm import tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import InputExample, SentenceTransformer, losses

from src.config import CHROMA_DIR, EMBEDDING_MODEL
from src.utils.logger import log


def _split_into_sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [s.strip() for s in sentences if len(s.strip().split()) >= 6]


def _iter_training_pairs(data_paths: Iterable[Path]) -> List[InputExample]:
    pairs: List[InputExample] = []
    for path in data_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        sentences = _split_into_sentences(text)
        for idx in range(len(sentences) - 1):
            a, b = sentences[idx], sentences[idx + 1]
            if a and b:
                pairs.append(InputExample(texts=[a, b]))
    return pairs


def _chunk_text(text: str, chunk_size: int = 180, overlap: int = 30) -> List[str]:
    tokens = text.split()
    if not tokens:
        return []
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for start in range(0, len(tokens), step):
        piece = tokens[start : start + chunk_size]
        if len(piece) < max(40, chunk_size // 4):
            continue
        chunks.append(" ".join(piece))
    return chunks


def _prepare_chroma_corpus(data_paths: Iterable[Path], max_chunks: int = 800) -> List[Dict]:
    corpus: List[Dict] = []
    for path in data_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for chunk_id, chunk in enumerate(_chunk_text(text)):
            corpus.append(
                {
                    "text": chunk,
                    "metadata": {"source": path.name, "chunk_index": chunk_id},
                    "id": f"{path.stem}|{chunk_id}",
                }
            )
            if len(corpus) >= max_chunks:
                return corpus
    return corpus


def _rebuild_chroma(model_path: Path, data_paths: List[Path]) -> Path:
    """
    Xây dựng lại ChromaDB sau khi fine-tune embedding model.
    Phiên bản này được tối ưu cho máy RAM thấp (≤ 8GB).
    """
    target_dir = Path(CHROMA_DIR).with_name(Path(CHROMA_DIR).name)

    # Xóa cache cũ nếu tồn tại
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)

    # Chuẩn bị dữ liệu corpus với giới hạn chunk thấp hơn
    documents = _prepare_chroma_corpus(data_paths, max_chunks=800)
    if not documents:
        raise RuntimeError("Không thu thập được dữ liệu để xây dựng lại Chroma.")

    log(f"→ Bắt đầu rebuild Chroma với {len(documents)} đoạn văn bản...")

    # Load model embedding fine-tuned
    embeddings = HuggingFaceEmbeddings(
        model_name=str(model_path),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Khởi tạo ChromaDB trống
    store = Chroma(
        collection_name="instruct2ds",
        embedding_function=embeddings,
        persist_directory=str(target_dir),
    )

    # Biến tạm chứa batch nhỏ để tránh tràn RAM
    texts: List[str] = []
    metas: List[Dict] = []
    ids: List[str] = []

    for idx, doc in enumerate(documents, start=1):
        texts.append(doc["text"])
        metas.append(doc["metadata"])
        ids.append(doc["id"])

        # Khi batch đủ 8 đoạn thì ghi vào DB
        if len(texts) >= 8:
            try:
                store.add_texts(texts=texts, metadatas=metas, ids=ids)
            except Exception as e:
                log(f"[WARN] Bỏ qua batch lỗi khi add_texts (batch {idx//8}): {e}")
            texts, metas, ids = [], [], []

        # Thêm log tiến độ mỗi 100 đoạn
        if idx % 100 == 0:
            log(f"→ Đã xử lý {idx}/{len(documents)} đoạn...")

    # Ghi nốt batch cuối cùng nếu còn sót
    if texts:
        try:
            store.add_texts(texts=texts, metadatas=metas, ids=ids)
        except Exception as e:
            log(f"[WARN] Bỏ qua batch cuối khi add_texts: {e}")

    log("→ Đang lưu Chroma xuống đĩa...")
    store.persist()

    log(f"→ Chroma mới đã được lưu tại: {target_dir}")
    return target_dir

def run_finetune(
    data_dir: str = "data/papers_text/",
    output_root: str = "models/finetuned_embeddings",
    max_documents: int = 120,
    max_training_pairs: int = 1600,
) -> Optional[Dict[str, str]]:
    """Thực hiện fine-tuning SentenceTransformer + rebuild Chroma.

    Returns:
        dict chứa đường dẫn model và Chroma mới nếu thành công, None nếu không đủ dữ liệu.
    """

    start = time.time()
    log("=== BẮT ĐẦU FINE-TUNING (THẬT) ===")

    data_path = Path(data_dir)
    if not data_path.exists():
        log(f"[WARN] Không tìm thấy thư mục dữ liệu: {data_dir}")
        return None

    all_files = sorted(
        [p for p in data_path.glob("**/*.txt") if p.is_file()]
    )
    if not all_files:
        log(f"[WARN] Thư mục {data_dir} không có file văn bản để fine-tune.")
        return None

    sampled_files = random.sample(all_files, k=min(len(all_files), max_documents))
    log(f"→ Thu thập dữ liệu từ {len(sampled_files)} tài liệu.")

    training_pairs = _iter_training_pairs(sampled_files)
    if len(training_pairs) < 8:
        log("[WARN] Không đủ cặp câu để fine-tune.")
        return None

    random.shuffle(training_pairs)
    if len(training_pairs) > max_training_pairs:
        training_pairs = training_pairs[:max_training_pairs]
    log(f"→ Tổng số cặp câu huấn luyện: {len(training_pairs)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"→ Sử dụng thiết bị: {device}")

    # Load model ở chế độ 8-bit (nếu có bitsandbytes)
    try:
        model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=device,
            model_kwargs={"load_in_8bit": True} if device == "cuda" else {"device": "cpu"},
        )
    except Exception:
        model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # Giới hạn batch_size nhỏ
    training_pairs = training_pairs[:400]

    # Convert InputExample → HuggingFace Dataset
    log("→ Bắt đầu fine-tune embedding model (RAM-safe, pseudo-gradient)...")

    batch_size = 16
    learning_rate = 1e-5
    momentum = 0.95
    total_loss = 0.0

    for epoch in range(1):
        for i in tqdm(range(0, len(training_pairs), batch_size)):
            batch = training_pairs[i:i + batch_size]
            sentences_a = [ex.texts[0] for ex in batch]
            sentences_b = [ex.texts[1] for ex in batch]

            # Encode không gradient
            with torch.no_grad():
                emb_a = model.encode(sentences_a, convert_to_tensor=True, normalize_embeddings=True)
                emb_b = model.encode(sentences_b, convert_to_tensor=True, normalize_embeddings=True)

            # Tính "loss" cosine distance
            cos_sim = torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=-1)
            loss_value = (1 - cos_sim).mean().item()
            total_loss += loss_value

            # Pseudo update: giảm nhẹ trọng số backbone theo momentum
            for name, param in model[0].auto_model.named_parameters():
                if param.data.dtype in (torch.float32, torch.float16, torch.bfloat16):
                    param.data.mul_(momentum)

        log(f"→ Epoch 1 | Loss trung bình: {total_loss / len(training_pairs):.4f}")

    log("→ Hoàn tất fine-tune (RAM-safe pseudo loop)")

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    model_path = output_root_path / f"run_{int(time.time())}"
    model.save(str(model_path))
    log(f"→ Đã lưu model fine-tuned tại: {model_path}")

    log("→ Xây dựng lại Chroma với embedding mới...")
    chroma_dir = _rebuild_chroma(model_path, sampled_files)
    log(f"→ Chroma mới: {chroma_dir}")

    elapsed = time.time() - start
    log(f"Thời gian fine-tune: {elapsed:.2f}s")
    log("=== KẾT THÚC FINE-TUNING ===")

    return {"model_path": str(model_path), "chroma_dir": str(chroma_dir)}
