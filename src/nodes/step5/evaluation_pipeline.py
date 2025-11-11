import time
import csv
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re
import random
from typing import List, Dict

from src.utils.logger import log
from src.config import EMBEDDING_MODEL, CHROMA_DIR
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from src.nodes.step5.finetune_manager import run_finetune
from src.llm import make_llm

# Load embeddings & DB
def load_embeddings_and_db():
    """Load embedding + ChromaDB một lần, dùng lại toàn bộ."""
    log("[INIT] Khởi tạo mô hình embedding & ChromaDB (cache dùng lại)...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = Chroma(
        collection_name="instruct2ds",
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return embeddings, vectordb


# Prompt sinh câu hỏi
CV_QG_PROMPT = """Bạn là trợ lý học thuật. Dựa DUY NHẤT vào ĐOẠN TRÍCH SAU:
---
{chunk}
---
Hãy sinh 2 câu hỏi NGẮN gọn về Computer Vision (ECCV), và TRẢ LỜI được từ đoạn trích. 
Mỗi câu trả lời ≤ 30 từ. Xuất đúng JSON (list) dạng:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
] 
Không thêm văn bản ngoài JSON.
"""


def sample_docs_for_qg(vectordb, n_docs=10) -> List[str]:
    raw = vectordb._collection.get(limit=500)
    texts = raw.get("documents", []) if raw else []
    random.shuffle(texts)
    return texts[:n_docs]


def generate_grounded_qa_pairs(llm, vectordb, n_questions=15) -> List[Dict[str, str]]:
    qa_pairs = []
    need_docs = max(1, (n_questions + 1) // 2)
    chunks = sample_docs_for_qg(vectordb, n_docs=need_docs)

    for ch in chunks:
        prompt = CV_QG_PROMPT.format(chunk=ch[:3000])
        out = llm.invoke(prompt)
        text = getattr(out, "content", str(out)).strip()

        try:
            json_str_match = re.search(r"\[.*\]", text, re.DOTALL)
            if not json_str_match:
                continue
            items = json.loads(json_str_match.group(0))
            for it in items:
                q = it.get("question", "").strip()
                a = it.get("answer", "").strip()
                if len(q) > 8 and len(a) > 1:
                    qa_pairs.append({"question": q, "answer_gold": a})
                    if len(qa_pairs) >= n_questions:
                        return qa_pairs
        except Exception:
            continue
    return qa_pairs


# Metric tính toán
def f1_token_level(pred: str, gold: str) -> float:
    if not pred or not gold:
        return 0.0
    p = pred.lower().split()
    g = gold.lower().split()
    common = {}
    for tok in p:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in g:
        if common.get(tok, 0) > 0:
            overlap += 1
            common[tok] -= 1
    prec = overlap / max(1, len(p))
    rec = overlap / max(1, len(g))
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def semantic_similarity(embeddings, text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    try:
        vec_a = embeddings.embed_query(text_a)
        vec_b = embeddings.embed_query(text_b)
        return float(cosine_similarity([vec_a], [vec_b])[0][0])
    except Exception:
        return 0.0


# Đánh giá độ chính xác
def evaluate_rag_accuracy(llm, embeddings, vectordb, qa_items, top_k=6):
    log("=== BẮT ĐẦU ĐÁNH GIÁ RAG ===")
    os.makedirs("logs/evaluated", exist_ok=True)
    csv_path = "logs/evaluated/eval_report.csv"
    acc_img = "logs/evaluated/accuracy_distribution.png"

    correct = 0
    total = 0
    rows = []

    cos_scores = []
    f1_scores = []

    for item in tqdm(qa_items, desc="Đánh giá RAG"):
        q = item["question"].strip()
        a_gold = item["answer_gold"].strip()

        docs = vectordb.similarity_search(q, k=top_k)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = (
            "Trả lời ngắn gọn (≤30 từ) DỰA VÀO NGỮ CẢNH. "
            "Nếu không đủ dữ kiện, nói 'Không đủ thông tin trong ngữ cảnh'.\n\n"
            f"### NGỮ CẢNH:\n{context}\n\n"
            f"### CÂU HỎI:\n{q}\n\n"
            "### TRẢ LỜI:"
        )

        try:
            out = llm.invoke(prompt)
            a_pred = getattr(out, "content", str(out)).strip()
        except Exception as e:
            log(f"[WARN] Lỗi LLM: {e}")
            a_pred = ""

        cos = semantic_similarity(embeddings, a_pred, a_gold)
        f1 = f1_token_level(a_pred, a_gold)
        cos_scores.append(cos)
        f1_scores.append(f1)

        label = "Đúng" if (cos >= 0.75 or f1 >= 0.75) else "Sai"
        correct += (label == "Đúng")
        total += 1

        rows.append({
            "question": q,
            "gold": a_gold,
            "pred": a_pred,
            "cosine": round(cos, 4),
            "f1": round(f1, 4),
            "label": label
        })

    acc = (correct / total * 100) if total else 0.0

    # CSV log
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "gold", "pred", "cosine", "f1", "label"])
        writer.writeheader()
        writer.writerows(rows)

    # Biểu đồ phân phối
    plt.figure(figsize=(7, 4))
    plt.hist(cos_scores, bins=10, alpha=0.6, label="Cosine", color="blue")
    plt.hist(f1_scores, bins=10, alpha=0.5, label="F1", color="orange")
    plt.axvline(0.75, color="red", linestyle="--", label="Threshold 0.75")
    plt.title("Distribution of Similarity Scores (Qwen2.5)")
    plt.xlabel("Similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_img)

    log("=== KẾT QUẢ ĐÁNH GIÁ RAG ===")
    log(f"Tổng số câu hỏi: {total}")
    log(f"Số câu đúng: {correct}")
    log(f"Độ chính xác trung bình: {acc:.2f}%")
    log(f"→ Báo cáo chi tiết: {csv_path}")
    log(f"→ Biểu đồ phân phối: {acc_img}")

    return acc


# Pipeline chính Step5
def evaluation_pipeline_node(state: dict) -> dict:
    log("=== BẮT ĐẦU BƯỚC 5 (Qwen2.5 - Ollama) ===")

    embeddings, vectordb = load_embeddings_and_db()
    llm = make_llm("qwen2.5:7b")

    qa_items = generate_grounded_qa_pairs(llm, vectordb, n_questions=15)
    log(f"→ Số câu hỏi sinh ra: {len(qa_items)}")

    acc = evaluate_rag_accuracy(llm, embeddings, vectordb, qa_items, top_k=6)

    if acc < 80:
        log(f"[AUTO] Accuracy {acc:.2f}% < 80% → Tiến hành fine-tuning...")
        run_finetune()
    else:
        log(f"[OK] Accuracy {acc:.2f}% đạt yêu cầu, không cần fine-tuning.")

    state["response"] = f"Đánh giá hoàn tất (Ollama). Độ chính xác trung bình: {acc:.2f}%"
    state.setdefault("trace", []).append(f"[evaluate_qwen] accuracy={acc:.2f}%")
    return state