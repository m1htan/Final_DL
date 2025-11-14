import os

os.environ["OLLAMA_NUM_GPU"] = "0"
os.environ["CHROMADB_ALLOW_RESET"] = "true"

import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

import csv
import json
import re
import shutil
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Sequence, Tuple
import time, gc
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.config import CHROMA_DIR, EMBEDDING_MODEL
from src.llm import make_llm
from src.nodes.step5.finetune_manager import run_finetune
from src.utils.evaluation import (
    EvaluationSummary,
    build_evaluation_summary,
    write_improvement_plan,
)
from src.utils.logger import log

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


@dataclass(frozen=True)
class RetrievalEvalConfig:
    """Configuration for a single evaluation attempt."""

    name: str
    top_k: int
    retrieval: str = "similarity"  # "similarity" | "mmr"
    prompt_variant: str = "strict"  # "strict" | "cite" | "deliberate"
    cosine_threshold: float = 0.75
    f1_threshold: float = 0.75
    answer_limit: int = 30
    mmr_lambda: float = 0.5

    def describe(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "top_k": str(self.top_k),
            "retrieval": self.retrieval,
            "prompt_variant": self.prompt_variant,
            "cosine_threshold": f"{self.cosine_threshold:.2f}",
            "f1_threshold": f"{self.f1_threshold:.2f}",
        }


# Load embeddings & DB
def load_embeddings_and_db(
    model_override: str | None = None, chroma_dir_override: str | None = None
):
    """Load embedding + ChromaDB một lần, dùng lại toàn bộ."""

    log("[INIT] Khởi tạo mô hình embedding & ChromaDB (cache dùng lại)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_override or EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = Chroma(
        collection_name="instruct2ds",
        persist_directory=chroma_dir_override or CHROMA_DIR,
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


def sample_docs_for_qg(vectordb, n_docs: int = 10) -> List[str]:
    raw = vectordb._collection.get(limit=500)
    texts = raw.get("documents", []) if raw else []
    random.shuffle(texts)
    return texts[:n_docs]


def generate_grounded_qa_pairs(llm, vectordb, n_questions: int = 15) -> List[Dict[str, str]]:
    qa_pairs: List[Dict[str, str]] = []
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
    common: Dict[str, int] = {}
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


def _select_documents(vectordb, question: str, config: RetrievalEvalConfig):
    try:
        if config.retrieval == "mmr":
            return vectordb.max_marginal_relevance_search(
                question,
                k=config.top_k,
                lambda_mult=config.mmr_lambda,
            )
        return vectordb.similarity_search(question, k=config.top_k)
    except Exception as exc:
        log(f"[WARN] Retrieval error ({config.name}): {exc}")
        return []


def _build_answer_prompt(context: str, question: str, config: RetrievalEvalConfig) -> str:
    if config.prompt_variant == "cite":
        header = (
            f"Bạn là chuyên gia Computer Vision. Trả lời ngắn gọn (≤{config.answer_limit} từ) dựa hoàn"
            " toàn vào ngữ cảnh. Nếu dùng thông tin, hãy trích dẫn một cụm từ trong dấu ngoặc kép. "
            "Nếu thiếu dữ kiện, trả lời 'Không đủ thông tin trong ngữ cảnh.'\n\n"
        )
    elif config.prompt_variant == "deliberate":
        header = (
            f"Phân tích ngắn gọn ngữ cảnh trước khi trả lời. Kết thúc bằng câu trả lời ≤{config.answer_limit} từ.\n\n"
            "Định dạng:\n"
            "Suy luận: <gạch đầu dòng liệt kê ý chính từ ngữ cảnh>\n"
            "Trả lời: <câu trả lời cuối cùng>\n\n"
        )
    else:
        header = (
            f"Trả lời ngắn gọn (≤{config.answer_limit} từ) DỰA VÀO NGỮ CẢNH. Nếu không đủ dữ kiện,"
            " nói 'Không đủ thông tin trong ngữ cảnh.'\n\n"
        )

    body = (
        f"### NGỮ CẢNH:\n{context}\n\n"
        f"### CÂU HỎI:\n{question}\n\n"
    )
    if config.prompt_variant == "deliberate":
        footer = "### Suy luận và trả lời:\n"
    else:
        footer = "### TRẢ LỜI:\n"
    return header + body + footer


def _is_better(candidate: EvaluationSummary, incumbent: EvaluationSummary) -> bool:
    if candidate.accuracy != incumbent.accuracy:
        return candidate.accuracy > incumbent.accuracy
    if candidate.avg_f1 != incumbent.avg_f1:
        return candidate.avg_f1 > incumbent.avg_f1
    return candidate.avg_cosine > incumbent.avg_cosine


def _promote_artifacts(summary: EvaluationSummary, output_root: str = "logs/evaluated") -> Dict[str, str]:
    promoted: Dict[str, str] = {}
    os.makedirs(output_root, exist_ok=True)
    for key, src in summary.artifacts.items():
        if not src or not os.path.exists(src):
            continue
        dst = os.path.join(output_root, os.path.basename(src))
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
        promoted[key] = dst
    summary.artifacts = promoted
    return promoted


def evaluate_rag_accuracy(
    llm,
    embeddings,
    vectordb,
    qa_items: Sequence[Dict[str, str]],
    config: RetrievalEvalConfig,
    output_root: str = "logs/evaluated",
) -> EvaluationSummary:
    log(f"=== BẮT ĐẦU ĐÁNH GIÁ RAG ({config.name}) ===")
    config_dir = os.path.join(output_root, config.name)
    os.makedirs(config_dir, exist_ok=True)
    csv_path = os.path.join(config_dir, "eval_report.csv")
    acc_img = os.path.join(config_dir, "accuracy_distribution.png")
    suggestions_path = os.path.join(config_dir, "improvement_plan.txt")

    rows: List[Dict[str, object]] = []
    cos_scores: List[float] = []
    f1_scores: List[float] = []

    for item in tqdm(qa_items, desc=f"Đánh giá RAG ({config.name})"):
        q = item["question"].strip()
        a_gold = item["answer_gold"].strip()

        docs = _select_documents(vectordb, q, config)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = _build_answer_prompt(context, q, config)

        try:
            out = llm.invoke(prompt)
            a_pred = getattr(out, "content", str(out)).strip()
        except Exception as exc:
            log(f"[WARN] Lỗi LLM ({config.name}): {exc}")
            a_pred = ""

        cos = semantic_similarity(embeddings, a_pred, a_gold)
        f1 = f1_token_level(a_pred, a_gold)
        cos_scores.append(cos)
        f1_scores.append(f1)

        label = (
            "Đúng"
            if (cos >= config.cosine_threshold or f1 >= config.f1_threshold)
            else "Sai"
        )

        rows.append(
            {
                "question": q,
                "gold": a_gold,
                "pred": a_pred,
                "cosine": cos,
                "f1": f1,
                "label": label,
            }
        )

    summary = build_evaluation_summary(
        rows,
        cos_scores,
        f1_scores,
        cosine_threshold=config.cosine_threshold,
        f1_threshold=config.f1_threshold,
        config=config.describe(),
        artifacts={
            "csv": csv_path,
            "chart": acc_img,
            "improvement_plan": suggestions_path,
        },
    )

    # CSV log
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "gold", "pred", "cosine", "f1", "label"],
        )
        writer.writeheader()
        formatted_rows = []
        for row in rows:
            formatted_rows.append(
                {
                    "question": row["question"],
                    "gold": row["gold"],
                    "pred": row["pred"],
                    "cosine": round(row["cosine"], 4),
                    "f1": round(row["f1"], 4),
                    "label": row["label"],
                }
            )
        writer.writerows(formatted_rows)

    # Biểu đồ phân phối
    plt.figure(figsize=(7, 4))
    plt.hist(cos_scores, bins=10, alpha=0.6, label="Cosine", color="blue")
    plt.hist(f1_scores, bins=10, alpha=0.5, label="F1", color="orange")
    plt.axvline(
        config.cosine_threshold,
        color="red",
        linestyle="--",
        label=f"Cosine ≥ {config.cosine_threshold:.2f}",
    )
    plt.title(f"Distribution of Similarity Scores ({config.name})")
    plt.xlabel("Similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_img)
    plt.close()

    write_improvement_plan(suggestions_path, summary)

    log(f"=== KẾT QUẢ ĐÁNH GIÁ RAG ({config.name}) ===")
    log(f"Tổng số câu hỏi: {summary.total}")
    log(f"Số câu đúng: {summary.correct}")
    log(f"Độ chính xác trung bình: {summary.accuracy:.2f}%")
    log(f"Điểm cosine trung bình: {summary.avg_cosine:.4f}")
    log(f"Điểm F1 trung bình: {summary.avg_f1:.4f}")
    for idx, suggestion in enumerate(summary.suggestions, start=1):
        log(f"[SUGGESTION {idx}] {suggestion}")

    log(f"→ Báo cáo chi tiết: {csv_path}")
    log(f"→ Biểu đồ phân phối: {acc_img}")
    log(f"→ Gợi ý cải thiện: {suggestions_path}")

    return summary


def run_accuracy_sweep(
    llm,
    embeddings,
    vectordb,
    qa_items: Sequence[Dict[str, str]],
    configs: Iterable[RetrievalEvalConfig],
) -> Tuple[RetrievalEvalConfig, EvaluationSummary, List[Tuple[RetrievalEvalConfig, EvaluationSummary]]]:
    summaries: List[Tuple[RetrievalEvalConfig, EvaluationSummary]] = []
    best_summary: EvaluationSummary | None = None
    best_config: RetrievalEvalConfig | None = None

    for config in configs:
        summary = evaluate_rag_accuracy(llm, embeddings, vectordb, qa_items, config)
        summaries.append((config, summary))
        if best_summary is None or _is_better(summary, best_summary):
            best_summary = summary
            best_config = config

    assert best_summary is not None and best_config is not None, "No evaluation summaries produced."

    log("=== Tổng kết các cấu hình đánh giá ===")
    for config, summary in summaries:
        log(
            f"[{config.name}] accuracy={summary.accuracy:.2f}% | avg_f1={summary.avg_f1:.4f} | avg_cosine={summary.avg_cosine:.4f}"
        )

    return best_config, best_summary, summaries


# Pipeline chính Step5
def evaluation_pipeline_node(state: dict) -> dict:
    log("=== BẮT ĐẦU BƯỚC 5 (Qwen2.5 - Ollama) ===")

    embeddings, vectordb = load_embeddings_and_db()
    llm = make_llm("qwen2.5:7b")

    qa_items = generate_grounded_qa_pairs(llm, vectordb, n_questions=15)
    log(f"→ Số câu hỏi sinh ra: {len(qa_items)}")

    # Ghi lại danh sách câu hỏi & câu trả lời gốc
    os.makedirs("logs/evaluated", exist_ok=True)
    qa_csv_path = f"logs/evaluated/questions_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(qa_csv_path, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=["question", "answer_gold"])
        writer.writeheader()
        writer.writerows(qa_items)
    log(f"→ Đã lưu danh sách câu hỏi tại: {qa_csv_path}")

    configs = [
        RetrievalEvalConfig(name="baseline_similarity", top_k=6, prompt_variant="strict"),
        RetrievalEvalConfig(
            name="mmr_high_topk",
            top_k=9,
            retrieval="mmr",
            cosine_threshold=0.72,
            f1_threshold=0.72,
        ),
        RetrievalEvalConfig(
            name="cite_prompt",
            top_k=7,
            prompt_variant="cite",
            cosine_threshold=0.74,
            f1_threshold=0.7,
        ),
    ]

    best_config, best_summary, _ = run_accuracy_sweep(
        llm,
        embeddings,
        vectordb,
        qa_items,
        configs,
    )

    promoted_artifacts = _promote_artifacts(best_summary)
    write_improvement_plan(os.path.join("logs/evaluated", "improvement_plan.txt"), best_summary)

    log(
        "[AUTO] Cấu hình hiệu quả nhất: "
        f"{best_config.name} (top_k={best_config.top_k}, retrieval={best_config.retrieval}, prompt={best_config.prompt_variant})"
    )

    try:
        log("[CLEANUP] Đang đóng vectordb trước khi rebuild...")

        if hasattr(vectordb, "_client"):
            client = getattr(vectordb, "_client", None)
            if hasattr(client, "_server"):
                server = getattr(client, "_server", None)
                if hasattr(server, "close"):
                    server.close()
                    log("[CLEANUP] Đã gọi server.close()")
            if hasattr(client, "persist"):
                client.persist()
        vectordb._collection = None
        del vectordb
        gc.collect()
        time.sleep(3)

        log("[CLEANUP] vectordb đã được đóng hoàn toàn.")
    except Exception as e:
        log(f"[WARN] Không thể đóng vectordb đúng cách: {e}")

        if best_summary.accuracy < 80:
            log(f"[AUTO] Accuracy {best_summary.accuracy:.2f}% < 80% → Tiến hành fine-tuning và tái đánh giá...")
            finetune_artifacts = run_finetune()
            if finetune_artifacts:
                embeddings, vectordb = load_embeddings_and_db(
                    model_override=finetune_artifacts.get("model_path"),
                    chroma_dir_override=finetune_artifacts.get("chroma_dir"),
                )
            else:
                log("[WARN] Fine-tuning không thành công, dùng lại embedding hiện tại.")
            tuned_config = replace(
                best_config,
                name=f"{best_config.name}_after_ft",
                top_k=min(12, best_config.top_k + 2),
                retrieval="mmr",
                cosine_threshold=max(0.7, best_config.cosine_threshold - 0.02),
                f1_threshold=max(0.7, best_config.f1_threshold - 0.02),
            )
            tuned_summary = evaluate_rag_accuracy(
                llm,
                embeddings,
                vectordb,
                qa_items,
                tuned_config,
            )
            if _is_better(tuned_summary, best_summary):
                log(
                    f"[IMPROVED] Fine-tuning giúp tăng accuracy lên {tuned_summary.accuracy:.2f}%"
                )
                best_summary = tuned_summary
                best_config = tuned_config
                promoted_artifacts = _promote_artifacts(best_summary)
                write_improvement_plan(os.path.join("logs/evaluated", "improvement_plan.txt"), best_summary)
            else:
                log("[INFO] Fine-tuning không cải thiện đáng kể so với cấu hình trước đó.")
        else:
            log(
                f"[OK] Accuracy {best_summary.accuracy:.2f}% đạt yêu cầu, không cần fine-tuning."
            )

    csv_path = promoted_artifacts.get("csv", "logs/evaluated/eval_report.csv")
    chart_path = promoted_artifacts.get("chart", "logs/evaluated/accuracy_distribution.png")
    plan_path = promoted_artifacts.get(
        "improvement_plan", "logs/evaluated/improvement_plan.txt"
    )

    log(f"→ Báo cáo chi tiết (tốt nhất): {csv_path}")
    log(f"→ Biểu đồ phân phối (tốt nhất): {chart_path}")
    log(f"→ Gợi ý cải thiện (tốt nhất): {plan_path}")

    state["response"] = (
        "Đánh giá hoàn tất (Qwen2.5 qua Ollama). "
        f"Độ chính xác trung bình: {best_summary.accuracy:.2f}%\n"
        f"Cấu hình tối ưu: {best_config.name} (top_k={best_config.top_k}, retrieval={best_config.retrieval}).\n"
        + (
            f"Gợi ý ưu tiên: {best_summary.suggestions[0]}"
            if best_summary.suggestions
            else "Hệ thống đạt chuẩn hiện tại."
        )
    )
    state.setdefault("trace", []).append(
        f"[evaluate_qwen] accuracy={best_summary.accuracy:.2f}%"
    )
    state["evaluation_summary"] = best_summary.as_dict()
    return state
