"""Utility helpers to summarise evaluation runs and propose improvements."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

@dataclass
class EvaluationExample:
    """Represents a single evaluated question/answer pair."""

    question: str
    gold: str
    pred: str
    cosine: float
    f1: float
    label: str


@dataclass
class EvaluationSummary:
    """Aggregated metrics collected during an evaluation run."""

    total: int
    correct: int
    accuracy: float
    avg_cosine: float
    avg_f1: float
    cosine_threshold: float
    f1_threshold: float
    low_confidence_examples: List[EvaluationExample] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    config: Dict[str, str] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        """Return a serialisable representation."""

        payload = asdict(self)
        payload["low_confidence_examples"] = [asdict(ex) for ex in self.low_confidence_examples]
        return payload


def build_evaluation_summary(
    rows: Iterable[Dict],
    cos_scores: Iterable[float],
    f1_scores: Iterable[float],
    cosine_threshold: float,
    f1_threshold: float,
    *,
    config: Optional[Dict[str, str]] = None,
    artifacts: Optional[Dict[str, str]] = None,
) -> EvaluationSummary:
    """Create an :class:`EvaluationSummary` from raw evaluation artefacts."""

    rows = list(rows)
    cos_scores = list(cos_scores)
    f1_scores = list(f1_scores)

    total = len(rows)
    correct = sum(1 for row in rows if row.get("label") == "Đúng")
    accuracy = (correct / total * 100.0) if total else 0.0

    avg_cosine = sum(cos_scores) / len(cos_scores) if cos_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    low_confidence_examples: List[EvaluationExample] = []
    for row in rows:
        if row.get("label") != "Đúng":
            low_confidence_examples.append(_row_to_example(row))
        else:
            below_threshold = (
                row.get("cosine", 0.0) < cosine_threshold
                and row.get("f1", 0.0) < f1_threshold
            )
            if below_threshold:
                low_confidence_examples.append(_row_to_example(row))

    summary = EvaluationSummary(
        total=total,
        correct=correct,
        accuracy=accuracy,
        avg_cosine=avg_cosine,
        avg_f1=avg_f1,
        cosine_threshold=cosine_threshold,
        f1_threshold=f1_threshold,
        low_confidence_examples=low_confidence_examples,
    )
    summary.config = config or {}
    summary.artifacts = artifacts or {}
    summary.suggestions = _generate_improvement_suggestions(summary)
    return summary


def _row_to_example(row: Dict) -> EvaluationExample:
    return EvaluationExample(
        question=row.get("question", ""),
        gold=row.get("gold", ""),
        pred=row.get("pred", ""),
        cosine=float(row.get("cosine", 0.0)),
        f1=float(row.get("f1", 0.0)),
        label=row.get("label", ""),
    )


def _generate_improvement_suggestions(summary: EvaluationSummary) -> List[str]:
    suggestions: List[str] = []

    if summary.accuracy < 95:
        suggestions.append(
            "Tăng cường giai đoạn truy xuất: thử nghiệm các giá trị top_k lớn hơn, "
            "lọc theo điểm số và đảm bảo dữ liệu embedding đã được cập nhật mới nhất."
        )

    if summary.avg_cosine < summary.cosine_threshold:
        suggestions.append(
            "Điều chỉnh mô hình embedding (fine-tune hoặc chọn checkpoint mạnh hơn) "
            "và bật chuẩn hóa vector để cải thiện độ tương đồng ngữ nghĩa."
        )

    if summary.avg_f1 < summary.f1_threshold:
        suggestions.append(
            "Tinh chỉnh prompt trả lời: yêu cầu trích dẫn trực tiếp từ ngữ cảnh và "
            "thử nghiệm các chỉ dẫn súc tích hơn để tránh thông tin ngoài ngữ cảnh."
        )

    if summary.low_confidence_examples:
        suggestions.append(
            "Rà soát thủ công các câu hỏi có điểm thấp, bổ sung dữ liệu huấn luyện/"
            "augmentation cho những chủ đề tương ứng."
        )

    if summary.config.get("retrieval", "") != "mmr" and summary.accuracy < 70:
        suggestions.append(
            "Kết hợp chiến lược truy xuất đa dạng (MMR, reranker) để giảm trùng lặp"
            " và bao phủ thông tin tốt hơn."
        )

    if not suggestions:
        suggestions.append("Hệ thống đạt độ chính xác cao; tiếp tục giám sát định kỳ.")

    return suggestions


def write_improvement_plan(path: str, summary: EvaluationSummary, max_examples: int = 5) -> None:
    """Persist a textual improvement report to ``path``."""

    lines: List[str] = []
    lines.append("# Gợi ý cải thiện độ chính xác")
    lines.append("")
    lines.append(f"Tổng số câu hỏi: {summary.total}")
    lines.append(f"Số câu chính xác: {summary.correct}")
    lines.append(f"Độ chính xác trung bình: {summary.accuracy:.2f}%")
    if summary.config:
        lines.append("")
        lines.append("## Cấu hình đánh giá tối ưu")
        for key, value in summary.config.items():
            lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("## Đề xuất chính")
    for idx, suggestion in enumerate(summary.suggestions, start=1):
        lines.append(f"{idx}. {suggestion}")

    if summary.low_confidence_examples:
        lines.append("")
        lines.append("## Các ví dụ cần xem xét thêm")
        for example in summary.low_confidence_examples[:max_examples]:
            lines.append("-")
            lines.append(f"  Câu hỏi: {example.question}")
            lines.append(f"  Gold: {example.gold}")
            lines.append(f"  Pred: {example.pred}")
            lines.append(
                f"  Cosine={example.cosine:.4f}, F1={example.f1:.4f}, Label={example.label}"
            )

    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))

def load_evaluation_rows(csv_path: str) -> List[Dict[str, object]]:
    """Load evaluation rows from the CSV artefact produced by the pipeline."""

    resolved = Path(csv_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Không tìm thấy file kết quả: {csv_path}")

    with resolved.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows: List[Dict[str, object]] = []
        for raw in reader:
            row: Dict[str, object] = dict(raw)
            for key in ("cosine", "f1"):
                try:
                    row[key] = float(row.get(key, 0.0))
                except (TypeError, ValueError):
                    row[key] = 0.0
            rows.append(row)
    return rows


def summarise_csv(
    csv_path: str,
    *,
    cosine_threshold: float,
    f1_threshold: float,
    config: Optional[Dict[str, str]] = None,
    artifacts: Optional[Dict[str, str]] = None,
) -> EvaluationSummary:
    """Convenience wrapper to create a summary directly from a CSV artefact."""

    rows = load_evaluation_rows(csv_path)
    cos_scores = [float(row.get("cosine", 0.0)) for row in rows]
    f1_scores = [float(row.get("f1", 0.0)) for row in rows]
    return build_evaluation_summary(
        rows,
        cos_scores,
        f1_scores,
        cosine_threshold,
        f1_threshold,
        config=config,
        artifacts=artifacts,
    )


def compare_evaluation_runs(
    baseline_csv: str,
    candidate_csv: str,
    *,
    cosine_threshold: float,
    f1_threshold: float,
) -> Tuple[EvaluationSummary, EvaluationSummary, Dict[str, float]]:
    """Compare two evaluation CSV artefacts and report their metric deltas."""

    baseline_summary = summarise_csv(
        baseline_csv,
        cosine_threshold=cosine_threshold,
        f1_threshold=f1_threshold,
        config={"label": "baseline"},
        artifacts={"csv": baseline_csv},
    )
    candidate_summary = summarise_csv(
        candidate_csv,
        cosine_threshold=cosine_threshold,
        f1_threshold=f1_threshold,
        config={"label": "candidate"},
        artifacts={"csv": candidate_csv},
    )

    deltas = {
        "accuracy": candidate_summary.accuracy - baseline_summary.accuracy,
        "avg_cosine": candidate_summary.avg_cosine - baseline_summary.avg_cosine,
        "avg_f1": candidate_summary.avg_f1 - baseline_summary.avg_f1,
    }

    return baseline_summary, candidate_summary, deltas