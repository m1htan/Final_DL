import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.evaluation import compare_evaluation_runs


def _write_eval_csv(path: Path, rows):
    fieldnames = [
        "question",
        "gold",
        "pred",
        "cosine",
        "f1",
        "label",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.mark.parametrize(
    "baseline_correct,candidate_correct",
    [
        (2, 4),
        (0, 3),
    ],
)
def test_finetuned_chroma_improves_accuracy(tmp_path, baseline_correct, candidate_correct):
    baseline_rows = []
    for idx in range(4):
        baseline_rows.append(
            {
                "question": f"Q{idx}",
                "gold": "gold",
                "pred": "pred" if idx < baseline_correct else "wrong",
                "cosine": 0.3 + idx * 0.05,
                "f1": 0.1 + idx * 0.02,
                "label": "Đúng" if idx < baseline_correct else "Sai",
            }
        )

    candidate_rows = []
    for idx in range(4):
        candidate_rows.append(
            {
                "question": f"Q{idx}",
                "gold": "gold",
                "pred": "pred",
                "cosine": 0.4 + idx * 0.1,
                "f1": 0.2 + idx * 0.05,
                "label": "Đúng" if idx < candidate_correct else "Sai",
            }
        )

    baseline_path = tmp_path / "baseline.csv"
    candidate_path = tmp_path / "candidate.csv"
    _write_eval_csv(baseline_path, baseline_rows)
    _write_eval_csv(candidate_path, candidate_rows)

    baseline, candidate, delta = compare_evaluation_runs(
        str(baseline_path),
        str(candidate_path),
        cosine_threshold=0.5,
        f1_threshold=0.2,
    )

    assert candidate.accuracy > baseline.accuracy
    assert delta["accuracy"] == pytest.approx(candidate.accuracy - baseline.accuracy)
    assert delta["avg_cosine"] > 0
    assert delta["avg_f1"] > 0