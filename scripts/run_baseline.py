"""Tune the TF-IDF + logistic regression baseline on validation, save it, and write a report.

Usage: python scripts/run_baseline.py
Outputs: models/baseline.joblib, reports/baseline_validation.json
"""

import json
import time

from intentbench import baseline
from intentbench.data import load_splits
from intentbench.evaluate import REPORTS_DIR, environment_info


def main() -> None:
    start = time.perf_counter()
    splits = load_splits()
    print(f"Tuning on {len(splits.train)} train / {len(splits.validation)} validation examples")
    best, grid = baseline.tune(splits)
    baseline.save(best)

    best_row = max(grid, key=lambda r: r["macro_f1"])
    report = {
        "approach": "tfidf_logreg",
        "split": "validation",
        "selected_by": "macro_f1",
        "best": best_row,
        "grid": grid,
        "total_seconds": round(time.perf_counter() - start, 1),
        "environment": environment_info(),
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "baseline_validation.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"\nBest: {best_row['features']}, C={best_row['C']:g}")
    for key in ("accuracy", "in_scope_accuracy", "macro_f1", "oos_precision", "oos_recall"):
        print(f"  {key:<18} {best_row[key]:.4f}")
    print(f"Saved {baseline.MODEL_PATH.name} and {out.name} in {report['total_seconds']}s")


if __name__ == "__main__":
    main()
