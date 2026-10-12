"""Choose the confidence threshold below which the served model answers out_of_scope.

Usage:
  python scripts/choose_threshold.py

Uses validation only. For each threshold t, a prediction whose top probability is below t
becomes out-of-scope. Rule: catch as many out-of-scope queries as possible (highest OOS recall)
while giving up at most 1 point of in-scope accuracy, i.e. wrongly rejecting at most ~1 in 100
valid queries. Accuracy alone would barely move (only 3% of validation is out-of-scope), and a
balanced score (in-scope accuracy + OOS recall) / 2 pushed t to 0.98, rejecting 6% of valid
queries, because the model is over-confident and its probabilities sit near 1.

Outputs: reports/threshold.json, reports/figures/threshold_curve.png
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from intentbench.data import load_splits
from intentbench.evaluate import (
    FIGURES_DIR,
    INK,
    MUTED,
    REPORTS_DIR,
    SERIES,
    classification_metrics,
    use_plot_style,
)
from intentbench.finetune import DistilBertClassifier

THRESHOLDS = np.round(np.arange(0.0, 1.0, 0.01), 2)
MAX_IN_SCOPE_DROP = 0.01


def apply_threshold(proba: np.ndarray, threshold: float, oos_id: int) -> np.ndarray:
    pred = proba.argmax(-1)
    return np.where(proba.max(-1) < threshold, oos_id, pred)


def main() -> None:
    splits = load_splits()
    val = splits.validation
    proba = DistilBertClassifier().predict_proba(val["text"].tolist())

    rows = []
    for t in THRESHOLDS:
        m = classification_metrics(
            val["label"],
            apply_threshold(proba, t, splits.oos_id),
            splits.oos_id,
            len(splits.labels),
        )
        rows.append({"threshold": float(t), **m})
    no_threshold = rows[0]
    floor = no_threshold["in_scope_accuracy"] - MAX_IN_SCOPE_DROP
    allowed = [r for r in rows if r["in_scope_accuracy"] >= floor]
    best = max(allowed, key=lambda r: (r["oos_recall"], -r["threshold"]))

    report = {
        "model": "distilbert",
        "split": "validation",
        "selected_by": f"highest oos_recall with in_scope_accuracy >= {floor:.4f} "
        f"(at most {MAX_IN_SCOPE_DROP:.0%} below no threshold)",
        "threshold": best["threshold"],
        "at_threshold": best,
        "without_threshold": no_threshold,
        "curve": rows,
    }
    (REPORTS_DIR / "threshold.json").write_text(json.dumps(report, indent=2))

    use_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label, color in [
        ("in_scope_accuracy", "In-scope accuracy", SERIES[0]),
        ("oos_recall", "Out-of-scope recall", SERIES[1]),
        ("oos_precision", "Out-of-scope precision", SERIES[2]),
    ]:
        ax.plot([r["threshold"] for r in rows], [r[key] for r in rows], label=label, color=color)
    ax.axvline(best["threshold"], color=MUTED, linestyle="--", linewidth=1)
    ax.annotate(
        f"chosen: {best['threshold']:.2f}",
        xy=(best["threshold"], 0.25),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK,
    )
    ax.set_xlabel("Confidence threshold (top probability below it → out_of_scope)")
    ax.set_ylabel("Validation score")
    ax.axhline(floor, color=SERIES[0], linestyle=":", linewidth=1)
    ax.set_ylim(0.2, 1.0)
    ax.set_title("DistilBERT: choosing the out-of-scope threshold on validation", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_curve.png", dpi=150)

    print(f"Chosen threshold: {best['threshold']:.2f}")
    for name, r in [("no threshold", no_threshold), ("at threshold", best)]:
        print(
            f"  {name:<13} in-scope acc {r['in_scope_accuracy']:.3f}  OOS recall "
            f"{r['oos_recall']:.2f}  OOS precision {r['oos_precision']:.2f}  "
            f"macro-F1 {r['macro_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
