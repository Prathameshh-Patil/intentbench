"""Metrics shared by every approach, plus the environment record that goes into each report.

Metric definitions (all approaches use exactly these):
- accuracy: share of all examples (in-scope and out-of-scope) labelled correctly.
- in_scope_accuracy: accuracy on examples whose true label is one of the 150 intents.
- macro_f1: F1 computed separately for each of the 150 intents, then averaged with equal
  weight. Out-of-scope examples still count: predicting an intent for one is a false positive.
- oos_precision / oos_recall / oos_f1: out-of-scope detection treated as a yes/no task.
"""

import importlib.metadata
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

LIBRARIES = ["datasets", "pandas", "numpy", "scikit-learn", "torch", "transformers"]


def classification_metrics(y_true, y_pred, oos_id: int, n_labels: int) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    in_scope_labels = [i for i in range(n_labels) if i != oos_id]
    is_in_scope = y_true != oos_id
    oos_p, oos_r, oos_f1, _ = precision_recall_fscore_support(
        y_true == oos_id, y_pred == oos_id, average="binary", zero_division=0
    )
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "in_scope_accuracy": float((y_true[is_in_scope] == y_pred[is_in_scope]).mean()),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=in_scope_labels, average="macro", zero_division=0)
        ),
        "oos_precision": float(oos_p),
        "oos_recall": float(oos_r),
        "oos_f1": float(oos_f1),
        "n": int(len(y_true)),
    }


def environment_info() -> dict:
    """Python, OS, CPU and library versions, so every reported number can be traced."""
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    versions = {}
    for lib in LIBRARIES:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "libraries": versions,
    }


# Colours for every figure in the project: one surface, ink for text, fixed series order.
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def use_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "font.size": 10,
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )
