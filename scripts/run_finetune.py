"""Fine-tune DistilBERT on train, keep the best epoch by validation macro-F1, plot the curves.

Usage: python scripts/run_finetune.py
Outputs: models/distilbert/, reports/finetune_validation.json,
         reports/figures/finetune_curves.png
"""

import json

import matplotlib.pyplot as plt

from intentbench import finetune
from intentbench.data import load_splits
from intentbench.evaluate import (
    FIGURES_DIR,
    REPORTS_DIR,
    SERIES,
    classification_metrics,
    environment_info,
    use_plot_style,
)


def plot_curves(history: list[dict], best_epoch: int, path) -> None:
    use_plot_style()
    epochs = [r["epoch"] for r in history]
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(10, 3.6))

    for key, label, color in [
        ("train_loss", "training", SERIES[0]),
        ("loss", "validation", SERIES[1]),
    ]:
        values = [r[key] for r in history]
        ax_loss.plot(epochs, values, color=color, marker="o", markersize=5, label=label)
        ax_loss.annotate(
            label,
            (epochs[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color="#52514e",
        )
    ax_loss.set_title("Loss per epoch", loc="left")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.legend(loc="upper right")

    f1 = [r["macro_f1"] for r in history]
    ax_f1.plot(epochs, f1, color=SERIES[0], marker="o", markersize=5)
    best_f1 = f1[epochs.index(best_epoch)]
    ax_f1.annotate(
        f"kept: epoch {best_epoch}\nmacro-F1 {best_f1:.3f}",
        (best_epoch, best_f1),
        xytext=(0, -34),
        textcoords="offset points",
        ha="center",
        color="#52514e",
        arrowprops={"arrowstyle": "-", "color": "#52514e"},
    )
    ax_f1.set_title("Validation macro-F1 per epoch", loc="left")
    ax_f1.set_xlabel("epoch")

    for ax in (ax_loss, ax_f1):
        ax.set_xticks(epochs)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    splits = load_splits()
    trainer, timing = finetune.train(splits)
    history = finetune.loss_history(trainer)
    best = max(history, key=lambda r: r["macro_f1"])

    # Sanity check: reload the saved model from disk and confirm it is the best epoch.
    clf = finetune.DistilBertClassifier()
    reloaded = classification_metrics(
        splits.validation["label"],
        clf.predict(splits.validation["text"].tolist()),
        splits.oos_id,
        len(splits.labels),
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_curves(history, best["epoch"], FIGURES_DIR / "finetune_curves.png")

    report = {
        "approach": "distilbert",
        "split": "validation",
        "base_model": finetune.BASE_MODEL,
        "selected_by": "macro_f1",
        "hyperparams": finetune.HYPERPARAMS,
        "best_epoch": best["epoch"],
        "best": reloaded,
        "history": history,
        **timing,
        "device": clf.device,
        "environment": environment_info(),
    }
    out = REPORTS_DIR / "finetune_validation.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"\nBest epoch {best['epoch']} (reloaded from {finetune.MODEL_DIR.name}/):")
    for key in ("accuracy", "in_scope_accuracy", "macro_f1", "oos_precision", "oos_recall"):
        print(f"  {key:<18} {reloaded[key]:.4f}")
    print(f"Training took {timing['train_seconds']}s on {clf.device}")


if __name__ == "__main__":
    main()
