"""Fine-tune DistilBERT for 151-way intent classification with the Hugging Face Trainer.

Trains on train, evaluates on validation after every epoch, and keeps the epoch with the best
validation macro-F1. Out-of-scope is the 151st class, as in the baseline.
"""

import shutil

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from intentbench.data import SEED, Splits
from intentbench.evaluate import MODELS_DIR, classification_metrics

BASE_MODEL = "distilbert-base-uncased"
MODEL_DIR = MODELS_DIR / "distilbert"
MAX_LENGTH = 64  # longest query is 28 words; 64 tokens leaves room for subword splits

HYPERPARAMS = {
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 32,
    "num_train_epochs": 8,
    "weight_decay": 0.01,
    "warmup_steps": 0.1,  # a float below 1 means "10% of all training steps"
}


def _to_dataset(df, tokenizer) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}))
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH), batched=True
    )


def train(splits: Splits, output_dir=MODEL_DIR) -> tuple[Trainer, dict]:
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(splits.labels),
        id2label=dict(enumerate(splits.labels)),
        label2id=splits.label2id,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return classification_metrics(labels, logits.argmax(-1), splits.oos_id, len(splits.labels))

    checkpoints = output_dir.parent / f"{output_dir.name}-checkpoints"
    args = TrainingArguments(
        output_dir=str(checkpoints),
        **HYPERPARAMS,
        per_device_eval_batch_size=128,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_total_limit=2,
        seed=SEED,
        data_seed=SEED,
        dataloader_pin_memory=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_to_dataset(splits.train, tokenizer),
        eval_dataset=_to_dataset(splits.validation, tokenizer),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    result = trainer.train()

    trainer.save_model(str(output_dir))  # the best epoch, thanks to load_best_model_at_end
    shutil.rmtree(checkpoints, ignore_errors=True)
    return trainer, {"train_seconds": round(result.metrics["train_runtime"], 1)}


def loss_history(trainer: Trainer) -> list[dict]:
    """One row per epoch: training loss, validation loss and validation metrics."""
    rows: dict[int, dict] = {}
    for entry in trainer.state.log_history:
        if "epoch" not in entry:
            continue
        row = rows.setdefault(round(entry["epoch"]), {"epoch": round(entry["epoch"])})
        if "loss" in entry:
            row["train_loss"] = entry["loss"]
        if "eval_loss" in entry:
            row.update(
                {k.removeprefix("eval_"): v for k, v in entry.items() if k.startswith("eval_")}
            )
    return [r for _, r in sorted(rows.items()) if "train_loss" in r and "loss" in r]


class DistilBertClassifier:
    """Loads a saved fine-tuned model and predicts class probabilities for raw texts."""

    def __init__(self, model_dir=MODEL_DIR, device: str | None = None):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                texts[i : i + batch_size],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            out.append(torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.predict_proba(texts).argmax(-1)
