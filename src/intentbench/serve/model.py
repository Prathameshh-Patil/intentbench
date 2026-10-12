"""Load the fine-tuned model once and turn its probabilities into an API answer.

The model comes from INTENTBENCH_MODEL: a local folder (default `models/distilbert`) or a
Hugging Face Hub repo id. The out-of-scope threshold comes from OOS_THRESHOLD, or else from
`reports/threshold.json`, written by `scripts/choose_threshold.py` on validation data.
"""

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = ROOT / "models" / "distilbert"
THRESHOLD_FILE = ROOT / "reports" / "threshold.json"
OOS_DATASET_LABEL = "oos"
OOS_NAME = "out_of_scope"
MAX_LENGTH = 64  # same as training
TOP_K = 3
BASE_MODEL = "distilbert-base-uncased"


def default_threshold() -> float:
    if os.environ.get("OOS_THRESHOLD"):
        return float(os.environ["OOS_THRESHOLD"])
    return float(json.loads(THRESHOLD_FILE.read_text())["threshold"])


def decide(proba: list[float], labels: list[str], threshold: float) -> dict:
    """Top-3 intents, and the final label: out_of_scope if the top probability is too low."""
    names = [OOS_NAME if n == OOS_DATASET_LABEL else n for n in labels]
    ranked = sorted(range(len(proba)), key=lambda i: proba[i], reverse=True)
    best = ranked[0]
    confidence = float(proba[best])
    label = OOS_NAME if confidence < threshold else names[best]
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "top_intents": [
            {"intent": names[i], "confidence": round(float(proba[i]), 4)} for i in ranked[:TOP_K]
        ],
    }


class Predictor:
    def __init__(self, model: str | Path | None = None, threshold: float | None = None):
        source = model or os.environ.get("INTENTBENCH_MODEL") or DEFAULT_MODEL
        self.source = str(source)
        self.threshold = default_threshold() if threshold is None else threshold
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source).eval()
        config = self.model.config
        self.labels = [config.id2label[i] for i in range(config.num_labels)]
        self.base_model = BASE_MODEL

    @torch.no_grad()
    def proba(self, text: str) -> list[float]:
        enc = self.tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        return torch.softmax(self.model(**enc).logits, dim=-1)[0].tolist()

    def classify(self, text: str) -> dict:
        return decide(self.proba(text), self.labels, self.threshold)
