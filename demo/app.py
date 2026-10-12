"""Gradio demo: type a message and see all three approaches answer side by side.

Run locally:  python demo/app.py   (then open http://localhost:7860)

Models come from `models/` if present, otherwise from the Hugging Face Hub repo in
INTENTBENCH_HF_REPO. The LLM column is off by default; when on, it uses the same few-shot prompt
that was evaluated, and LLM_DAILY_CAP limits fresh (uncached) LLM calls per day so visitors
can't run up a bill.
"""

import json
import os
import time
from datetime import date
from pathlib import Path

import gradio as gr
import joblib
import numpy as np

from intentbench import llm
from intentbench.serve.model import Predictor, decide

ROOT = Path(__file__).resolve().parents[1]
HF_REPO = os.environ.get("INTENTBENCH_HF_REPO", "Prathameshh-Patil/intentbench-distilbert")
FEW_SHOT_REPORT = ROOT / "reports" / "llm_gemini_few_validation.json"
LLM_DAILY_CAP = int(os.environ.get("LLM_DAILY_CAP", "200"))
EXAMPLES = [
    "what's the balance in my checking account",
    "can you freeze my card, I think it was stolen",
    "how long should I boil an egg",
    "set an alarm for 6am",
    "who won the world cup in 2010",
    "asdf qwerty",
]


def _local_or_hub(local: Path, hub_file: str | None = None) -> str:
    """A local model path if it exists, otherwise the Hub repo (or a file inside it)."""
    if local.exists():
        return str(local)
    if hub_file is None:
        return HF_REPO
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, hub_file)


bert = Predictor(model=_local_or_hub(ROOT / "models" / "distilbert"))
tfidf = joblib.load(_local_or_hub(ROOT / "models" / "baseline.joblib", "baseline/baseline.joblib"))
LABELS = bert.labels  # same label order for every approach (the dataset's)
_llm_clf = None
_llm_usage = {"day": date.today(), "count": 0}


def _llm_classifier() -> llm.LLMClassifier:
    global _llm_clf
    if _llm_clf is None:
        examples = json.loads(FEW_SHOT_REPORT.read_text())["few_shot_examples"]
        _llm_clf = llm.LLMClassifier(LABELS, examples)
    return _llm_clf


def _llm_row(text: str) -> list:
    if _llm_usage["day"] != date.today():
        _llm_usage.update(day=date.today(), count=0)
    if _llm_usage["count"] >= LLM_DAILY_CAP:
        return ["LLM (few-shot)", "daily limit reached, try tomorrow", "", ""]
    try:
        start = time.perf_counter()
        p = _llm_classifier().classify(text)
        ms = (time.perf_counter() - start) * 1000
    except Exception as e:  # missing key, quota, network: show it instead of crashing the page
        return ["LLM (few-shot)", f"unavailable ({type(e).__name__})", "", ""]
    if not p.cached:
        _llm_usage["count"] += 1
    name = llm.OOS_NAME if LABELS[p.label] == "oos" else LABELS[p.label]
    return ["LLM (few-shot)", name, "n/a", f"{ms:.0f}" + (" (cached)" if p.cached else "")]


def classify(text: str, use_llm: bool):
    text = (text or "").strip()[:500]
    if not text:
        return []

    start = time.perf_counter()
    proba = tfidf.predict_proba([text])[0]
    ms = (time.perf_counter() - start) * 1000
    best = int(np.argmax(proba))
    name = llm.OOS_NAME if LABELS[best] == "oos" else LABELS[best]
    rows = [["TF-IDF + logistic regression", name, f"{proba[best]:.2f}", f"{ms:.1f}"]]

    start = time.perf_counter()
    out = decide(bert.proba(text), LABELS, bert.threshold)
    ms = (time.perf_counter() - start) * 1000
    rows.append(["DistilBERT + threshold", out["label"], f"{out['confidence']:.2f}", f"{ms:.1f}"])

    if use_llm:
        rows.append(_llm_row(text))
    return rows


with gr.Blocks(title="IntentBench") as demo:
    gr.Markdown(
        "# IntentBench\n"
        "Three ways to sort a short message into one of 150 intents (banking, travel, "
        "small talk, ...) or **out_of_scope**. Type a message and compare. "
        f"DistilBERT answers out_of_scope when its confidence is below {bert.threshold:.2f}. "
        "[Code and results](https://github.com/Prathameshh-Patil/intentbench)"
    )
    text = gr.Textbox(label="Message", placeholder="e.g. what's my credit limit", max_lines=3)
    use_llm = gr.Checkbox(
        label=f"Also ask the LLM (Gemini, few-shot; up to {LLM_DAILY_CAP} new calls per day)",
        value=False,
    )
    button = gr.Button("Classify", variant="primary")
    table = gr.Dataframe(
        headers=["Approach", "Answer", "Confidence", "Time (ms)"],
        datatype=["str", "str", "str", "str"],
        interactive=False,
    )
    gr.Examples(EXAMPLES, inputs=text)
    button.click(classify, [text, use_llm], table)
    text.submit(classify, [text, use_llm], table)

if __name__ == "__main__":
    demo.launch()
