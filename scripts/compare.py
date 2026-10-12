"""Final comparison: run every approach on the held-out test set, once.

Usage:
  python scripts/compare.py

- TF-IDF + logistic regression and DistilBERT are scored on the full test set and on the fixed
  LLM test sample; the LLM is scored on the sample only (cost).
- DistilBERT is also scored with the out-of-scope threshold chosen on validation
  (scripts/choose_threshold.py), because that is the configuration the API serves.
- Only the LLM variant (zero- or few-shot) with the higher validation macro-F1 is run on test:
  choosing between variants is tuning, and tuning happens on validation.
- Latency: one request at a time on this machine's CPU (median and p95). LLM latency is the
  network round trip recorded when each answer was first fetched (the cache stores it).
- Outputs: reports/results.json, reports/comparison.md, reports/test_predictions.csv
"""

import json
import random
import time

import numpy as np
import pandas as pd
import torch

from intentbench import baseline, llm
from intentbench.data import LLM_TEST_SAMPLE, LLM_VALIDATION_SAMPLE, SEED, llm_sample, load_splits
from intentbench.evaluate import (
    MODELS_DIR,
    REPORTS_DIR,
    classification_metrics,
    environment_info,
)
from intentbench.finetune import MODEL_DIR, DistilBertClassifier

LATENCY_QUERIES = 300  # single requests timed per local model
WARMUP_QUERIES = 10


def latency(predict_one, texts: list[str]) -> dict:
    for t in texts[:WARMUP_QUERIES]:
        predict_one(t)
    seconds = []
    for t in texts:
        start = time.perf_counter()
        predict_one(t)
        seconds.append(time.perf_counter() - start)
    ms = np.array(seconds) * 1000
    return {
        "median_ms": round(float(np.median(ms)), 2),
        "p95_ms": round(float(np.percentile(ms, 95)), 2),
        "n": len(ms),
    }


def dir_size_mb(path) -> float:
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return round(sum(p.stat().st_size for p in files) / 1e6, 1)


def chosen_llm_variant(provider: str) -> tuple[str, dict]:
    reports = {}
    for variant in ("zero", "few"):
        path = REPORTS_DIR / f"llm_{provider}_{variant}_validation.json"
        if not path.exists():
            raise SystemExit(f"Missing {path.name}: run scripts/run_llm.py --variant {variant}")
        reports[variant] = json.loads(path.read_text())
    best = max(reports, key=lambda v: reports[v]["metrics"]["macro_f1"])
    return best, {v: r["metrics"]["macro_f1"] for v, r in reports.items()}


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    splits = load_splits()
    test = splits.test
    sample = llm_sample(test, LLM_TEST_SAMPLE)
    n_labels, oos = len(splits.labels), splits.oos_id

    def score(y_true, y_pred):
        return classification_metrics(y_true, y_pred, oos, n_labels)

    timing_texts = test["text"].sample(LATENCY_QUERIES, random_state=SEED).tolist()
    preds = pd.DataFrame({"text": test["text"], "true": test["intent"]})
    results = {}

    print("TF-IDF + logistic regression ...")
    pipe = baseline.load()
    preds["tfidf_logreg"] = [splits.labels[i] for i in pipe.predict(test["text"])]
    results["tfidf_logreg"] = {
        "name": "TF-IDF + logistic regression",
        "test": score(test["label"], pipe.predict(test["text"])),
        "llm_sample": score(sample["label"], pipe.predict(sample["text"])),
        "latency": latency(lambda t: pipe.predict_proba([t]), timing_texts),
        "latency_device": "cpu",
        "model_size_mb": dir_size_mb(MODELS_DIR / "baseline.joblib"),
        "cost_per_1000_usd": 0.0,
    }

    print("DistilBERT ...")
    bert = DistilBertClassifier()  # GPU if present, for the bulk predictions
    bert_proba = bert.predict_proba(test["text"].tolist())
    bert_test = bert_proba.argmax(-1)
    preds["distilbert"] = [splits.labels[i] for i in bert_test]
    bert_cpu = DistilBertClassifier(device="cpu")  # serving runs on CPU, so time it there
    results["distilbert"] = {
        "name": "DistilBERT (fine-tuned)",
        "test": score(test["label"], bert_test),
        "llm_sample": score(sample["label"], bert_test[sample.index]),
        "latency": latency(lambda t: bert_cpu.predict_proba([t]), timing_texts),
        "latency_device": "cpu",
        "model_size_mb": dir_size_mb(MODEL_DIR),
        "cost_per_1000_usd": 0.0,
    }
    # The served configuration: same model, out_of_scope below the validation-chosen threshold.
    threshold = json.loads((REPORTS_DIR / "threshold.json").read_text())["threshold"]
    bert_thr = np.where(bert_proba.max(-1) < threshold, oos, bert_test)
    preds["distilbert_threshold"] = [splits.labels[i] for i in bert_thr]
    results["distilbert_threshold"] = {
        **results["distilbert"],
        "name": f"DistilBERT + threshold {threshold:.2f} (served)",
        "threshold": threshold,
        "test": score(test["label"], bert_thr),
        "llm_sample": score(sample["label"], bert_thr[sample.index]),
    }

    provider = llm.provider()
    variant, val_f1 = chosen_llm_variant(provider.name)
    print(f"LLM ({provider.model}, {variant}-shot, chosen on validation: {val_f1}) ...")
    examples = None
    if variant == "few":
        val_sample = llm_sample(splits.validation, LLM_VALIDATION_SAMPLE)
        examples = llm.select_few_shot_examples(splits, splits.validation.drop(val_sample.index))
    clf = llm.LLMClassifier(splits.labels, examples)
    llm_preds = []
    for i, text in enumerate(sample["text"], start=1):
        llm_preds.append(clf.classify(text))
        if i % 50 == 0:
            print(f"  {i}/{len(sample)}")
    preds["llm"] = None
    preds.loc[sample.index, "llm"] = [splits.labels[p.label] for p in llm_preds]
    llm_ms = np.array([p.seconds for p in llm_preds]) * 1000
    results["llm"] = {
        "name": f"LLM {variant}-shot ({provider.model})",
        "variant": variant,
        "validation_macro_f1_by_variant": val_f1,
        "llm_sample": score(sample["label"], [p.label for p in llm_preds]),
        "latency": {
            "median_ms": round(float(np.median(llm_ms)), 2),
            "p95_ms": round(float(np.percentile(llm_ms, 95)), 2),
            "n": len(llm_ms),
        },
        "latency_device": "network round trip to the provider",
        "model_size_mb": None,
        "cost_per_1000_usd": round(float(np.mean([p.cost_usd for p in llm_preds])) * 1000, 4),
        "price_source": provider.price_source,
        "invalid_after_retry": sum(not p.valid for p in llm_preds),
    }

    report = {
        "test_size": len(test),
        "llm_sample_size": len(sample),
        "results": results,
        "environment": environment_info(),
    }
    (REPORTS_DIR / "results.json").write_text(json.dumps(report, indent=2))
    preds["in_llm_sample"] = preds.index.isin(sample.index)
    preds.to_csv(REPORTS_DIR / "test_predictions.csv", index_label="row")
    (REPORTS_DIR / "comparison.md").write_text(comparison_markdown(report))
    print((REPORTS_DIR / "comparison.md").read_text())


def comparison_markdown(report: dict) -> str:
    r = report["results"]
    env = report["environment"]

    def row(key, metrics_key):
        x, m = r[key], r[key][metrics_key]
        size = f"{x['model_size_mb']:.0f} MB" if x["model_size_mb"] else "hosted"
        cost = f"${x['cost_per_1000_usd']:.3f}" if x["cost_per_1000_usd"] else "$0 (self-hosted)"
        return (
            f"| {x['name']} | {m['accuracy']:.3f} | {m['in_scope_accuracy']:.3f} | "
            f"{m['macro_f1']:.3f} | {m['oos_precision']:.2f} | {m['oos_recall']:.2f} | "
            f"{x['latency']['median_ms']:.1f} | {x['latency']['p95_ms']:.1f} | {size} | {cost} |"
        )

    header = (
        "| Approach | Accuracy | In-scope acc. | Macro-F1 (150 intents) | OOS precision | "
        "OOS recall | Median latency (ms) | p95 latency (ms) | Model size | Cost / 1,000 req. |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    llm_r = r["llm"]
    return f"""# Final comparison on the held-out test set

Generated by `python scripts/compare.py` on {env["timestamp"]}. The test set was not used for any
choice: every setting, threshold and the LLM variant were picked on validation.

## All three approaches on the same {report["llm_sample_size"]}-row test sample

The LLM is run on a fixed random sample (seed 42) to fit the free API quota, so every approach is
also scored on exactly these rows.

{header}
{row("tfidf_logreg", "llm_sample")}
{row("distilbert", "llm_sample")}
{row("distilbert_threshold", "llm_sample")}
{row("llm", "llm_sample")}

## Local models on the full test set ({report["test_size"]} rows)

{header}
{row("tfidf_logreg", "test")}
{row("distilbert", "test")}
{row("distilbert_threshold", "test")}

## How these numbers were measured

- **Macro-F1** is over the 150 intents; out-of-scope (OOS) detection is reported separately as
  precision and recall of the `oos` label. Rows without "threshold" take the top class (argmax);
  the served row answers `oos` when the top probability is below the threshold chosen on
  validation by `scripts/choose_threshold.py`.
- **Latency**: one request at a time, median and p95 over {r["tfidf_logreg"]["latency"]["n"]} test
  queries after a warm-up, on {env["cpu"]} CPU ({env["platform"]}), Python {env["python"]},
  torch {env["libraries"].get("torch")}, scikit-learn {env["libraries"].get("scikit-learn")}.
  LLM latency is the network round trip when each answer was first fetched.
- **LLM**: {llm_r["name"]}, chosen over the other variant by validation macro-F1
  ({llm_r["validation_macro_f1_by_variant"]}). Invalid answers after retry on test:
  {llm_r["invalid_after_retry"]}.
- **Cost** uses the provider's paid-tier price per token: {llm_r["price_source"]}. The local
  models have no per-request fee; the hardware they run on is not priced in.
- **Model size** is the saved model on disk.
"""


if __name__ == "__main__":
    main()
