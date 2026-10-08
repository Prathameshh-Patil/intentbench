"""Run the zero-shot or few-shot LLM classifier on the fixed validation sample.

Usage:
  python scripts/run_llm.py --variant zero --limit 20   # dry run: 20 requests + cost estimate
  python scripts/run_llm.py --variant zero              # full validation sample
  python scripts/run_llm.py --variant few
Outputs (full run only): reports/llm_<variant>_validation.json and ..._requests.csv
"""

import argparse
import json

import numpy as np
import pandas as pd

from intentbench import baseline, llm
from intentbench.data import LLM_TEST_SAMPLE, LLM_VALIDATION_SAMPLE, llm_sample, load_splits
from intentbench.evaluate import REPORTS_DIR, classification_metrics, environment_info
from intentbench.finetune import DistilBertClassifier


def latency_stats(seconds: list[float]) -> dict:
    if not seconds:
        return {}
    return {
        "median_s": round(float(np.median(seconds)), 3),
        "p95_s": round(float(np.percentile(seconds, 95)), 3),
        "n": len(seconds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["zero", "few"], required=True)
    parser.add_argument("--limit", type=int, help="only the first N sample rows (dry run)")
    args = parser.parse_args()

    splits = load_splits()
    sample = llm_sample(splits.validation, LLM_VALIDATION_SAMPLE)
    examples = None
    if args.variant == "few":
        dev = splits.validation.drop(sample.index)  # never the rows we score on
        examples = llm.select_few_shot_examples(splits, dev)
    clf = llm.LLMClassifier(splits.labels, examples)
    rows = sample.head(args.limit) if args.limit else sample

    preds = []
    for i, text in enumerate(rows["text"], start=1):
        preds.append(clf.classify(text))
        if args.limit or i % 50 == 0:
            p = preds[-1]
            print(
                f"{i:>4}/{len(rows)}  {splits.labels[p.label]:<24} in={p.input_tokens:<5} "
                f"out={p.output_tokens:<3} {p.seconds:5.2f}s  ${p.cost_usd:.6f}"
                + ("  (cached)" if p.cached else "")
            )

    requests = pd.DataFrame(
        {
            "text": rows["text"].values,
            "true": [splits.labels[i] for i in rows["label"]],
            "pred": [splits.labels[p.label] for p in preds],
            **{
                k: [getattr(p, k) for p in preds]
                for k in ("valid", "cached", "input_tokens", "output_tokens", "seconds", "cost_usd")
            },
        }
    )
    metrics = classification_metrics(
        rows["label"], [p.label for p in preds], splits.oos_id, len(splits.labels)
    )
    mean_cost = requests["cost_usd"].mean()
    print(f"\nPrompt (system) tokens per request: ~{int(requests['input_tokens'].median())}")
    print(f"Mean cost per request at paid-tier prices: ${mean_cost:.6f}")
    print(f"  -> per 1,000 requests: ${mean_cost * 1000:.3f}")
    planned = LLM_VALIDATION_SAMPLE + LLM_TEST_SAMPLE
    print(
        f"  -> this variant on validation ({LLM_VALIDATION_SAMPLE}) + test ({LLM_TEST_SAMPLE}): "
        f"${mean_cost * planned:.2f}"
    )
    print(f"Invalid answers after retry: {int((~requests['valid']).sum())}")
    print(f"Accuracy on these {len(rows)}: {metrics['accuracy']:.3f}")

    if args.limit:
        print("\nDry run only: no report written.")
        return

    # Every approach on the very same rows, so numbers are directly comparable.
    reference = {
        "tfidf_logreg": classification_metrics(
            sample["label"],
            baseline.load().predict(sample["text"]),
            splits.oos_id,
            len(splits.labels),
        ),
        "distilbert": classification_metrics(
            sample["label"],
            DistilBertClassifier().predict(sample["text"].tolist()),
            splits.oos_id,
            len(splits.labels),
        ),
    }
    fresh = requests[~requests["cached"]]
    report = {
        "approach": f"llm_{args.variant}_shot",
        "split": "validation",
        "sample_size": len(rows),
        "model": llm.MODEL,
        "metrics": metrics,
        "other_approaches_on_same_sample": reference,
        "tokens": {
            "input_mean": round(float(requests["input_tokens"].mean()), 1),
            "output_mean": round(float(requests["output_tokens"].mean()), 1),
        },
        "cost": {
            "per_request_usd": round(float(mean_cost), 7),
            "per_1000_requests_usd": round(float(mean_cost * 1000), 4),
            "price_input_per_m": llm.PRICE_INPUT_PER_M,
            "price_output_per_m": llm.PRICE_OUTPUT_PER_M,
            "price_source": llm.PRICE_SOURCE,
        },
        "latency_uncached": latency_stats(fresh["seconds"].tolist()),
        "latency_all_original_calls": latency_stats(requests["seconds"].tolist()),
        "invalid_after_retry": int((~requests["valid"]).sum()),
        "few_shot_examples": examples,
        "system_prompt": clf.system,
        "environment": environment_info(),
    }
    stem = f"llm_{args.variant}_validation"
    (REPORTS_DIR / f"{stem}.json").write_text(json.dumps(report, indent=2))
    requests.to_csv(REPORTS_DIR / f"{stem}_requests.csv", index=False)
    print(f"\nWrote reports/{stem}.json and {stem}_requests.csv")
    for name, m in {"llm": metrics, **reference}.items():
        print(
            f"  {name:<14} macro-F1 {m['macro_f1']:.3f}  in-scope acc {m['in_scope_accuracy']:.3f}"
            f"  OOS recall {m['oos_recall']:.2f}  OOS precision {m['oos_precision']:.2f}"
        )


if __name__ == "__main__":
    main()
