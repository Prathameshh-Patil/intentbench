---
language: en
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: distilbert-base-uncased
datasets:
  - clinc/clinc_oos
metrics:
  - accuracy
  - f1
---

# IntentBench DistilBERT: 150-intent classifier with out-of-scope detection

A `distilbert-base-uncased` model fine-tuned to sort a short user message (one sentence, like a request to a voice assistant or chatbot) into one of 150 intents, or `oos` (out-of-scope). It is the model served by the [IntentBench](https://github.com/Prathameshh-Patil/intentbench) API and demo. All numbers below come from scripts in that repo.

## Data

[CLINC150](https://huggingface.co/datasets/clinc/clinc_oos), `plus` configuration, pinned to revision `155b9c7` (Larson et al., 2019, CC BY 3.0). The 150 intents cover 10 domains: banking, credit cards, work, travel, utility, auto and commute, kitchen and dining, home, small talk, and meta. The `plus` configuration adds more out-of-scope training data.

| Split | Examples | Out-of-scope |
|---|---|---|
| train | 15,201 | 250 (1.6%) |
| validation | 3,096 | 100 (3.2%) |
| test | 5,500 | 1,000 (18.2%) |

The official splits are used. Queries that duplicate a test or validation query, ignoring case and punctuation, were removed from train (49) and validation (4). Test is untouched.

## Training

| Setting | Value |
|---|---|
| Base model | `distilbert-base-uncased` (66M parameters) |
| Head | new 151-way linear classifier (`oos` is the 151st class) |
| Learning rate | 5e-5, linear decay, 10% warm-up |
| Batch size | 32 |
| Epochs | 8, keeping the epoch with the best validation macro-F1 (epoch 8) |
| Weight decay | 0.01 |
| Max length | 64 tokens |
| Seed | 42 |
| Hardware | Apple M5 GPU (MPS), 746 s |

Reproduce with `python scripts/run_finetune.py` (or `notebooks/02_finetune_on_gpu.ipynb` on Colab).

**Out-of-scope threshold.** At inference, if the top probability is below **0.69**, the answer is `out_of_scope`. The threshold was chosen on validation as the highest out-of-scope recall that costs at most 1 point of in-scope accuracy (`scripts/choose_threshold.py`, `reports/threshold.json`).

## Metrics (held-out test set, evaluated once)

Full test set, 5,500 queries (`scripts/compare.py`, `reports/comparison.md`):

| | Accuracy | In-scope accuracy | Macro-F1 (150 intents) | OOS precision | OOS recall |
|---|---|---|---|---|---|
| This model, threshold 0.69 (as served) | 0.918 | 0.949 | 0.939 | 0.87 | 0.78 |
| This model, no threshold | 0.892 | 0.958 | 0.920 | 0.98 | 0.59 |
| TF-IDF + logistic regression (baseline) | 0.843 | 0.917 | 0.877 | 0.91 | 0.51 |

On the 400-row sample where all three approaches were run, the served model scores accuracy 0.910, against 0.897 for few-shot Gemini (`gemini-3.5-flash-lite`).

**Speed and size.** Median latency is 12 ms per single request (p95 24 ms) on an Apple M5 CPU, and 269 MB on disk.

## Intended use

- Routing short, single-intent English messages in the domains above, for example in a demo assistant, or as a starting point for a production intent classifier after retraining on your own intents.
- Teaching and benchmarking: comparing a classical model, a fine-tuned transformer and an LLM on the same data.

## Limits

- **Out-of-scope detection is the weak spot.** About 1 in 5 out-of-scope test queries still gets an intent, especially when it sits next to a real domain ("add mary to my phone plan" becomes `sync_device`). The threshold also rejects about 1 in 100 valid queries.
- **Probabilities are over-confident.** Validation loss rose after epoch 4 while accuracy kept improving. Treat the confidence as a ranking score, not a calibrated probability.
- **Narrow domain.** It knows only these 150 intents. Anything else should come out as `out_of_scope` at best, and it can't learn new intents without retraining.
- **Data.** The queries are crowd-sourced, short, English-only and US-centric. They include typos and some label noise (for example, name-change requests labelled `pin_change`). Performance on long messages, other languages or other phrasing styles is unknown.
- **Not for high-stakes decisions.** A wrong intent in banking (for example `transfer` or `freeze_account`) should always be confirmed with the user before any action.

## How to use

```python
from intentbench.serve.model import Predictor

p = Predictor("Prathameshh-Patil/intentbench-distilbert")  # or a local folder
p.classify("can you freeze my card, I think it was stolen")
# {'label': 'report_lost_card', 'confidence': 0.98, 'top_intents': [...]}
```

Or with plain Transformers: `pipeline("text-classification", model="Prathameshh-Patil/intentbench-distilbert")`. Note that the plain pipeline does not apply the 0.69 threshold.

**Getting the model files.** The repo does not include model weights (they're git-ignored). Either train them with `python scripts/run_baseline.py` and `python scripts/run_finetune.py`, or download them from the Hugging Face Hub repo above once it's published. The demo downloads them automatically when `models/` is empty.
