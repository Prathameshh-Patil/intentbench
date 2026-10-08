# Design decisions

One line per decision, with the reason.

- **src layout (`src/intentbench/`)**: tests run against the installed package, so packaging mistakes show up early.
- **Python 3.12 for the virtual environment** (not the system default 3.14): PyTorch and Transformers ship the most dependable pre-built packages for 3.12.
- **MIT license**: short, permissive, standard for portfolio code.
- **Model weights, dataset caches and LLM cache are git-ignored**: they are large or regenerable; the model card explains how to obtain the models.
- **LLM response cache lives in `.llm_cache/`**: a fixed, ignored location so re-runs are free and reproducible.
- **Git remote uses SSH, not HTTPS**: the `gh` token lacked the `workflow` scope needed to push `.github/workflows/`; SSH authenticates with your own key instead.
- **Dataset id `clinc/clinc_oos`, pinned to commit `155b9c7`**: the old short name `clinc_oos` no longer resolves in `datasets` 5.x; pinning the commit means a re-run downloads identical data.
- **Use the official CLINC150 splits instead of re-splitting with a seed**: published results use them, so our numbers are comparable; `SEED = 42` is still defined in `data.py` for every random step later (sampling, training).
- **Remove cross-split near-duplicates (case/punctuation-insensitive)**: 49 train rows duplicated validation/test queries and 4 validation rows duplicated test queries. Test is untouched; leaked rows are dropped from train and validation, so no score is inflated by memorisation.
- **Shared metric code in `evaluate.py` from Phase 2**: every approach is scored by the same function, so definitions (e.g. macro-F1 over the 150 intents) can't drift between approaches.
- **Baseline model selection by validation macro-F1 over the 150 intents**: it is the headline metric in the brief; out-of-scope gets its own threshold work in Phase 7.
- **Baseline grid: word 1-gram / word 1-2-gram / char 2-5-gram x C in {1, 10, 100, 1000}**: char n-grams were added because the data has typos; C=1000 was added after C=100 won at the edge of the first grid (it scored lower, so the optimum is inside the grid).
- **`out_of_scope` is the 151st class for the baseline**: the simplest option; whether a confidence threshold helps is decided in Phase 7 on validation.
- **Fine-tune locally on the Mac GPU (MPS) instead of a hosted GPU**: Apple M5, ~70 s per epoch, 8 epochs in 12.4 min; `02_finetune_on_gpu.ipynb` still exists as a thin Colab wrapper around the same script for people without a GPU.
- **`accelerate` added as a dependency**: the Hugging Face `Trainer` will not run without it; it is part of the Transformers training stack, not a new tool choice.
- **DistilBERT settings: lr 5e-5, batch 32, 8 epochs, 10% warmup, weight decay 0.01, keep best epoch by validation macro-F1**: standard values for fine-tuning BERT-size models on small classification tasks; no hyperparameter search, because the baseline comparison is the point, not squeezing out 0.2%.
- **No extra epochs although the best epoch was the last**: validation macro-F1 is flat from epoch 5 (+0.003) and the linear learning-rate schedule reaches zero at epoch 8; validation loss already rose after epoch 4 (over-confidence), which matters for the Phase 7 threshold.
- **CI installs CPU-only PyTorch**: CI never trains; the default Linux wheel bundles GBs of CUDA libraries.
- **LLM provider: Google Gemini, model `gemini-3.5-flash-lite`**: you had a Gemini key; Flash-Lite is Google's cheapest, fastest model with a free tier, which suits a high-volume, short-answer task. Thinking is set to `minimal` (its default) to keep latency and cost low.
- **Provider isolated in one function, `llm.call_llm`**: swapping provider means rewriting only that function.
- **Cost uses the paid "Standard" price ($0.30 / $2.50 per 1M input / output tokens)** from https://ai.google.dev/gemini-api/docs/pricing (last updated 2026-10-07, checked 2026-10-08), even though the runs used the free tier: it is what the approach would really cost in production.
- **Strict parsing + one retry instead of a schema enum**: Gemini rejects a response-schema enum with all 151 labels (it accepted up to 97). The schema still forces `{"intent": "<string>"}`; only an exact intent name is accepted; an invalid answer is retried once with a note, then counted as invalid and mapped to out-of-scope.
- **`oos` is shown to the LLM as `out_of_scope`**: the model has no way to know what "oos" means; the dataset label is mapped back when scoring.
- **Fixed LLM samples: 500 validation rows, 400 test rows (`data.llm_sample`, seed 42)**: the Gemini free tier allows 500 requests per day per model, and we chose to stay free; the test sample was cut from 1,000 to 400 rows (800 requests for both variants, about 2 days). Cost: out-of-scope numbers on this sample rest on fewer examples. Every approach is also scored on exactly these rows.
- **Few-shot examples: 3 train examples for each intent in the baseline's 15 most-confused pairs, plus 10 out-of-scope examples**: "confusing" is measured, not guessed from names (name overlap flagged 86 intents, mostly noise). Confusions are measured on validation rows *outside* the LLM's scored sample, so the example choice is not tuned to the rows it is graded on.
- **Response cache in `.llm_cache/`, one JSON file per request, keyed by a hash of model + system prompt + query**: a re-run costs nothing and gives identical predictions; changing the prompt changes the key, so stale answers are never reused.
- **No `python-dotenv`**: a 6-line `load_env()` reads `.env`, avoiding a new dependency.
- **Client-side pacing at `GEMINI_RPM` requests per minute (default 15)**: the free tier rejects more than 15 requests per minute for this model; pacing avoids the errors instead of retrying through them. Latency is timed from when the request is sent, so the wait is not counted.
- **Stop immediately when the daily quota is exhausted**: retrying a per-day 429 is pointless; the cache keeps finished answers, so the run resumes where it stopped after the reset.
