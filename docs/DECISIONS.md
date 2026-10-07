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
