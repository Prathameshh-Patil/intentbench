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
