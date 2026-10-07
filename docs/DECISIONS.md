# Design decisions

One line per decision, with the reason.

- **src layout (`src/intentbench/`)**: tests run against the installed package, so packaging mistakes show up early.
- **Python 3.12 for the virtual environment** (not the system default 3.14): PyTorch and Transformers ship the most dependable pre-built packages for 3.12.
- **MIT license**: short, permissive, standard for portfolio code.
- **Model weights, dataset caches and LLM cache are git-ignored**: they are large or regenerable; the model card explains how to obtain the models.
- **LLM response cache lives in `.llm_cache/`**: a fixed, ignored location so re-runs are free and reproducible.
