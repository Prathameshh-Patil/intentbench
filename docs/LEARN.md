# LEARN

What was built in each phase, why it is designed that way, and what the alternative would have been.

## Phase 0: Machine setup and repository

**What we built.** An empty but complete project skeleton: the folder layout, a README stub, an MIT license, a `.gitignore`, an empty `.env.example`, and a public GitHub repository with `main` pushed.

**Why it is designed this way.**
- *Folder layout first.* Deciding where things live before writing code keeps each file small and with one job (`data.py` only loads data, `baseline.py` only does the baseline, and so on). The `src/intentbench/` "src layout" means tests import the installed package, not a stray file from the current folder, which catches packaging mistakes early.
- *`.gitignore` before the first commit.* Once a file is committed, it lives in git history forever, even if you delete it later. Ignoring `.env` (secrets), model weights (hundreds of MB) and caches from day one means they can never be committed by accident.
- *`.env.example`.* Shows anyone cloning the repo which settings they must provide, without exposing real values. The real `.env` stays only on your machine.
- *`.gitkeep` files.* Git tracks files, not folders. An empty placeholder keeps the planned folders visible on GitHub.
- *MIT license.* Without a license, nobody is legally allowed to reuse public code. MIT is the shortest common permissive license, so it's a sensible default for a portfolio project.

**Alternatives.** A project template generator (such as cookiecutter) could have produced this skeleton, but it adds files you didn't choose and can't explain. A flat layout (code at the repo root) is simpler, but it lets tests pass by accident because they import local files instead of the installed package.

## Phase 1: Environment, data and exploration

**What we built.**
- `pyproject.toml`: the project's name, dependencies (`datasets`, `pandas`, `matplotlib`) and dev tools (`pytest`, `ruff`, Jupyter), plus ruff and pytest settings.
- A Python 3.12 virtual environment in `.venv/`, with the project installed in editable mode (`pip install -e ".[dev]"`).
- `src/intentbench/data.py`: `load_splits()` returns train/validation/test DataFrames (`text`, `label`, `intent`) and the label list. `python -m intentbench.data` prints the split sizes.
- `tests/test_data.py`: checks that no query appears in two splits, that test is the official 5,500 rows, and that the label maps are consistent.
- `notebooks/01_explore.ipynb` and `reports/figures/query_length.png`.
- `.github/workflows/ci.yml`: runs ruff and pytest on every push and pull request.

**Why it is designed this way.**
- *One loader for everything.* Every approach (baseline, DistilBERT, LLM) calls `load_splits()`. If they each loaded data their own way, a small difference such as lowercasing or a different dedupe could make the comparison unfair without anyone noticing.
- *Official splits instead of re-splitting with a seed.* CLINC150 already ships fixed splits that published papers use. Keeping them means our numbers can be compared with the literature. `SEED = 42` still lives in `data.py` for every later random step.
- *Pinned dataset revision.* The Hub dataset could be edited one day. Pinning the commit hash means `load_splits()` returns identical data forever.
- *Leakage removal.* We found queries that appear in two splits once case and punctuation are ignored ("bye" vs "bye!"). A model gets those right by memory, not by understanding, which inflates scores. We dropped 49 from train and 4 from validation, and left test untouched. The test `test_splits_never_overlap` makes sure this can never come back silently.
- *Why test is used only once.* Every time you look at a test score and change something, you are tuning to the test set, a little. Do it enough times and the test score stops predicting performance on new data. So we make all choices on validation and look at test once, at the very end.
- *CI.* Your Mac is not a clean machine. It has caches and packages the project never declared. CI installs from scratch on Linux, so a green tick means "a stranger can install and run this".

**What I noticed in the data.**
- Train has under 2% out-of-scope; test has 18%. Out-of-scope detection will be the hardest part.
- Queries are short (median 8 words). Out-of-scope ones are the same length, so length gives no hint.
- Some duplicates carry contradictory labels (label noise), and there are real typos.

**Alternatives.**
- Re-splitting with `train_test_split(seed=42)` would also be reproducible, but the numbers would no longer be comparable with published work.
- `uv` or Poetry would manage the environment faster. Plain `venv` + `pip` has nothing extra to learn and is what the brief's stack implies.
- Exact-match dedupe would have missed 40+ near-duplicates; embedding-based dedupe would catch paraphrases too, but it is harder to explain and would remove legitimate variety.

## Phase 2: Classical baseline

**What we built.**
- `src/intentbench/evaluate.py`: the metric function every approach uses, and `environment_info()` (Python, CPU, library versions) stamped into every report.
- `src/intentbench/baseline.py`: a scikit-learn `Pipeline` of `TfidfVectorizer` then `LogisticRegression`, plus a small grid search on validation.
- `scripts/run_baseline.py`: tunes, saves `models/baseline.joblib` (41 MB, git-ignored) and writes `reports/baseline_validation.json`. It runs in about 85 seconds on an M-series Mac.

**Validation results (best: char 2-5-grams, C=100).** Macro-F1 0.914, in-scope accuracy 0.915, out-of-scope recall 0.69, out-of-scope precision 0.72.

**TF-IDF, in plain words.** Each query becomes a long vector with one slot per vocabulary item (a word, a word pair, or a short run of characters).
- *TF (term frequency):* how often the item appears in this query.
- *IDF (inverse document frequency):* how rare the item is across all training queries.

Multiplying them makes common filler ("what", "is", "my") count for little, while rare, telling items ("transfer", "w2", "calories") count for a lot. With `analyzer="char_wb"`, the items are 2-5 character pieces like `"inte"` or `"rest"`. So the typo "intetest" still shares most of its pieces with "interest", which is why char n-grams won here.

**Logistic regression with 151 classes.** The model learns one weight per (feature, class) pair. For a query, it adds up the weights of the features present to get a score for each class. *Softmax* then turns the 151 scores into probabilities that sum to 1. Training adjusts the weights to make the correct class's probability high. `C` controls regularization, a penalty that keeps weights small:
- *Small C* means strong penalty and a simpler model that may underfit (C=1 was worst for every feature type).
- *Large C* lets the model fit training data closely and risk overfitting (C=1000 dropped again).

The best value sits in between, and the grid showed that.

**Macro-F1 versus accuracy.**
- *Accuracy* is the share of predictions that are right, so big classes dominate it.
- *Macro-F1* computes F1 (the balance of precision and recall) for each intent separately, then averages them with equal weight. One badly handled intent pulls it down even if it's rare.

Our 150 intents are balanced, so the two are close here. The difference shows up with imbalanced data, and with out-of-scope: wrongly predicting `transfer` for an out-of-scope query is a false positive that lowers `transfer`'s precision, and so macro-F1.

**Why a strong baseline matters.** Without it, "DistilBERT gets 95%" sounds impressive but means nothing. The real question is how much better than a 90-second, 41 MB, CPU-only model it is, and whether that gain pays for the extra cost. A weak baseline (for example, untuned with default settings) makes every fancier model look good. A tuned one keeps the comparison honest.

**Alternatives.**
- A linear SVM would score about the same, but it gives no calibrated probabilities, and we need confidence scores for the API and the out-of-scope threshold.
- Naive Bayes is faster but usually weaker on this kind of task.
- Proper k-fold cross-validation would give more stable tuning, but CLINC already provides a validation split, and folds would cost 5x the time for a baseline.
