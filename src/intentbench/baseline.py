"""Classical baseline: TF-IDF features + multinomial logistic regression.

Out-of-scope is treated as just another class (the 151st), learned from the 250 train examples.
"""

import itertools

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from intentbench.data import Splits
from intentbench.evaluate import MODELS_DIR, classification_metrics

MODEL_PATH = MODELS_DIR / "baseline.joblib"

# The settings we tune on validation.
# - features: words, word pairs, or character 2-5-grams (robust to typos like "intetest")
# - C: inverse regularization strength; larger C = weaker regularization = fits train harder
FEATURES = {
    "word 1-gram": {"analyzer": "word", "ngram_range": (1, 1)},
    "word 1-2-gram": {"analyzer": "word", "ngram_range": (1, 2)},
    "char 2-5-gram": {"analyzer": "char_wb", "ngram_range": (2, 5)},
}
C_VALUES = [1.0, 10.0, 100.0, 1000.0]


def build_pipeline(analyzer: str = "word", ngram_range=(1, 2), C: float = 10.0) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, sublinear_tf=True),
            ),
            ("clf", LogisticRegression(C=C, max_iter=2000)),
        ]
    )


def tune(splits: Splits) -> tuple[Pipeline, list[dict]]:
    """Fit one pipeline per setting on train, score on validation, return the best by macro-F1."""
    results, best, best_f1 = [], None, -1.0
    for (name, features), C in itertools.product(FEATURES.items(), C_VALUES):
        pipe = build_pipeline(**features, C=C)
        pipe.fit(splits.train["text"], splits.train["label"])
        y_pred = pipe.predict(splits.validation["text"])
        metrics = classification_metrics(
            splits.validation["label"], y_pred, splits.oos_id, len(splits.labels)
        )
        results.append({"features": name, "C": C, **metrics})
        print(f"  {name:<14} C={C:<6g} macro-F1 {metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] > best_f1:
            best, best_f1 = pipe, metrics["macro_f1"]
    return best, results


def save(pipe: Pipeline, path=MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load(path=MODEL_PATH) -> Pipeline:
    return joblib.load(path)
