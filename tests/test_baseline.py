import pytest

from intentbench.baseline import build_pipeline, load, save
from intentbench.evaluate import classification_metrics

TEXTS = ["what is my balance", "check my balance", "play a song", "play some music"]
LABELS = [0, 0, 1, 1]


def test_pipeline_learns_toy_data(tmp_path):
    pipe = build_pipeline(C=100.0).fit(TEXTS, LABELS)
    assert list(pipe.predict(["my balance please", "play music"])) == [0, 1]
    assert pipe.predict_proba(["balance"]).shape == (1, 2)

    save(pipe, tmp_path / "m.joblib")
    assert list(load(tmp_path / "m.joblib").predict(["play a song"])) == [1]


def test_classification_metrics():
    # labels 0 and 1 are intents, 2 is out-of-scope
    y_true = [0, 0, 1, 2, 2]
    y_pred = [0, 1, 1, 2, 0]
    m = classification_metrics(y_true, y_pred, oos_id=2, n_labels=3)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["in_scope_accuracy"] == pytest.approx(2 / 3)
    assert m["oos_precision"] == pytest.approx(1.0)
    assert m["oos_recall"] == pytest.approx(0.5)
    # intent 0: P=1/2 R=1/2 F1=0.5; intent 1: P=1/2 R=1 F1=2/3
    assert m["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 2)
