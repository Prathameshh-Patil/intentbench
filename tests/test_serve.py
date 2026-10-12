import pytest
from fastapi.testclient import TestClient

from intentbench.serve.app import create_app
from intentbench.serve.model import decide
from intentbench.serve.schemas import MAX_TEXT_CHARS

LABELS = ["balance", "oos", "transfer", "freeze_account"]


class FakePredictor:
    """Stands in for the real model so the API tests run in milliseconds, without weights."""

    source = "fake"
    base_model = "distilbert-base-uncased"
    labels = LABELS
    threshold = 0.5

    def classify(self, text):
        proba = [0.1, 0.05, 0.05, 0.8] if "freeze" in text else [0.4, 0.3, 0.2, 0.1]
        return decide(proba, self.labels, self.threshold)


@pytest.fixture
def client():
    with TestClient(create_app(FakePredictor)) as c:  # `with` runs the startup that loads it
        yield c


def test_decide_confident():
    out = decide([0.1, 0.05, 0.05, 0.8], LABELS, threshold=0.5)
    assert out["label"] == "freeze_account"
    assert out["confidence"] == pytest.approx(0.8)
    assert [x["intent"] for x in out["top_intents"]] == [
        "freeze_account",
        "balance",
        "out_of_scope",
    ]


def test_decide_below_threshold_is_out_of_scope():
    out = decide([0.4, 0.3, 0.2, 0.1], LABELS, threshold=0.5)
    assert out["label"] == "out_of_scope"
    assert out["top_intents"][0]["intent"] == "balance"  # still shows the best guesses


def test_decide_predicted_oos_uses_api_name():
    assert decide([0.0, 0.9, 0.05, 0.05], LABELS, threshold=0.5)["label"] == "out_of_scope"


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["num_labels"] == 4
    assert info["oos_threshold"] == 0.5
    assert info["max_text_chars"] == MAX_TEXT_CHARS


def test_classify(client):
    r = client.post("/classify", json={"text": "  please freeze my card  "})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "freeze_account"
    assert len(body["top_intents"]) == 3


@pytest.mark.parametrize(
    "payload",
    [{"text": ""}, {"text": "   \n "}, {"text": "x" * (MAX_TEXT_CHARS + 1)}, {}, {"text": 42}],
)
def test_classify_rejects_bad_input(client, payload):
    assert client.post("/classify", json=payload).status_code == 422


def test_classify_accepts_max_length(client):
    assert client.post("/classify", json={"text": "a" * MAX_TEXT_CHARS}).status_code == 200
