import pytest

from intentbench import llm

LABELS = ["balance", "oos", "transfer"]


@pytest.fixture(autouse=True)
def groq_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")


@pytest.fixture
def fake_llm(monkeypatch, tmp_path):
    """Replace the provider call with a scripted fake and use a temporary cache."""
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    calls = []

    def install(*answers):
        def fake(system, user, choices):
            calls.append(user)
            return llm.LLMResponse(
                answers[len(calls) - 1], input_tokens=100, output_tokens=5, seconds=0.5
            )

        monkeypatch.setattr(llm, "call_llm", fake)
        return calls

    return install


def test_valid_answer_is_parsed_and_cached(fake_llm):
    calls = fake_llm('{"intent": "transfer"}')
    clf = llm.LLMClassifier(LABELS)

    first = clf.classify("send money to mom")
    assert first.label == 2 and first.valid and not first.cached
    assert first.cost_usd == pytest.approx(llm.cost_usd(100, 5, p=llm.PROVIDERS["groq"]))

    second = clf.classify("send money to mom")  # served from disk, no new call
    assert second.cached and second.label == 2
    assert len(calls) == 1


def test_out_of_scope_maps_to_oos_label(fake_llm):
    fake_llm('{"intent": "out_of_scope"}')
    assert llm.LLMClassifier(LABELS).classify("what is the meaning of life").label == 1


def test_invalid_answer_gets_one_retry(fake_llm):
    calls = fake_llm("not json", '{"intent": "balance"}')
    pred = llm.LLMClassifier(LABELS).classify("how much do i have")
    assert pred.label == 0 and pred.valid
    assert len(calls) == 2
    assert pred.input_tokens == 200  # both calls are paid for


def test_still_invalid_after_retry_falls_back_to_oos(fake_llm):
    fake_llm('{"intent": "made_up"}', '{"intent": "also_made_up"}')
    pred = llm.LLMClassifier(LABELS).classify("hmm")
    assert pred.label == 1 and not pred.valid


def test_prompt_lists_every_intent_and_examples():
    system = llm.build_system_prompt(LABELS, {"balance": ["what's my balance"]})
    assert "balance" in system and "transfer" in system and "out_of_scope" in system
    assert '"what\'s my balance" -> balance' in system
    assert "\noos\n" not in system


def test_daily_quota_stops_immediately(monkeypatch):
    attempts = []

    def send(model, system, user, choices):
        attempts.append(1)
        raise llm.DailyQuotaExceeded("daily limit")

    monkeypatch.setitem(llm._SENDERS, "groq", send)
    monkeypatch.setattr(llm, "_wait_for_rate_limit", lambda rpm: None)
    with pytest.raises(llm.DailyQuotaExceeded):
        llm.call_llm("system", "user", LABELS)
    assert len(attempts) == 1


def test_temporary_errors_are_retried(monkeypatch):
    outcomes = [llm._Retryable(503), llm._Retryable("APITimeoutError"), "ok"]

    def send(model, system, user, choices):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return llm.LLMResponse('{"intent": "balance"}', input_tokens=10, output_tokens=2, seconds=0)

    monkeypatch.setitem(llm._SENDERS, "groq", send)
    monkeypatch.setattr(llm, "_wait_for_rate_limit", lambda rpm: None)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    assert llm.call_llm("system", "user", LABELS).text == '{"intent": "balance"}'
    assert outcomes == []


def test_cost_counts_cached_input_at_the_cached_price():
    p = llm.PROVIDERS["groq"]
    cost = llm.cost_usd(1000, 100, cached_input_tokens=800, p=p)
    expected = (
        200 * p.price_input_per_m + 800 * p.price_cached_input_per_m + 100 * p.price_output_per_m
    ) / 1e6
    assert cost == pytest.approx(expected)
