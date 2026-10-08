import pytest

from intentbench import llm

LABELS = ["balance", "oos", "transfer"]


@pytest.fixture
def fake_llm(monkeypatch, tmp_path):
    """Replace the provider call with a scripted fake and use a temporary cache."""
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    calls = []

    def install(*answers):
        def fake(system, user, model=llm.MODEL):
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
    assert first.cost_usd == pytest.approx(llm.cost_usd(100, 5))

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
