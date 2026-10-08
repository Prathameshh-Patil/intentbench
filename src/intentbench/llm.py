"""Zero-shot and few-shot intent classification with an LLM.

- The provider (Gemini) is used only inside `call_llm`. To switch provider, rewrite that one
  function; everything else stays the same.
- Output is forced into the JSON shape {"intent": "<string>"} by a response schema, then parsed
  strictly: only an exact intent name is accepted, and an invalid answer gets one retry.
  (A schema enum of all 151 names would be stricter, but Gemini rejects enums that large.)
- Every response is cached on disk in `.llm_cache/`, keyed by model + prompt + query, so a
  re-run costs nothing and returns exactly the same predictions.
"""

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from functools import cache

import pandas as pd

from intentbench.data import OOS_LABEL, SEED, Splits
from intentbench.evaluate import ROOT

MODEL = "gemini-3.5-flash-lite"
# USD per 1M tokens, Gemini API paid tier, "Standard" (real-time) rates.
PRICE_INPUT_PER_M = 0.30
PRICE_OUTPUT_PER_M = 2.50
PRICE_SOURCE = (
    "https://ai.google.dev/gemini-api/docs/pricing (gemini-3.5-flash-lite, Standard paid tier; "
    "page last updated 2026-10-07, checked 2026-10-08)"
)

CACHE_DIR = ROOT / ".llm_cache"
OOS_NAME = "out_of_scope"  # the LLM sees this instead of the dataset's terse "oos"
MAX_RATE_LIMIT_RETRIES = 8


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float


def load_env(path=ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip())


@cache
def _gemini_client():
    from google import genai

    load_env()
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set. Put it in .env (see .env.example).")
    return genai.Client()


def call_llm(system: str, user: str, model: str = MODEL) -> LLMResponse:
    """Send one request; the answer is forced to be JSON {"intent": "<string>"}.

    This is the only provider-specific function in the project.
    """
    from google.genai import errors, types

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0,
        seed=SEED,
        max_output_tokens=50,
        response_mime_type="application/json",
        response_json_schema={
            "type": "object",
            "properties": {"intent": {"type": "string"}},
            "required": ["intent"],
        },
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        start = time.perf_counter()
        try:
            r = _gemini_client().models.generate_content(model=model, contents=user, config=config)
        except errors.APIError as e:
            # 429 = rate limited, 5xx = temporary server problem: wait and try again.
            if e.code in (429, 500, 503) and attempt < MAX_RATE_LIMIT_RETRIES:
                time.sleep(min(2**attempt, 60))
                continue
            raise
        seconds = time.perf_counter() - start
        u = r.usage_metadata
        return LLMResponse(
            text=r.text or "",
            input_tokens=u.prompt_token_count or 0,
            # thinking tokens are billed as output
            output_tokens=(u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
            seconds=seconds,
        )
    raise RuntimeError("unreachable")


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * PRICE_INPUT_PER_M + output_tokens * PRICE_OUTPUT_PER_M) / 1e6


def _display_names(labels: list[str]) -> list[str]:
    return [OOS_NAME if name == OOS_LABEL else name for name in labels]


def build_system_prompt(labels: list[str], examples: dict[str, list[str]] | None = None) -> str:
    names = sorted(n for n in _display_names(labels) if n != OOS_NAME)
    parts = [
        "You classify short messages that users send to a virtual assistant into exactly one "
        "intent. The assistant handles banking, credit cards, travel, home, auto, work, "
        "kitchen and dining, utilities, and small talk.",
        "Intents:\n" + "\n".join(names),
        f"If the message does not clearly match one of these intents, answer {OOS_NAME}. "
        "Many messages are out of scope; do not force a match.",
    ]
    if examples:
        lines = [f'"{text}" -> {intent}' for intent, texts in examples.items() for text in texts]
        parts.append("Examples for intents that are easy to mix up:\n" + "\n".join(lines))
    parts.append('Answer with JSON: {"intent": "<intent name>"}.')
    return "\n\n".join(parts)


def select_few_shot_examples(
    splits: Splits, dev: pd.DataFrame, n_pairs: int = 15, per_intent: int = 3, n_oos: int = 10
) -> dict[str, list[str]]:
    """Pick train examples for the intents the baseline confuses most on `dev`.

    `dev` must not overlap the rows the LLM is scored on, so the choice of examples is not
    tuned to the evaluation rows. Out-of-scope always gets examples: every approach struggles
    with it.
    """
    from intentbench import baseline

    pred = baseline.load().predict(dev["text"])
    pairs = Counter((t, p) for t, p in zip(dev["label"], pred, strict=True) if t != p).most_common(
        n_pairs
    )
    confusing = sorted({i for pair, _ in pairs for i in pair} - {splits.oos_id})

    train = splits.train
    examples = {}
    for label_id in confusing:
        rows = train[train["label"] == label_id].sample(per_intent, random_state=SEED)
        examples[splits.labels[label_id]] = rows["text"].tolist()
    oos_rows = train[train["label"] == splits.oos_id].sample(n_oos, random_state=SEED)
    examples[OOS_NAME] = oos_rows["text"].tolist()
    return examples


@dataclass
class Prediction:
    label: int
    raw: str
    valid: bool
    cached: bool
    input_tokens: int
    output_tokens: int
    seconds: float
    cost_usd: float


class LLMClassifier:
    def __init__(
        self, labels: list[str], examples: dict[str, list[str]] | None = None, model: str = MODEL
    ):
        self.labels = labels
        self.model = model
        self.system = build_system_prompt(labels, examples)
        self.choices = _display_names(labels)
        self.name_to_id = {name: i for i, name in enumerate(self.choices)}

    def _parse(self, text: str) -> int | None:
        try:
            intent = json.loads(text)["intent"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
        if not isinstance(intent, str):
            return None
        return self.name_to_id.get(intent.strip().lower())

    def classify(self, query: str) -> Prediction:
        key = hashlib.sha256(json.dumps([self.model, self.system, query]).encode()).hexdigest()
        path = CACHE_DIR / f"{key}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            fields = {k: saved[k] for k in Prediction.__dataclass_fields__ if k != "cached"}
            return Prediction(**fields, cached=True)

        user = f"Message: {query}"
        responses = [call_llm(self.system, user, self.model)]
        label = self._parse(responses[0].text)
        if label is None:  # one retry that says what was wrong
            reminder = (
                f"{user}\n\nYour previous answer {responses[0].text.strip()!r} is not one of the "
                "listed intents. Answer with exactly one intent name from the list."
            )
            responses.append(call_llm(self.system, reminder, self.model))
            label = self._parse(responses[-1].text)

        in_tok = sum(r.input_tokens for r in responses)
        out_tok = sum(r.output_tokens for r in responses)
        pred = Prediction(
            label=label if label is not None else self.name_to_id[OOS_NAME],
            raw=responses[-1].text,
            valid=label is not None,
            cached=False,
            input_tokens=in_tok,
            output_tokens=out_tok,
            seconds=sum(r.seconds for r in responses),
            cost_usd=cost_usd(in_tok, out_tok),
        )
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps({**asdict(pred), "query": query, "model": self.model}))
        return pred
