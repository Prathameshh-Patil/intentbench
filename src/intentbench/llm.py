"""Zero-shot and few-shot intent classification with an LLM.

- Two providers are supported (Gemini, used for all reported results, and Groq), chosen with
  LLM_PROVIDER in .env. Each has one small `_send_*` function; `call_llm` adds pacing, timing
  and retries around it. Adding a provider means one more `_send_*` function and a `Provider`.
- The answer must be JSON {"intent": "<name>"}. On Groq, strict structured output restricts
  <name> to the 151 valid intents (constrained decoding). Gemini rejects an enum that large, so
  there the answer is parsed strictly. On both, an invalid answer gets one retry.
- Every response is cached on disk in `.llm_cache/`, keyed by model + prompt + query, so a
  re-run costs nothing and returns exactly the same predictions.
"""

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from functools import cache

import pandas as pd

from intentbench.data import OOS_LABEL, SEED, Splits
from intentbench.evaluate import ROOT

CACHE_DIR = ROOT / ".llm_cache"
OOS_NAME = "out_of_scope"  # the LLM sees this instead of the dataset's terse "oos"
MAX_RETRIES = 8
REQUEST_TIMEOUT_S = 30


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    # USD per 1M tokens, published on-demand prices
    price_input_per_m: float
    price_cached_input_per_m: float
    price_output_per_m: float
    price_source: str
    default_rpm: int  # free-tier requests per minute


PROVIDERS = {
    "groq": Provider(
        name="groq",
        model="openai/gpt-oss-120b",
        price_input_per_m=0.15,
        price_cached_input_per_m=0.075,
        price_output_per_m=0.60,
        price_source="https://console.groq.com/docs/model/openai/gpt-oss-120b (checked 2026-10-08)",
        default_rpm=30,
    ),
    "gemini": Provider(
        name="gemini",
        model="gemini-3.5-flash-lite",
        price_input_per_m=0.30,
        price_cached_input_per_m=0.30,  # implicit caching not used in our runs
        price_output_per_m=2.50,
        price_source=(
            "https://ai.google.dev/gemini-api/docs/pricing (gemini-3.5-flash-lite, Standard "
            "paid tier; page last updated 2026-10-07, checked 2026-10-08)"
        ),
        default_rpm=15,
    ),
}


def load_env(path=ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip())


def provider() -> Provider:
    load_env()
    return PROVIDERS[os.environ.get("LLM_PROVIDER") or "gemini"]


class DailyQuotaExceeded(RuntimeError):
    """The provider's daily request limit is used up; retrying today is pointless."""


class _Retryable(Exception):
    """A temporary failure (rate limit, server error, timeout): wait and try again."""


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float
    cached_input_tokens: int = 0


def _require_key(var: str) -> None:
    load_env()
    if not os.environ.get(var):
        raise RuntimeError(f"{var} is not set. Put it in .env (see .env.example).")


@cache
def _gemini_client():
    from google import genai
    from google.genai import types

    _require_key("GEMINI_API_KEY")
    # Timeout so a stuck request can't hang the run; SDK retries off, `call_llm` retries.
    options = types.HttpOptions(
        timeout=REQUEST_TIMEOUT_S * 1000, retry_options=types.HttpRetryOptions(attempts=1)
    )
    return genai.Client(http_options=options)


@cache
def _groq_client():
    from groq import Groq

    _require_key("GROQ_API_KEY")
    return Groq(timeout=REQUEST_TIMEOUT_S, max_retries=0)


def _send_gemini(model: str, system: str, user: str, choices: list[str]) -> LLMResponse:
    import httpx
    from google.genai import errors, types

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0,
        seed=SEED,
        max_output_tokens=50,
        response_mime_type="application/json",
        response_json_schema={
            "type": "object",
            "properties": {"intent": {"type": "string"}},  # enum of 151 is rejected
            "required": ["intent"],
        },
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )
    try:
        r = _gemini_client().models.generate_content(model=model, contents=user, config=config)
    except httpx.TransportError as e:
        raise _Retryable(type(e).__name__) from e
    except errors.APIError as e:
        if e.code == 429 and "PerDay" in str(e.details):
            raise DailyQuotaExceeded("Gemini daily request limit reached.") from e
        if e.code in (429, 500, 502, 503, 504):
            raise _Retryable(e.code) from e
        raise
    u = r.usage_metadata
    return LLMResponse(
        text=r.text or "",
        input_tokens=u.prompt_token_count or 0,
        # thinking tokens are billed as output
        output_tokens=(u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
        seconds=0.0,
    )


def _send_groq(model: str, system: str, user: str, choices: list[str]) -> LLMResponse:
    import groq

    schema = {
        "type": "object",
        "properties": {"intent": {"type": "string", "enum": choices}},
        "required": ["intent"],
        "additionalProperties": False,
    }
    try:
        r = _groq_client().chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            seed=SEED,
            reasoning_effort="low",  # gpt-oss always reasons a little; keep it short
            max_completion_tokens=1024,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "intent", "strict": True, "schema": schema},
            },
        )
    except groq.RateLimitError as e:
        if "per day" in str(e).lower():
            raise DailyQuotaExceeded("Groq daily request limit reached.") from e
        raise _Retryable(429) from e
    except groq.InternalServerError as e:
        raise _Retryable(e.status_code) from e
    except (groq.APITimeoutError, groq.APIConnectionError) as e:
        raise _Retryable(type(e).__name__) from e
    u = r.usage
    details = getattr(u, "prompt_tokens_details", None)
    return LLMResponse(
        text=r.choices[0].message.content or "",
        input_tokens=u.prompt_tokens,
        cached_input_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
        output_tokens=u.completion_tokens,  # includes reasoning tokens
        seconds=0.0,
    )


_SENDERS = {"gemini": _send_gemini, "groq": _send_groq}
_last_request_at = 0.0


def _wait_for_rate_limit(rpm: float) -> None:
    """Space requests so we stay under `rpm` requests per minute."""
    global _last_request_at
    wait = _last_request_at + 60 / rpm - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def call_llm(system: str, user: str, choices: list[str]) -> LLMResponse:
    """Send one request to the configured provider, with pacing, timing and retries."""
    p = provider()
    send = _SENDERS[p.name]
    rpm = float(os.environ.get(f"{p.name.upper()}_RPM") or p.default_rpm)
    for attempt in range(MAX_RETRIES + 1):
        _wait_for_rate_limit(rpm)
        start = time.perf_counter()  # latency excludes the pacing wait
        try:
            response = send(p.model, system, user, choices)
        except _Retryable as e:
            if attempt == MAX_RETRIES:
                raise
            wait = min(2**attempt, 60)
            print(f"    retry {attempt + 1}/{MAX_RETRIES} in {wait}s: {e}")
            time.sleep(wait)
            continue
        response.seconds = time.perf_counter() - start
        return response
    raise RuntimeError("unreachable")


def cost_usd(
    input_tokens: int, output_tokens: int, cached_input_tokens: int = 0, p: Provider | None = None
) -> float:
    p = p or provider()
    fresh = input_tokens - cached_input_tokens
    return (
        fresh * p.price_input_per_m
        + cached_input_tokens * p.price_cached_input_per_m
        + output_tokens * p.price_output_per_m
    ) / 1e6


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
    cached_input_tokens: int = 0


class LLMClassifier:
    def __init__(self, labels: list[str], examples: dict[str, list[str]] | None = None):
        self.labels = labels
        self.provider = provider()
        self.model = self.provider.model
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
            known = {f.name for f in fields(Prediction)} - {"cached"}
            return Prediction(**{k: v for k, v in saved.items() if k in known}, cached=True)

        user = f"Message: {query}"
        responses = [call_llm(self.system, user, self.choices)]
        label = self._parse(responses[0].text)
        if label is None:  # one retry that says what was wrong
            reminder = (
                f"{user}\n\nYour previous answer {responses[0].text.strip()!r} is not one of the "
                "listed intents. Answer with exactly one intent name from the list."
            )
            responses.append(call_llm(self.system, reminder, self.choices))
            label = self._parse(responses[-1].text)

        in_tok = sum(r.input_tokens for r in responses)
        cached_tok = sum(r.cached_input_tokens for r in responses)
        out_tok = sum(r.output_tokens for r in responses)
        pred = Prediction(
            label=label if label is not None else self.name_to_id[OOS_NAME],
            raw=responses[-1].text,
            valid=label is not None,
            cached=False,
            input_tokens=in_tok,
            output_tokens=out_tok,
            seconds=sum(r.seconds for r in responses),
            cost_usd=cost_usd(in_tok, out_tok, cached_tok, self.provider),
            cached_input_tokens=cached_tok,
        )
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps({**asdict(pred), "query": query, "model": self.model}))
        return pred
