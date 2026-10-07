"""Load CLINC150 (plus), remove train/eval leakage, and expose fixed splits and label maps.

Every other module gets its data from `load_splits()`, so all approaches see exactly
the same examples.

Run `python -m intentbench.data` to print the split sizes.
"""

import re
from dataclasses import dataclass
from functools import cache

import pandas as pd
from datasets import load_dataset

DATASET_NAME = "clinc/clinc_oos"
DATASET_CONFIG = "plus"
# Pinned dataset commit on the Hugging Face Hub, so a re-run downloads identical data.
DATASET_REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"
# One seed for everything random in the project (sampling, model init, shuffling).
SEED = 42
OOS_LABEL = "oos"


@dataclass(frozen=True)
class Splits:
    """Train/validation/test DataFrames with columns `text`, `label` (int), `intent` (str)."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    labels: list[str]  # labels[i] is the intent name for label id i
    removed_from_train: int  # rows dropped because they duplicate a validation/test query
    removed_from_validation: int  # rows dropped because they duplicate a test query

    @property
    def label2id(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.labels)}

    @property
    def oos_id(self) -> int:
        return self.label2id[OOS_LABEL]


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace: 'Bye!' and 'bye' become the same."""
    text = re.sub(r"[^a-z0-9 ]", "", text.lower())
    return " ".join(text.split())


@cache
def load_splits() -> Splits:
    """Load the official splits and remove queries that leak across splits.

    Two queries count as the same if they match after `normalize`. The test set stays exactly
    as published, so final results compare with the literature.
    """
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, revision=DATASET_REVISION)
    labels = ds["train"].features["intent"].names

    frames = {}
    for split in ("train", "validation", "test"):
        df = ds[split].to_pandas().rename(columns={"intent": "label"})
        df["intent"] = df["label"].map(lambda i: labels[i])
        frames[split] = df

    # Test stays untouched. Validation drops queries that also appear in test; train drops
    # queries that appear in either.
    test = frames["test"]
    test_texts = set(test["text"].map(normalize))
    validation = frames["validation"]
    val_leaked = validation["text"].map(normalize).isin(test_texts)
    validation = validation[~val_leaked].reset_index(drop=True)

    held_out = test_texts | set(validation["text"].map(normalize))
    train = frames["train"]
    train_leaked = train["text"].map(normalize).isin(held_out)
    train = train[~train_leaked].reset_index(drop=True)

    return Splits(
        train=train,
        validation=validation,
        test=test,
        labels=labels,
        removed_from_train=int(train_leaked.sum()),
        removed_from_validation=int(val_leaked.sum()),
    )


def main() -> None:
    s = load_splits()
    print(f"Dataset: {DATASET_NAME} ({DATASET_CONFIG}) @ {DATASET_REVISION[:7]}")
    print(f"Labels: {len(s.labels)} (150 intents + '{OOS_LABEL}')")
    for name in ("train", "validation", "test"):
        df = getattr(s, name)
        n_oos = int((df["label"] == s.oos_id).sum())
        print(f"{name:<11} {len(df):>6} examples  ({n_oos} out-of-scope)")
    print(f"Removed from train as duplicates of validation/test: {s.removed_from_train}")
    print(f"Removed from validation as duplicates of test: {s.removed_from_validation}")


if __name__ == "__main__":
    main()
