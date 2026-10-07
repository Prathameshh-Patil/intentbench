import itertools

from intentbench.data import OOS_LABEL, load_splits, normalize


def test_splits_never_overlap():
    s = load_splits()
    texts = {
        name: set(getattr(s, name)["text"].map(normalize))
        for name in ("train", "validation", "test")
    }
    for a, b in itertools.combinations(texts, 2):
        overlap = texts[a] & texts[b]
        assert not overlap, f"{a} and {b} share {len(overlap)} queries, e.g. {sorted(overlap)[:3]}"


def test_test_split_is_the_official_one():
    s = load_splits()
    assert len(s.test) == 5500
    assert len(s.validation) + s.removed_from_validation == 3100


def test_label_maps():
    s = load_splits()
    assert len(s.labels) == 151
    assert s.labels[s.oos_id] == OOS_LABEL
    for df in (s.train, s.validation, s.test):
        assert (df["intent"] == df["label"].map(lambda i: s.labels[i])).all()


def test_normalize():
    assert normalize("  Bye!  ") == "bye"
    assert normalize("What's   UP?") == "whats up"
