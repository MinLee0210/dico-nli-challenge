import pytest

from src.augment import augment, reverse_negatives, transitive
from src.data import DicoExample, make_synthetic_dico, twin_indices
from src.labels import reverse_label


def ex(iid, t1, t2, label, langs=("en", "en")):
    return DicoExample(iid, iid.split("__")[0], langs[0], langs[1], t1, t2, label)


def test_reverse_negatives_creates_twins():
    data = make_synthetic_dico(n_pairs=30, seed=0)
    added = reverse_negatives(data)
    n_neg = sum(e.label == "NEGATIVE_OTHER" for e in data)
    assert len(added) == n_neg
    full = data + added
    for e, t in zip(full, twin_indices(full)):
        assert t is not None  # every row now has a twin
    assert reverse_negatives(full) == []  # idempotent


def test_transitive_chains_entailment():
    data = [
        ex(
            "p1__en-en__original",
            "US drone strike",
            "drone strike",
            "FORWARD_ENTAILMENT",
        ),
        ex(
            "p1__en-en__flipped",
            "drone strike",
            "US drone strike",
            "BACKWARD_ENTAILMENT",
        ),
        ex("p2__en-en__original", "strike", "drone strike", "BACKWARD_ENTAILMENT"),
        ex("p2__en-en__flipped", "drone strike", "strike", "FORWARD_ENTAILMENT"),
    ]
    added = transitive(data)
    assert len(added) == 2
    original, flipped = added
    assert (original.text1, original.text2, original.label) == (
        "US drone strike",
        "strike",
        "FORWARD_ENTAILMENT",
    )
    assert flipped.label == reverse_label(original.label)
    assert twin_indices(added) == [1, 0]


def test_transitive_equivalence_cycle_and_languages():
    data = [
        ex("p1__en-es__original", "car", "coche", "EQUIVALENCE", ("en", "es")),
        ex("p2__es-en__original", "coche", "automobile", "EQUIVALENCE", ("es", "en")),
    ]
    [orig, flip] = transitive(data)
    assert orig.label == "FORWARD_ENTAILMENT" or orig.label == "EQUIVALENCE"
    assert {orig.text1_lang, orig.text2_lang} == {"en"}
    # "coche" (es) must not chain with an English "coche"
    other = [ex("p3__en-en__original", "coche", "x", "FORWARD_ENTAILMENT")]
    assert all(e.text1 != "x" and e.text2 != "x" for e in transitive(data + other))


def test_augment_rejects_unknown():
    with pytest.raises(ValueError):
        augment([], ["mixup"])
