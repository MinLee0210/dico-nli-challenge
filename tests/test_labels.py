import pytest

from src.labels import (
    BACKWARD_ENTAILMENT,
    EQUIVALENCE,
    FORWARD_ENTAILMENT,
    LABELS,
    NEGATIVE_OTHER,
    REVERSE_PERM,
    is_flipped,
    parse_instance_id,
    reverse_instance_id,
    reverse_label,
    twin_key,
)


def test_reverse_label_algebra():
    assert reverse_label(EQUIVALENCE) == EQUIVALENCE
    assert reverse_label(FORWARD_ENTAILMENT) == BACKWARD_ENTAILMENT
    assert reverse_label(BACKWARD_ENTAILMENT) == FORWARD_ENTAILMENT
    assert reverse_label(NEGATIVE_OTHER) == NEGATIVE_OTHER
    for label in LABELS:
        assert reverse_label(reverse_label(label)) == label


def test_reverse_perm_matches_reverse_label():
    for i, label in enumerate(LABELS):
        assert LABELS[REVERSE_PERM[i]] == reverse_label(label)


def test_reverse_label_rejects_unknown():
    with pytest.raises(ValueError):
        reverse_label("CONTRADICTION")


def test_reverse_instance_id_monolingual_and_mixed():
    assert reverse_instance_id("dico_phrasis_0000001__en-en__original") == (
        "dico_phrasis_0000001__en-en__flipped"
    )
    # Mixed track: languages swap together with the direction tag.
    assert reverse_instance_id("dico_phrasis_0000001__en-es__original") == (
        "dico_phrasis_0000001__es-en__flipped"
    )
    iid = "dico_phrasis_0000001__eu-es__flipped"
    assert reverse_instance_id(reverse_instance_id(iid)) == iid


def test_twin_key_pairs_exactly_two_ids():
    a = "p__en-es__original"
    b = "p__es-en__flipped"
    c = "p__en-es__flipped"
    assert twin_key(a) == twin_key(b)
    assert twin_key(a) != twin_key(c)


def test_parse_and_flipped():
    assert parse_instance_id("p__en-eu__flipped") == ("p", "en", "eu", "flipped")
    assert is_flipped("p__en-eu__flipped")
    assert not is_flipped("p__en-eu__original")
    with pytest.raises(ValueError):
        parse_instance_id("no-structure")
