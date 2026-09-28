import numpy as np
import pytest

from src.decoding import decode
from src.labels import LABEL2ID, reverse_label


def logp(**scores) -> np.ndarray:
    """A log-prob row with the given label scores (others very unlikely)."""
    row = np.full(4, -10.0)
    for name, value in scores.items():
        row[LABEL2ID[name]] = value
    return row


ORIG = "p__en-en__original"
FLIP = "p__en-en__flipped"


def test_independent_can_be_inconsistent():
    lp = np.stack([logp(FORWARD_ENTAILMENT=-0.1), logp(FORWARD_ENTAILMENT=-0.1)])
    assert decode([ORIG, FLIP], lp, "independent") == [
        "FORWARD_ENTAILMENT",
        "FORWARD_ENTAILMENT",
    ]


def test_twin_decoding_is_consistent():
    # original weakly FE; flipped strongly BE (== FE in canonical frame).
    lp = np.stack(
        [
            logp(FORWARD_ENTAILMENT=-0.9, EQUIVALENCE=-0.6),
            logp(BACKWARD_ENTAILMENT=-0.05),
        ]
    )
    labels = decode([ORIG, FLIP], lp, "twin")
    assert labels == ["FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"]
    assert labels[1] == reverse_label(labels[0])


def test_source_mode_pools_all_views():
    ids = [
        "p__en-es__original",
        "p__es-en__flipped",
        "p__en-eu__original",
        "p__eu-en__flipped",
    ]
    # Three views say EQ, one says FE -> EQ everywhere.
    lp = np.stack(
        [
            logp(FORWARD_ENTAILMENT=-0.1),
            logp(EQUIVALENCE=-0.1),
            logp(EQUIVALENCE=-0.1),
            logp(EQUIVALENCE=-0.1),
        ]
    )
    assert decode(ids, lp, "source") == ["EQUIVALENCE"] * 4
    # twin mode keeps the two twin groups separate
    twin = decode(ids, lp, "twin")
    assert twin[2:] == ["EQUIVALENCE", "EQUIVALENCE"]


def test_neg_bias_shifts_negative_decisions():
    lp = np.stack([logp(NEGATIVE_OTHER=-0.7, EQUIVALENCE=-0.6)])
    assert decode([ORIG], lp, "twin") == ["EQUIVALENCE"]
    assert decode([ORIG], lp, "twin", neg_bias=0.5) == ["NEGATIVE_OTHER"]


def test_structural_prior():
    single = "q__en-en__original"
    lp = np.stack(
        [
            logp(NEGATIVE_OTHER=-0.01),
            logp(NEGATIVE_OTHER=-0.01, EQUIVALENCE=-5.0),
            logp(EQUIVALENCE=-0.01),
        ]
    )
    labels = decode([ORIG, FLIP, single], lp, "twin", structural_prior=True)
    assert labels == ["EQUIVALENCE", "EQUIVALENCE", "NEGATIVE_OTHER"]


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        decode([ORIG], np.zeros((2, 4)), "twin")
    with pytest.raises(ValueError):
        decode([ORIG], np.zeros((1, 4)), "bogus")
