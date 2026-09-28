"""DiCo-NLI label inventory, reversal algebra, and instance-id conventions.

Labels are indexed in a fixed order so a probability vector can be reversed
with a single index permutation:

    p_rev = p[..., REVERSE_PERM]

`NEGATIVE_OTHER` is not reversible in the official scorer (it never takes part
in SoftCons/HardCons); for aggregation purposes it maps to itself.

Instance ids follow `<pair_id>__<lang1>-<lang2>__<original|flipped>`. The
reversed twin of an instance swaps the languages and the direction tag — this
rule matches `reverse_pair_id` on every train/dev reference file.
"""

from typing import Tuple

EQUIVALENCE = "EQUIVALENCE"
FORWARD_ENTAILMENT = "FORWARD_ENTAILMENT"
BACKWARD_ENTAILMENT = "BACKWARD_ENTAILMENT"
NEGATIVE_OTHER = "NEGATIVE_OTHER"

LABELS: Tuple[str, ...] = (
    EQUIVALENCE,
    FORWARD_ENTAILMENT,
    BACKWARD_ENTAILMENT,
    NEGATIVE_OTHER,
)
REVERSIBLE_LABELS: Tuple[str, ...] = LABELS[:3]
NUM_LABELS = len(LABELS)
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
NEGATIVE_ID = LABEL2ID[NEGATIVE_OTHER]

# Index permutation implementing Rev on label ids / probability vectors:
# EQ -> EQ, FE -> BE, BE -> FE, NEG -> NEG.
REVERSE_PERM: Tuple[int, ...] = (0, 2, 1, 3)

ORIGINAL = "original"
FLIPPED = "flipped"


def reverse_label(label: str) -> str:
    """Rev(label). `NEGATIVE_OTHER` maps to itself (not scored for consistency)."""
    if label not in LABEL2ID:
        raise ValueError(f"unknown label {label!r}; expected one of {LABELS}")
    return LABELS[REVERSE_PERM[LABEL2ID[label]]]


def parse_instance_id(instance_id: str) -> Tuple[str, str, str, str]:
    """Split an instance id into `(pair_id, lang1, lang2, direction)`."""
    parts = instance_id.split("__")
    if len(parts) != 3 or "-" not in parts[1]:
        raise ValueError(f"unexpected instance_id format: {instance_id!r}")
    pair_id, langs, direction = parts
    lang1, lang2 = langs.split("-", 1)
    return pair_id, lang1, lang2, direction


def reverse_instance_id(instance_id: str) -> str:
    """The id of the reversed twin: languages swapped, original <-> flipped."""
    pair_id, lang1, lang2, direction = parse_instance_id(instance_id)
    if direction == ORIGINAL:
        other = FLIPPED
    elif direction == FLIPPED:
        other = ORIGINAL
    else:
        raise ValueError(f"unknown direction tag {direction!r} in {instance_id!r}")
    return f"{pair_id}__{lang2}-{lang1}__{other}"


def is_flipped(instance_id: str) -> bool:
    """True when the instance is in the flipped orientation of its source pair.

    All `original` instances of a source pair share one orientation across
    languages, so `original` is used as the canonical frame when aggregating.
    """
    return parse_instance_id(instance_id)[3] == FLIPPED


def maybe_reverse_label(label: str, flip: bool) -> str:
    return reverse_label(label) if flip else label


def twin_key(instance_id: str) -> str:
    """Key shared by exactly an instance and its reversed twin.

    Not the language set: in the mixed track `en-es__original` pairs with
    `es-en__flipped` while `en-es__flipped` pairs with `es-en__original`.
    """
    return min(instance_id, reverse_instance_id(instance_id))
