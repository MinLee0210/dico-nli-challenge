"""Consistency-aware decoding: per-instance log-probs -> official labels.

Every instance is mapped into its source pair's canonical frame (`original`
direction; `flipped` rows get Rev applied), log-probs are averaged within a
group, one canonical label is chosen, and it is mapped back per instance. Any
group containing both directions therefore decodes to a Rev-compatible pair,
which is what SoftCons scores.

Grouping modes:
    independent  every instance alone (plain argmax; the baseline)
    twin         an instance and its reversed twin
    source       every instance sharing a pair_id — all language views in the
                 given files, e.g. the 12 Track 4 views, or Tracks 1-4 at once

`neg_bias` is added to the averaged NEGATIVE_OTHER score before the argmax; tune
it on dev to trade Weighted F1 against HardCons.

`structural_prior` uses the file layout: a group with a reversed twin present
is decoded among the three reversible labels, and a group with no twin is
decoded as NEGATIVE_OTHER. This holds on 100% of train/dev but is a property
of how the data was built, not of the phrases — see docs/RESEARCH.md before
using it in a submission.
"""

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from src.labels import (
    LABELS,
    NEGATIVE_ID,
    REVERSE_PERM,
    is_flipped,
    reverse_instance_id,
    twin_key,
)

DECODING_MODES = ("independent", "twin", "source")


def _group_key(instance_id: str, mode: str) -> str:
    if mode == "independent":
        return instance_id
    if mode == "twin":
        return twin_key(instance_id)
    if mode == "source":
        return instance_id.split("__", 1)[0]
    raise ValueError(f"unknown decoding mode {mode!r}; choose from {DECODING_MODES}")


def decode(
    instance_ids: Sequence[str],
    log_probs: np.ndarray,
    mode: str = "twin",
    neg_bias: float = 0.0,
    structural_prior: bool = False,
) -> List[str]:
    """Decode `(N, 4)` log-probabilities into one label per instance id."""
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_probs.shape != (len(instance_ids), len(LABELS)):
        raise ValueError(
            f"log_probs must be ({len(instance_ids)}, {len(LABELS)}), got {log_probs.shape}"
        )
    perm = list(REVERSE_PERM)
    flipped = np.array([is_flipped(i) for i in instance_ids])
    canonical = np.where(flipped[:, None], log_probs[:, perm], log_probs)

    present = set(instance_ids)
    groups: Dict[str, List[int]] = defaultdict(list)
    for row, iid in enumerate(instance_ids):
        groups[_group_key(iid, mode)].append(row)

    labels: List[str] = [""] * len(instance_ids)
    for rows in groups.values():
        score = canonical[rows].mean(axis=0)
        score[NEGATIVE_ID] += neg_bias
        if structural_prior:
            has_twin = any(
                reverse_instance_id(instance_ids[r]) in present for r in rows
            )
            if has_twin:
                score[NEGATIVE_ID] = -np.inf
            else:
                score = np.full_like(score, -np.inf)
                score[NEGATIVE_ID] = 0.0
        choice = int(np.argmax(score))
        for r in rows:
            labels[r] = LABELS[perm[choice] if flipped[r] else choice]
    return labels
