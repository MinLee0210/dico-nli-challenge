"""Label-safe training-data augmentation derived from the training set itself.

The released data is small (~1.3k reversible source pairs per track), so these
add examples whose labels follow logically from gold labels, without any
external resource:

    reverse_negatives   (B, A) for every NEGATIVE_OTHER (A, B). PhrasIS relations
                        folded into NEG (similar, related, opposite,
                        unaligned) are symmetric, so the reversed pair is NEG
                        too. It gets the twin id of the original, which puts
                        both in one batch and under the consistency loss, and
                        trains a symmetric NEG decision.
    transitive          entailment closure over phrases shared between pairs:
                        a ⊨ b and b ⊨ c give a ⊨ c (EQ counts as both
                        directions), for (a, c) not already labeled. New
                        pairs get both directions under a fresh pair_id.
                        Phrases are matched per language, so this works on
                        every track, including mixed.

Apply to training files only (`TrainingConfig.augment`). Heavier options
(translation, LLM generation, lexical resources) are covered in
docs/RESEARCH.md.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Set, Tuple

from src.data import DicoExample
from src.labels import (
    BACKWARD_ENTAILMENT,
    EQUIVALENCE,
    FORWARD_ENTAILMENT,
    NEGATIVE_OTHER,
    is_flipped,
    reverse_instance_id,
)

Node = Tuple[str, str]  # (language, normalized text)


def _node(lang: str, text: str) -> Node:
    return lang, text.lower().strip()


def reverse_negatives(examples: Sequence[DicoExample]) -> List[DicoExample]:
    """Reversed copies of NEGATIVE_OTHER rows that have no twin yet."""
    present = {ex.instance_id for ex in examples}
    added = []
    for ex in examples:
        if ex.label != NEGATIVE_OTHER:
            continue
        twin_id = reverse_instance_id(ex.instance_id)
        if twin_id in present:
            continue
        added.append(
            DicoExample(
                twin_id,
                ex.pair_id,
                ex.text2_lang,
                ex.text1_lang,
                ex.text2,
                ex.text1,
                NEGATIVE_OTHER,
            )
        )
        present.add(twin_id)
    return added


def transitive(
    examples: Sequence[DicoExample], prefix: str = "dico_aug_trans"
) -> List[DicoExample]:
    """New FE (or EQ) pairs implied by chaining gold entailments."""
    entails: Dict[Node, Set[Node]] = defaultdict(set)
    surface: Dict[Node, str] = {}
    labeled: Set[Tuple[Node, Node]] = set()
    for ex in examples:
        if is_flipped(ex.instance_id):
            continue  # the original row carries the same relation
        a = _node(ex.text1_lang, ex.text1)
        b = _node(ex.text2_lang, ex.text2)
        surface.setdefault(a, ex.text1)
        surface.setdefault(b, ex.text2)
        labeled.update({(a, b), (b, a)})
        if ex.label == FORWARD_ENTAILMENT:
            entails[a].add(b)
        elif ex.label == BACKWARD_ENTAILMENT:
            entails[b].add(a)
        elif ex.label == EQUIVALENCE:
            entails[a].add(b)
            entails[b].add(a)

    implied: Dict[Tuple[Node, Node], str] = {}
    for a in list(entails):
        for b in list(entails[a]):
            for c in entails.get(b, ()):
                if c == a or (a, c) in labeled or (c, a) in implied:
                    continue
                both = a in entails.get(c, ())
                implied[(a, c)] = EQUIVALENCE if both else FORWARD_ENTAILMENT

    added = []
    for n, ((a, c), label) in enumerate(sorted(implied.items())):
        pair_id = f"{prefix}_{n:07d}"
        la, lc = a[0], c[0]
        original = DicoExample(
            f"{pair_id}__{la}-{lc}__original",
            pair_id,
            la,
            lc,
            surface[a],
            surface[c],
            label,
        )
        flipped = DicoExample(
            f"{pair_id}__{lc}-{la}__flipped",
            pair_id,
            lc,
            la,
            surface[c],
            surface[a],
            BACKWARD_ENTAILMENT if label == FORWARD_ENTAILMENT else EQUIVALENCE,
        )
        added.extend([original, flipped])
    return added


AUGMENTATIONS: Dict[str, Callable[[Sequence[DicoExample]], List[DicoExample]]] = {
    "reverse_negatives": reverse_negatives,
    "transitive": transitive,
}


def augment(examples: Sequence[DicoExample], names: Sequence[str]) -> List[DicoExample]:
    """`examples` plus every requested augmentation (each computed on the originals)."""
    unknown = [n for n in names if n not in AUGMENTATIONS]
    if unknown:
        raise ValueError(
            f"unknown augmentation(s) {unknown}; choose from {list(AUGMENTATIONS)}"
        )
    out = list(examples)
    for name in names:
        out.extend(AUGMENTATIONS[name](examples))
    return out
