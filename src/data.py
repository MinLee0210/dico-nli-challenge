"""Dataset plumbing for DiCo-NLI phrase pairs.

One example is an ordered phrase pair `(text1, text2)` with an optional gold
label. Files are the official CSVs:

    participant_labeled   instance_id,pair_id,text1_lang,text2_lang,text1,text2,label
    participant_unlabeled instance_id,pair_id,text1_lang,text2_lang,text1,text2
    reference             ... + reverse_pair_id

Several files can be concatenated (e.g. all four tracks) — instance ids are
unique across tracks because they embed the language pair.

Tokenization happens in `PairCollator` (dynamic padding), and
`TwinBatchSampler` keeps an instance and its reversed twin in the same batch so
the reversal-consistency loss can see both directions.
"""

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler

from src.labels import (
    LABEL2ID,
    LABELS,
    NEGATIVE_OTHER,
    REVERSIBLE_LABELS,
    reverse_instance_id,
    reverse_label,
)

REQUIRED_COLUMNS = ("instance_id", "pair_id", "text1", "text2")
PREDICTION_HEADER = ("instance_id", "label")


@dataclass(frozen=True)
class DicoExample:
    instance_id: str
    pair_id: str
    text1_lang: str
    text2_lang: str
    text1: str
    text2: str
    label: Optional[str] = None

    @property
    def label_id(self) -> int:
        if self.label is None:
            raise ValueError(f"{self.instance_id} has no gold label")
        return LABEL2ID[self.label]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_dico_csv(path: str | Path) -> List[DicoExample]:
    """Read one official DiCo-NLI CSV. `label` is optional (unlabeled/test)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DiCo-NLI file not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        has_label = "label" in reader.fieldnames
        examples = []
        for row in reader:
            label = row["label"].strip() if has_label and row["label"] else None
            if label is not None and label not in LABEL2ID:
                raise ValueError(f"{path}: unknown label {label!r}")
            examples.append(
                DicoExample(
                    instance_id=row["instance_id"].strip(),
                    pair_id=row["pair_id"].strip(),
                    text1_lang=row.get("text1_lang", "").strip(),
                    text2_lang=row.get("text2_lang", "").strip(),
                    text1=_clean(row["text1"]),
                    text2=_clean(row["text2"]),
                    label=label,
                )
            )
    ids = [ex.instance_id for ex in examples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} contains duplicate instance ids")
    return examples


def read_dico_files(paths: Iterable[str | Path]) -> List[DicoExample]:
    examples: List[DicoExample] = []
    for path in paths:
        examples.extend(read_dico_csv(path))
    ids = [ex.instance_id for ex in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate instance ids across the given files")
    return examples


def write_predictions(
    path: str | Path, instance_ids: Sequence[str], labels: Sequence[str]
) -> None:
    """Write a submission CSV with exactly `instance_id,label`."""
    if len(instance_ids) != len(labels):
        raise ValueError("instance_ids and labels must have the same length")
    for label in labels:
        if label not in LABEL2ID:
            raise ValueError(f"invalid label {label!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(PREDICTION_HEADER)
        writer.writerows(zip(instance_ids, labels))


def twin_indices(examples: Sequence[DicoExample]) -> List[Optional[int]]:
    """For each example, the index of its reversed twin in `examples` (or None)."""
    index = {ex.instance_id: i for i, ex in enumerate(examples)}
    return [index.get(reverse_instance_id(ex.instance_id)) for ex in examples]


SYNTHETIC_WORDS = (
    "the red ship sails north a big dog runs fast old man reads new law "
    "every child eats some bread in southern city protest against bill"
).split()


def make_synthetic_dico(
    n_pairs: int = 40,
    langs: Sequence[Tuple[str, str]] = (("en", "en"),),
    negative_fraction: float = 0.2,
    seed: int = 0,
) -> List[DicoExample]:
    """Synthetic data with the official structure, for smoke runs and tests.

    Reversible source pairs get an `original` and a `flipped` row per language
    pair (labels related by Rev); NEGATIVE_OTHER pairs get only `original`.
    Texts are random words, so the labels are not learnable — this exercises
    plumbing, not modeling.
    """
    rng = random.Random(seed)
    examples = []
    for n in range(n_pairs):
        pair_id = f"dico_synth_{n:07d}"
        a = " ".join(rng.choices(SYNTHETIC_WORDS, k=rng.randint(1, 4)))
        b = " ".join(rng.choices(SYNTHETIC_WORDS, k=rng.randint(1, 4)))
        label = (
            NEGATIVE_OTHER
            if rng.random() < negative_fraction
            else rng.choice(REVERSIBLE_LABELS)
        )
        for l1, l2 in langs:
            examples.append(
                DicoExample(
                    f"{pair_id}__{l1}-{l2}__original", pair_id, l1, l2, a, b, label
                )
            )
            if label != NEGATIVE_OTHER:
                examples.append(
                    DicoExample(
                        f"{pair_id}__{l2}-{l1}__flipped",
                        pair_id,
                        l2,
                        l1,
                        b,
                        a,
                        reverse_label(label),
                    )
                )
    return examples


def write_dico_csv(path: str | Path, examples: Sequence[DicoExample]) -> None:
    """Write examples in the participant_labeled layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("instance_id", "pair_id", "text1_lang", "text2_lang", "text1", "text2")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns + ("label",))
        for ex in examples:
            writer.writerow([getattr(ex, c) for c in columns] + [ex.label or ""])


class DicoDataset(Dataset):
    """Yields the example index; `PairCollator` looks the example up and tokenizes."""

    def __init__(self, examples: Sequence[DicoExample]):
        self.examples = list(examples)
        self.twins = twin_indices(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> int:
        return index

    @property
    def has_labels(self) -> bool:
        return bool(self.examples) and all(ex.label is not None for ex in self.examples)

    @property
    def num_classes(self) -> int:
        return len(LABELS)

    @property
    def labels(self) -> torch.Tensor:
        return torch.tensor([ex.label_id for ex in self.examples], dtype=torch.long)


class PairCollator:
    """Tokenize a list of example indices into a padded batch.

    Batch keys: input_ids, attention_mask, (token_type_ids), index, twin, and
    labels when the dataset is labeled. `twin[i]` is the in-batch position of
    row i's reversed twin, or -1.
    """

    def __init__(
        self,
        dataset: DicoDataset,
        tokenizer,
        max_length: int = 128,
        with_swap: bool = False,
        pair_template: Optional[str] = None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Also emit the (text2, text1) encoding as `sw_*` keys.
        self.with_swap = with_swap
        # One prompt per pair (decoder backbones) instead of a tokenizer pair.
        self.pair_template = pair_template

    def _encode(self, firsts: List[str], seconds: List[str]):
        kwargs = dict(
            padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        if self.pair_template is None:
            return self.tokenizer(firsts, seconds, **kwargs)
        prompts = [self.pair_template.format(a=a, b=b) for a, b in zip(firsts, seconds)]
        return self.tokenizer(prompts, **kwargs)

    def __call__(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        examples = [self.dataset.examples[i] for i in indices]
        text1 = [ex.text1 for ex in examples]
        text2 = [ex.text2 for ex in examples]
        batch = dict(self._encode(text1, text2))
        if self.with_swap:
            sw = self._encode(text2, text1)
            batch.update({f"sw_{k}": v for k, v in sw.items()})
        position = {idx: pos for pos, idx in enumerate(indices)}
        batch["index"] = torch.tensor(indices, dtype=torch.long)
        batch["twin"] = torch.tensor(
            [
                position.get(self.dataset.twins[i], -1)
                if self.dataset.twins[i] is not None
                else -1
                for i in indices
            ],
            dtype=torch.long,
        )
        if all(ex.label is not None for ex in examples):
            batch["labels"] = torch.tensor(
                [ex.label_id for ex in examples], dtype=torch.long
            )
        return batch


class TwinBatchSampler(Sampler[List[int]]):
    """Batches that never split an instance from its reversed twin.

    Units are twin pairs or singletons; units are shuffled (when `shuffle`) and
    packed greedily, so a batch holds `batch_size` or `batch_size - 1` rows.
    """

    def __init__(
        self,
        dataset: DicoDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        fraction: float = 1.0,
    ):
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2 to keep twins together")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        # Each epoch visits a random `fraction` of the units (finer-grained
        # validation when many near-duplicate views make one epoch too long).
        self.fraction = fraction
        self.epoch = 0
        seen, units = set(), []
        for i, j in enumerate(dataset.twins):
            if i in seen:
                continue
            unit = [i] if j is None else [i, j]
            seen.update(unit)
            units.append(unit)
        self.units = units

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> List[List[int]]:
        units = list(self.units)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(units)
        units = units[: max(1, round(len(units) * self.fraction))]
        batches, current = [], []
        for unit in units:
            if current and len(current) + len(unit) > self.batch_size:
                batches.append(current)
                current = []
            current.extend(unit)
        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._batches()
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches())
