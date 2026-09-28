"""Summarize DiCo-NLI CSVs: labels, twin structure, languages, lengths.

Surfaces the structural facts the modeling relies on — every reversible item
has its reversed twin in the same file, NEGATIVE_OTHER items are singletons,
and a source pair_id spans several language views.

Usage:
    uv run python scripts/data/inspect_data.py data/raw/dico/final_data/train/*participant_labeled.csv
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import read_dico_csv, twin_indices  # noqa: E402
from src.labels import NEGATIVE_OTHER, is_flipped  # noqa: E402


def summarize(path: Path) -> None:
    examples = read_dico_csv(path)
    twins = twin_indices(examples)
    labels = Counter(ex.label or "<unlabeled>" for ex in examples)
    langs = Counter(f"{ex.text1_lang}-{ex.text2_lang}" for ex in examples)
    views = Counter(Counter(ex.pair_id for ex in examples).values())
    singletons = [ex for ex, t in zip(examples, twins) if t is None]
    single_labels = Counter(ex.label or "<unlabeled>" for ex in singletons)
    words = [len(ex.text1.split()) + len(ex.text2.split()) for ex in examples]
    flipped = sum(is_flipped(ex.instance_id) for ex in examples)

    print(f"\n{path.name}")
    print(
        f"  rows: {len(examples)}  source pairs: {len({ex.pair_id for ex in examples})}"
    )
    print("  labels: " + "  ".join(f"{k}:{v}" for k, v in sorted(labels.items())))
    print("  lang pairs: " + "  ".join(f"{k}:{v}" for k, v in sorted(langs.items())))
    print(
        "  rows per pair_id: " + "  ".join(f"{k}x{v}" for k, v in sorted(views.items()))
    )
    print(
        f"  with twin: {len(examples) - len(singletons)}  singletons: {len(singletons)}"
    )
    print(
        "  singleton labels: "
        + "  ".join(f"{k}:{v}" for k, v in sorted(single_labels.items()))
    )
    if singletons and set(single_labels) == {NEGATIVE_OTHER}:
        print("  note: every singleton is NEGATIVE_OTHER (structural cue)")
    print(f"  flipped rows: {flipped}")
    print(f"  words per pair: mean {sum(words) / len(words):.1f}  max {max(words)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.files:
        summarize(path)


if __name__ == "__main__":
    main()
