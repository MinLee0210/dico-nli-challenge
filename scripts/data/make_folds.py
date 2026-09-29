"""Grouped K-fold splits of train+dev, identical across the four tracks.

A source pair (`pair_id`) never straddles folds, and every track uses the same
assignment, so fold k of Track 1 and fold k of Track 4 hold out the same pairs.
Fold k is written in the official layout, so a config only changes `data_root`:

    <out>/f<k>/train/dico_nli_train_track<N>_participant_labeled.csv
    <out>/f<k>/dev/dico_nli_dev_track<N>_participant_labeled.csv

    uv run python scripts/data/make_folds.py --k 5
    uv run python scripts/training/train.py --config configs/runs/e8_mdeberta_fine.yaml \\
        --data_root data/folds/f0
"""

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import read_dico_csv, write_dico_csv  # noqa: E402

TRACKS = (1, 2, 3, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/raw/dico/final_data"))
    parser.add_argument("--out", type=Path, default=Path("data/folds"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tracks = {
        t: read_dico_csv(
            args.data_root / "train" / f"dico_nli_train_track{t}_participant_labeled.csv"
        )
        + read_dico_csv(
            args.data_root / "dev" / f"dico_nli_dev_track{t}_participant_labeled.csv"
        )
        for t in TRACKS
    }
    pair_ids = sorted({ex.pair_id for exs in tracks.values() for ex in exs})
    random.Random(args.seed).shuffle(pair_ids)
    fold_of = {p: i % args.k for i, p in enumerate(pair_ids)}

    for k in range(args.k):
        for t, exs in tracks.items():
            held = [ex for ex in exs if fold_of[ex.pair_id] == k]
            kept = [ex for ex in exs if fold_of[ex.pair_id] != k]
            root = args.out / f"f{k}"
            write_dico_csv(root / "train" / f"dico_nli_train_track{t}_participant_labeled.csv", kept)
            write_dico_csv(root / "dev" / f"dico_nli_dev_track{t}_participant_labeled.csv", held)
        print(f"fold {k}: {sum(fold_of[p] == k for p in pair_ids)} source pairs held out")


if __name__ == "__main__":
    main()
