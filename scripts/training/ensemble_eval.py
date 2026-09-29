"""Offline ensemble / decoding search over saved dev log-probs.

Reads `results/<dir>/logprobs/<run>__<file-stem>.npz` written by
`src.pipelines.predict`, averages the chosen runs, decodes all four tracks
jointly, and prints per-track and macro (ALL-4) Weighted F1 / SoftCons /
HardCons for each decoding setting.

    uv run python scripts/training/ensemble_eval.py --logprob_dir results/dev/logprobs \
        --runs e3_mdeberta e7_xlmr_large --split dev
"""

import argparse
import itertools
from pathlib import Path

import numpy as np

from src.data import read_dico_csv
from src.decoding import decode
from src.pipelines.eval import dico_metrics

TRACKS = (1, 2, 3, 4)


def load(logprob_dir: Path, runs, split: str, data_root: Path):
    files = {
        t: data_root / split / f"dico_nli_{split}_track{t}_participant_labeled.csv"
        for t in TRACKS
    }
    examples = {t: read_dico_csv(p) for t, p in files.items()}
    ids = [ex.instance_id for t in TRACKS for ex in examples[t]]
    per_run = []
    for run in runs:
        rows = {}
        for t in TRACKS:
            with np.load(logprob_dir / f"{run}__{files[t].stem}.npz") as d:
                rows.update(zip(d["ids"].tolist(), d["log_probs"]))
        per_run.append(np.stack([rows[i] for i in ids]))
    return examples, ids, per_run


def score(examples, ids, log_probs, per_track=False, **kw):
    """Decode all tracks jointly, or each track file on its own (`per_track`,
    which never pools information across tracks)."""
    if per_track:
        labels, start = [], 0
        for t in TRACKS:
            n = len(examples[t])
            labels += decode(ids[start : start + n], log_probs[start : start + n], **kw)
            start += n
    else:
        labels = decode(ids, log_probs, **kw)
    out, start = [], 0
    for t in TRACKS:
        n = len(examples[t])
        s = dico_metrics(examples[t], labels[start : start + n])
        out.append((s["weighted_f1"], s["soft_cons"], s["hard_cons"]))
        start += n
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logprob_dir", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--data_root", type=Path, default=Path("data/raw/dico/final_data"))
    ap.add_argument("--per_track", action="store_true", help="decode each track alone")
    ap.add_argument("--neg_biases", type=float, nargs="+", default=[0.0])
    args = ap.parse_args()

    examples, ids, per_run = load(args.logprob_dir, args.runs, args.split, args.data_root)
    avg = np.mean(per_run, axis=0)
    for mode, prior, bias in itertools.product(
        ("independent", "twin", "source"), (False, True), args.neg_biases
    ):
        s = score(examples, ids, avg, per_track=args.per_track, mode=mode, neg_bias=bias, structural_prior=prior)
        m = s.mean(axis=0)
        tracks = " | ".join(f"T{t} {a:.3f}/{b:.3f}/{c:.3f}" for t, (a, b, c) in zip(TRACKS, s))
        print(
            f"{mode:11s} prior={int(prior)} bias={bias:+.1f}  "
            f"ALL {m[0]:.3f}/{m[1]:.3f}/{m[2]:.3f}  ||  {tracks}"
        )


if __name__ == "__main__":
    main()
