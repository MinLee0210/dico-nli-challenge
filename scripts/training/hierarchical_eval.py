"""Hierarchical composition without the structural prior.

p(NEG) comes from clean models (trained on all four labels); the choice among
EQ / FE / BE comes from prior-aware models (trained on reversible labels only,
which are more accurate at that 3-way decision):

    log p(NEG)  = log g
    log p(c)    = log (1 - g) + log q(c),  c in {EQ, FE, BE}

where g is the clean ensemble's mean NEG probability and q is the prior-aware
ensemble's distribution renormalised over the three reversible labels. No
dataset-layout information is used. Each track is decoded on its own.

    PYTHONPATH=. uv run python scripts/training/hierarchical_eval.py \
        --gate e12_mdeberta_s2 e12_xlmr_s2 --rev e15_mdeberta_sym_prior e11_xlmr_prior
"""

import argparse
from pathlib import Path

import numpy as np

from src.data import read_dico_csv, write_predictions
from src.decoding import decode
from src.labels import NEGATIVE_ID
from src.pipelines.eval import dico_metrics

TRACKS = (1, 2, 3, 4)


def load_track(logprob_dir, runs, stem, ids):
    out = []
    for r in runs:
        path = logprob_dir / f"{r}__{stem}.npz"
        if not path.exists():
            continue
        with np.load(path) as d:
            table = dict(zip(d["ids"].tolist(), d["log_probs"]))
        out.append(np.stack([table[i] for i in ids]))
    return out


def compose(gate_lp, rev_lp):
    g = np.mean([np.exp(x[:, NEGATIVE_ID]) for x in gate_lp], axis=0).clip(1e-6, 1 - 1e-6)
    q = np.mean([np.exp(x) for x in rev_lp], axis=0)
    q[:, NEGATIVE_ID] = 0.0
    q = q / q.sum(axis=1, keepdims=True)
    out = np.log(np.clip(q * (1 - g)[:, None], 1e-12, None))
    out[:, NEGATIVE_ID] = np.log(g)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logprob_dir", type=Path, default=Path("results/dev/logprobs"))
    ap.add_argument("--data_root", type=Path, default=Path("data/raw/dico/final_data"))
    ap.add_argument("--gate", nargs="+", default=None, help="clean runs giving p(NEG)")
    ap.add_argument("--rev", nargs="+", default=None, help="prior-aware runs for EQ/FE/BE")
    ap.add_argument("--select", action="store_true",
                    help="greedy per-track selection of gate and rev members (dev-optimistic)")
    ap.add_argument("--neg_biases", type=float, nargs="+", default=[0.0])
    ap.add_argument("--out_dir", type=Path, default=None, help="write predictions (first bias)")
    args = ap.parse_args()
    if args.select:
        return select(args)

    for bias in args.neg_biases:
        rows = []
        for t in TRACKS:
            stem = f"dico_nli_dev_track{t}_participant_labeled"
            exs = read_dico_csv(args.data_root / "dev" / f"{stem}.csv")
            ids = [e.instance_id for e in exs]
            lp = compose(load_track(args.logprob_dir, args.gate, stem, ids),
                         load_track(args.logprob_dir, args.rev, stem, ids))
            labels = decode(ids, lp, mode="source", neg_bias=bias)
            m = dico_metrics(exs, labels)
            rows.append([m["weighted_f1"], m["soft_cons"], m["hard_cons"]])
            if args.out_dir is not None and bias == args.neg_biases[0]:
                write_predictions(args.out_dir / f"track{t}_predictions.csv", ids, labels)
        rows = np.array(rows)
        m = rows.mean(axis=0)
        print(f"bias={bias:+.1f}  ALL-4 {100*m[0]:.1f}/{100*m[1]:.1f}/{100*m[2]:.1f}  ||  " +
              " | ".join(f"T{t} {100*a:.1f}/{100*b:.1f}/{100*c:.1f}" for t, (a, b, c) in zip(TRACKS, rows)))


def select(args, max_steps: int = 12) -> None:
    """Per track: greedily add gate (clean) or rev (prior-aware) members,
    whichever raises the mean of Weighted F1, SoftCons and HardCons most."""
    import json

    runs = sorted({p.name.split("__")[0] for p in args.logprob_dir.glob("*.npz")})
    gate_pool = [r for r in runs if "prior" not in r]
    rev_pool = [r for r in runs if "prior" in r]
    totals, chosen_all = [], {}
    for t in TRACKS:
        stem = f"dico_nli_dev_track{t}_participant_labeled"
        exs = read_dico_csv(args.data_root / "dev" / f"{stem}.csv")
        ids = [e.instance_id for e in exs]
        cache = {r: load_track(args.logprob_dir, [r], stem, ids) for r in runs}
        gp = [r for r in gate_pool if cache[r]]
        rp = [r for r in rev_pool if cache[r]]

        def score(g, q):
            lp = compose([cache[r][0] for r in g], [cache[r][0] for r in q])
            m = dico_metrics(exs, decode(ids, lp, mode="source"))
            v = np.array([m["weighted_f1"], m["soft_cons"], m["hard_cons"]])
            return v.mean(), v

        best, g, q = max((score([a], [b])[0], [a], [b]) for a in gp for b in rp)
        for _ in range(max_steps):
            cands = [(score(g + [a], q)[0], g + [a], q) for a in gp] + \
                    [(score(g, q + [b])[0], g, q + [b]) for b in rp]
            v, g2, q2 = max(cands, key=lambda c: c[0])
            if v <= best + 1e-6:
                break
            best, g, q = v, g2, q2
        v = score(g, q)[1]
        totals.append(v)
        chosen_all[f"track{t}"] = {"gate": g, "rev": q}
        print(f"T{t}: {100*v[0]:.1f}/{100*v[1]:.1f}/{100*v[2]:.1f}  gate={g} rev={q}")
        if args.out_dir is not None:
            lp = compose([cache[r][0] for r in g], [cache[r][0] for r in q])
            write_predictions(args.out_dir / f"track{t}_predictions.csv", ids,
                              decode(ids, lp, mode="source"))
    m = np.mean(totals, axis=0)
    print(f"ALL-4 {100*m[0]:.1f} / {100*m[1]:.1f} / {100*m[2]:.1f}")
    if args.out_dir is not None:
        (args.out_dir / "selection.json").write_text(json.dumps(chosen_all, indent=1))


if __name__ == "__main__":
    main()
