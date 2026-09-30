"""Greedy forward ensemble selection (Caruana et al., 2004) over saved dev log-probs.

Starts from the best single run and repeatedly adds the run (with replacement)
that most improves the objective: the mean of ALL-4 Weighted F1, SoftCons and
HardCons, decoding each track on its own. Stops when no addition helps.

Selection uses the dev set, so its score is optimistic; report it as such.

    PYTHONPATH=. uv run python scripts/training/ensemble_select.py \
        --logprob_dir results/dev/logprobs --prior
"""

import argparse
from pathlib import Path

import numpy as np

from scripts.training.ensemble_eval import TRACKS, load
from src.decoding import decode
from src.pipelines.eval import dico_metrics


def all4(examples, ids, log_probs, prior):
    index = {iid: i for i, iid in enumerate(ids)}
    rows = []
    for t in TRACKS:
        t_ids = [e.instance_id for e in examples[t]]
        labels = decode(t_ids, log_probs[[index[i] for i in t_ids]], mode="source",
                        structural_prior=prior)
        m = dico_metrics(examples[t], labels)
        rows.append([m["weighted_f1"], m["soft_cons"], m["hard_cons"]])
    return np.array(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logprob_dir", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, default=Path("data/raw/dico/final_data"))
    ap.add_argument("--prior", action="store_true", help="decode with the structural prior")
    ap.add_argument("--exclude", nargs="*", default=[], help="run-name substrings to skip")
    ap.add_argument("--max_steps", type=int, default=15)
    ap.add_argument("--per_track", action="store_true",
                    help="select a separate ensemble for each track (allows T1-only runs on T1)")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="per-track mode: write track<N>_predictions.csv and selection.json here")
    args = ap.parse_args()
    if args.per_track:
        return select_per_track(args)

    stems = [f"dico_nli_dev_track{t}_participant_labeled" for t in TRACKS]
    runs = sorted(
        r for r in {p.name.split("__")[0] for p in args.logprob_dir.glob("*.npz")}
        if all((args.logprob_dir / f"{r}__{s}.npz").exists() for s in stems)
        and not any(x in r for x in args.exclude)
        # prior-aware models never learned NEG: only valid with the prior
        and (args.prior or not any(k in r for k in ("prior",)))
    )
    examples, ids, per_run = load(args.logprob_dir, runs, "dev", args.data_root)
    lp = dict(zip(runs, per_run))

    def objective(members):
        s = all4(examples, ids, np.mean([lp[m] for m in members], axis=0), args.prior)
        return s.mean(axis=0).mean(), s

    single = sorted(((objective([r])[0], r) for r in runs), reverse=True)
    print("top single runs:", [(r, round(100 * v, 2)) for v, r in single[:5]])
    chosen = [single[0][1]]
    best, _ = objective(chosen)
    for _ in range(args.max_steps):
        cand = max(((objective(chosen + [r])[0], r) for r in runs), key=lambda x: x[0])
        if cand[0] <= best + 1e-6:
            break
        best = cand[0]
        chosen.append(cand[1])
        print(f"+ {cand[1]:32s} objective {100 * best:.2f}")
    _, s = objective(chosen)
    m = s.mean(axis=0)
    print(f"\nselected ({len(chosen)}): {chosen}")
    print(f"ALL-4 {100*m[0]:.1f} / {100*m[1]:.1f} / {100*m[2]:.1f}  ||  " +
          " | ".join(f"T{t} {100*a:.1f}/{100*b:.1f}/{100*c:.1f}" for t, (a, b, c) in zip(TRACKS, s)))


def select_per_track(args) -> None:
    import json

    from src.data import read_dico_csv, write_predictions

    totals, selection = [], {}
    for t in TRACKS:
        stem = f"dico_nli_dev_track{t}_participant_labeled"
        exs = read_dico_csv(args.data_root / "dev" / f"{stem}.csv")
        ids = [e.instance_id for e in exs]
        lp = {}
        for p in sorted(args.logprob_dir.glob(f"*__{stem}.npz")):
            r = p.name.split("__")[0]
            if any(x in r for x in args.exclude) or (not args.prior and "prior" in r):
                continue
            with np.load(p) as d:
                table = dict(zip(d["ids"].tolist(), d["log_probs"]))
            lp[r] = np.stack([table[i] for i in ids])

        def objective(members):
            labels = decode(ids, np.mean([lp[m] for m in members], axis=0), mode="source",
                            structural_prior=args.prior)
            m = dico_metrics(exs, labels)
            v = np.array([m["weighted_f1"], m["soft_cons"], m["hard_cons"]])
            return v.mean(), v

        runs = list(lp)
        chosen = [max(runs, key=lambda r: objective([r])[0])]
        best = objective(chosen)[0]
        for _ in range(args.max_steps):
            v, r = max(((objective(chosen + [r])[0], r) for r in runs), key=lambda x: x[0])
            if v <= best + 1e-6:
                break
            best, chosen = v, chosen + [r]
        v = objective(chosen)[1]
        totals.append(v)
        selection[f"track{t}"] = chosen
        if args.out_dir is not None:
            labels = decode(ids, np.mean([lp[m] for m in chosen], axis=0), mode="source",
                            structural_prior=args.prior)
            write_predictions(args.out_dir / f"track{t}_predictions.csv", ids, labels)
        print(f"T{t}: {100*v[0]:.1f}/{100*v[1]:.1f}/{100*v[2]:.1f}  <- {chosen}")
    m = np.mean(totals, axis=0)
    print(f"ALL-4 {100*m[0]:.1f} / {100*m[1]:.1f} / {100*m[2]:.1f}")
    if args.out_dir is not None:
        (args.out_dir / "selection.json").write_text(json.dumps(selection, indent=1))


if __name__ == "__main__":
    main()
