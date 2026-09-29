"""Emit LaTeX table rows for the technical report from saved dev log-probs.

Each system is a named set of runs; its log-probs are averaged and decoded
per track (never pooling across tracks) in `source` mode, with and without
`structural_prior`. Rows are `name & T1 F1/S/H & ... & ALL-4 F1/S/H \\\\`.

    PYTHONPATH=. uv run python scripts/training/report_tables.py \
        --logprob_dir results/dev/logprobs --out docs/report/tables
"""

import argparse
from pathlib import Path

import numpy as np

from scripts.training.ensemble_eval import load
from src.decoding import decode
from src.pipelines.eval import dico_metrics

C7 = [
    "e8_mdeberta_fine", "e12_mdeberta_s1", "e12_mdeberta_s2", "e12_mdeberta_s3",
    "e7_xlmr_large", "e12_xlmr_s1", "e12_xlmr_s2",
]
P5 = [
    "e11_mdeberta_prior", "e11_xlmr_prior", "e14_mdeberta_prior_s1",
    "e14_mdeberta_prior_s2", "e14_xlmr_prior_s1",
]
# (runs, prior_aware). Prior-aware models never learned NEG, so they appear
# only in the structural-prior table.
SYSTEMS = {
    "mmBERT-base": (["e3_mmbert"], False),
    "mDeBERTa-v3-base NLI": (["e3_mdeberta"], False),
    "mDeBERTa, fine-grained val. (A)": (["e8_mdeberta_fine"], False),
    r"\quad A + rev.-neg. aug.": (["e13_mdeberta_revneg_s42"], False),
    r"\quad A + NEG weight 2": (["e18_mdeberta_negw2"], False),
    r"\quad A + focal ($\gamma$=2)": (["e18_mdeberta_focal2"], False),
    r"\quad A + iSTS answers-students": (["e19_mdeberta_ists"], False),
    r"\quad A, symmetric arch.": (["e9_mdeberta_sym"], False),
    "XLM-R-large XNLI (B)": (["e7_xlmr_large"], False),
    r"\quad B + rev.-neg. aug.": (["e13_xlmr_revneg"], False),
    r"\quad B + iSTS answers-students": (["e19_xlmr_ists"], False),
    "Ens. clean-7 (A, B + 5 seeds)": (C7, False),
    "mDeBERTa, prior-aware": (["e11_mdeberta_prior"], True),
    "mDeBERTa, prior-aware, symmetric": (["e15_mdeberta_sym_prior"], True),
    "XLM-R-large, prior-aware": (["e11_xlmr_prior"], True),
    "XLM-R-large, prior-aware, symmetric": (["e17_xlmr_sym_prior"], True),
    "Ens. prior-aware-5": (P5, True),
    "Ens. prior-aware-5 + clean-7": (P5 + C7, True),
    "Ens. prior-14 (+ 2 symmetric)": (
        P5 + ["e15_mdeberta_sym_prior", "e17_xlmr_sym_prior"] + C7, True),
}


def fmt(row) -> str:
    return " & ".join(f"{100 * v:.1f}" for v in row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logprob_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, default=Path("data/raw/dico/final_data"))
    ap.add_argument("--extra", nargs="*", default=[], help="name=run1,run2 systems")
    args = ap.parse_args()

    systems = dict(SYSTEMS)
    for spec in args.extra:
        name, runs = spec.split("=", 1)
        systems[name] = (runs.split(","), True)
    available = {p.name.split("__")[0] for p in args.logprob_dir.glob("*.npz")}

    args.out.mkdir(parents=True, exist_ok=True)
    for prior in (False, True):
        lines = []
        for name, (runs, prior_aware) in systems.items():
            if prior_aware and not prior:
                continue
            if not all(r in available for r in runs):
                print(f"skip {name}: missing {set(runs) - available}")
                continue
            examples, ids, per_run = load(args.logprob_dir, runs, "dev", args.data_root)
            s = score_tracks(examples, ids, np.mean(per_run, axis=0), prior)
            lines.append(f"{name} & " + " & ".join(fmt(r) for r in s) + r" \\")
        path = args.out / f"dev_{'prior' if prior else 'clean'}.tex"
        path.write_text("\n".join(lines) + "\n")
        print(f"wrote {path}")


def score_tracks(examples, ids, log_probs, prior):
    """Per-track (F1, Soft, Hard) decoded track by track, plus the ALL-4 mean."""
    index = {iid: i for i, iid in enumerate(ids)}
    rows = []
    for t in (1, 2, 3, 4):
        t_ids = [ex.instance_id for ex in examples[t]]
        labels = decode(
            t_ids,
            log_probs[[index[i] for i in t_ids]],
            mode="source",
            structural_prior=prior,
        )
        m = dico_metrics(examples[t], labels)
        rows.append([m["weighted_f1"], m["soft_cons"], m["hard_cons"]])
    rows.append(np.mean(rows, axis=0).tolist())
    return rows


if __name__ == "__main__":
    main()
