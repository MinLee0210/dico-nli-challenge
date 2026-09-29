"""Stacked NEG detector over ensemble log-probs + cheap pair features.

The reversible labels are decoded well; NEG vs. reversible is the bottleneck.
This fits a small classifier that predicts, per source group (pair_id within a
track file), whether the pair is NEGATIVE_OTHER. Inputs: the pooled canonical
log-probs of each ensemble member and lexical features of the pair. Groups
predicted reversible are decoded among EQ/FE/BE from the ensemble; groups
predicted NEG get NEG. No dataset-layout information (twin presence) is used.

Scores are out-of-fold: grouped K-fold by pair_id (shared across tracks, so no
source pair is in both train and test folds).

    PYTHONPATH=. uv run --with scikit-learn python scripts/training/neg_stacker.py \
        --logprob_dir results/dev/logprobs --runs e8_mdeberta_fine e7_xlmr_large
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.training.ensemble_eval import TRACKS, load
from src.labels import LABELS, NEGATIVE_ID, REVERSE_PERM, is_flipped
from src.pipelines.eval import dico_metrics

NEGATION = {"no", "not", "never", "without", "nor", "ez", "ez-", "gabe", "sin", "nunca", "ni"}


def tokens(s: str):
    return re.findall(r"\w+", s.lower())


def pair_features(t1: str, t2: str) -> list:
    a, b = tokens(t1), tokens(t2)
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    na = {w for w in sa if w.isdigit()}
    nb = {w for w in sb if w.isdigit()}
    return [
        inter / union,                              # Jaccard
        inter / (len(sa) or 1),                     # coverage of a by b
        inter / (len(sb) or 1),                     # coverage of b by a
        float(sa <= sb or sb <= sa),                # one side contains the other
        len(a), len(b), abs(len(a) - len(b)),
        float(bool(na) and bool(nb) and na != nb),  # conflicting numbers
        float(bool(sa & NEGATION) != bool(sb & NEGATION)),
        float(t1.strip().lower() == t2.strip().lower()),
    ]


def build(examples, ids, per_run):
    """One row per (track, pair_id) group: features, target, member rows."""
    index = {iid: i for i, iid in enumerate(ids)}
    perm = list(REVERSE_PERM)
    groups = defaultdict(list)
    for t in TRACKS:
        for ex in examples[t]:
            groups[(t, ex.instance_id.split("__", 1)[0])].append(ex)
    keys, X, y, canon = [], [], [], []
    for key, exs in groups.items():
        rows = [index[ex.instance_id] for ex in exs]
        flips = [is_flipped(ex.instance_id) for ex in exs]
        # Decoding may pool the twin (legitimate twin decoding) ...
        pooled = [
            np.stack([lp[r][perm] if f else lp[r] for r, f in zip(rows, flips)]).mean(axis=0)
            for lp in per_run
        ]
        ens = np.mean(pooled, axis=0)
        # ... but stacker features use the `original` row only: averaging two
        # rows vs. one would leak whether a reversed twin exists.
        ref = next((ex for ex in exs if not is_flipped(ex.instance_id)), exs[0])
        feats = []
        for lp in per_run:
            m = lp[index[ref.instance_id]]
            rev = np.delete(m, NEGATIVE_ID)
            feats += [m[NEGATIVE_ID], rev.max(), m[NEGATIVE_ID] - rev.max()]
        feats += pair_features(ref.text1, ref.text2)
        keys.append(key)
        X.append(feats)
        y.append(int(ref.label_id == NEGATIVE_ID))
        canon.append(ens)
    return keys, np.array(X), np.array(y), np.array(canon), groups


def decode_groups(keys, canon, is_neg, groups):
    """Instance labels: NEG where predicted, else ensemble argmax over EQ/FE/BE."""
    perm = list(REVERSE_PERM)
    out = {}
    for key, c, neg in zip(keys, canon, is_neg):
        if neg:
            choice = NEGATIVE_ID
        else:
            s = c.copy()
            s[NEGATIVE_ID] = -np.inf
            choice = int(np.argmax(s))
        for ex in groups[key]:
            lab = perm[choice] if is_flipped(ex.instance_id) else choice
            out[ex.instance_id] = LABELS[lab]
    return out


def evaluate(examples, labels_by_id):
    rows = []
    for t in TRACKS:
        exs = examples[t]
        m = dico_metrics(exs, [labels_by_id[ex.instance_id] for ex in exs])
        rows.append((m["weighted_f1"], m["soft_cons"], m["hard_cons"]))
    rows = np.array(rows)
    return rows, rows.mean(axis=0)


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--logprob_dir", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data_root", type=Path, default=Path("data/raw/dico/final_data"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6])
    args = ap.parse_args()

    examples, ids, per_run = load(args.logprob_dir, args.runs, "dev", args.data_root)
    keys, X, y, canon, groups = build(examples, ids, per_run)
    pair_ids = np.array([k[1] for k in keys])
    print(f"{len(keys)} groups, NEG rate {y.mean():.3f}, {X.shape[1]} features")

    # Baseline: ensemble argmax on the pooled canonical scores.
    base = decode_groups(keys, canon, canon.argmax(axis=1) == NEGATIVE_ID, groups)
    _, m = evaluate(examples, base)
    print(f"baseline ensemble        ALL {m[0]:.3f}/{m[1]:.3f}/{m[2]:.3f}")

    models = {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "gbdt": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300),
    }
    for name, make in models.items():
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=args.folds).split(X, y, pair_ids):
            clf = make().fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        acc = ((oof > 0.5) == y).mean()
        for t in args.thresholds:
            labels = decode_groups(keys, canon, oof > t, groups)
            rows, m = evaluate(examples, labels)
            per = " | ".join(f"T{k} {a:.3f}/{b:.3f}/{c:.3f}" for k, (a, b, c) in zip(TRACKS, rows))
            print(f"{name:6s} thr={t:.2f} (acc@.5 {acc:.3f})  ALL {m[0]:.3f}/{m[1]:.3f}/{m[2]:.3f}  ||  {per}")


if __name__ == "__main__":
    main()
