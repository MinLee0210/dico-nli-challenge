"""Convert iSTS 2016 answers-students chunk alignments to DiCo-NLI CSV rows.

Only the answers-students subset is used: it is not part of PhrasIS. The
headlines and images subsets ARE the PhrasIS source (and contain its test
pairs), so this script refuses them.

Mapping (chunk1, chunk2):
    EQUI -> EQUIVALENCE           SPE1 -> FORWARD_ENTAILMENT (chunk1 more specific)
    SPE2 -> BACKWARD_ENTAILMENT   SIMI / REL / OPPO -> NEGATIVE_OTHER
Dropped: NOALI / unaligned chunks, and types with _FACT / _POL modifiers.
Reversible pairs get both directions (like DiCo twins); NEG pairs one row.

    uv run python scripts/data/ists_to_dico.py --wa <file.wa> [...] --out data/external/ists_answers_students.csv
"""

import argparse
import csv
import re
from pathlib import Path

from src.labels import REVERSE_PERM, LABELS

TYPE_MAP = {
    "EQUI": "EQUIVALENCE",
    "SPE1": "FORWARD_ENTAILMENT",
    "SPE2": "BACKWARD_ENTAILMENT",
    "SIMI": "NEGATIVE_OTHER",
    "REL": "NEGATIVE_OTHER",
    "OPPO": "NEGATIVE_OTHER",
}
LINE = re.compile(r"^(.*?)<==>(.*?)//\s*(\S+)\s*//\s*(\S+)\s*//\s*(.*?)\s*<==>\s*(.*?)\s*$")


def reverse(label: str) -> str:
    return LABELS[REVERSE_PERM[LABELS.index(label)]]


def parse(path: Path):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.match(line)
        if not m:
            continue
        ids1, ids2, typ, _score, c1, c2 = m.groups()
        if ids1.strip() == "0" or ids2.strip() == "0" or "-not aligned-" in (c1 + c2):
            continue
        if typ not in TYPE_MAP:  # NOALI, *_FACT, *_POL, ...
            continue
        c1, c2 = c1.strip(), c2.strip()
        if not (c1 and c2):
            continue
        # Identical strings carry no signal unless labelled EQUI.
        if c1.lower() != c2.lower() or typ == "EQUI":
            yield c1, c2, TYPE_MAP[typ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wa", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for p in args.wa:
        if "answers-students" not in p.name:
            raise SystemExit(f"refusing {p.name}: only answers-students is PhrasIS-free")

    seen, rows = set(), []
    for p in args.wa:
        for c1, c2, label in parse(p):
            key = (c1.lower(), c2.lower())
            if key in seen or (key[1], key[0]) in seen:
                continue
            seen.add(key)
            pid = f"ists{len(seen)}"
            base = dict(pair_id=pid, text1_lang="en", text2_lang="en")
            rows.append({**base, "instance_id": f"{pid}__en-en__original",
                         "text1": c1, "text2": c2, "label": label})
            if label != "NEGATIVE_OTHER":
                rows.append({**base, "instance_id": f"{pid}__en-en__flipped",
                             "text1": c2, "text2": c1, "label": reverse(label)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["instance_id", "pair_id", "text1_lang", "text2_lang", "text1", "text2", "label"]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r[c] for c in cols} for r in rows)
    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"wrote {args.out}: {len(rows)} rows, {len(seen)} pairs, {counts}")


if __name__ == "__main__":
    main()
