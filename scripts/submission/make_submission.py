"""Validate track prediction CSVs and pack them into a CodaBench ZIP.

Checks each `track<N>_predictions.csv` against the phase's submission template
(or participant file): exact `instance_id,label` header, every template id
present once, no unknown ids, only official labels. Then writes the files at
the ZIP root — CodaBench rejects nested paths and extra files.

Usage:
    uv run python scripts/submission/make_submission.py \
        --pred_dir results/predictions/dev \
        --templates data/raw/dico/final_data/dev/dico_nli_dev_track{1,2,3,4}_submission_template.csv \
        --out results/submissions/dev_submission.zip
"""

import argparse
import csv
import re
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.labels import LABELS  # noqa: E402


def track_of(path: Path) -> str:
    match = re.search(r"track(\d)", path.name)
    if not match:
        raise ValueError(f"cannot tell the track of {path.name}")
    return match.group(1)


def validate(pred_path: Path, template_path: Path) -> list[str]:
    """Return a list of problems (empty when the file is valid)."""
    with open(template_path, newline="", encoding="utf-8") as f:
        expected = [row["instance_id"] for row in csv.DictReader(f)]
    with open(pred_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)

    problems = []
    if header != ["instance_id", "label"]:
        problems.append(f"header must be instance_id,label, got {header}")
    ids = [r[0] for r in rows if r]
    if len(ids) != len(set(ids)):
        problems.append("duplicate instance ids")
    missing = set(expected) - set(ids)
    unknown = set(ids) - set(expected)
    if missing:
        problems.append(f"{len(missing)} missing ids, e.g. {sorted(missing)[0]}")
    if unknown:
        problems.append(f"{len(unknown)} unknown ids, e.g. {sorted(unknown)[0]}")
    bad = {r[1] for r in rows if len(r) != 2 or r[1] not in LABELS}
    if any(len(r) != 2 for r in rows):
        problems.append("every row must have exactly two columns")
    if bad:
        problems.append(f"invalid labels: {sorted(bad)[:3]}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--templates", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    templates = {track_of(p): p for p in args.templates}
    files, failed = [], False
    for track, template in sorted(templates.items()):
        pred = args.pred_dir / f"track{track}_predictions.csv"
        if not pred.exists():
            print(f"track{track}: no prediction file, skipped")
            continue
        problems = validate(pred, template)
        status = "OK" if not problems else "INVALID"
        print(f"track{track}: {status}  {pred}")
        for p in problems:
            print(f"    - {p}")
        failed |= bool(problems)
        files.append(pred)

    if failed:
        sys.exit("refusing to package an invalid submission")
    if not files:
        sys.exit("no prediction files found")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.name)
    print(f"wrote {args.out}: {', '.join(p.name for p in files)}")
    if len(files) < 4:
        print("note: four-track macro scores need all four track files")


if __name__ == "__main__":
    main()
