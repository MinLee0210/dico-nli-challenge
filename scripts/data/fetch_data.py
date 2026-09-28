"""Fetch the official DiCo-NLI task repository (data, scorer, starter kit).

Clones https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI into
`data/raw/dico` (gitignored), or fast-forwards it when already present. The
task repo is GPL-3.0; it is kept out of this repo's history and only used as
data plus an external scorer.

Layout afterwards:
    data/raw/dico/final_data/{train,dev}/dico_nli_<split>_track<N>_*.csv
    data/raw/dico/trial_data/...
    data/raw/dico/evaluation_functions/   (official scorer, python -m)

Usage:
    uv run python scripts/data/fetch_data.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_REPO = "https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI.git"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "raw" / "dico")
    parser.add_argument("--repo", type=str, default=TASK_REPO)
    args = parser.parse_args()

    if (args.out / ".git").exists():
        cmd = ["git", "-C", str(args.out), "pull", "--ff-only"]
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--depth", "1", args.repo, str(args.out)]
    print("$ " + " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        sys.exit("fetch failed")

    head = subprocess.run(
        ["git", "-C", str(args.out), "log", "-1", "--format=%h %cs %s"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"task repo at {args.out}  ({head})")
    for split in ("train", "dev"):
        files = sorted((args.out / "final_data" / split).glob("*.csv"))
        print(f"  final_data/{split}: {len(files)} files")


if __name__ == "__main__":
    main()
