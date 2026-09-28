"""Score prediction files with the organizers' official scorer.

Runs `python -m evaluation_functions` from the fetched task repo
(`scripts/data/fetch_data.py`) — it is GPL-3.0 and deliberately not vendored.
Use it to confirm numbers from `src.pipelines.eval.dico_metrics` before
reporting them.

Usage:
    uv run python scripts/submission/official_score.py \
        --gold data/raw/dico/final_data/dev/dico_nli_dev_track1_reference.csv \
        --predictions results/predictions/dev/track1_predictions.csv \
        --output_dir results/predictions/dev/track1_official
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_REPO = REPO_ROOT / "data" / "raw" / "dico"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="*_reference.csv")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--task_repo", type=Path, default=TASK_REPO)
    args = parser.parse_args()

    if not (args.task_repo / "evaluation_functions").exists():
        sys.exit(
            f"official scorer not found under {args.task_repo}; run scripts/data/fetch_data.py"
        )

    env = {**os.environ, "PYTHONPATH": str(args.task_repo)}
    cmd = [
        sys.executable,
        "-m",
        "evaluation_functions",
        "--gold",
        str(args.gold.resolve()),
        "--predictions",
        str(args.predictions.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
    ]
    code = subprocess.run(cmd, env=env).returncode
    if code != 0:
        sys.exit(code)
    print((args.output_dir / "scores.txt").read_text(), end="")


if __name__ == "__main__":
    main()
