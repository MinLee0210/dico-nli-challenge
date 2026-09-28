"""Evaluate a trained checkpoint on labeled DiCo-NLI files.

Reports Weighted F1 / SoftCons / HardCons (plus per-label P/R/F1) under every
decoding mode, or one mode with --decoding, optionally writing a JSON summary.

Usage:
    uv run python scripts/training/evaluate.py \
        --ckpt checkpoints/<run>/best.pt \
        --data data/raw/dico/final_data/dev/dico_nli_dev_track1_participant_labeled.csv
"""

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import DicoDataset, PairCollator, read_dico_files  # noqa: E402
from src.decoding import DECODING_MODES, decode  # noqa: E402
from src.pipelines.eval import dico_metrics, format_report, predict_log_probs  # noqa: E402
from src.pipelines.predict import load_model  # noqa: E402
from src.utils.io_utils import save_json  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--decoding", choices=DECODING_MODES, default=None)
    parser.add_argument("--neg_bias", type=float, default=0.0)
    parser.add_argument("--structural_prior", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    device = detect_device()
    model, cfg, tokenizer = load_model(args.ckpt, device)
    dataset = DicoDataset(read_dico_files(args.data))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PairCollator(dataset, tokenizer, cfg.max_length),
    )
    log_probs, _ = predict_log_probs(model, loader, device)
    ids = [ex.instance_id for ex in dataset.examples]

    results = {}
    for mode in [args.decoding] if args.decoding else DECODING_MODES:
        preds = decode(
            ids,
            log_probs,
            mode,
            neg_bias=args.neg_bias,
            structural_prior=args.structural_prior,
        )
        results[mode] = dico_metrics(dataset.examples, preds)
        print(f"\n[decoding={mode}]")
        print(format_report(results[mode]))

    if args.json:
        save_json(results, args.json)
        print(f"saved: {args.json}")


if __name__ == "__main__":
    main()
