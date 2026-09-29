"""Checkpoint(s) -> DiCo-NLI submission CSVs.

Scores one or more input CSVs (participant labeled/unlabeled, any track) with
one or more checkpoints, averages log-probabilities across checkpoints (and
any saved `--logprobs` files from earlier runs), decodes with a consistency
mode, and writes `instance_id,label` files named for CodaBench
(`track<N>_predictions.csv`). Per-checkpoint log-probs are saved next to the
predictions so ensembles and decoding settings can be re-tuned without
re-running models.

All inputs are decoded jointly, so `--decoding source` pools every view of a
source pair across the given files (e.g. Tracks 1-4 together).

Usage:
    uv run python -m src.pipelines.predict \\
        --ckpt checkpoints/<run>/best.pt \\
        --inputs data/raw/dico/final_data/dev/dico_nli_dev_track1_participant_labeled.csv \\
        --out_dir results/predictions/dev --decoding twin
"""

import argparse
import re
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import ModelConfig
from src.data import DicoDataset, PairCollator, read_dico_csv, write_predictions
from src.decoding import DECODING_MODES, decode
from src.modules.model import PairClassifier, build_tokenizer
from src.pipelines.eval import dico_metrics, format_report, predict_log_probs
from src.utils.io_utils import save_json
from src.utils.model_utils import detect_device


def config_from_checkpoint(ckpt: dict) -> ModelConfig:
    """Rebuild ModelConfig from a checkpoint's saved `extra["cfg"]`."""
    saved = ckpt.get("extra", {}).get("cfg", {})
    known = {f.name for f in fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in saved.items() if k in known})


def load_model(ckpt_path: Path, device: torch.device):
    """Returns (model, cfg, tokenizer) rebuilt from a checkpoint."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location=str(device), weights_only=False)
    cfg = config_from_checkpoint(raw)
    model = PairClassifier(cfg).to(device)
    model.load_state_dict(raw["model"])
    model.eval()
    return model, cfg, build_tokenizer(cfg)


def output_name(input_path: Path) -> str:
    """`track<N>_predictions.csv` when the file name names a track."""
    match = re.search(r"track(\d)", input_path.name)
    return (
        f"track{match.group(1)}_predictions.csv"
        if match
        else f"{input_path.stem}_predictions.csv"
    )


def load_logprobs(paths: Sequence[Path]) -> Dict[str, List[np.ndarray]]:
    """id -> list of log-prob rows, from `.npz` files with `ids` / `log_probs`."""
    table: Dict[str, List[np.ndarray]] = defaultdict(list)
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            for iid, row in zip(data["ids"].tolist(), data["log_probs"]):
                table[iid].append(row)
    return table


@torch.no_grad()
def predict(
    inputs: Sequence[Path],
    out_dir: Path,
    ckpts: Sequence[Path] = (),
    logprob_files: Sequence[Path] = (),
    decoding: str = "twin",
    neg_bias: float = 0.0,
    structural_prior: bool = False,
    batch_size: int = 64,
) -> Dict[str, List[str]]:
    """Write one prediction CSV per input; return {input path: labels}."""
    if not ckpts and not logprob_files:
        raise ValueError("give at least one --ckpt or --logprobs file")
    out_dir = Path(out_dir)
    per_file = {Path(p): read_dico_csv(p) for p in inputs}
    examples = [ex for exs in per_file.values() for ex in exs]
    ids = [ex.instance_id for ex in examples]

    table = load_logprobs(logprob_files)
    device = detect_device()
    for ckpt in ckpts:
        model, cfg, tokenizer = load_model(Path(ckpt), device)
        for path, exs in per_file.items():
            dataset = DicoDataset(exs)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=PairCollator(dataset, tokenizer, cfg.max_length, cfg.symmetric),
            )
            log_probs, _ = predict_log_probs(model, loader, device)
            logprob_dir = out_dir / "logprobs"
            logprob_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                logprob_dir / f"{Path(ckpt).parent.name}__{path.stem}.npz",
                ids=np.array([ex.instance_id for ex in exs]),
                log_probs=log_probs,
            )
            for ex, row in zip(exs, log_probs):
                table[ex.instance_id].append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    missing = [i for i in ids if i not in table]
    if missing:
        raise ValueError(
            f"{len(missing)} instances have no log-probs, e.g. {missing[0]}"
        )
    averaged = np.stack([np.mean(table[i], axis=0) for i in ids])
    labels = decode(
        ids, averaged, decoding, neg_bias=neg_bias, structural_prior=structural_prior
    )

    result, start = {}, 0
    for path, exs in per_file.items():
        file_labels = labels[start : start + len(exs)]
        start += len(exs)
        target = out_dir / output_name(path)
        write_predictions(target, [ex.instance_id for ex in exs], file_labels)
        print(f"saved: {target}  ({len(exs)} rows)")
        if all(ex.label is not None for ex in exs):
            scores = dico_metrics(exs, file_labels)
            print(format_report(scores))
            save_json(scores, out_dir / f"{target.stem}_scores.json")
        result[str(path)] = file_labels
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--logprobs", type=Path, nargs="*", default=[], help="saved .npz log-probs"
    )
    parser.add_argument("--decoding", choices=DECODING_MODES, default="twin")
    parser.add_argument("--neg_bias", type=float, default=0.0)
    parser.add_argument("--structural_prior", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    predict(
        args.inputs,
        args.out_dir,
        ckpts=args.ckpt,
        logprob_files=args.logprobs,
        decoding=args.decoding,
        neg_bias=args.neg_bias,
        structural_prior=args.structural_prior,
        batch_size=args.batch_size,
    )
