"""End-to-end smoke test on synthetic DiCo-NLI data with a tiny random BERT.

Writes synthetic train/dev CSVs with the official structure, saves a tiny
offline backbone, trains two epochs with the consistency loss on, reloads the
best checkpoint, and writes Track-style predictions. No network, no GPU, no
real dataset required — it checks plumbing, not accuracy.

Usage:
    uv run python scripts/training/smoke_test.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import ModelConfig  # noqa: E402
from src.data import (  # noqa: E402
    SYNTHETIC_WORDS,
    make_synthetic_dico,
    read_dico_csv,
    write_dico_csv,
)
from src.modules.model import PairClassifier, build_hf_model, build_tokenizer  # noqa: E402
from src.pipelines.config import TrainingConfig  # noqa: E402
from src.pipelines.predict import predict  # noqa: E402
from src.pipelines.train import train  # noqa: E402
from src.utils.model_utils import save_tiny_backbone  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp(prefix="dico_smoke_"))
    try:
        data = tmp / "data"
        write_dico_csv(
            data / "train_track1.csv", make_synthetic_dico(n_pairs=60, seed=0)
        )
        write_dico_csv(data / "dev_track1.csv", make_synthetic_dico(n_pairs=20, seed=1))
        backbone = save_tiny_backbone(tmp / "tiny-bert", SYNTHETIC_WORDS)

        model_cfg = ModelConfig(backbone=backbone, max_length=32)
        model = PairClassifier(model_cfg, build_hf_model(model_cfg))
        cfg = TrainingConfig(
            data_root=str(data),
            train_files=["train_track1.csv"],
            val_files=["dev_track1.csv"],
            ckpt_dir=str(tmp / "checkpoints"),
            result_dir=str(tmp / "results"),
            run_name="smoke",
            epochs=2,
            batch_size=8,
            log_every=5,
            lr=1e-3,
            consistency_weight=0.5,
            lr_scheduler={
                "type": "warmup_linear",
                "t_max": "auto",
                "warmup_ratio": 0.1,
            },
        )
        train(cfg, model=model, tokenizer=build_tokenizer(model_cfg))

        best = tmp / "checkpoints" / "smoke" / "best.pt"
        if not best.exists():
            raise RuntimeError("no best.pt produced")
        out = tmp / "predictions"
        predict([data / "dev_track1.csv"], out, ckpts=[best], decoding="twin")

        pred_file = out / "track1_predictions.csv"
        header = pred_file.read_text().splitlines()[0]
        assert header == "instance_id,label", header
        rows = len(pred_file.read_text().splitlines()) - 1
        assert rows == len(read_dico_csv(data / "dev_track1.csv"))
        print("\nsmoke test: PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
