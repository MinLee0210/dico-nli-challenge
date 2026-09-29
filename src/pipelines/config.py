"""Training-loop hyperparameters — separate from `src.config.ModelConfig`
(model architecture only). Controllable via a YAML file (see
`configs/train.yaml`) with individual fields overridable from the CLI.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from src.utils.io_utils import read_yaml


@dataclass
class TrainingConfig:
    # --- data ---
    # Paths are relative to `data_root` (the task repo's final_data/, fetched by
    # scripts/data/fetch_data.py). Lists so several tracks can be concatenated.
    data_root: str = "data/raw/dico/final_data"
    train_files: List[str] = field(
        default_factory=lambda: ["train/dico_nli_train_track1_participant_labeled.csv"]
    )
    val_files: List[str] = field(
        default_factory=lambda: ["dev/dico_nli_dev_track1_participant_labeled.csv"]
    )
    # Optional extra labeled files reported every validation pass. When empty,
    # the val split doubles as the reported/monitored split.
    test_files: List[str] = field(default_factory=list)
    # Label-safe augmentations applied to the training set only; any of
    # "reverse_negatives", "transitive" (see src/augment.py).
    augment: List[str] = field(default_factory=list)
    seed: int = 42

    # --- output ---
    ckpt_dir: str = "checkpoints"
    result_dir: str = "results"
    run_name: Optional[str] = None

    # --- optimization ---
    epochs: int = 10
    batch_size: int = 16
    accum_steps: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    amp: bool = False
    lr: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    log_every: int = 50
    eval_every: int = 1
    # Fraction of twin units visited per epoch (<1 = validate more often).
    epoch_fraction: float = 1.0
    # Train on the reversible labels only (for use with `structural_prior`,
    # which decides NEGATIVE_OTHER from the file layout instead of the model).
    drop_negatives: bool = False
    ckpt_every: int = 0  # 0 = only best.pt (HF backbones make epoch_*.pt large)

    # --- loss ---
    label_smoothing: float = 0.0
    # Weight of the symmetric KL between p(x) and Rev(p(x_rev)); 0 disables.
    consistency_weight: float = 0.0
    # Per-label CE weights in LABELS order (EQ, FE, BE, NEG); None = unweighted.
    class_weights: Optional[list] = None
    # Focal-loss gamma; 0 = plain cross-entropy.
    focal_gamma: float = 0.0

    # --- decoding (see src/decoding.py) ---
    decoding: str = "twin"  # independent | twin | source
    neg_bias: float = 0.0
    structural_prior: bool = False

    # Optional ModelConfig overrides, e.g.
    # {"backbone": "microsoft/deberta-v3-base", "max_length": 64}.
    arch: Optional[dict] = None

    # --- callbacks ---
    # None (or {"type": "none"}) disables LR scheduling. `t_max: auto` and
    # `warmup_ratio` are resolved from the number of optimizer steps. See
    # src/callbacks/lr_scheduler.py.
    lr_scheduler: Optional[dict] = None
    early_stopping: Optional[dict] = None
    save_best: bool = True
    # Which metric BestCheckpoint / early stopping monitor: weighted_f1,
    # soft_cons, hard_cons, dico_mean (mean of the three) are maximized;
    # val_loss is minimized.
    best_metric: str = "dico_mean"
    best_mode: str = "max"

    # None (or {"enabled": false}) disables W&B logging.
    wandb: Optional[dict] = None

    resume_from: Optional[str] = None


def load_training_config(
    config_path: Optional[Union[str, Path]] = None, **cli_overrides
) -> TrainingConfig:
    """Build a TrainingConfig from defaults, a YAML file, then CLI overrides
    (highest precedence, but only applied for keys the caller actually passed —
    argparse defaults of None are treated as "not set").
    """
    cfg = TrainingConfig()

    if config_path is not None:
        yaml_data = read_yaml(config_path)
        known_fields = set(asdict(cfg).keys())
        for key, value in yaml_data.items():
            if key not in known_fields:
                raise ValueError(
                    f"unknown training config key {key!r} in {config_path}"
                )
            setattr(cfg, key, value)

    for key, value in cli_overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    return cfg
