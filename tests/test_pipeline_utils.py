from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.data import DicoDataset, PairCollator, TwinBatchSampler, make_synthetic_dico
from src.pipelines._utils import announce_training, label_counts, module_param_counts
from src.pipelines.config import TrainingConfig, load_training_config
from src.pipelines.train import resolve_scheduler_cfg


def test_label_counts_labeled_and_unlabeled():
    dataset = DicoDataset(make_synthetic_dico(n_pairs=20, seed=0))
    assert sum(label_counts(dataset).values()) == len(dataset)

    class Plain(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, i):
            return 0

    assert label_counts(Plain()) is None


def test_module_param_counts_sum_to_total(tiny_model):
    total = sum(p.numel() for p in tiny_model.parameters())
    assert sum(n for _, n in module_param_counts(tiny_model)) == total


def test_resolve_scheduler_cfg():
    cfg = resolve_scheduler_cfg(
        {"type": "warmup_linear", "t_max": "auto", "warmup_ratio": 0.1}, 200
    )
    assert cfg == {"type": "warmup_linear", "t_max": 200, "warmup_steps": 20}
    assert resolve_scheduler_cfg(None, 10) is None


def test_train_yaml_loads():
    root = Path(__file__).resolve().parents[1]
    for name in sorted(p.name for p in (root / "configs").glob("*.yaml")):
        cfg = load_training_config(root / "configs" / name)
        assert cfg.train_files and cfg.decoding in ("independent", "twin", "source")


def test_announce_training_prints_config_data_and_model(
    capsys, tiny_model, tiny_cfg, tiny_tokenizer
):
    dataset = DicoDataset(make_synthetic_dico(n_pairs=12, seed=1))
    loader = DataLoader(
        dataset,
        batch_sampler=TwinBatchSampler(dataset, 4),
        collate_fn=PairCollator(dataset, tiny_tokenizer, 32),
    )
    announce_training(
        model=tiny_model,
        model_cfg=tiny_cfg,
        train_cfg=TrainingConfig(seed=7),
        train_loader=loader,
        train_dataset=dataset,
        device=torch.device("cpu"),
        run_name="run-x",
        ckpt_dir=Path("ckpt"),
        result_dir=Path("res"),
    )
    out = capsys.readouterr().out
    assert "RUN  run-x" in out
    assert "model config (ModelConfig)" in out
    assert "training config (TrainingConfig)" in out
    assert "data: train sample" in out
    assert "TOTAL" in out
    assert "forward check (untrained)" in out
    assert "logits" in out
