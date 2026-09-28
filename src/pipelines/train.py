"""DiCo-NLI training entrypoint.

Fine-tunes a Hugging Face cross-encoder on ordered phrase pairs with
cross-entropy plus an optional reversal-consistency term. Batches keep each
instance with its reversed twin. Validation decodes with the configured
consistency mode and reports Weighted F1 / SoftCons / HardCons. Controlled via
a YAML config (see configs/train.yaml) with individual CLI overrides. Callbacks
(LR schedule, early stopping, best checkpoint, W&B) are wired from that file.

Usage:
    uv run python -m src.pipelines.train --config configs/train.yaml
    uv run python -m src.pipelines.train --config configs/train.yaml --epochs 5 --lr 3e-5
"""

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from src.callbacks import (
    BestCheckpoint,
    EarlyStopping,
    TrainContext,
    TrainerState,
    build_lr_scheduler,
)
from src.augment import augment
from src.callbacks.wandb_callback import WandbCallback
from src.config import ModelConfig
from src.data import DicoDataset, PairCollator, TwinBatchSampler, read_dico_files
from src.modules.loss import DicoLoss
from src.modules.model import PairClassifier, build_tokenizer, model_inputs
from src.pipelines._utils import announce_training
from src.pipelines.config import TrainingConfig, load_training_config
from src.pipelines.eval import HEADLINE_METRICS, eval_per_epoch, format_report
from src.utils.io_utils import load_env, save_checkpoint, save_json
from src.utils.model_utils import detect_device, get_run_name

REPO_ROOT = Path(__file__).resolve().parents[2]

# Metrics copied into TrainerState.extra — what callbacks can monitor.
MONITORED_METRICS = HEADLINE_METRICS + ("dico_mean", "macro_f1", "accuracy")


def _paths(data_root: str, names: Optional[List[str]]) -> List[Path]:
    paths = [Path(data_root) / name for name in names or []]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "data files not found (run scripts/data/fetch_data.py?): "
            + ", ".join(missing)
        )
    return paths


def build_model(model_cfg: ModelConfig, device: torch.device) -> PairClassifier:
    return PairClassifier(model_cfg).to(device)


def build_optimizer(model: torch.nn.Module, train_cfg: TrainingConfig):
    """AdamW without weight decay on biases and normalization weights."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_norm = "norm" in name.lower() or name.endswith(".bias")
        (no_decay if is_norm else decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
    )


def resolve_scheduler_cfg(cfg: Optional[dict], total_steps: int) -> Optional[dict]:
    """Fill `t_max: auto` and `warmup_ratio` from the run's optimizer steps."""
    if not cfg:
        return cfg
    cfg = dict(cfg)
    if cfg.get("t_max") == "auto":
        cfg["t_max"] = total_steps
    if "warmup_ratio" in cfg:
        cfg["warmup_steps"] = max(int(cfg.pop("warmup_ratio") * total_steps), 1)
    return cfg


def build_callbacks(
    train_cfg: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    run_name: str,
    wandb_config: dict,
    total_steps: int,
):
    callbacks = []

    lr_cb = build_lr_scheduler(
        optimizer, resolve_scheduler_cfg(train_cfg.lr_scheduler, total_steps)
    )
    if lr_cb is not None:
        callbacks.append(lr_cb)

    if train_cfg.early_stopping and train_cfg.early_stopping.get("enabled", True):
        es_kwargs = {
            k: v for k, v in train_cfg.early_stopping.items() if k != "enabled"
        }
        callbacks.append(EarlyStopping(**es_kwargs))

    if train_cfg.save_best:
        callbacks.append(
            BestCheckpoint(monitor=train_cfg.best_metric, mode=train_cfg.best_mode)
        )

    # after BestCheckpoint: relies on <ckpt_dir>/best.pt already existing
    if train_cfg.wandb and train_cfg.wandb.get("enabled", False):
        wb = train_cfg.wandb
        callbacks.append(
            WandbCallback(
                project_name=wb.get("project", "dico-nli"),
                run_name=run_name,
                config=wandb_config,
                entity=wb.get("entity"),
                monitor=wb.get("monitor", train_cfg.best_metric),
                mode=wb.get("mode", train_cfg.best_mode),
                log_artifacts=wb.get("log_artifacts", True),
                group=wb.get("group"),
            )
        )

    return callbacks


def build_loaders(
    train_cfg: TrainingConfig,
    tokenizer,
    max_length: int,
    device: torch.device,
) -> tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], DicoDataset]:
    """Returns (train_loader, val_loader, test_loader, train_dataset)."""
    train_paths = _paths(train_cfg.data_root, train_cfg.train_files)
    if not train_paths:
        raise ValueError("train_files must list at least one file")
    train_examples = read_dico_files(train_paths)
    if train_cfg.augment:
        n = len(train_examples)
        train_examples = augment(train_examples, train_cfg.augment)
        print(f"augment {train_cfg.augment}: {n} -> {len(train_examples)} examples")
    train_dataset = DicoDataset(train_examples)
    if not train_dataset.has_labels:
        raise ValueError("training files must be labeled")

    pin_memory = train_cfg.pin_memory and device.type == "cuda"

    def eval_loader(names: List[str]) -> Optional[DataLoader]:
        paths = _paths(train_cfg.data_root, names)
        if not paths:
            return None
        dataset = DicoDataset(read_dico_files(paths))
        return DataLoader(
            dataset,
            batch_size=train_cfg.batch_size * 2,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=pin_memory,
            collate_fn=PairCollator(dataset, tokenizer, max_length),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=TwinBatchSampler(
            train_dataset, train_cfg.batch_size, shuffle=True, seed=train_cfg.seed
        ),
        num_workers=train_cfg.num_workers,
        pin_memory=pin_memory,
        collate_fn=PairCollator(train_dataset, tokenizer, max_length),
    )
    return (
        train_loader,
        eval_loader(train_cfg.val_files),
        eval_loader(train_cfg.test_files),
        train_dataset,
    )


def _optimizer_step(
    model,
    optimizer,
    scaler,
    callbacks,
    ctx,
    pending_losses,
    step,
    epoch,
    log_every,
    max_grad_norm,
):
    if max_grad_norm and max_grad_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    step += 1
    avg_loss = sum(pending_losses) / len(pending_losses)
    state = TrainerState(step=step, train_loss=avg_loss, epoch=epoch)
    for cb in callbacks:
        cb.on_step_end(ctx, state)
    if step % log_every == 0 or step == 1:
        lr = optimizer.param_groups[0]["lr"]
        print(f"  epoch {epoch:3d}  step {step:5d}  loss={avg_loss:.4f}  lr={lr:.2e}")
    return step, avg_loss


def train_per_epoch(
    model,
    criterion,
    loader,
    optimizer,
    scaler,
    accum_steps,
    device,
    callbacks,
    ctx,
    step,
    epoch,
    log_every,
    max_grad_norm=1.0,
):
    model.train()
    epoch_losses, pending = [], []
    optimizer.zero_grad()

    for batch in loader:
        labels = batch["labels"].to(device)
        twin = batch["twin"].to(device)

        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits, _ = model(**model_inputs(batch, device))
            loss = criterion(logits.float(), labels, twin)

        scaler.scale(loss / accum_steps).backward()
        pending.append(loss.item())

        if len(pending) == accum_steps:
            step, avg_loss = _optimizer_step(
                model,
                optimizer,
                scaler,
                callbacks,
                ctx,
                pending,
                step,
                epoch,
                log_every,
                max_grad_norm,
            )
            epoch_losses.append(avg_loss)
            pending = []

    if pending:
        step, avg_loss = _optimizer_step(
            model,
            optimizer,
            scaler,
            callbacks,
            ctx,
            pending,
            step,
            epoch,
            log_every,
            max_grad_norm,
        )
        epoch_losses.append(avg_loss)

    return (
        sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan"),
        step,
    )


def train(
    train_cfg: TrainingConfig,
    model: Optional[PairClassifier] = None,
    tokenizer=None,
):
    """Run training. `model` / `tokenizer` may be injected (tests, smoke run);
    otherwise they are built from `train_cfg.arch` over ModelConfig defaults.
    """
    load_env(REPO_ROOT / ".env")  # WANDB_API_KEY / HF token for callbacks
    device = detect_device()
    print(f"device: {device}")

    torch.manual_seed(train_cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_cfg.seed)
    elif device.type == "mps":
        torch.mps.manual_seed(train_cfg.seed)

    model_cfg = (
        model.cfg if model is not None else ModelConfig(**(train_cfg.arch or {}))
    )
    tokenizer = tokenizer if tokenizer is not None else build_tokenizer(model_cfg)
    train_loader, val_loader, test_loader, train_dataset = build_loaders(
        train_cfg, tokenizer, model_cfg.max_length, device
    )
    print(f"train examples: {len(train_dataset)}  backbone: {model_cfg.backbone}")

    model = model.to(device) if model is not None else build_model(model_cfg, device)
    criterion = DicoLoss(
        label_smoothing=train_cfg.label_smoothing,
        consistency_weight=train_cfg.consistency_weight,
    ).to(device)
    optimizer = build_optimizer(model, train_cfg)
    scaler = torch.amp.GradScaler(
        "cuda" if device.type == "cuda" else "cpu",
        enabled=(train_cfg.amp and device.type == "cuda"),
    )

    run_name = train_cfg.run_name or get_run_name(
        Path(model_cfg.backbone).name,
        "dico",
        train_cfg.lr,
        train_cfg.batch_size,
    )
    ckpt_dir = Path(train_cfg.ckpt_dir) / run_name
    result_dir = Path(train_cfg.result_dir) / run_name

    total_steps = train_cfg.epochs * math.ceil(
        len(train_loader) / max(train_cfg.accum_steps, 1)
    )
    callbacks = build_callbacks(
        train_cfg,
        optimizer,
        run_name,
        {**asdict(model_cfg), **asdict(train_cfg)},
        total_steps,
    )
    early_stoppers = [cb for cb in callbacks if isinstance(cb, EarlyStopping)]

    wandb_run_id = None
    for cb in callbacks:
        if isinstance(cb, WandbCallback) and cb.enabled and cb.run is not None:
            wandb_run_id = cb.run.id

    ctx = TrainContext(
        model=model,
        optimizer=optimizer,
        ckpt_dir=ckpt_dir,
        cfg_dict=asdict(model_cfg),
        wandb_run_id=wandb_run_id,
    )
    for cb in callbacks:
        cb.on_train_start(ctx)

    if train_cfg.resume_from:
        raw = torch.load(train_cfg.resume_from, map_location=str(device))
        model.load_state_dict(raw["model"])
        if "optimizer" in raw:
            optimizer.load_state_dict(raw["optimizer"])
        step = int(raw.get("step", 0))
        start_epoch = int(raw.get("extra", {}).get("epoch", 0)) + 1
        print(f"resumed at step {step}, continuing from epoch {start_epoch}")
    else:
        step, start_epoch = 0, 1

    announce_training(
        model=model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_loader=train_loader,
        train_dataset=train_dataset,
        device=device,
        run_name=run_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
    )

    decode_kwargs = dict(
        decoding=train_cfg.decoding,
        neg_bias=train_cfg.neg_bias,
        structural_prior=train_cfg.structural_prior,
    )
    history = []
    stopped_early = False
    epoch = start_epoch
    for epoch in range(start_epoch, train_cfg.epochs + 1):
        train_loss, step = train_per_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            train_cfg.accum_steps,
            device,
            callbacks,
            ctx,
            step,
            epoch,
            train_cfg.log_every,
            train_cfg.max_grad_norm,
        )
        record = {"epoch": epoch, "step": step, "loss": train_loss}
        print(f"epoch {epoch:3d}/{train_cfg.epochs}  train_loss={train_loss:.4f}")

        should_validate = epoch % train_cfg.eval_every == 0 or epoch == train_cfg.epochs
        if should_validate:
            val_result = (
                eval_per_epoch(
                    model, val_loader, device, criterion=criterion, **decode_kwargs
                )
                if val_loader is not None
                else None
            )
            # Report the test split when present, else fall back to the val
            # split so a run without an extra labeled split still has metrics.
            metric_result = (
                eval_per_epoch(
                    model, test_loader, device, criterion=criterion, **decode_kwargs
                )
                if test_loader is not None
                else val_result
            )
            extra = {}
            if metric_result is not None:
                extra = {k: metric_result[k] for k in MONITORED_METRICS}
                extra["metric_loss"] = metric_result["loss"]
                record.update(extra)
                print(format_report(metric_result))
            state = TrainerState(
                step=step,
                train_loss=train_loss,
                epoch=epoch,
                val_loss=val_result["loss"] if val_result is not None else None,
                extra=extra,
            )
            record["val_loss"] = state.val_loss
            for cb in callbacks:
                cb.on_validation_end(ctx, state)
            if any(es.should_stop for es in early_stoppers):
                stopped_early = True

        history.append(record)

        periodic = train_cfg.ckpt_every > 0 and epoch % train_cfg.ckpt_every == 0
        final_without_best = not train_cfg.save_best and (
            epoch == train_cfg.epochs or stopped_early
        )
        if periodic or final_without_best:
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            save_checkpoint(
                model,
                optimizer,
                step,
                ckpt_path,
                extra={
                    "cfg": asdict(model_cfg),
                    "epoch": epoch,
                    "wandb_run_id": ctx.wandb_run_id,
                },
            )
            print(f"  saved checkpoint: {ckpt_path}")

        if stopped_early:
            break

    for cb in callbacks:
        cb.on_train_end(ctx)

    save_json(history, ckpt_dir / "train_log.json")
    save_json(history, result_dir / "train_log.json")
    print(
        f"\n{'stopped early' if stopped_early else 'finished'} "
        f"at epoch {epoch} (step {step})"
    )
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML, see configs/train.yaml"
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--train_files", type=str, nargs="+", default=None)
    parser.add_argument("--val_files", type=str, nargs="+", default=None)
    parser.add_argument("--test_files", type=str, nargs="+", default=None)
    parser.add_argument("--augment", type=str, nargs="*", default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--accum_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--ckpt_every", type=int, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--consistency_weight", type=float, default=None)
    parser.add_argument(
        "--decoding", type=str, default=None, choices=["independent", "twin", "source"]
    )
    parser.add_argument("--neg_bias", type=float, default=None)
    parser.add_argument("--structural_prior", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--pin_memory", action="store_true", default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    train_cfg = load_training_config(args.config, **overrides)
    train(train_cfg)
