"""Presentation helpers for the training entrypoint.

Kept separate from `src.pipelines.train` so the training loop stays readable and
this introspection can be reused (a debug script) or tested on its own. Nothing
here affects training — it only prints.

Output is rendered with `rich` when it is installed (panels, tables, JSON
highlighting) and falls back to aligned plain text otherwise. `rich` is an
optional extra: `uv sync --extra rich`.
"""

import json
from collections import Counter, OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.config import ModelConfig
from src.pipelines.config import TrainingConfig

try:  # optional pretty-printing dependency
    from rich.console import Console
    from rich.json import JSON
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table

    _HAS_RICH = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _HAS_RICH = False

WIDTH = 72


def banner(title: str) -> None:
    print(f"\n{title}\n{'-' * WIDTH}")


def indent(text: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def label_counts(dataset: Dataset) -> Optional[Dict[str, int]]:
    """Per-label example counts when the dataset carries gold labels, else None."""
    examples = getattr(dataset, "examples", None)
    if not examples or any(ex.label is None for ex in examples):
        return None
    return dict(sorted(Counter(ex.label for ex in examples).items()))


def twin_summary(dataset: Dataset) -> Optional[str]:
    twins = getattr(dataset, "twins", None)
    if twins is None:
        return None
    with_twin = sum(t is not None for t in twins)
    return f"{with_twin} rows with reversed twin, {len(twins) - with_twin} singletons"


def module_param_counts(
    model: torch.nn.Module, depth: int = 3
) -> List[Tuple[str, int]]:
    """Parameter counts grouped by the first `depth` name components."""
    groups: "OrderedDict[str, int]" = OrderedDict()
    for name, param in model.named_parameters():
        key = ".".join(name.split(".")[:depth])
        groups[key] = groups.get(key, 0) + param.numel()
    return list(groups.items())


def _sample_rows(train_dataset: Dataset, n: int) -> List[Tuple[str, str, str]]:
    examples = getattr(train_dataset, "examples", [])[:n]
    return [(ex.text1, ex.text2, ex.label or "?") for ex in examples]


def announce_training(
    *,
    model: torch.nn.Module,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    train_loader: DataLoader,
    train_dataset: Dataset,
    device: torch.device,
    run_name: str,
    ckpt_dir: Path,
    result_dir: Path,
    sample_rows: int = 4,
) -> None:
    """Print configs, a data sample, and the model layout before training."""
    common = dict(
        model=model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_loader=train_loader,
        train_dataset=train_dataset,
        device=device,
        run_name=run_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
        sample_rows=sample_rows,
    )
    if _HAS_RICH:
        _announce_rich(**common)
    else:
        _announce_plain(**common)


def _forward_check(model: torch.nn.Module, batch: dict, device) -> str:
    """A no-grad forward on a real batch; never raises."""
    from src.modules.model import model_inputs

    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits, feature = model(**model_inputs(batch, device))
        model.train(was_training)
        preview = " ".join(f"{v:+.3f}" for v in logits[0].tolist())
        return (
            f"input_ids {tuple(batch['input_ids'].shape)} -> logits "
            f"{tuple(logits.shape)}  feature {tuple(feature.shape)}\n"
            f"logits[0] = [{preview}]"
        )
    except Exception as e:  # noqa: BLE001 - a shape peek must never block training
        return f"forward check skipped: {e}"


# --- plain-text fallback ------------------------------------------------------


def _announce_plain(
    *,
    model,
    model_cfg,
    train_cfg,
    train_loader,
    train_dataset,
    device,
    run_name,
    ckpt_dir,
    result_dir,
    sample_rows,
) -> None:
    print("\n" + "=" * WIDTH)
    print(f"RUN  {run_name}")
    print("=" * WIDTH)
    print(f"device:      {device}   seed: {train_cfg.seed}")
    print(f"checkpoints: {ckpt_dir}")
    print(f"results:     {result_dir}")

    banner("model config (ModelConfig)")
    print(indent(json.dumps(asdict(model_cfg), indent=2, default=str)))
    banner("training config (TrainingConfig)")
    print(indent(json.dumps(asdict(train_cfg), indent=2, default=str)))

    counts = label_counts(train_dataset)
    banner(f"data: train sample  ({len(train_dataset)} examples)")
    if counts is not None:
        print("label counts: " + "  ".join(f"{k}:{v}" for k, v in counts.items()))
    summary = twin_summary(train_dataset)
    if summary:
        print(summary)
    for text1, text2, label in _sample_rows(train_dataset, sample_rows):
        print(f"  {text1!r}  ->  {text2!r}   [{label}]")

    batch = next(iter(train_loader))
    print(f"batch: input_ids {tuple(batch['input_ids'].shape)}")

    banner(f"model: {type(model).__name__}  ({model_cfg.backbone})")
    total = 0
    for name, n in module_param_counts(model):
        total += n
        print(f"  {name:<44s} {n:>14,d}")
    print(f"  {'TOTAL':<44s} {total:>14,d}")

    banner("forward check (untrained)")
    print(_forward_check(model, batch, device))

    print("\n" + "=" * WIDTH + "\n")


# --- rich rendering -----------------------------------------------------------


def _announce_rich(
    *,
    model,
    model_cfg,
    train_cfg,
    train_loader,
    train_dataset,
    device,
    run_name,
    ckpt_dir,
    result_dir,
    sample_rows,
) -> None:
    console = Console(highlight=False)
    console.print()
    console.print(Rule(f"[bold]RUN  {run_name}[/bold]"))

    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold cyan", justify="right")
    info.add_column()
    info.add_row("device", str(device))
    info.add_row("seed", str(train_cfg.seed))
    info.add_row("checkpoints", str(ckpt_dir))
    info.add_row("results", str(result_dir))
    console.print(info)

    console.print(
        Panel(
            JSON(json.dumps(asdict(model_cfg), default=str)),
            title="model config (ModelConfig)",
            border_style="blue",
        )
    )
    console.print(
        Panel(
            JSON(json.dumps(asdict(train_cfg), default=str)),
            title="training config (TrainingConfig)",
            border_style="blue",
        )
    )

    # --- data sample ---
    counts = label_counts(train_dataset)
    batch = next(iter(train_loader))
    caption = f"batch: input_ids {tuple(batch['input_ids'].shape)}"
    summary = twin_summary(train_dataset)
    if summary:
        caption += f"   {summary}"
    if counts is not None:
        caption += "\n" + "  ".join(f"{k}:{v}" for k, v in counts.items())
    table = Table(
        title=f"data: train sample  ({len(train_dataset)} examples)",
        border_style="green",
        caption=caption,
    )
    table.add_column("text1")
    table.add_column("text2")
    table.add_column("label", style="bold")
    for text1, text2, label in _sample_rows(train_dataset, sample_rows):
        table.add_row(text1, text2, label)
    console.print(table)

    # --- model layout ---
    model_table = Table(
        title=f"model: {type(model).__name__}  ({model_cfg.backbone})",
        border_style="magenta",
    )
    model_table.add_column("module")
    model_table.add_column("params", justify="right")
    total = 0
    for name, n in module_param_counts(model):
        total += n
        model_table.add_row(name, f"{n:,d}")
    model_table.add_section()
    model_table.add_row("TOTAL", f"{total:,d}", style="bold")
    console.print(model_table)

    # --- forward sanity check ---
    console.print(
        Panel(
            _forward_check(model, batch, device),
            title="forward check (untrained)",
            border_style="yellow",
        )
    )
    console.print()
