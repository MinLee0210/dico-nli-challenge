"""DiCo-NLI evaluation: Weighted F1, SoftCons, HardCons, plus diagnostics.

`dico_metrics` re-implements the official scorer's definitions
(evaluation_functions/metrics.py in the task repo, GPL-3.0, not vendored here):

    weighted_f1  support-weighted F1 over the four labels
    soft_cons    over gold-reversible twin pairs: first prediction is reversible
                 and the second equals its reverse (gold is ignored)
    hard_cons    over the same pairs: both predictions equal gold

`scripts/submission/official_score.py` runs the official scorer to confirm.

The per-pass work lives in `eval_per_epoch`: forward over a loader, decode with
the configured consistency mode, score. `evaluate` optionally persists it.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import DicoExample
from src.decoding import decode
from src.labels import LABELS, REVERSIBLE_LABELS, reverse_instance_id, reverse_label
from src.modules.loss import DicoLoss
from src.modules.model import PairClassifier, model_inputs
from src.utils.io_utils import save_json

HEADLINE_METRICS = ("weighted_f1", "soft_cons", "hard_cons")


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def reversible_pairs(examples: Sequence[DicoExample]) -> List[Tuple[int, int]]:
    """Index pairs (i, j) of gold-reversible twins, each pair listed once."""
    index = {ex.instance_id: i for i, ex in enumerate(examples)}
    pairs = []
    for i, ex in enumerate(examples):
        if ex.label not in REVERSIBLE_LABELS:
            continue
        j = index.get(reverse_instance_id(ex.instance_id))
        if j is not None and i < j:
            pairs.append((i, j))
    return pairs


def dico_metrics(
    examples: Sequence[DicoExample], predictions: Sequence[str]
) -> Dict[str, object]:
    """Official scores plus macro-F1, accuracy, per-label P/R/F1, confusion."""
    if len(examples) != len(predictions):
        raise ValueError("one prediction per example is required")
    if not examples:
        raise ValueError("cannot score zero examples")

    k = len(LABELS)
    lid = {label: i for i, label in enumerate(LABELS)}
    confusion = np.zeros((k, k), dtype=np.int64)
    for ex, pred in zip(examples, predictions):
        confusion[lid[ex.label], lid[pred]] += 1

    total = int(confusion.sum())
    per_label, weighted_f1, macro_values = {}, 0.0, []
    for c, label in enumerate(LABELS):
        tp = int(confusion[c, c])
        support = int(confusion[c].sum())
        predicted = int(confusion[:, c].sum())
        precision = _safe_div(tp, predicted)
        recall = _safe_div(tp, support)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        weighted_f1 += support / total * f1
        if support:
            macro_values.append(f1)

    pairs = reversible_pairs(examples)
    soft = hard = 0
    for i, j in pairs:
        pi, pj = predictions[i], predictions[j]
        if pi in REVERSIBLE_LABELS and pj == reverse_label(pi):
            soft += 1
        if pi == examples[i].label and pj == examples[j].label:
            hard += 1

    result = {
        "weighted_f1": weighted_f1,
        "soft_cons": _safe_div(soft, len(pairs)),
        "hard_cons": _safe_div(hard, len(pairs)),
        "macro_f1": float(np.mean(macro_values)),
        "accuracy": float(np.trace(confusion)) / total,
        "reversible_pairs": len(pairs),
        "per_label": per_label,
        "confusion": confusion.tolist(),
    }
    result["dico_mean"] = float(np.mean([result[m] for m in HEADLINE_METRICS]))
    return result


@torch.no_grad()
def predict_log_probs(
    model: PairClassifier,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[DicoLoss] = None,
) -> Tuple[np.ndarray, float]:
    """Log-probs `(N, 4)` in dataset order, and the mean loss (nan if unlabeled)."""
    model.eval()
    n = len(loader.dataset)
    out = np.zeros((n, len(LABELS)), dtype=np.float32)
    seen = np.zeros(n, dtype=bool)
    losses = []
    for batch in loader:
        logits, _ = model(**model_inputs(batch, device))
        idx = batch["index"].numpy()
        out[idx] = F.log_softmax(logits.float(), dim=-1).cpu().numpy()
        seen[idx] = True
        if criterion is not None and "labels" in batch:
            losses.append(criterion(logits, batch["labels"].to(device)).item())
    if not seen.all():
        raise RuntimeError("loader did not cover every example")
    return out, float(np.mean(losses)) if losses else float("nan")


def eval_per_epoch(
    model: PairClassifier,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[DicoLoss] = None,
    decoding: str = "twin",
    neg_bias: float = 0.0,
    structural_prior: bool = False,
) -> Dict[str, object]:
    """One full pass over a labeled `loader`: decode, score, attach the loss."""
    examples = loader.dataset.examples
    log_probs, loss = predict_log_probs(model, loader, device, criterion)
    ids = [ex.instance_id for ex in examples]
    preds = decode(
        ids, log_probs, decoding, neg_bias=neg_bias, structural_prior=structural_prior
    )
    result = dico_metrics(examples, preds)
    result["loss"] = loss
    result["decoding"] = decoding
    return result


def evaluate(
    model: PairClassifier,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[DicoLoss] = None,
    decoding: str = "twin",
    neg_bias: float = 0.0,
    structural_prior: bool = False,
    save_json_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """Evaluate `model` and optionally write the metrics dict to JSON."""
    result = eval_per_epoch(
        model,
        loader,
        device,
        criterion=criterion,
        decoding=decoding,
        neg_bias=neg_bias,
        structural_prior=structural_prior,
    )
    if save_json_path is not None:
        save_json(result, save_json_path)
    return result


def format_report(result: Dict[str, object]) -> str:
    lines = [
        f"weighted-F1={result['weighted_f1'] * 100:6.2f}  "
        f"SoftCons={result['soft_cons'] * 100:6.2f}  "
        f"HardCons={result['hard_cons'] * 100:6.2f}  "
        f"(macro-F1={result['macro_f1'] * 100:6.2f}  "
        f"acc={result['accuracy'] * 100:6.2f}  "
        f"loss={result.get('loss', float('nan')):.4f})"
    ]
    for label, m in result["per_label"].items():
        lines.append(
            f"  {label:<20s} P={m['precision'] * 100:6.2f}  "
            f"R={m['recall'] * 100:6.2f}  F1={m['f1'] * 100:6.2f}  n={m['support']}"
        )
    return "\n".join(lines)
