"""Training loss and the logit-derived helpers used at evaluation time.

`DicoLoss` = cross-entropy + `consistency_weight` * reversal-consistency term.
The consistency term is a symmetric KL between p(x) and Rev(p(x_rev)) for
every row whose reversed twin is in the same batch (see `TwinBatchSampler`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.labels import REVERSE_PERM


def reverse_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Apply Rev to a (..., 4) distribution: swap FE and BE columns."""
    return log_probs[..., list(REVERSE_PERM)]


def reversal_consistency_loss(logits: torch.Tensor, twin: torch.Tensor) -> torch.Tensor:
    """Mean symmetric KL(p(x) || Rev p(x_rev)) over rows with an in-batch twin.

    `twin[i]` is the batch position of row i's reversed twin, or -1. Each pair
    is visited from both sides, which is what makes the KL symmetric.
    """
    has_twin = twin >= 0
    if not has_twin.any():
        return logits.new_zeros(())
    log_p = F.log_softmax(logits, dim=-1)
    rows = has_twin.nonzero(as_tuple=True)[0]
    own = log_p[rows]
    other = reverse_log_probs(log_p[twin[rows]])
    return F.kl_div(other, own, log_target=True, reduction="batchmean")


class DicoLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, consistency_weight: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.consistency_weight = consistency_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        twin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.ce(logits, labels)
        if self.consistency_weight > 0 and twin is not None:
            loss = loss + self.consistency_weight * reversal_consistency_loss(
                logits, twin
            )
        return loss


@torch.no_grad()
def probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Softmax probabilities, (B, num_classes)."""
    return F.softmax(logits, dim=1)


@torch.no_grad()
def predict(logits: torch.Tensor) -> torch.Tensor:
    """Argmax class predictions, (B,)."""
    return logits.argmax(dim=1)
