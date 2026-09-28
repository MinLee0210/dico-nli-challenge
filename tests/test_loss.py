import torch

from src.labels import REVERSE_PERM
from src.modules.loss import DicoLoss, reversal_consistency_loss, reverse_log_probs


def test_consistency_loss_zero_for_consistent_twins():
    a = torch.tensor([[2.0, 1.0, -1.0, 0.0]])
    b = a[:, list(REVERSE_PERM)]  # exactly Rev of a
    logits = torch.cat([a, b])
    twin = torch.tensor([1, 0])
    assert reversal_consistency_loss(logits, twin).item() < 1e-6


def test_consistency_loss_positive_for_same_direction_prediction():
    a = torch.tensor([[0.0, 3.0, -3.0, 0.0]])  # FE
    logits = torch.cat([a, a])  # twin also FE -> inconsistent
    twin = torch.tensor([1, 0])
    assert reversal_consistency_loss(logits, twin).item() > 1.0


def test_consistency_loss_ignores_rows_without_twin():
    logits = torch.randn(3, 4)
    assert reversal_consistency_loss(logits, torch.tensor([-1, -1, -1])).item() == 0.0


def test_reverse_log_probs_is_involution():
    x = torch.randn(5, 4)
    assert torch.equal(reverse_log_probs(reverse_log_probs(x)), x)


def test_dico_loss_adds_weighted_consistency():
    torch.manual_seed(0)
    logits = torch.randn(4, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    twin = torch.tensor([1, 0, -1, -1])
    plain = DicoLoss()(logits, labels, twin)
    weighted = DicoLoss(consistency_weight=1.0)(logits, labels, twin)
    assert weighted.item() >= plain.item()
    weighted.backward()
    assert logits.grad is not None
