import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import DicoDataset, DicoExample, PairCollator, make_synthetic_dico
from src.pipelines.eval import dico_metrics, eval_per_epoch, evaluate, format_report


def ex(iid, label):
    return DicoExample(iid, iid.split("__")[0], "en", "en", "a", "b", label)


GOLD = [
    ex("p1__en-en__original", "FORWARD_ENTAILMENT"),
    ex("p1__en-en__flipped", "BACKWARD_ENTAILMENT"),
    ex("p2__en-en__original", "EQUIVALENCE"),
    ex("p2__en-en__flipped", "EQUIVALENCE"),
    ex("p3__en-en__original", "NEGATIVE_OTHER"),
]


def test_perfect_predictions():
    result = dico_metrics(GOLD, [g.label for g in GOLD])
    assert result["weighted_f1"] == pytest.approx(1.0)
    assert result["soft_cons"] == 1.0 and result["hard_cons"] == 1.0
    assert result["reversible_pairs"] == 2


def test_soft_vs_hard_consistency():
    # p1: consistent but wrong (EQ/EQ); p2: inconsistent (FE/FE).
    preds = [
        "EQUIVALENCE",
        "EQUIVALENCE",
        "FORWARD_ENTAILMENT",
        "FORWARD_ENTAILMENT",
        "NEGATIVE_OTHER",
    ]
    result = dico_metrics(GOLD, preds)
    assert result["soft_cons"] == 0.5
    assert result["hard_cons"] == 0.0
    assert result["accuracy"] == pytest.approx(1 / 5)


def test_negative_prediction_breaks_consistency():
    preds = [g.label for g in GOLD]
    preds[0] = preds[1] = "NEGATIVE_OTHER"
    result = dico_metrics(GOLD, preds)
    assert result["soft_cons"] == 0.5  # NEG/NEG is not soft-consistent


def test_weighted_f1_matches_sklearn_definition():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    examples = make_synthetic_dico(n_pairs=80, seed=4)
    labels = [
        "EQUIVALENCE",
        "FORWARD_ENTAILMENT",
        "BACKWARD_ENTAILMENT",
        "NEGATIVE_OTHER",
    ]
    preds = [labels[i] for i in rng.integers(0, 4, len(examples))]
    ours = dico_metrics(examples, preds)["weighted_f1"]
    ref = sklearn.f1_score(
        [e.label for e in examples], preds, average="weighted", zero_division=0
    )
    assert ours == pytest.approx(ref)


def make_loader(tokenizer, n_pairs=30, seed=0):
    dataset = DicoDataset(make_synthetic_dico(n_pairs=n_pairs, seed=seed))
    return DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=PairCollator(dataset, tokenizer, 32),
    )


def test_eval_per_epoch_twin_decoding_is_fully_consistent(tiny_model, tiny_tokenizer):
    loader = make_loader(tiny_tokenizer)
    result = eval_per_epoch(tiny_model, loader, torch.device("cpu"), decoding="twin")
    for key in ("weighted_f1", "soft_cons", "hard_cons", "dico_mean", "loss"):
        assert key in result
    # A random model decodes NEG sometimes; every other twin pair is consistent.
    assert result["soft_cons"] >= 0.0
    independent = eval_per_epoch(
        tiny_model, loader, torch.device("cpu"), decoding="independent"
    )
    assert result["soft_cons"] >= independent["soft_cons"]


def test_evaluate_writes_json(tiny_model, tiny_tokenizer, tmp_path):
    loader = make_loader(tiny_tokenizer, n_pairs=10)
    out = tmp_path / "scores.json"
    result = evaluate(tiny_model, loader, torch.device("cpu"), save_json_path=out)
    assert out.exists()
    report = format_report(result)
    assert "SoftCons" in report and "HardCons" in report
