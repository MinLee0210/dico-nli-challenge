import pytest
import torch

from src.config import ModelConfig
from src.data import DicoDataset, PairCollator, make_synthetic_dico
from src.labels import NUM_LABELS
from src.modules.model import model_inputs


def test_forward_shapes(tiny_model, tiny_tokenizer):
    dataset = DicoDataset(make_synthetic_dico(n_pairs=6, seed=0))
    batch = PairCollator(dataset, tiny_tokenizer, 32)(list(range(len(dataset))))
    with torch.no_grad():
        logits, feature = tiny_model(**model_inputs(batch, torch.device("cpu")))
    assert logits.shape == (len(dataset), NUM_LABELS)
    assert feature.shape == (len(dataset), tiny_model.hf.config.hidden_size)
    assert torch.isfinite(logits).all()


def test_model_inputs_filters_batch_keys():
    batch = {
        "input_ids": torch.zeros(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.zeros(2, dtype=torch.long),
        "twin": torch.zeros(2, dtype=torch.long),
    }
    assert set(model_inputs(batch, torch.device("cpu"))) == {
        "input_ids",
        "attention_mask",
    }


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(backbone="")
    with pytest.raises(ValueError):
        ModelConfig(num_labels=3)
    with pytest.raises(ValueError):
        ModelConfig(classifier_dropout=1.0)
    assert ModelConfig(backbone="x").tokenizer_name == "x"
    assert ModelConfig(backbone="x", tokenizer="y").tokenizer_name == "y"


def test_symmetric_model_is_reversal_equivariant(tiny_cfg, tiny_tokenizer):
    from dataclasses import replace

    from src.labels import REVERSE_PERM
    from src.modules.model import PairClassifier, build_hf_model

    cfg = replace(tiny_cfg, symmetric=True)
    model = PairClassifier(cfg, build_hf_model(cfg)).eval()
    dataset = DicoDataset(make_synthetic_dico(n_pairs=6, seed=0))
    idx = list(range(len(dataset)))
    fwd = PairCollator(dataset, tiny_tokenizer, 32, with_swap=True)(idx)
    # The same batch with text1/text2 exchanged: sw_* becomes the main encoding.
    keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in fwd]
    rev = {
        **{k: fwd[f"sw_{k}"] for k in keys},
        **{f"sw_{k}": fwd[k] for k in keys},
    }
    cpu = torch.device("cpu")
    with torch.no_grad():
        a, _ = model(**model_inputs(fwd, cpu))
        b, _ = model(**model_inputs(rev, cpu))
    assert torch.allclose(a, b[:, list(REVERSE_PERM)], atol=1e-5)
