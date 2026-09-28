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
