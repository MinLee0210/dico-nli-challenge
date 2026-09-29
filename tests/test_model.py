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


def _swap(batch):
    keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in batch]
    return {**{k: batch[f"sw_{k}"] for k in keys}, **{f"sw_{k}": batch[k] for k in keys}}


def test_entail2_head_is_normalised_and_reversal_equivariant(tiny_cfg, tiny_tokenizer):
    from dataclasses import replace

    from src.labels import REVERSE_PERM
    from src.modules.model import PairClassifier, build_hf_model

    cfg = replace(tiny_cfg, head="entail2")
    hf = build_hf_model(cfg)
    assert hf.config.num_labels == 2  # no NLI head to reuse -> binary head
    model = PairClassifier(cfg, hf).eval()
    dataset = DicoDataset(make_synthetic_dico(n_pairs=6, seed=0))
    fwd = PairCollator(dataset, tiny_tokenizer, 32, with_swap=True)(list(range(len(dataset))))
    cpu = torch.device("cpu")
    with torch.no_grad():
        a, _ = model(**model_inputs(fwd, cpu))
        b, _ = model(**model_inputs(_swap(fwd), cpu))
    assert torch.allclose(a.exp().sum(-1), torch.ones(len(dataset)), atol=1e-5)
    assert torch.allclose(a, b[:, list(REVERSE_PERM)], atol=1e-5)


def test_entailment_index_reads_nli_label_names():
    from types import SimpleNamespace

    from src.modules.model import entailment_index

    assert entailment_index(SimpleNamespace(id2label={0: "contradiction", 1: "neutral", 2: "entailment"})) == 2
    assert entailment_index(SimpleNamespace(id2label={0: "ENTAILMENT", 1: "neutral"})) == 0
    assert entailment_index(SimpleNamespace(id2label={0: "LABEL_0", 1: "LABEL_1"})) is None


def test_config_rejects_bad_head_combinations():
    with pytest.raises(ValueError):
        ModelConfig(backbone="x", head="nope")
    with pytest.raises(ValueError):
        ModelConfig(backbone="x", head="entail2", symmetric=True)
    with pytest.raises(ValueError):
        ModelConfig(backbone="x", load_in_4bit=True)
    assert ModelConfig(backbone="x", head="entail2").uses_swap


def test_pair_template_builds_one_prompt_per_pair(tiny_tokenizer):
    dataset = DicoDataset(make_synthetic_dico(n_pairs=3, seed=0))
    ex = dataset.examples[0]
    batch = PairCollator(dataset, tiny_tokenizer, 32, pair_template="{a} ship {b}")([0])
    expected = tiny_tokenizer(f"{ex.text1} ship {ex.text2}")["input_ids"]
    assert batch["input_ids"][0].tolist() == expected
    assert "token_type_ids" not in batch or batch["token_type_ids"].sum() == 0


def test_lora_checkpoint_keeps_only_trainable_tensors(tiny_cfg):
    pytest.importorskip("peft")
    from dataclasses import replace

    from src.modules.model import PairClassifier, build_hf_model

    cfg = replace(tiny_cfg, lora={"r": 2, "alpha": 4, "target_modules": ["query", "value"]})
    model = PairClassifier(cfg, build_hf_model(cfg))
    state = model.trainable_state_dict()
    assert state and all("lora_" in k or "classifier" in k for k in state)
    assert len(state) < len(model.state_dict())
