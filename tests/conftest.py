import pytest

from src.config import ModelConfig
from src.data import SYNTHETIC_WORDS
from src.modules.model import PairClassifier, build_hf_model, build_tokenizer
from src.utils.model_utils import save_tiny_backbone


@pytest.fixture(scope="session")
def tiny_backbone(tmp_path_factory) -> str:
    """Path to a saved tiny random BERT + tokenizer (offline)."""
    return save_tiny_backbone(tmp_path_factory.mktemp("tiny-bert"), SYNTHETIC_WORDS)


@pytest.fixture
def tiny_cfg(tiny_backbone) -> ModelConfig:
    return ModelConfig(backbone=tiny_backbone, max_length=32)


@pytest.fixture
def tiny_model(tiny_cfg) -> PairClassifier:
    return PairClassifier(tiny_cfg, build_hf_model(tiny_cfg)).eval()


@pytest.fixture
def tiny_tokenizer(tiny_cfg):
    return build_tokenizer(tiny_cfg)
