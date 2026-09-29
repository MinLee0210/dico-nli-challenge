"""Model architecture for the DiCo-NLI pair classifier.

This is the ARCHITECTURE-ONLY config — which pretrained backbone, how inputs
are truncated, head dropout. Training-loop hyperparameters (epochs, lr, data
paths, loss weights, decoding) live in `src.pipelines.config.TrainingConfig`.
The split lets a checkpoint rebuild its model without the training YAML.
"""

from dataclasses import dataclass
from typing import Optional

from src.labels import NUM_LABELS


@dataclass
class ModelConfig:
    """A Hugging Face cross-encoder with a 4-way classification head.

        (text1, text2) -> tokenizer pair encoding -> backbone -> [CLS] feature
                       -> classification head -> logits over LABELS

    Any `AutoModelForSequenceClassification`-compatible id or local path works
    (DeBERTa-v3, mDeBERTa, XLM-R, BERnaT, ...).
    """

    backbone: str = "microsoft/mdeberta-v3-base"
    # Tokenizer id/path; defaults to `backbone`.
    tokenizer: Optional[str] = None
    num_labels: int = NUM_LABELS
    # Phrases average ~6 words per pair; 128 subword tokens is ample.
    max_length: int = 128
    # Overrides the backbone's classifier/pooler dropout when set.
    classifier_dropout: Optional[float] = None
    # Rebuild the tokenizer's pair template from the model config's CLS/SEP
    # ids, for checkpoints whose post-processor is broken (HiTZ/JaunBERT).
    fix_pair_template: bool = False
    # Reversal-equivariant model: score (a, b) and (b, a) with the shared
    # encoder and return 0.5 * (f(a, b) + Rev f(b, a)), so f(b, a) = Rev f(a, b)
    # holds exactly. Costs 2x compute; needs `with_swap` batches.
    symmetric: bool = False

    def __post_init__(self) -> None:
        if not self.backbone:
            raise ValueError("backbone must be a model id or path")
        if self.num_labels != NUM_LABELS:
            raise ValueError(f"num_labels must be {NUM_LABELS}, got {self.num_labels}")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be > 0, got {self.max_length}")
        if self.classifier_dropout is not None and not (
            0.0 <= self.classifier_dropout < 1.0
        ):
            raise ValueError(
                f"classifier_dropout must be in [0, 1), got {self.classifier_dropout}"
            )

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer or self.backbone
