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
    # "flat": one 4-way head. "entail2": score entailment in each direction,
    # q = p(a |= b), q' = p(b |= a), and compose EQ = qq', FE = q(1-q'),
    # BE = (1-q)q', NEG = (1-q)(1-q'). Reuses an NLI checkpoint's entailment
    # output instead of re-initialising the head; exactly reversal-equivariant.
    head: str = "flat"
    # Decoder backbones read one prompt, not a tokenizer pair; `{a}` / `{b}`
    # are replaced by text1 / text2. None = native pair encoding.
    pair_template: Optional[str] = None
    # PEFT LoRA on the backbone, e.g. {"r": 16, "alpha": 32, "dropout": 0.05,
    # "target_modules": "all-linear"}. The classification head stays trainable.
    lora: Optional[dict] = None
    # 4-bit NF4 base weights (QLoRA); needs `lora` and bitsandbytes.
    load_in_4bit: bool = False
    # Dtype of the backbone weights: "float32" (fp16 AMP) or "bfloat16" (LLMs).
    torch_dtype: str = "float32"
    gradient_checkpointing: bool = False

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
        if self.head not in ("flat", "entail2"):
            raise ValueError(f"head must be 'flat' or 'entail2', got {self.head!r}")
        if self.head == "entail2" and self.symmetric:
            raise ValueError("head 'entail2' is already reversal-equivariant; drop symmetric")
        if self.load_in_4bit and not self.lora:
            raise ValueError("load_in_4bit needs a lora block")
        if self.torch_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"torch_dtype must be float32 or bfloat16, got {self.torch_dtype!r}")

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer or self.backbone

    @property
    def uses_swap(self) -> bool:
        """Whether batches must also carry the (text2, text1) encoding."""
        return self.symmetric or self.head == "entail2"
