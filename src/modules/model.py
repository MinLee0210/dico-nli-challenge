"""DiCo-NLI pair classifier: a Hugging Face cross-encoder with a 4-way head.

    (input_ids, attention_mask[, token_type_ids]) (B, T)
      -> AutoModelForSequenceClassification
      -> logits (B, 4), feature = last-layer first-token state (B, H)

`forward` returns `(logits, feature)` — the same contract as the template — so
the training loop, evaluator, and tests do not depend on the backbone family.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.labels import ID2LABEL, LABEL2ID, REVERSE_PERM

# Config attributes that control head dropout across backbone families
# (BERT/RoBERTa/XLM-R: classifier_dropout; DeBERTa-v2/v3: cls_dropout;
# ALBERT: classifier_dropout_prob).
_DROPOUT_ATTRS = ("classifier_dropout", "cls_dropout", "classifier_dropout_prob")


def build_tokenizer(cfg: ModelConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:  # decoder-only backbones
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.fix_pair_template:
        _fix_pair_template(tokenizer, cfg.backbone)
    return tokenizer


def _fix_pair_template(tokenizer, backbone: str) -> None:
    """Rebuild `[CLS] A [SEP] B [SEP]` from the model config's special ids.

    Some checkpoints ship a post-processor with the wrong specials (e.g.
    HiTZ/JaunBERT encodes a pair as `<unk> A<s><unk> B<s>`).
    """
    from tokenizers.processors import TemplateProcessing
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(backbone)
    cls_id = getattr(hf_cfg, "cls_token_id", None) or hf_cfg.bos_token_id
    sep_id = getattr(hf_cfg, "sep_token_id", None) or hf_cfg.eos_token_id
    cls = tokenizer.convert_ids_to_tokens(cls_id)
    sep = tokenizer.convert_ids_to_tokens(sep_id)
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single=f"{cls} $A {sep}",
        pair=f"{cls} $A {sep} $B:1 {sep}:1",
        special_tokens=[(cls, cls_id), (sep, sep_id)],
    )


def build_hf_model(cfg: ModelConfig) -> nn.Module:
    from transformers import AutoConfig, AutoModelForSequenceClassification

    hf_cfg = AutoConfig.from_pretrained(
        cfg.backbone,
        num_labels=cfg.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if cfg.classifier_dropout is not None:
        for attr in _DROPOUT_ATTRS:
            if hasattr(hf_cfg, attr):
                setattr(hf_cfg, attr, cfg.classifier_dropout)
    # Checkpoints fine-tuned on 3-way NLI ship a head of a different shape;
    # re-initialise it for our 4 labels. Load fp32 master weights: transformers 5
    # defaults to the checkpoint dtype, which breaks the fp16 GradScaler.
    return AutoModelForSequenceClassification.from_pretrained(
        cfg.backbone,
        config=hf_cfg,
        ignore_mismatched_sizes=True,
        dtype=torch.float32,
    )


class PairClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig, hf_model: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.hf = hf_model if hf_model is not None else build_hf_model(cfg)
        if getattr(self.hf.config, "pad_token_id", None) is None:
            self.hf.config.pad_token_id = self.hf.config.eos_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        sw_input_ids: Optional[torch.Tensor] = None,
        sw_attention_mask: Optional[torch.Tensor] = None,
        sw_token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits (B, num_labels), feature (B, hidden)).

        With `cfg.symmetric` and the swapped encoding given, logits are
        0.5 * (f(a, b) + Rev f(b, a)): exactly reversal-equivariant.
        """
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.hf(**kwargs, output_hidden_states=True)
        feature = out.hidden_states[-1][:, 0]
        logits = out.logits
        if self.cfg.symmetric and sw_input_ids is not None:
            sw_kwargs = {"input_ids": sw_input_ids, "attention_mask": sw_attention_mask}
            if sw_token_type_ids is not None:
                sw_kwargs["token_type_ids"] = sw_token_type_ids
            sw_logits = self.hf(**sw_kwargs).logits
            logits = 0.5 * (logits + sw_logits[..., list(REVERSE_PERM)])
        return logits, feature


MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "sw_input_ids",
    "sw_attention_mask",
    "sw_token_type_ids",
)


def model_inputs(batch: dict, device: torch.device) -> dict:
    """The tensors `PairClassifier.forward` consumes, moved to `device`."""
    return {k: batch[k].to(device) for k in MODEL_INPUT_KEYS if k in batch}
