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


def entailment_index(hf_cfg) -> Optional[int]:
    """Index of the `entailment` output of an NLI checkpoint, else None."""
    for i, name in (getattr(hf_cfg, "id2label", None) or {}).items():
        if str(name).lower().startswith("entail"):
            return int(i)
    return None


def _head_labels(cfg: ModelConfig, hf_cfg) -> dict:
    if cfg.head == "flat":
        return {"num_labels": cfg.num_labels, "id2label": ID2LABEL, "label2id": LABEL2ID}
    if entailment_index(hf_cfg) is not None:
        return {}  # keep the NLI head: its entailment output is q(a |= b)
    names = {0: "not_entailment", 1: "entailment"}
    return {"num_labels": 2, "id2label": names, "label2id": {v: k for k, v in names.items()}}


def build_hf_model(cfg: ModelConfig) -> nn.Module:
    from transformers import AutoConfig, AutoModelForSequenceClassification

    hf_cfg = AutoConfig.from_pretrained(cfg.backbone)
    for key, value in _head_labels(cfg, hf_cfg).items():
        setattr(hf_cfg, key, value)
    if cfg.classifier_dropout is not None:
        for attr in _DROPOUT_ATTRS:
            if hasattr(hf_cfg, attr):
                setattr(hf_cfg, attr, cfg.classifier_dropout)
    # Composite (VLM) configs are read through their text sub-config.
    for sub in (hf_cfg, hf_cfg.get_text_config()):
        if getattr(sub, "pad_token_id", None) is None:
            sub.pad_token_id = build_tokenizer(cfg).pad_token_id

    kwargs = {}
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    # A head of a different shape (3-way NLI -> 4 labels) is re-initialised.
    # float32 by default: transformers 5 otherwise loads the checkpoint dtype,
    # which breaks the fp16 GradScaler.
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.backbone,
        config=hf_cfg,
        ignore_mismatched_sizes=True,
        dtype=getattr(torch, cfg.torch_dtype),
        **kwargs,
    )
    if cfg.lora:
        model = _apply_lora(model, cfg)
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def _apply_lora(model: nn.Module, cfg: ModelConfig) -> nn.Module:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    lora = dict(cfg.lora)
    config = LoraConfig(
        r=lora.pop("r", 16),
        lora_alpha=lora.pop("alpha", 32),
        lora_dropout=lora.pop("dropout", 0.05),
        target_modules=lora.pop("target_modules", "all-linear"),
        task_type="SEQ_CLS",  # keeps the classification head fully trainable
        **lora,
    )
    return get_peft_model(model, config)


def place(model: nn.Module, device: torch.device) -> nn.Module:
    """Move to `device`; 4-bit models are already placed at load time."""
    return model if model.cfg.load_in_4bit else model.to(device)


class PairClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig, hf_model: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.hf = hf_model if hf_model is not None else build_hf_model(cfg)
        if getattr(self.hf.config, "pad_token_id", None) is None:
            self.hf.config.pad_token_id = self.hf.config.eos_token_id
        if cfg.head == "entail2":
            idx = entailment_index(self.hf.config)
            self.entail_idx = 1 if idx is None else idx

    def _entail_log_probs(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(log q, log (1 - q)) with q = p(entailment) from the NLI logits."""
        log_p = torch.log_softmax(logits.float(), dim=-1)
        log_q = log_p[:, self.entail_idx]
        others = torch.cat(
            [log_p[:, : self.entail_idx], log_p[:, self.entail_idx + 1 :]], dim=-1
        )
        return log_q, torch.logsumexp(others, dim=-1)

    def trainable_state_dict(self) -> dict:
        """What a checkpoint must store: everything, or only the trainable
        (LoRA + head) tensors when the backbone is frozen under LoRA."""
        state = self.state_dict()
        if not self.cfg.lora:
            return state
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in state.items() if k in keep}

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
        if not self.cfg.uses_swap:
            return logits, feature
        if sw_input_ids is None:
            raise ValueError("this model needs the swapped (text2, text1) encoding")
        sw_kwargs = {"input_ids": sw_input_ids, "attention_mask": sw_attention_mask}
        if sw_token_type_ids is not None:
            sw_kwargs["token_type_ids"] = sw_token_type_ids
        sw_logits = self.hf(**sw_kwargs).logits
        if self.cfg.symmetric:
            return 0.5 * (logits + sw_logits[..., list(REVERSE_PERM)]), feature
        # entail2: normalised log-probs over (EQ, FE, BE, NEG) in LABELS order.
        fwd, not_fwd = self._entail_log_probs(logits)
        bwd, not_bwd = self._entail_log_probs(sw_logits)
        composed = torch.stack(
            [fwd + bwd, fwd + not_bwd, not_fwd + bwd, not_fwd + not_bwd], dim=-1
        )
        return composed, feature


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
