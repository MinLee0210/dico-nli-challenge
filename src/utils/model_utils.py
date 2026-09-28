import random
import string
from datetime import datetime

import torch


def get_run_name(
    model_name: str,
    dset_name: str,
    lr: float,
    batch_size: int,
    experiment_type: str = "",
    random_suffix: bool = True,
    random_suffix_len: int = 6,
) -> str:
    """Generate a unique, sortable run name with a timestamp (YYYY-MM-DD)."""
    now = datetime.now().strftime("%Y-%m-%d")
    rand_suffix = (
        "".join(
            random.Random().choices(
                string.ascii_letters + string.digits, k=random_suffix_len
            )
        )
        if random_suffix
        else ""
    )
    return f"{model_name}_{dset_name}_{experiment_type}_lr{lr}_bs{batch_size}_{now}_{rand_suffix}"


def count_parameters(model: torch.nn.Module, verbose: bool = True) -> dict:
    """Count parameters in a PyTorch model.

    from src.utils.model_utils import count_parameters
    count_parameters(model)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = n_all - n_trainable
    if verbose:
        print(
            "Parameter Count: all {:,d}; trainable {:,d}; frozen {:,d}".format(
                n_all, n_trainable, n_frozen
            )
        )
    return {
        "n_all": n_all,
        "n_trainable": n_trainable,
        "n_frozen": n_frozen,
    }


def save_tiny_backbone(path, words=(), num_labels: int = 4) -> str:
    """Save a randomly initialised 2-layer BERT + WordPiece tokenizer to `path`.

    Offline stand-in for a real backbone (tests, smoke runs): anything that
    takes a model id accepts this directory. Returns `str(path)`.
    """
    from pathlib import Path

    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        BertTokenizerFast,
    )

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = specials + sorted({w.lower() for w in words} - set(specials))
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    BertTokenizerFast(vocab_file=str(path / "vocab.txt")).save_pretrained(path)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=num_labels,
    )
    BertForSequenceClassification(config).save_pretrained(path)
    return str(path)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
