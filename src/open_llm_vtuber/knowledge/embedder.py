"""Embedding model wrapper for the file knowledge base.

Uses `transformers` (already a project dependency) to load a
sentence-embedding model and produce L2-normalized embeddings. The model is
loaded lazily and cached (module-level singleton) so the VTuber server and all
connected clients share a single instance.
"""

import threading
from pathlib import Path

import numpy as np
from loguru import logger

_model = None
_tokenizer = None
_load_lock = threading.Lock()

# HF hub caches use symlinks which fail on some Windows setups (WinError
# 1314). We keep a plain-copy local mirror under models/<model-name> and
# prefer it when present so no network/symlink is ever needed at runtime.
_MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models"


def _resolve_model_path(model_name: str) -> str:
    local = _MODEL_DIR / model_name.split("/")[-1]
    if (local / "config.json").is_file() and (local / "model.safetensors").is_file():
        return str(local)
    return model_name


def _load(model_name: str, device: str):
    global _model, _tokenizer
    with _load_lock:
        if _model is not None:
            return _model, _tokenizer
        from transformers import AutoModel, AutoTokenizer

        resolved = _resolve_model_path(model_name)
        logger.info(
            f"knowledge: loading embedding model '{resolved}' on {device} ..."
        )
        _tokenizer = AutoTokenizer.from_pretrained(resolved)
        _model = AutoModel.from_pretrained(resolved)
        _model.eval()
        _model.to(device)
        logger.info("knowledge: embedding model ready")
        return _model, _tokenizer


def embed_texts(texts, model_name="BAAI/bge-m3", device="cpu") -> np.ndarray:
    """Embed a list of strings into an (N, D) float32 array, L2-normalized."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    import torch

    model, tokenizer = _load(model_name, device)
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    # mean pooling over non-padding tokens
    attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
    pooled = (outputs.last_hidden_state * attention_mask).sum(
        dim=1
    ) / attention_mask.sum(dim=1)
    vectors = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return vectors.cpu().numpy().astype(np.float32)


def free_model() -> None:
    """Release the loaded embedding model (mainly for tests)."""
    global _model, _tokenizer
    _model = None
    _tokenizer = None
