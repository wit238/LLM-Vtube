"""Embedding model wrapper for the file knowledge base.

Uses `transformers` (already a project dependency) to load a
sentence-embedding model and produce L2-normalized embeddings. The model is
loaded lazily and cached (module-level singleton) so the VTuber server and all
connected clients share a single instance.
"""

import threading

import numpy as np
from loguru import logger

_model = None
_tokenizer = None
_load_lock = threading.Lock()


def _load(model_name: str, device: str):
    global _model, _tokenizer
    with _load_lock:
        if _model is not None:
            return _model, _tokenizer
        from transformers import AutoModel, AutoTokenizer

        logger.info(
            f"knowledge: loading embedding model '{model_name}' on {device} ..."
        )
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name)
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
