"""File knowledge base (RAG over a markdown folder).

On every server start (`build_index`) the folder is scanned and compared
against a persisted cache keyed by file-hash in `cache_dir`:
  - new / changed files are chunked + embedded,
  - removed files are dropped,
  - the vectors (numpy) + chunk metadata (JSON) are written to disk,
so restarting is cheap and answers are always grounded in the current files.

At conversation time `retrieve(query)` embeds the query once and returns the
top_k most similar chunks (cosine similarity, L2-normalized vectors).
"""

import hashlib
import json
import threading
from pathlib import Path

import numpy as np
from loguru import logger

from .chunker import chunk_markdown, load_folder_files
from .embedder import embed_texts

CACHE_META = "index.json"
CACHE_VECTORS = "vectors.npy"


def _file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class KnowledgeBase:
    """Thread-safe index over markdown files with cosine retrieval."""

    def __init__(self, config) -> None:
        self.config = config
        self._chunks: list = []  # list of dicts {file, heading, text}
        self._vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._lock = threading.Lock()
        self._manifest: dict = {}  # {relpath: sha256}
        self._ready = False

    # ---- indexing -------------------------------------------------------

    def build_index(self) -> None:
        cfg = self.config
        if not getattr(cfg, "enabled", True):
            return
        cache_dir = Path(cfg.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = cache_dir / CACHE_META
        vec_path = cache_dir / CACHE_VECTORS

        files = load_folder_files(cfg.folder_path, cfg.extension)
        new_manifest = {rel: _file_hash(content) for rel, content in files}
        chunks = self._load_chunks(meta_path)
        vectors = self._load_vectors(vec_path)
        chunk_manifest = self._load_manifest_from_chunks(chunks)

        changed = [
            rel for rel, h in new_manifest.items() if chunk_manifest.get(rel) != h
        ]
        removed = [rel for rel in chunk_manifest if rel not in new_manifest]

        # rebuild everything if the cache is structurally inconsistent
        if chunks and vectors.shape[0] == 0:
            changed = list(new_manifest)

        if changed or removed or not chunks:
            logger.info(
                f"knowledge: indexing {cfg.folder_path} "
                f"({len(new_manifest)} files, "
                f"{len(changed)} changed, {len(removed)} removed)"
            )
            chunks = [
                c
                for c in chunks
                if c["file"] not in removed and c["file"] not in changed
            ]
            for rel, content in files:
                new_chunks = chunk_markdown(
                    content,
                    rel,
                    chunk_size=cfg.chunk_size,
                    chunk_overlap=cfg.chunk_overlap,
                )
                chunk_dicts = [
                    {
                        "file": c.file,
                        "heading": c.heading,
                        "text": c.to_display_text(cfg.include_sources),
                        "hash": new_manifest.get(rel, ""),
                    }
                    for c in new_chunks
                ]
                chunks.extend(chunk_dicts)

            all_texts = [c["text"] for c in chunks] or [""]
            vectors = embed_texts(
                all_texts,
                cfg.embedding_model,
                cfg.embedding_device,
            )
            self._manifest = new_manifest
            self._chunks = chunks
            self._vectors = vectors
            self._save_cache(meta_path, vec_path)
        else:
            self._manifest = new_manifest
            self._chunks = chunks
            self._vectors = vectors

        # Warm the embedding model now so the first user query isn't delayed
        # by model loading (cold load ~10s, warm embed ~0.1s).
        try:
            embed_texts([""], cfg.embedding_model, cfg.embedding_device)
        except Exception as e:
            logger.warning(f"knowledge: embedding model warm-up failed: {e}")

        self._ready = True
        logger.info(
            f"knowledge: ready - {len(self._chunks)} chunks from "
            f"{len(new_manifest)} file(s)"
        )

    # ---- retrieval ------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int = None, min_score: float = None
    ) -> list[dict]:
        """Return the top_k most relevant chunk dicts for `query`."""
        cfg = self.config
        top_k = top_k if top_k is not None else cfg.top_k
        min_score = min_score if min_score is not None else cfg.min_score
        if not self._ready or not query.strip():
            return []
        with self._lock:
            if self._vectors.shape[0] == 0 or self._vectors.shape[1] == 0:
                return []
            q = embed_texts([query], cfg.embedding_model, cfg.embedding_device)
            if q.shape[0] != 1:
                return []
            scores = self._vectors @ q[0]  # (N,) cosine since both normalized
            order = np.argsort(scores)[::-1]
            results = []
            for i in order:
                if scores[i] < min_score:
                    break
                c = self._chunks[i]
                results.append(
                    {
                        "file": c["file"],
                        "heading": c.get("heading", ""),
                        "text": c["text"],
                        "score": float(scores[i]),
                    }
                )
                if len(results) >= top_k:
                    break
            return results

    def is_ready(self) -> bool:
        return self._ready

    # ---- persistence helpers ---------------------------------------------

    def _load_manifest(self, path: Path) -> dict:
        try:
            if path.is_file():
                return json.loads(path.read_text("utf-8"))
        except Exception as e:
            logger.warning(f"knowledge: could not read index meta: {e}")
        return {}

    def _load_chunks(self, path: Path) -> list:
        try:
            data = self._load_manifest(path)
            return data.get("chunks", [])
        except Exception:
            return []

    def _load_vectors(self, path: Path) -> np.ndarray:
        try:
            if path.is_file():
                return np.load(path).astype(np.float32)
        except Exception as e:
            logger.warning(f"knowledge: could not load vectors: {e}")
        return np.zeros((0, 0), dtype=np.float32)

    def _load_manifest_from_chunks(self, chunks: list) -> dict:
        """Derive the file->hash manifest from chunks (best effort)."""
        return {c["file"]: c.get("hash", "") for c in chunks}

    def _save_cache(self, meta_path: Path, vec_path: Path) -> None:
        payload = {
            "manifest": self._manifest,
            "chunks": self._chunks,
        }
        meta_path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
        np.save(vec_path, self._vectors)
