from __future__ import annotations

import hashlib

import numpy as np

from orbweaver.config import settings


def embed_text(text: str, dim: int | None = None) -> list[float]:
    dim = dim or settings.embedding_dim
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        if settings.embedding_model.startswith("hash://"):
            raise ImportError("using hash embedder")
        model = SentenceTransformer(settings.embedding_model)
        vec = model.encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vec.tolist()]
    except Exception:
        out: list[float] = []
        seed = text.encode("utf-8")
        while len(out) < dim:
            seed = hashlib.blake2b(seed, digest_size=32).digest()
            for b in seed:
                out.append((b / 127.5) - 1.0)
                if len(out) >= dim:
                    break
        arr = np.array(out[:dim], dtype=float)
        n = float(np.linalg.norm(arr) or 1.0)
        return (arr / n).tolist()
