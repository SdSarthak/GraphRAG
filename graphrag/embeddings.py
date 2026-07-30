"""Text embeddings.

The default embedder is a deterministic hashing vectoriser: it needs no model
download, no API key and no warm-up, which keeps the whole pipeline runnable
offline. It uses sub-linear term frequencies over word unigrams and bigrams,
hashed into a fixed-width vector with signed buckets to limit collisions, then
L2-normalised so dot products are cosine similarities.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, List, Sequence

import numpy as np

from .text import tokenize


class HashingEmbedder:
    """Deterministic hashing vectoriser (the ``hashing-*`` embedding models)."""

    def __init__(self, dim: int = 512, ngram_range: Sequence[int] = (1, 2)) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.min_n, self.max_n = int(ngram_range[0]), int(ngram_range[-1])
        if self.min_n < 1 or self.max_n < self.min_n:
            raise ValueError("invalid ngram_range")

    @property
    def name(self) -> str:
        return f"hashing-{self.dim}"

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Embed a batch of texts into an ``(n, dim)`` float32 matrix."""
        rows = [self.embed_one(text) for text in texts]
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(rows)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text into a ``(dim,)`` L2-normalised vector."""
        vector = np.zeros(self.dim, dtype=np.float32)
        counts: dict = {}
        tokens = tokenize(text)
        for gram in self._ngrams(tokens):
            counts[gram] = counts.get(gram, 0) + 1

        for gram, count in counts.items():
            index, sign = self._bucket(gram)
            vector[index] += sign * (1.0 + math.log(count))

        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector

    def _ngrams(self, tokens: Sequence[str]) -> List[str]:
        grams: List[str] = []
        for size in range(self.min_n, self.max_n + 1):
            if len(tokens) < size:
                continue
            for start in range(len(tokens) - size + 1):
                grams.append(" ".join(tokens[start : start + size]))
        return grams

    def _bucket(self, gram: str) -> tuple:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of ``matrix`` and ``vector``."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm == 0.0:
        return np.zeros((matrix.shape[0],), dtype=np.float32)
    denominator = np.where(matrix_norms == 0.0, 1.0, matrix_norms) * vector_norm
    return (matrix @ vector) / denominator


def build_embedder(config) -> HashingEmbedder:
    """Create the embedder named by ``config.embedding_model``."""
    model = str(config.embedding_model)
    if model.startswith("hashing"):
        return HashingEmbedder(dim=config.embedding_dim)
    raise ValueError(
        f"Unknown embedding model {model!r}. Supported: 'hashing-<dim>'."
    )
