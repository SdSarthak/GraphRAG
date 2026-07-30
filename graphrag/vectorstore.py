"""An in-memory vector store with numpy persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .embeddings import cosine_similarity


class VectorStore:
    """Dense vector index over chunk ids.

    Small enough to stay exact (brute-force cosine) which is the right default
    for corpora that fit in memory; the interface leaves room for an ANN
    backend later.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.ids: List[str] = []
        self._positions: Dict[str, int] = {}
        self.matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._positions

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Add or replace vectors for the given ids."""
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors must have the same length")
        if vectors.size and vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected vectors of dimension {self.dim}, got {vectors.shape[1]}"
            )

        new_ids: List[str] = []
        new_rows: List[np.ndarray] = []
        for chunk_id, vector in zip(ids, vectors):
            position = self._positions.get(chunk_id)
            if position is None:
                new_ids.append(chunk_id)
                new_rows.append(np.asarray(vector, dtype=np.float32))
            else:
                self.matrix[position] = vector

        if new_ids:
            block = np.vstack(new_rows).astype(np.float32)
            self.matrix = (
                block if self.matrix.size == 0 else np.vstack([self.matrix, block])
            )
            for offset, chunk_id in enumerate(new_ids):
                self._positions[chunk_id] = len(self.ids) + offset
            self.ids.extend(new_ids)

    def search(
        self, query: np.ndarray, top_k: int = 5, min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Return the ``top_k`` ``(chunk_id, cosine_score)`` pairs."""
        if not self.ids or top_k <= 0:
            return []
        scores = cosine_similarity(self.matrix, np.asarray(query, dtype=np.float32))
        count = min(top_k, scores.shape[0])
        best = np.argpartition(-scores, count - 1)[:count]
        ordered = best[np.argsort(-scores[best])]
        return [
            (self.ids[index], float(scores[index]))
            for index in ordered
            if float(scores[index]) > min_score
        ]

    def get(self, chunk_id: str) -> Optional[np.ndarray]:
        position = self._positions.get(chunk_id)
        return None if position is None else self.matrix[position]

    # -- persistence -----------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if path.suffix != ".npz":  # numpy appends the suffix silently
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ids=np.array(self.ids, dtype=object),
            matrix=self.matrix,
            dim=np.array([self.dim]),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VectorStore":
        with np.load(Path(path), allow_pickle=True) as data:
            dim = int(data["dim"][0])
            store = cls(dim=dim)
            ids = [str(value) for value in data["ids"].tolist()]
            matrix = np.asarray(data["matrix"], dtype=np.float32)
        store.ids = ids
        store.matrix = matrix if matrix.size else np.zeros((0, dim), dtype=np.float32)
        store._positions = {chunk_id: index for index, chunk_id in enumerate(ids)}
        return store
