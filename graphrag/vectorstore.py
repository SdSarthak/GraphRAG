"""An in-memory vector store with numpy persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .embeddings import cosine_similarity
from .storage import atomic_write_binary


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
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            # A single embedding is a common caller mistake and used to blow up
            # later with an opaque IndexError on ``vectors.shape[1]``.
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-dimensional, got {vectors.ndim}")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors must have the same length")
        if vectors.size and vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected vectors of dimension {self.dim}, got {vectors.shape[1]}"
            )

        new_ids: List[str] = []
        new_rows: List[np.ndarray] = []
        pending: Dict[str, int] = {}
        for chunk_id, vector in zip(ids, vectors):
            position = self._positions.get(chunk_id)
            if position is not None:
                self.matrix[position] = vector
            elif chunk_id in pending:
                # A duplicate id inside one batch used to append a second row,
                # leaving len(store) wrong, the position map pointing at the
                # stale row and search returning the same chunk twice.
                new_rows[pending[chunk_id]] = np.asarray(vector, dtype=np.float32)
            else:
                pending[chunk_id] = len(new_ids)
                new_ids.append(chunk_id)
                new_rows.append(np.asarray(vector, dtype=np.float32))

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
        atomic_write_binary(
            path,
            lambda stream: np.savez_compressed(
                stream,
                ids=np.array(self.ids, dtype=object),
                matrix=self.matrix,
                dim=np.array([self.dim]),
            ),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VectorStore":
        path = Path(path)
        try:
            with np.load(path, allow_pickle=True) as data:
                missing = [key for key in ("ids", "matrix", "dim") if key not in data]
                if missing:
                    raise ValueError(
                        f"{path} is not a GraphRAG vector store "
                        f"(missing: {', '.join(missing)})"
                    )
                dim = int(data["dim"][0])
                ids = [str(value) for value in data["ids"].tolist()]
                matrix = np.asarray(data["matrix"], dtype=np.float32)
        except OSError as exc:
            raise OSError(f"could not read vector store {path}: {exc}") from exc

        store = cls(dim=dim)
        if matrix.size:
            # A truncated or hand-edited file would otherwise map scores onto
            # the wrong ids and return confidently wrong search results.
            if matrix.ndim != 2 or matrix.shape[1] != dim:
                raise ValueError(
                    f"{path} holds vectors of the wrong shape "
                    f"{matrix.shape} for dimension {dim}"
                )
            if matrix.shape[0] != len(ids):
                raise ValueError(
                    f"{path} is corrupt: {len(ids)} ids but "
                    f"{matrix.shape[0]} vectors"
                )
        elif ids:
            raise ValueError(f"{path} is corrupt: {len(ids)} ids but no vectors")

        duplicates = len(ids) - len(set(ids))
        if duplicates:
            raise ValueError(f"{path} is corrupt: {duplicates} duplicate chunk ids")

        store.ids = ids
        store.matrix = matrix if matrix.size else np.zeros((0, dim), dtype=np.float32)
        store._positions = {chunk_id: index for index, chunk_id in enumerate(ids)}
        return store
