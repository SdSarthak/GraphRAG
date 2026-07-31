"""Document containers, chunking and corpus loading."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

from .text import normalize

TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".rst", ".log")
_SKIP_DIRECTORIES = frozenset(
    {"__pycache__", "node_modules", "venv", "env", "site-packages"}
)

logger = logging.getLogger(__name__)


def _hash_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


@dataclass
class Document:
    """A source document before chunking."""

    text: str
    source: str = "inline"
    id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = normalize(self.text)
        if not self.id:
            self.id = _hash_id("doc", f"{self.source}:{self.text}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(
            text=data["text"],
            source=data.get("source", "inline"),
            id=data.get("id", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Chunk:
    """A retrievable slice of a document."""

    text: str
    document_id: str
    source: str = "inline"
    index: int = 0
    id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = normalize(self.text)
        if not self.id:
            self.id = _hash_id("chunk", f"{self.document_id}:{self.index}:{self.text}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "document_id": self.document_id,
            "source": self.source,
            "index": self.index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            text=data["text"],
            document_id=data["document_id"],
            source=data.get("source", "inline"),
            index=data.get("index", 0),
            id=data.get("id", ""),
            metadata=data.get("metadata", {}),
        )


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping word windows.

    Args:
        text: Raw text to split.
        chunk_size: Maximum number of words per chunk.
        chunk_overlap: Number of words shared between consecutive chunks.

    Returns:
        List of chunk strings. Empty input yields an empty list.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = normalize(text).split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = chunk_size - chunk_overlap
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_document(
    document: Document, chunk_size: int, chunk_overlap: int
) -> List[Chunk]:
    """Split a document into :class:`Chunk` objects."""
    return [
        Chunk(
            text=piece,
            document_id=document.id,
            source=document.source,
            index=index,
            metadata=dict(document.metadata),
        )
        for index, piece in enumerate(chunk_text(document.text, chunk_size, chunk_overlap))
    ]


def coerce_documents(
    documents: Iterable[Union[str, Document, Dict[str, Any]]],
    source: str = "inline",
) -> List[Document]:
    """Normalise mixed input (strings, dicts, Documents) into Documents."""
    result: List[Document] = []
    for item in documents:
        if isinstance(item, Document):
            result.append(item)
        elif isinstance(item, dict):
            result.append(Document.from_dict(item))
        elif isinstance(item, str):
            result.append(Document(text=item, source=source))
        else:
            raise TypeError(f"Unsupported document type: {type(item)!r}")
    return [doc for doc in result if doc.text]


def iter_text_files(
    path: Union[str, os.PathLike], suffixes: Sequence[str] = TEXT_SUFFIXES
) -> Iterator[Path]:
    """Yield text files under ``path`` (a file or a directory tree).

    Walks with ``os.walk``: ``Path.rglob`` follows directory symlinks, so a
    corpus containing a symlink cycle made indexing run forever.
    """
    root = Path(path)
    if root.is_file():
        yield root
        return
    if not root.exists():
        raise FileNotFoundError(f"No such file or directory: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a readable file or directory: {root}")

    wanted = {suffix.lower() for suffix in suffixes}
    for directory, subdirectories, names in os.walk(root, followlinks=False):
        # Version control and virtualenv directories are never corpus content
        # and are usually the bulk of the files under a project root.
        subdirectories[:] = sorted(
            name
            for name in subdirectories
            if not name.startswith(".") and name not in _SKIP_DIRECTORIES
        )
        for name in sorted(names):
            candidate = Path(directory) / name
            if candidate.suffix.lower() in wanted:
                yield candidate


def load_documents(
    path: Union[str, os.PathLike],
    suffixes: Sequence[str] = TEXT_SUFFIXES,
    encoding: str = "utf-8",
    max_files: Optional[int] = None,
) -> List[Document]:
    """Load documents from a file or directory.

    Args:
        path: File or directory to read.
        suffixes: File extensions treated as text.
        encoding: Text encoding, decoded with ``errors="replace"``.
        max_files: Optional cap on the number of files read.

    Returns:
        A list of non-empty documents.
    """
    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be positive")

    documents: List[Document] = []
    skipped = 0
    for file_path in iter_text_files(path, suffixes):
        if max_files is not None and len(documents) >= max_files:
            logger.info("Stopped after %d files (max_files)", max_files)
            break
        try:
            raw = file_path.read_text(encoding=encoding, errors="replace")
        except OSError as exc:
            # A single unreadable file must not abort the corpus, but
            # swallowing it without a word made permission problems invisible.
            skipped += 1
            logger.warning("Skipping %s: %s", file_path, exc)
            continue
        if not raw.strip():
            continue
        documents.append(Document(text=raw, source=str(file_path)))
    if skipped:
        logger.warning("Skipped %d unreadable file(s) under %s", skipped, path)
    return documents
