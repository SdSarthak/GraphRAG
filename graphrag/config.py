"""Configuration for the GraphRAG system."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

ENV_PREFIX = "GRAPHRAG_"


def _load_dotenv() -> None:
    """Load a local .env file when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional dependency
        return
    load_dotenv(override=False)


@dataclass
class GraphRAGConfig:
    """All tunables for indexing, retrieval and generation.

    Every field can be overridden with an environment variable named
    ``GRAPHRAG_<FIELD_NAME_UPPERCASE>`` (see ``.env.example``).
    """

    # --- Generation -----------------------------------------------------
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = 8192
    llm_effort: str = "medium"  # low | medium | high | xhigh | max
    api_key: Optional[str] = None

    # --- Indexing -------------------------------------------------------
    embedding_model: str = "hashing-512"
    embedding_dim: int = 512
    chunk_size: int = 180  # words per chunk
    chunk_overlap: int = 40  # words shared between neighbouring chunks
    extractor: str = "rule"  # rule | llm
    max_entities_per_chunk: int = 12
    min_entity_length: int = 3

    # --- Retrieval ------------------------------------------------------
    retrieval_top_k: int = 5
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    graph_weight: float = 1.0
    rrf_k: int = 60
    pagerank_alpha: float = 0.85
    graph_expansion_depth: int = 2
    max_context_chars: int = 6000

    # --- Storage --------------------------------------------------------
    storage_dir: str = "storage"

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.extractor not in {"rule", "llm"}:
            raise ValueError("extractor must be either 'rule' or 'llm'")
        if self.llm_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError(
                "llm_effort must be one of: low, medium, high, xhigh, max"
            )

    # --- Factories ------------------------------------------------------
    @classmethod
    def from_env(cls, **overrides: Any) -> "GraphRAGConfig":
        """Build a config from environment variables, then apply overrides."""
        _load_dotenv()
        values: Dict[str, Any] = {}
        for field in fields(cls):
            raw = os.environ.get(ENV_PREFIX + field.name.upper())
            if raw is None or raw == "":
                continue
            values[field.name] = _coerce(raw, field.type)

        if "api_key" not in values:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                values["api_key"] = api_key

        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("api_key", None)  # never serialise credentials
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphRAGConfig":
        known = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def _coerce(raw: str, annotation: Any) -> Any:
    """Convert an environment string into the dataclass field type."""
    text = str(annotation)
    if "int" in text and "Optional" not in text:
        return int(raw)
    if "float" in text:
        return float(raw)
    if "bool" in text:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return raw
