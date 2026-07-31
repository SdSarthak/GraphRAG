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
    llm_timeout: float = 120.0  # seconds before a request is abandoned
    llm_max_retries: int = 2  # transient 429/5xx retries inside the SDK
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
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.extractor not in {"rule", "llm"}:
            raise ValueError("extractor must be either 'rule' or 'llm'")
        if self.llm_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError(
                "llm_effort must be one of: low, medium, high, xhigh, max"
            )

        # Everything below used to be accepted unchecked, and the bad values
        # did not fail loudly: rrf_k=-1 divided by zero mid-query,
        # pagerank_alpha>1 made the power iteration diverge instead of rank,
        # and a non-positive top_k silently returned no results at all.
        for name in ("llm_max_tokens", "max_entities_per_chunk", "retrieval_top_k",
                     "max_context_chars"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.llm_timeout <= 0.0:
            raise ValueError("llm_timeout must be positive")
        if self.llm_max_retries < 0:
            raise ValueError("llm_max_retries must not be negative")
        if self.min_entity_length < 1:
            raise ValueError("min_entity_length must be at least 1")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        if self.graph_expansion_depth < 0:
            raise ValueError("graph_expansion_depth must not be negative")
        if not 0.0 < self.pagerank_alpha < 1.0:
            raise ValueError("pagerank_alpha must be strictly between 0 and 1")
        for name in ("vector_weight", "keyword_weight", "graph_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must not be negative")
        if max(self.vector_weight, self.keyword_weight, self.graph_weight) <= 0.0:
            raise ValueError(
                "at least one of vector_weight, keyword_weight or graph_weight "
                "must be greater than zero, otherwise retrieval returns nothing"
            )
        if not str(self.storage_dir).strip():
            raise ValueError("storage_dir must not be empty")

    # --- Factories ------------------------------------------------------
    @classmethod
    def from_env(
        cls, defaults: Optional[Dict[str, Any]] = None, **overrides: Any
    ) -> "GraphRAGConfig":
        """Build a config from environment variables, then apply overrides.

        Precedence, lowest first: dataclass defaults, ``defaults`` (typically
        the config stored alongside an existing index), environment
        variables, explicit ``overrides``.
        """
        _load_dotenv()
        values: Dict[str, Any] = {}
        if defaults:
            known = {field.name for field in fields(cls)}
            values.update(
                {
                    key: value
                    for key, value in defaults.items()
                    if key in known and value is not None
                }
            )
        for field in fields(cls):
            variable = ENV_PREFIX + field.name.upper()
            raw = os.environ.get(variable)
            if raw is None or raw.strip() == "":
                continue
            try:
                values[field.name] = _coerce(raw, field.type)
            except ValueError as exc:
                # Without this the user sees "invalid literal for int() with
                # base 10: 'big'" and no hint about which variable is wrong.
                raise ValueError(
                    f"{variable}={raw!r} is not a valid value for "
                    f"{field.name} ({exc})"
                ) from exc

        if "api_key" not in values:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                values["api_key"] = api_key

        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**cls._coerced(values))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("api_key", None)  # never serialise credentials
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphRAGConfig":
        """Rebuild a config from a saved (or hand-edited) ``config.json``."""
        if not isinstance(data, dict):
            raise ValueError("config data must be a JSON object")
        return cls(**cls._coerced(data))

    @classmethod
    def _coerced(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Keep known keys, converting strings to the declared field type.

        A saved or hand-edited config.json can hold "180" where 180 is meant;
        coercing here keeps validation type-correct instead of blowing up with
        a TypeError on the first comparison.
        """
        known = {field.name: field for field in fields(cls)}
        values: Dict[str, Any] = {}
        for key, value in data.items():
            field = known.get(key)
            if field is None:
                continue
            if isinstance(value, str) and "str" not in str(field.type):
                try:
                    value = _coerce(value, field.type)
                except ValueError as exc:
                    raise ValueError(f"invalid value for {key}: {exc}") from exc
            values[key] = value
        return values


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
