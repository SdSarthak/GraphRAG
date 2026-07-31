"""Boundary validation for configuration.

Every value here used to be accepted silently and then produced a crash or a
wrong result somewhere far away from the mistake.
"""

import json

import pytest

from graphrag.config import GraphRAGConfig


@pytest.mark.parametrize(
    "overrides",
    [
        {"rrf_k": -1},  # divided by zero inside rank fusion
        {"rrf_k": 0},
        {"retrieval_top_k": 0},  # returned nothing, with no explanation
        {"retrieval_top_k": -3},
        {"pagerank_alpha": 1.0},  # power iteration stops converging
        {"pagerank_alpha": 3.0},
        {"pagerank_alpha": 0.0},
        {"max_context_chars": 0},
        {"chunk_overlap": -5},
        {"graph_expansion_depth": -1},
        {"llm_max_tokens": 0},
        {"min_entity_length": 0},
        {"max_entities_per_chunk": 0},
        {"vector_weight": -1.0},
        {"storage_dir": "   "},
    ],
)
def test_invalid_values_are_rejected(overrides):
    with pytest.raises(ValueError):
        GraphRAGConfig(**overrides)


def test_all_zero_retrieval_weights_are_rejected():
    # Every signal disabled means retrieve() can only ever return [].
    with pytest.raises(ValueError):
        GraphRAGConfig(vector_weight=0.0, keyword_weight=0.0, graph_weight=0.0)


def test_a_single_enabled_signal_is_allowed():
    config = GraphRAGConfig(vector_weight=0.0, keyword_weight=0.0, graph_weight=1.0)
    assert config.graph_weight == 1.0


def test_bad_environment_value_names_the_variable(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_CHUNK_SIZE", "big")
    with pytest.raises(ValueError) as excinfo:
        GraphRAGConfig.from_env()
    assert "GRAPHRAG_CHUNK_SIZE" in str(excinfo.value)


def test_out_of_range_environment_value_is_rejected(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_RRF_K", "0")
    with pytest.raises(ValueError):
        GraphRAGConfig.from_env()


def test_whitespace_only_environment_value_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_LLM_EFFORT", "   ")
    assert GraphRAGConfig.from_env().llm_effort == "medium"


def test_from_dict_coerces_stringified_numbers():
    config = GraphRAGConfig.from_dict({"chunk_size": "200", "pagerank_alpha": "0.5"})
    assert config.chunk_size == 200
    assert config.pagerank_alpha == 0.5


def test_from_dict_rejects_garbage_instead_of_raising_typeerror():
    with pytest.raises(ValueError):
        GraphRAGConfig.from_dict({"chunk_size": "not-a-number"})


def test_from_dict_ignores_unknown_keys():
    config = GraphRAGConfig.from_dict({"chunk_size": 90, "made_up": 1})
    assert config.chunk_size == 90


def test_saved_config_round_trips():
    original = GraphRAGConfig(chunk_size=90, chunk_overlap=10, rrf_k=30)
    restored = GraphRAGConfig.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.to_dict() == original.to_dict()
