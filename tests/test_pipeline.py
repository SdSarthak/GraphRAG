import json

import pytest

from graphrag import GraphRAG, GraphRAGConfig
from graphrag.cli import main as cli_main
from graphrag.llm import ExtractiveLLM, build_answer_prompt, generate_answer

DOCUMENTS = [
    "Machine learning is a subset of artificial intelligence that lets "
    "computers learn patterns from data.",
    "Deep learning is a branch of machine learning that uses neural networks "
    "with multiple layers.",
    "Computer vision enables machines to interpret images. Convolutional "
    "neural networks power computer vision systems.",
    "Photosynthesis converts sunlight into chemical energy inside plant cells.",
]


@pytest.fixture()
def system() -> GraphRAG:
    config = GraphRAGConfig(embedding_dim=256, retrieval_top_k=3, chunk_size=60,
                            chunk_overlap=10)
    rag = GraphRAG(config=config, llm=ExtractiveLLM())
    rag.add_documents(DOCUMENTS, source="unit")
    return rag


# -- indexing -----------------------------------------------------------
def test_indexing_populates_every_store(system):
    stats = system.stats()

    assert stats["documents"] == len(DOCUMENTS)
    assert stats["chunks"] > 0
    assert stats["vectors"] == stats["chunks"]
    assert stats["graph"]["entities"] > 0
    assert stats["graph"]["relationships"] > 0
    assert len(system.bm25) == stats["chunks"]


def test_reindexing_the_same_documents_is_a_no_op(system):
    before = len(system.chunks)
    added = system.add_documents(DOCUMENTS, source="unit")

    assert added == 0
    assert len(system.chunks) == before


def test_same_text_from_a_different_source_is_a_distinct_document(system):
    before = len(system.documents)
    system.add_documents(DOCUMENTS[:1], source="another-corpus")

    assert len(system.documents) == before + 1


def test_add_documents_is_incremental(system):
    added = system.add_documents(
        ["Reinforcement learning trains agents with rewards and penalties."]
    )
    assert added > 0
    assert system.graph.has_entity("reinforcement learning")


def test_build_knowledge_graph_returns_the_graph():
    rag = GraphRAG(config=GraphRAGConfig(embedding_dim=128), llm=ExtractiveLLM())
    graph = rag.build_knowledge_graph(DOCUMENTS)

    assert len(graph) > 0
    assert graph is rag.graph


def test_add_path_indexes_files(tmp_path):
    (tmp_path / "notes.md").write_text(
        "Knowledge graphs link entities and relationships.", encoding="utf-8"
    )
    rag = GraphRAG(config=GraphRAGConfig(embedding_dim=128), llm=ExtractiveLLM())

    assert rag.add_path(tmp_path) > 0
    assert rag.graph.has_entity("knowledge graphs")


# -- querying -----------------------------------------------------------
def test_query_returns_grounded_answer_and_sources(system):
    result = system.query("What is deep learning?")

    assert result["answer"]
    assert result["sources"]
    assert result["entities"]
    assert result["context"].startswith("[1]")
    assert "deep learning" in " ".join(result["sources"]).lower()


def test_query_signals_include_all_three_retrievers(system):
    result = system.query("How do neural networks relate to computer vision?")
    signals = set()
    for item in result["retrieved"]:
        signals.update(item["signals"])

    assert {"vector", "keyword", "graph"} & signals == {"vector", "keyword", "graph"}


def test_query_ranks_off_topic_chunks_last(system):
    result = system.query("What is machine learning?", top_k=2)
    joined = " ".join(result["sources"]).lower()

    assert "photosynthesis" not in joined


def test_query_without_generation(system):
    result = system.query("What is machine learning?", generate=False)

    assert result["answer"] == "Generation skipped."
    assert result["retrieved"]


def test_query_on_empty_index_is_graceful():
    rag = GraphRAG(config=GraphRAGConfig(embedding_dim=64), llm=ExtractiveLLM())
    result = rag.query("anything at all?")

    assert result["sources"] == []
    assert "No documents" in result["answer"]


def test_query_rejects_empty_question(system):
    with pytest.raises(ValueError):
        system.query("   ")


def test_related_entities_traverses_the_graph(system):
    related = system.related_entities("deep learning", depth=1)
    assert related


# -- generation ---------------------------------------------------------
def test_extractive_llm_cites_passages():
    context = (
        "[1] (source: unit)\nDeep learning uses neural networks.\n\n"
        "[2] (source: unit)\nPhotosynthesis happens in plants."
    )
    answer = ExtractiveLLM().answer("What does deep learning use?", context)

    assert "neural networks" in answer
    assert "[1]" in answer
    assert "Photosynthesis" not in answer


def test_extractive_llm_admits_when_context_is_empty():
    assert "enough information" in ExtractiveLLM().answer("anything?", "")


def test_generate_answer_uses_the_prompt_contract():
    context = "[1] (source: s)\nKnowledge graphs link related entities."
    prompt = build_answer_prompt("What do knowledge graphs link?", context)

    assert "<context>" in prompt
    assert "Question: What do knowledge graphs link?" in prompt
    assert "entities" in generate_answer(
        ExtractiveLLM(), "What do knowledge graphs link?", context
    )


def test_generate_answer_without_context_admits_ignorance():
    assert "enough information" in generate_answer(ExtractiveLLM(), "anything?", "  ")


# -- persistence --------------------------------------------------------
def test_save_and_load_round_trip(system, tmp_path):
    target = system.save(tmp_path / "index")
    restored = GraphRAG.load(target, llm=ExtractiveLLM())

    assert len(restored.chunks) == len(system.chunks)
    assert len(restored.documents) == len(system.documents)
    assert len(restored.graph) == len(system.graph)
    assert len(restored.vector_store) == len(system.vector_store)

    original = system.query("What is deep learning?")
    reloaded = restored.query("What is deep learning?")
    assert reloaded["sources"] == original["sources"]


def test_saved_config_never_contains_credentials(system, tmp_path):
    config = GraphRAGConfig(embedding_dim=256, api_key="sk-ant-not-a-real-key")
    system.config = config
    target = system.save(tmp_path / "index")

    payload = json.loads((target / "config.json").read_text(encoding="utf-8"))
    assert "api_key" not in payload


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        GraphRAG.load(tmp_path / "missing")


# -- config -------------------------------------------------------------
def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_CHUNK_SIZE", "42")
    monkeypatch.setenv("GRAPHRAG_LLM_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

    config = GraphRAGConfig.from_env()

    assert config.chunk_size == 42
    assert config.llm_model == "claude-sonnet-5"
    assert config.api_key == "sk-ant-not-a-real-key"


def test_config_validates_chunking():
    with pytest.raises(ValueError):
        GraphRAGConfig(chunk_size=10, chunk_overlap=10)


def test_config_validates_extractor_name():
    with pytest.raises(ValueError):
        GraphRAGConfig(extractor="magic")


# -- cli ----------------------------------------------------------------
def test_cli_index_query_and_stats(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "ml.txt").write_text("\n".join(DOCUMENTS), encoding="utf-8")
    storage = str(tmp_path / "storage")

    assert cli_main(["--storage", storage, "index", str(corpus)]) == 0
    assert cli_main(["--storage", storage, "query", "What is deep learning?"]) == 0
    out = capsys.readouterr().out
    assert "Answer:" in out

    assert cli_main(["--storage", storage, "stats"]) == 0
    assert "entities" in capsys.readouterr().out

    graphml = tmp_path / "graph.graphml"
    assert cli_main(["--storage", storage, "export", str(graphml)]) == 0
    assert graphml.exists()


def test_cli_demo_runs(capsys):
    assert cli_main(["demo"]) == 0
    assert "GraphRAG" in capsys.readouterr().out


def test_cli_query_without_index_fails(tmp_path):
    with pytest.raises(SystemExit):
        cli_main(["--storage", str(tmp_path / "nope"), "query", "hello?"])
