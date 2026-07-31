# -*- coding: utf-8 -*-
"""CLI failure modes: bad arguments, unusable paths, backend errors."""

import json
from pathlib import Path

import pytest

from graphrag.cli import main as cli_main
from graphrag.pipeline import CONFIG_FILE, GRAPH_FILE

DOCUMENTS = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
]


@pytest.fixture()
def indexed(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "ml.txt").write_text("\n".join(DOCUMENTS), encoding="utf-8")
    storage = str(tmp_path / "storage")
    assert cli_main(["--storage", storage, "index", str(corpus)]) == 0
    return corpus, storage


def test_indexing_a_missing_path_fails_cleanly(tmp_path, capsys):
    code = cli_main(
        ["--storage", str(tmp_path / "s"), "index", str(tmp_path / "nope")]
    )
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_a_corpus_with_no_supported_files_is_reported(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "data.bin").write_bytes(b"\x00\x01")
    storage = tmp_path / "storage"

    code = cli_main(["--storage", str(storage), "index", str(corpus)])

    assert code == 1
    assert "no readable text" in capsys.readouterr().err
    # An empty index must not be left behind pretending the run worked.
    assert not storage.exists()


def test_invalid_option_values_are_rejected(indexed, capsys):
    _, storage = indexed
    assert cli_main(["--storage", storage, "query", "anything?", "--top-k", "0"]) == 1
    assert "error:" in capsys.readouterr().err

    assert cli_main(["--storage", storage, "entities", "deep learning", "--depth", "0"]) == 1
    assert "error:" in capsys.readouterr().err


def test_a_bad_environment_variable_is_reported(indexed, capsys, monkeypatch):
    _, storage = indexed
    monkeypatch.setenv("GRAPHRAG_RETRIEVAL_TOP_K", "lots")

    assert cli_main(["--storage", storage, "query", "what is deep learning?"]) == 1
    assert "GRAPHRAG_RETRIEVAL_TOP_K" in capsys.readouterr().err


def test_a_corrupt_index_is_reported_rather_than_traced(indexed, capsys):
    _, storage = indexed
    (Path(storage) / GRAPH_FILE).write_text("{broken", encoding="utf-8")

    assert cli_main(["--storage", storage, "stats"]) == 1
    assert "error:" in capsys.readouterr().err


def test_an_unknown_entity_exits_non_zero(indexed, capsys):
    _, storage = indexed
    assert cli_main(["--storage", storage, "entities", "quantum chromodynamics"]) == 1
    assert "not in the graph" in capsys.readouterr().err


def test_the_index_config_is_reused_when_querying(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "ml.txt").write_text("\n".join(DOCUMENTS), encoding="utf-8")
    storage = str(tmp_path / "storage")

    monkeypatch.setenv("GRAPHRAG_EMBEDDING_DIM", "128")
    assert cli_main(["--storage", storage, "index", str(corpus)]) == 0
    capsys.readouterr()

    # Querying in a fresh shell, without the variable set, used to build a
    # 512-wide embedder against a 128-wide index.
    monkeypatch.delenv("GRAPHRAG_EMBEDDING_DIM")
    assert cli_main(["--storage", storage, "query", "What is deep learning?"]) == 0
    assert "Answer:" in capsys.readouterr().out


def test_appending_reuses_the_stored_settings(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text(DOCUMENTS[0], encoding="utf-8")
    storage = str(tmp_path / "storage")

    monkeypatch.setenv("GRAPHRAG_EMBEDDING_DIM", "128")
    assert cli_main(["--storage", storage, "index", str(corpus)]) == 0
    monkeypatch.delenv("GRAPHRAG_EMBEDDING_DIM")

    (corpus / "b.txt").write_text(DOCUMENTS[1], encoding="utf-8")
    assert cli_main(["--storage", storage, "index", str(corpus), "--append"]) == 0

    stored = json.loads((tmp_path / "storage" / CONFIG_FILE).read_text(encoding="utf-8"))
    assert stored["embedding_dim"] == 128
    output = capsys.readouterr().out
    assert "Indexed" in output


def test_re_indexing_unchanged_content_says_so(indexed, capsys):
    corpus, storage = indexed
    assert cli_main(["--storage", storage, "index", str(corpus), "--append"]) == 0
    assert "unchanged" in capsys.readouterr().out


def test_exporting_an_empty_graph_is_refused(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("the of and", encoding="utf-8")
    storage = str(tmp_path / "storage")
    cli_main(["--storage", storage, "index", str(corpus)])
    capsys.readouterr()

    code = cli_main(["--storage", storage, "export", str(tmp_path / "g.graphml")])
    assert code == 1
    assert "empty" in capsys.readouterr().err


def test_non_ascii_answers_do_not_crash_the_console(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "zh.txt").write_text(
        "机器学习是人工智能的一个分支。深度学习使用神经网络。", encoding="utf-8"
    )
    storage = str(tmp_path / "storage")

    assert cli_main(["--storage", storage, "index", str(corpus)]) == 0
    assert cli_main(["--storage", storage, "query", "什么是深度学习"]) == 0
    assert "Answer:" in capsys.readouterr().out


def test_query_json_output_is_machine_readable(indexed, capsys):
    _, storage = indexed
    assert cli_main(["--storage", storage, "query", "deep learning?", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) >= {"answer", "sources", "entities", "context", "retrieved"}
