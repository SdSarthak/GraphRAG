"""Tests for the generation backends.

The Claude backend is exercised with a stub client, so nothing here touches the
network or needs an API key.
"""

import types

import pytest

from graphrag.config import GraphRAGConfig
from graphrag.llm import AnthropicLLM, ExtractiveLLM, build_llm


def _block(kind: str, text: str = "") -> types.SimpleNamespace:
    return types.SimpleNamespace(type=kind, text=text)


class _StubClient:
    """Stands in for ``anthropic.Anthropic``."""

    def __init__(self, response):
        self.response = response
        self.calls = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def make_llm(response) -> AnthropicLLM:
    llm = AnthropicLLM.__new__(AnthropicLLM)  # skip SDK client construction
    llm.model = "claude-opus-5"
    llm.max_tokens = 1024
    llm.effort = "medium"
    llm.client = _StubClient(response)
    llm._anthropic = types.SimpleNamespace(
        APIStatusError=RuntimeError, APIConnectionError=ConnectionError
    )
    return llm


def test_request_shape_matches_the_messages_api():
    llm = make_llm(types.SimpleNamespace(stop_reason="end_turn", content=[_block("text", "hi")]))
    answer = llm.complete(prompt="question?", system="be terse", max_tokens=256)

    assert answer == "hi"
    sent = llm.client.calls[0]
    assert sent["model"] == "claude-opus-5"
    assert sent["max_tokens"] == 256
    assert sent["system"] == "be terse"
    assert sent["messages"] == [{"role": "user", "content": "question?"}]
    assert sent["output_config"]["effort"] == "medium"
    # Sampling params and budget_tokens are rejected by current models.
    assert "temperature" not in sent
    assert "thinking" not in sent


def test_thinking_blocks_are_skipped_when_reading_text():
    llm = make_llm(
        types.SimpleNamespace(
            stop_reason="end_turn",
            content=[_block("thinking"), _block("text", "the answer")],
        )
    )
    assert llm.complete(prompt="q") == "the answer"


def test_refusal_is_handled_before_reading_content():
    llm = make_llm(types.SimpleNamespace(stop_reason="refusal", content=[]))
    assert llm.complete(prompt="q") == ""


def test_structured_output_requests_a_json_schema():
    llm = make_llm(
        types.SimpleNamespace(
            stop_reason="end_turn", content=[_block("text", '{"entities": []}')]
        )
    )
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    payload = llm.structured(prompt="extract", schema=schema)

    assert payload == {"entities": []}
    assert llm.client.calls[0]["output_config"]["format"] == {
        "type": "json_schema",
        "schema": schema,
    }


def test_structured_output_survives_non_json_responses():
    llm = make_llm(
        types.SimpleNamespace(stop_reason="end_turn", content=[_block("text", "sorry!")])
    )
    assert llm.structured(prompt="extract", schema={}) is None


def test_build_llm_falls_back_without_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm = build_llm(GraphRAGConfig(api_key=None))

    assert isinstance(llm, ExtractiveLLM)


def test_build_llm_selects_claude_when_a_key_is_present(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm = build_llm(GraphRAGConfig(api_key="sk-ant-not-a-real-key"))

    assert isinstance(llm, AnthropicLLM)
    assert llm.name == "claude-opus-5"


def test_build_llm_honours_allow_remote_false():
    llm = build_llm(GraphRAGConfig(api_key="sk-ant-not-a-real-key"), allow_remote=False)
    assert isinstance(llm, ExtractiveLLM)


def test_extractive_llm_limits_the_number_of_sentences():
    context = "\n".join(
        f"[{i}] (source: s)\nNeural networks learn representation number {i}."
        for i in range(1, 6)
    )
    answer = ExtractiveLLM(max_sentences=2).answer("What do neural networks learn?", context)

    assert answer.count("[") == 2


def test_extractive_llm_prefers_earlier_passages_on_ties():
    context = (
        "[1] (source: s)\nNeural networks learn representations.\n\n"
        "[2] (source: s)\nNeural networks learn representations."
    )
    answer = ExtractiveLLM(max_sentences=1).answer("What do neural networks learn?", context)

    assert "[1]" in answer


@pytest.mark.parametrize("question", ["", "   "])
def test_extractive_llm_with_no_context(question):
    assert "enough information" in ExtractiveLLM().answer(question, "")
