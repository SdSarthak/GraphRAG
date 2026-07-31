"""Tests for the generation backends.

The Claude backend is exercised with a stub client, so nothing here touches the
network or needs an API key.
"""

import logging
import types

import pytest

from graphrag.config import GraphRAGConfig
from graphrag.llm import (
    AnthropicLLM,
    ExtractiveLLM,
    _split_prompt,
    build_answer_prompt,
    build_llm,
    generate_answer,
)


class _StubStatusError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class _StubTimeout(Exception):
    pass


class _StubConnectionError(Exception):
    pass


def _block(kind: str, text: str = "") -> types.SimpleNamespace:
    return types.SimpleNamespace(type=kind, text=text)


class _StubClient:
    """Stands in for ``anthropic.Anthropic``."""

    def __init__(self, response, errors=()):
        self.response = response
        self.errors = list(errors)
        self.calls = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return self.response


def make_llm(response, errors=()) -> AnthropicLLM:
    llm = AnthropicLLM.__new__(AnthropicLLM)  # skip SDK client construction
    llm.model = "claude-opus-5"
    llm.max_tokens = 1024
    llm.effort = "medium"
    llm.client = _StubClient(response, errors)
    llm._anthropic = types.SimpleNamespace(
        APIStatusError=_StubStatusError,
        APITimeoutError=_StubTimeout,
        APIConnectionError=_StubConnectionError,
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


# -- API failure modes --------------------------------------------------
def test_api_status_errors_carry_an_actionable_message():
    llm = make_llm(None, errors=[_StubStatusError(401, "invalid x-api-key")])
    with pytest.raises(RuntimeError) as excinfo:
        llm.complete(prompt="q")

    message = str(excinfo.value)
    assert "401" in message and "ANTHROPIC_API_KEY" in message


def test_rate_limits_and_outages_say_what_to_do():
    for status, expected in ((429, "Rate limited"), (529, "temporarily unavailable")):
        llm = make_llm(None, errors=[_StubStatusError(status, "nope")])
        with pytest.raises(RuntimeError) as excinfo:
            llm.complete(prompt="q")
        assert expected in str(excinfo.value)


def test_timeouts_are_reported_as_timeouts():
    llm = make_llm(None, errors=[_StubTimeout("took too long")])
    with pytest.raises(RuntimeError) as excinfo:
        llm.complete(prompt="q")
    assert "timed out" in str(excinfo.value)


def test_connection_errors_are_reported():
    llm = make_llm(None, errors=[_StubConnectionError("dns failure")])
    with pytest.raises(RuntimeError) as excinfo:
        llm.complete(prompt="q")
    assert "Could not reach" in str(excinfo.value)


def test_an_old_sdk_without_output_config_is_retried_once():
    response = types.SimpleNamespace(stop_reason="end_turn", content=[_block("text", "ok")])
    llm = make_llm(
        response,
        errors=[TypeError("create() got an unexpected keyword argument 'output_config'")],
    )
    assert llm.complete(prompt="q") == "ok"
    assert len(llm.client.calls) == 2
    assert "output_config" not in llm.client.calls[1]


def test_an_unrelated_typeerror_is_not_swallowed():
    # Retrying a genuine bug just raises the same error one call later while
    # hiding where it came from.
    llm = make_llm(None, errors=[TypeError("unhashable type: 'dict'")])
    with pytest.raises(TypeError):
        llm.complete(prompt="q")
    assert len(llm.client.calls) == 1


def test_truncated_responses_are_flagged(caplog):
    llm = make_llm(
        types.SimpleNamespace(stop_reason="max_tokens", content=[_block("text", "half")])
    )
    with caplog.at_level(logging.WARNING):
        assert llm.complete(prompt="q") == "half"
    assert "truncated" in caplog.text


# -- prompt round trip --------------------------------------------------
def test_a_context_containing_the_word_question_does_not_hijack_the_query():
    # A FAQ corpus is full of "Question:" lines; the offline answerer used to
    # pick the first one it saw and answer that instead.
    context = "[1] (source: faq.md)\nQuestion: what is a transformer? It is a model."
    prompt = build_answer_prompt("What is deep learning?", context)
    question, recovered = _split_prompt(prompt)

    assert question == "What is deep learning?"
    assert recovered == context


def test_extractive_backend_answers_the_real_question_through_complete():
    context = (
        "[1] (source: faq.md)\nQuestion: what is a transformer?\n\n"
        "[2] (source: notes.md)\nDeep learning uses neural networks."
    )
    answer = generate_answer(ExtractiveLLM(), "What does deep learning use?", context)
    assert "neural networks" in answer


def test_source_paths_containing_brackets_are_not_treated_as_content():
    context = "[1] (source: C:\\notes (copy)\\a.txt)\nDeep learning uses neural networks."
    answer = ExtractiveLLM().answer("What does deep learning use?", context)

    assert "neural networks" in answer
    assert "a.txt" not in answer
