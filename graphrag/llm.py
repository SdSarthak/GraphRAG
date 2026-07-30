"""Answer generation backends.

``AnthropicLLM`` calls Claude through the official ``anthropic`` SDK.
``ExtractiveLLM`` is a deterministic offline fallback that composes an answer
from the retrieved passages themselves, so the pipeline stays fully functional
without an API key (in CI, in tests, and on a plane).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .text import content_tokens, split_sentences

logger = logging.getLogger(__name__)

ANSWER_SYSTEM = (
    "You answer questions strictly from the numbered context passages you are "
    "given. Cite the passages you use with bracketed numbers like [1] or "
    "[2]. If the context does not contain the answer, say so plainly instead "
    "of guessing. Keep the answer focused and concise."
)


def build_answer_prompt(question: str, context: str) -> str:
    """Render the user turn for answer generation."""
    return (
        "<context>\n"
        f"{context}\n"
        "</context>\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, citing passage numbers."
    )


class ExtractiveLLM:
    """Offline answerer: ranks context sentences against the question.

    Not a stub — it performs real extractive summarisation (term overlap with
    a position prior) and is the backend used whenever no API key is present.
    """

    name = "extractive"
    supports_structured_output = False

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        question, context = _split_prompt(prompt)
        return self.answer(question, context)

    def answer(self, question: str, context: str) -> str:
        if not context.strip():
            return "I don't have enough information in the indexed corpus to answer that."

        query_terms = set(content_tokens(question))
        scored: List[tuple] = []
        current_citation = ""
        current_rank = 0
        for raw in context.splitlines():
            line = raw.strip()
            if not line:
                continue
            marker = re.match(r"^\[(\d+)\]", line)
            if marker:
                current_rank = int(marker.group(1))
                current_citation = f"[{current_rank}]"
                line = line[marker.end() :].strip()
                line = re.sub(r"^\(source:[^)]*\)", "", line).strip()
            if not line:
                continue
            for sentence in split_sentences(line):
                terms = set(content_tokens(sentence))
                if not terms:
                    continue
                overlap = len(query_terms & terms)
                if overlap == 0:
                    continue
                coverage = overlap / max(1, len(query_terms))
                density = overlap / len(terms)
                # Trust the retriever: earlier passages carry a rank prior.
                rank_prior = 0.05 * max(0, current_rank - 1)
                score = coverage + 0.5 * density - rank_prior
                scored.append((score, sentence, current_citation))

        if not scored:
            return (
                "The retrieved passages do not directly answer that question. "
                "Closest available context is listed in the sources below."
            )

        scored.sort(key=lambda item: -item[0])
        seen: set = set()
        sentences: List[str] = []
        for _score, sentence, citation in scored:
            fingerprint = sentence.lower()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            sentences.append(f"{sentence} {citation}".strip())
            if len(sentences) >= self.max_sentences:
                break
        return " ".join(sentences)


class AnthropicLLM:
    """Claude-backed generation and schema-constrained extraction."""

    supports_structured_output = True

    def __init__(
        self,
        model: str = "claude-opus-5",
        api_key: Optional[str] = None,
        max_tokens: int = 8192,
        effort: str = "medium",
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The 'anthropic' package is required for AnthropicLLM. "
                "Install it with: pip install anthropic"
            ) from exc

        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self._anthropic = anthropic
        # The SDK resolves ANTHROPIC_API_KEY (or an `ant auth login` profile)
        # on its own when api_key is None.
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    @property
    def name(self) -> str:
        return self.model

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a text completion."""
        response = self._create(
            prompt=prompt,
            system=system or ANSWER_SYSTEM,
            max_tokens=max_tokens or self.max_tokens,
        )
        return self._text_of(response)

    def structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate JSON constrained to ``schema``; ``None`` on refusal."""
        response = self._create(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens or self.max_tokens,
            output_format={"type": "json_schema", "schema": schema},
        )
        text = self._text_of(response)
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Model returned non-JSON output for a structured request")
            return None
        return payload if isinstance(payload, dict) else None

    # -- internals -------------------------------------------------------
    def _create(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        output_format: Optional[Dict[str, Any]] = None,
    ) -> Any:
        output_config: Dict[str, Any] = {"effort": self.effort}
        if output_format is not None:
            output_config["format"] = output_format

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "output_config": output_config,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        try:
            return self.client.messages.create(**kwargs)
        except TypeError:
            # Older SDK builds do not accept output_config; retry without it so
            # generation still works (structured output degrades to free text).
            kwargs.pop("output_config", None)
            return self.client.messages.create(**kwargs)
        except self._anthropic.APIStatusError as exc:
            raise RuntimeError(
                f"Claude API error ({exc.status_code}): {exc.message}"
            ) from exc
        except self._anthropic.APIConnectionError as exc:
            raise RuntimeError(f"Could not reach the Claude API: {exc}") from exc

    @staticmethod
    def _text_of(response: Any) -> str:
        # Safety classifiers can decline: check stop_reason before content.
        if getattr(response, "stop_reason", None) == "refusal":
            logger.warning("Claude declined to answer this request")
            return ""
        parts = [
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        return "\n".join(parts).strip()


def _split_prompt(prompt: str) -> tuple:
    """Recover ``(question, context)`` from a rendered answer prompt."""
    context = ""
    match = re.search(r"<context>\n(.*?)\n</context>", prompt, re.DOTALL)
    if match:
        context = match.group(1)
    question_match = re.search(r"Question:\s*(.+)", prompt)
    question = question_match.group(1).strip() if question_match else prompt
    return question, context


def build_llm(config, allow_remote: bool = True) -> Any:
    """Pick a backend: Claude when credentials exist, extractive otherwise."""
    if not allow_remote:
        return ExtractiveLLM()

    import os

    has_credentials = bool(config.api_key or os.environ.get("ANTHROPIC_API_KEY"))
    if not has_credentials:
        logger.info(
            "No ANTHROPIC_API_KEY found; using the offline extractive answerer."
        )
        return ExtractiveLLM()

    try:
        return AnthropicLLM(
            model=config.llm_model,
            api_key=config.api_key,
            max_tokens=config.llm_max_tokens,
            effort=config.llm_effort,
        )
    except RuntimeError as exc:
        logger.warning("%s Falling back to the extractive answerer.", exc)
        return ExtractiveLLM()


def generate_answer(
    llm: Any, question: str, context: str, max_tokens: Optional[int] = None
) -> str:
    """Ask the configured backend for a grounded answer."""
    if not context.strip():
        return "I don't have enough information in the indexed corpus to answer that."
    prompt = build_answer_prompt(question, context)
    answer = llm.complete(prompt=prompt, system=ANSWER_SYSTEM, max_tokens=max_tokens)
    if not answer:
        return "The model did not return an answer for this question."
    return answer
