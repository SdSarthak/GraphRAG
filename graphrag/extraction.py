"""Entity and relationship extraction.

Two extractors are provided:

``RuleBasedExtractor``
    Deterministic, offline, dependency free. Combines proper-noun detection
    with frequency-ranked key phrases, then types the relations between
    co-occurring entities using a verb lexicon.

``LLMExtractor``
    Uses Claude with a strict JSON schema to pull a typed entity/relation set
    out of each chunk. Falls back to the rule-based extractor when the model
    is unavailable or returns nothing usable.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .text import (
    RELATION_VERBS,
    STOPWORDS,
    content_tokens,
    split_sentences,
    tokenize,
)

logger = logging.getLogger(__name__)

# "Deep Learning", "United States of America", "NASA"
_PROPER_NOUN_RE = re.compile(
    r"\b[A-Z][a-zA-Z0-9]*(?:[ -](?:of|de|and|the|for)?[ ]?[A-Z][a-zA-Z0-9]*)*\b"
)


@dataclass
class Entity:
    """A node candidate extracted from text."""

    name: str
    type: str = "concept"
    mentions: int = 1

    @property
    def key(self) -> str:
        """Canonical identifier used as the graph node id."""
        return self.name.strip().lower()

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "type": self.type, "mentions": self.mentions}


@dataclass
class Relationship:
    """A typed edge candidate between two entities."""

    source: str
    target: str
    type: str = "co_occurs"
    context: str = ""
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "context": self.context,
            "weight": self.weight,
        }


@dataclass
class ExtractionResult:
    """Entities and relationships found in a single chunk."""

    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.entities


class RuleBasedExtractor:
    """Deterministic extractor: proper nouns + frequency-ranked key phrases."""

    def __init__(
        self,
        max_entities: int = 12,
        min_entity_length: int = 3,
        max_phrase_words: int = 3,
    ) -> None:
        self.max_entities = max_entities
        self.min_entity_length = min_entity_length
        self.max_phrase_words = max_phrase_words

    # -- public API ------------------------------------------------------
    def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relationships from a piece of text."""
        if not text or not text.strip():
            return ExtractionResult()

        entities = self._extract_entities(text)
        if not entities:
            return ExtractionResult()

        relationships = self._extract_relationships(text, entities)
        return ExtractionResult(entities=entities, relationships=relationships)

    # -- entities --------------------------------------------------------
    def _extract_entities(self, text: str) -> List[Entity]:
        scores: Dict[str, float] = {}
        display: Dict[str, str] = {}
        types: Dict[str, str] = {}
        counts: Counter = Counter()

        for name, count in self._proper_nouns(text).items():
            key = name.lower()
            scores[key] = scores.get(key, 0.0) + count * 2.0  # proper nouns rank higher
            display.setdefault(key, name)
            types[key] = "proper_noun"
            counts[key] += count

        for phrase, count in self._key_phrases(text).items():
            key = phrase.lower()
            # Longer phrases are more specific entities, so they outrank their
            # own constituent words at equal frequency.
            length_bonus = 1.0 + 0.5 * (len(phrase.split()) - 1)
            scores[key] = scores.get(key, 0.0) + count * length_bonus
            display.setdefault(key, phrase)
            types.setdefault(key, "concept")
            counts[key] += count

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], -len(item[0].split()), item[0]),
        )
        entities: List[Entity] = []
        for key, _score in ranked:
            if self._is_subsumed(key, [e.key for e in entities]):
                continue
            entities.append(
                Entity(
                    name=display[key],
                    type=types.get(key, "concept"),
                    mentions=max(1, counts[key]),
                )
            )
            if len(entities) >= self.max_entities:
                break
        return entities

    def _proper_nouns(self, text: str) -> Dict[str, int]:
        found: Counter = Counter()
        for sentence in split_sentences(text):
            words = sentence.split()
            first_word = words[0] if words else ""
            for match in _PROPER_NOUN_RE.finditer(sentence):
                candidate = match.group(0).strip()
                # A capitalised first word is not evidence of a proper noun.
                if match.start() == 0 and candidate == first_word.strip(".,;:"):
                    continue
                if len(candidate) < self.min_entity_length:
                    continue
                if candidate.lower() in STOPWORDS:
                    continue
                found[candidate] += 1
        return dict(found)

    def _key_phrases(self, text: str) -> Dict[str, int]:
        phrases: Counter = Counter()
        for sentence in split_sentences(text):
            tokens = tokenize(sentence)
            window: List[str] = []
            for token in tokens + [""]:
                is_content = (
                    bool(token)
                    and token not in STOPWORDS
                    and len(token) >= self.min_entity_length
                    and not token.isdigit()
                )
                if is_content:
                    window.append(token)
                    continue
                for phrase in self._window_phrases(window):
                    phrases[phrase] += 1
                window = []
        return dict(phrases)

    def _window_phrases(self, window: Sequence[str]) -> List[str]:
        """All n-grams in a window, minus any spanning a relation verb.

        "deep learning uses neural networks" should yield the two entities and
        the ``uses`` relation between them, never the phrase "learning uses".
        """
        phrases: List[str] = []
        if not window:
            return phrases
        for size in range(1, min(self.max_phrase_words, len(window)) + 1):
            for start in range(0, len(window) - size + 1):
                gram = window[start : start + size]
                if any(token in RELATION_VERBS for token in gram):
                    continue
                phrases.append(" ".join(gram))
        return phrases

    @staticmethod
    def _is_subsumed(key: str, existing: Sequence[str]) -> bool:
        """Drop a candidate already covered by a longer accepted phrase."""
        for other in existing:
            if key == other:
                return True
            if f" {key} " in f" {other} ":
                return True
        return False

    # -- relationships ---------------------------------------------------
    def _extract_relationships(
        self, text: str, entities: Sequence[Entity]
    ) -> List[Relationship]:
        by_key = {entity.key: entity for entity in entities}
        relationships: List[Relationship] = []
        seen: set = set()

        for sentence in split_sentences(text):
            lowered = sentence.lower()
            positions: List[Tuple[int, str]] = []
            for key in by_key:
                index = lowered.find(key)
                if index >= 0:
                    positions.append((index, key))
            positions.sort()

            for i, (start_a, key_a) in enumerate(positions):
                for start_b, key_b in positions[i + 1 :]:
                    if key_a == key_b:
                        continue
                    between = lowered[start_a + len(key_a) : start_b]
                    rel_type = self._relation_type(between)
                    dedup_key = (key_a, key_b, rel_type)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    relationships.append(
                        Relationship(
                            source=key_a,
                            target=key_b,
                            type=rel_type,
                            context=sentence[:200],
                            weight=2.0 if rel_type != "co_occurs" else 1.0,
                        )
                    )
        return relationships

    @staticmethod
    def _relation_type(between: str) -> str:
        for token in tokenize(between):
            verb = RELATION_VERBS.get(token)
            if verb:
                return verb
        return "co_occurs"


EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                },
                "required": ["name", "type"],
                "additionalProperties": False,
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string"},
                },
                "required": ["source", "target", "type"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "relationships"],
    "additionalProperties": False,
}

EXTRACTION_SYSTEM = (
    "You build knowledge graphs. Given a passage, return the salient entities "
    "and the relationships between them. Use short noun phrases for entity "
    "names, lowercase snake_case for relationship types, and only include "
    "relationships whose endpoints both appear in the entity list."
)


class LLMExtractor:
    """Schema-constrained extraction with Claude, with a rule-based fallback."""

    def __init__(
        self,
        llm: Any,
        max_entities: int = 12,
        fallback: Optional[RuleBasedExtractor] = None,
    ) -> None:
        self.llm = llm
        self.max_entities = max_entities
        self.fallback = fallback or RuleBasedExtractor(max_entities=max_entities)

    def extract(self, text: str) -> ExtractionResult:
        if not text or not text.strip():
            return ExtractionResult()

        payload: Optional[Dict[str, Any]] = None
        if getattr(self.llm, "supports_structured_output", False):
            prompt = (
                f"Extract at most {self.max_entities} entities and the "
                "relationships between them from the passage below.\n\n"
                f"<passage>\n{text}\n</passage>"
            )
            try:
                payload = self.llm.structured(
                    system=EXTRACTION_SYSTEM,
                    prompt=prompt,
                    schema=EXTRACTION_SCHEMA,
                )
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("LLM extraction failed (%s); using rule-based", exc)
                payload = None

        result = self._parse(payload) if payload else ExtractionResult()
        if result.is_empty():
            return self.fallback.extract(text)
        return result

    def _parse(self, payload: Dict[str, Any]) -> ExtractionResult:
        entities: List[Entity] = []
        seen: set = set()
        for raw in payload.get("entities", [])[: self.max_entities]:
            name = str(raw.get("name", "")).strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            entities.append(Entity(name=name, type=str(raw.get("type") or "concept")))

        relationships: List[Relationship] = []
        for raw in payload.get("relationships", []):
            source = str(raw.get("source", "")).strip().lower()
            target = str(raw.get("target", "")).strip().lower()
            if not source or not target or source == target:
                continue
            if source not in seen or target not in seen:
                continue
            relationships.append(
                Relationship(
                    source=source,
                    target=target,
                    type=str(raw.get("type") or "related_to"),
                    weight=2.0,
                )
            )
        return ExtractionResult(entities=entities, relationships=relationships)


def build_extractor(config: Any, llm: Any = None) -> Any:
    """Create the extractor named by ``config.extractor``."""
    rule = RuleBasedExtractor(
        max_entities=config.max_entities_per_chunk,
        min_entity_length=config.min_entity_length,
    )
    if config.extractor == "llm" and llm is not None:
        return LLMExtractor(
            llm=llm, max_entities=config.max_entities_per_chunk, fallback=rule
        )
    return rule


def extract_query_entities(
    text: str, extractor: Any, max_entities: int = 8
) -> List[str]:
    """Return canonical entity keys for a query string.

    Queries are short, so this always uses the deterministic path: an LLM round
    trip per query would add latency without improving seed quality.
    """
    base = getattr(extractor, "fallback", extractor)
    if not isinstance(base, RuleBasedExtractor):
        base = RuleBasedExtractor(max_entities=max_entities)
    result = base.extract(text)
    keys = [entity.key for entity in result.entities[:max_entities]]
    if keys:
        return keys
    return content_tokens(text)[:max_entities]
