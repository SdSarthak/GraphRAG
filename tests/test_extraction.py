from graphrag.extraction import (
    Entity,
    ExtractionResult,
    LLMExtractor,
    Relationship,
    RuleBasedExtractor,
    extract_query_entities,
)

TEXT = (
    "Machine learning is a subset of artificial intelligence. "
    "Deep learning uses neural networks to model complex patterns."
)


def test_rule_based_extractor_finds_domain_entities():
    result = RuleBasedExtractor(max_entities=10).extract(TEXT)
    keys = {entity.key for entity in result.entities}

    assert result.entities
    assert "machine learning" in keys
    assert "neural networks" in keys


def test_rule_based_extractor_types_relationships():
    result = RuleBasedExtractor(max_entities=10).extract(
        "Deep learning uses neural networks."
    )
    types = {rel.type for rel in result.relationships}

    assert result.relationships
    assert "uses" in types


def test_relationship_endpoints_are_known_entities():
    result = RuleBasedExtractor(max_entities=10).extract(TEXT)
    keys = {entity.key for entity in result.entities}

    for relationship in result.relationships:
        assert relationship.source in keys
        assert relationship.target in keys
        assert relationship.source != relationship.target


def test_empty_text_yields_empty_result():
    result = RuleBasedExtractor().extract("   ")
    assert result.is_empty()
    assert result.relationships == []


def test_entity_key_is_lowercased():
    assert Entity(name="Neural Networks").key == "neural networks"


def test_max_entities_is_respected():
    long_text = " ".join(f"concept{i} matters here." for i in range(50))
    result = RuleBasedExtractor(max_entities=5).extract(long_text)
    assert len(result.entities) <= 5


def test_extract_query_entities_falls_back_to_tokens():
    keys = extract_query_entities("what is deep learning", RuleBasedExtractor())
    assert keys
    assert any("learning" in key for key in keys)


class _FakeLLM:
    supports_structured_output = True

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def structured(self, system, prompt, schema):
        self.calls += 1
        return self.payload


def test_llm_extractor_parses_model_output():
    llm = _FakeLLM(
        {
            "entities": [
                {"name": "Transformers", "type": "architecture"},
                {"name": "Attention", "type": "mechanism"},
            ],
            "relationships": [
                {"source": "transformers", "target": "attention", "type": "uses"},
                {"source": "transformers", "target": "unknown", "type": "uses"},
            ],
        }
    )
    result = LLMExtractor(llm=llm).extract("Transformers use attention.")

    assert llm.calls == 1
    assert {entity.key for entity in result.entities} == {"transformers", "attention"}
    # The relationship pointing at an unlisted entity is dropped.
    assert len(result.relationships) == 1
    assert result.relationships[0].type == "uses"


def test_llm_extractor_falls_back_when_model_fails():
    class _BrokenLLM:
        supports_structured_output = True

        def structured(self, system, prompt, schema):
            raise RuntimeError("api down")

    result = LLMExtractor(llm=_BrokenLLM()).extract(TEXT)
    assert result.entities  # rule-based fallback produced entities


def test_llm_extractor_falls_back_on_empty_payload():
    result = LLMExtractor(llm=_FakeLLM({"entities": [], "relationships": []})).extract(TEXT)
    assert result.entities


def test_extraction_result_round_trip_dicts():
    entity = Entity(name="A", type="concept", mentions=2)
    relationship = Relationship(source="a", target="b", type="uses")
    result = ExtractionResult(entities=[entity], relationships=[relationship])

    assert result.entities[0].to_dict()["mentions"] == 2
    assert result.relationships[0].to_dict()["type"] == "uses"
