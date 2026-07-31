# -*- coding: utf-8 -*-
"""Tokenisation edge cases, mostly around non-ASCII corpora.

Before these existed the tokeniser was ``[a-z0-9]``-only, so "café" became
"caf" and Cyrillic or CJK text produced no tokens at all - a whole corpus
could be indexed and remain permanently unretrievable.
"""

from graphrag.embeddings import HashingEmbedder
from graphrag.extraction import RuleBasedExtractor
from graphrag.text import content_tokens, is_cjk, split_sentences, tokenize


def test_accented_latin_is_not_mangled():
    assert tokenize("Le café était très bon") == ["le", "café", "était", "très", "bon"]


def test_german_and_cyrillic_tokens_survive():
    assert tokenize("Künstliche Intelligenz") == ["künstliche", "intelligenz"]
    assert tokenize("Машинное обучение") == ["машинное", "обучение"]


def test_cjk_is_segmented_per_character():
    tokens = tokenize("机器学习")
    assert tokens == ["机", "器", "学", "习"]
    assert all(is_cjk(token) for token in tokens)


def test_cjk_tokens_survive_the_content_length_filter():
    # Single characters are the unit of meaning; the length floor would
    # otherwise discard every term in the document.
    assert content_tokens("机器学习是人工智能的一个分支")


def test_english_contractions_stay_whole_but_elisions_split():
    assert "don't" in tokenize("I don't know")
    assert tokenize("l'apprentissage") == ["l", "apprentissage"]
    assert "est" in tokenize("Qu'est-ce que c'est")


def test_hyphenated_words_are_one_token():
    assert tokenize("state-of-the-art models") == ["state-of-the-art", "models"]


def test_decimals_do_not_split_sentences():
    assert split_sentences("Pi is 3.14 here. Next one.") == [
        "Pi is 3.14 here.",
        "Next one.",
    ]


def test_cjk_full_stop_splits_sentences():
    assert len(split_sentences("机器学习很好。深度学习也很好。")) == 2


def test_non_ascii_text_produces_a_non_zero_embedding():
    vector = HashingEmbedder(dim=64).embed_one("Машинное обучение")
    assert float(abs(vector).sum()) > 0.0


def test_proper_nouns_are_detected_in_accented_scripts():
    result = RuleBasedExtractor(max_entities=10).extract(
        "The theory came later. Émile Borel worked in Paris."
    )
    names = {entity.name for entity in result.entities}
    assert "Émile Borel" in names


def test_multi_word_proper_nouns_keep_their_connectors():
    result = RuleBasedExtractor(max_entities=10).extract(
        "Researchers met there. The Bank of England published a report."
    )
    names = {entity.name for entity in result.entities}
    assert any(name.endswith("Bank of England") for name in names)


def test_punctuation_terminates_a_proper_noun_phrase():
    # "Paris, France" is two entities, not one four-word phrase.
    spans = dict(
        (name, start)
        for start, name in RuleBasedExtractor._proper_noun_spans(
            "He flew to Paris, France yesterday"
        )
    )
    assert "Paris" in spans
    assert "France" in spans
