"""
Shared text utilities: tokenisation, stopwords and sentence splitting.

Everything in this module is deterministic and dependency free so that the rest
of the package can rely on it without pulling in optional NLP toolkits.
"""

from __future__ import annotations

import re
from typing import List

# Small, curated English stopword list. Deliberately kept short: aggressive
# filtering removes domain terms that make good graph nodes.
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are aren't as at
    be because been before being below between both but by can cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he her here hers
    herself him himself his how however i if in into is isn't it its itself
    just me more most must mustn't my myself no nor not of off on once only or
    other ought our ours ourselves out over own same shan't she should
    shouldn't so some such than that the their theirs them themselves then
    there these they this those through to too under until up upon very was
    wasn't we were weren't what when where which while who whom why with won't
    would wouldn't you your yours yourself yourselves
    """.split()
)

# Verbs that signal a typed relationship between two entities in a sentence.
RELATION_VERBS = {
    "is": "is_a",
    "are": "is_a",
    "was": "is_a",
    "were": "is_a",
    "uses": "uses",
    "use": "uses",
    "used": "uses",
    "using": "uses",
    "enables": "enables",
    "enable": "enables",
    "allows": "enables",
    "allow": "enables",
    "requires": "requires",
    "require": "requires",
    "contains": "contains",
    "includes": "contains",
    "include": "contains",
    "produces": "produces",
    "produce": "produces",
    "trains": "trains",
    "train": "trains",
    "supports": "supports",
    "support": "supports",
    "extends": "extends",
    "improves": "improves",
    "powers": "powers",
    "builds": "builds",
    "processes": "processes",
    "process": "processes",
    "analyzes": "analyzes",
    "analyses": "analyzes",
    "learns": "learns_from",
    "learn": "learns_from",
    "belongs": "part_of",
    "part": "part_of",
    "subset": "part_of",
    "link": "links_to",
    "links": "links_to",
    "linked": "links_to",
    "linking": "links_to",
    "connect": "connects",
    "connects": "connects",
    "combine": "combines",
    "combines": "combines",
    "convert": "converts",
    "converts": "converts",
    "generate": "generates",
    "generates": "generates",
    "represent": "represents",
    "represents": "represents",
    "describe": "describes",
    "describes": "describes",
    "transform": "transforms",
    "transforms": "transforms",
    "depend": "depends_on",
    "depends": "depends_on",
    "relate": "related_to",
    "relates": "related_to",
}

# Scripts written without spaces between words. A word-boundary tokeniser
# would collapse a whole Chinese or Japanese sentence into one useless token,
# so these are tokenised one character at a time instead (cheap unigram
# segmentation - crude, but it produces real, matchable terms where a
# whitespace tokeniser produces none).
_CJK_RANGES = (
    "぀-ヿ"  # hiragana + katakana
    "㐀-䶿"  # CJK extension A
    "一-鿿"  # CJK unified ideographs
    "豈-﫿"  # CJK compatibility ideographs
)

# ``[^\W_]`` is "letter or digit in any script" - unlike ``[a-zA-Z0-9]`` it
# keeps accented Latin, Cyrillic, Greek, Hangul and so on. Without this the
# whole pipeline silently discards non-ASCII corpora: "café" became "caf" and
# Cyrillic text tokenised to nothing at all, leaving zero embeddings and an
# empty keyword index.
_LETTER = f"[^\\W_{_CJK_RANGES}]"
_TOKEN_RE = re.compile(f"[{_CJK_RANGES}]|{_LETTER}+(?:['’\\-]{_LETTER}+)*")
_CJK_RE = re.compile(f"[{_CJK_RANGES}]")

# Romance-language elision: "l'apprentissage" is the article plus the noun and
# must split, while the English contraction "don't" must not. The two are told
# apart by which side is short - elisions have a one or two letter *prefix*,
# contractions a short suffix.
_ELISION_RE = re.compile(f"^({_LETTER}{{1,2}})['’]({_LETTER}.*)$")
# ASCII terminators need trailing whitespace so "3.14" and "U.S.A." stay
# intact; the CJK terminators are unambiguous and are usually not followed by
# a space at all.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s*|\n{2,}")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Collapse whitespace and strip surrounding blanks."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, punctuation removed.

    Unicode aware: accented and non-Latin scripts survive tokenisation, and
    CJK text is segmented into single characters.
    """
    tokens: List[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0).lower()
        elision = _ELISION_RE.match(token)
        if elision:
            tokens.append(elision.group(1))
            token = elision.group(2)
        tokens.append(token)
    return tokens


def is_cjk(token: str) -> bool:
    """True when the token comes from a script that has no word spacing."""
    return bool(_CJK_RE.search(token))


def content_tokens(text: str, min_length: int = 3) -> List[str]:
    """Tokens with stopwords and very short tokens removed.

    The length floor is skipped for CJK tokens: single characters are the
    whole unit of meaning there, so applying it would discard every term.
    """
    return [
        token
        for token in tokenize(text)
        if token not in STOPWORDS and (len(token) >= min_length or is_cjk(token))
    ]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on terminal punctuation or blank lines."""
    parts = (normalize(part) for part in _SENTENCE_RE.split(text))
    return [part for part in parts if part]
