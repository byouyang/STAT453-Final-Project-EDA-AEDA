"""Text normalization and tokenization shared by every experiment condition."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


# Words with internal apostrophes remain a token; every punctuation mark is a token.
# This deliberately preserves the six AEDA insertion marks through CNN input.
TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", flags=re.UNICODE)
PUNCTUATION_STRIP_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Canonical form for exact/whitespace/case overlap checks."""
    return WHITESPACE_PATTERN.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def punctuation_insensitive_key(text: str) -> str:
    """A conservative normalized key that catches punctuation-only variants."""
    canonical = normalize_text(text)
    return WHITESPACE_PATTERN.sub(" ", PUNCTUATION_STRIP_PATTERN.sub(" ", canonical)).strip()


def overlap_keys(text: str) -> dict[str, str]:
    return {
        "canonical": normalize_text(text),
        "punctuation_insensitive": punctuation_insensitive_key(text),
    }


def tokenize(text: str) -> list[str]:
    """Tokenize while retaining standalone punctuation such as ! and ,."""
    return TOKEN_PATTERN.findall(normalize_text(text))


def contains_expected_punctuation(tokens: Iterable[str]) -> bool:
    token_set = set(tokens)
    return "!" in token_set and "," in token_set

