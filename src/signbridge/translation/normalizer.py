"""Deterministic Text Normalizer for Stage 8 Translation.

Provides lightweight, pure-function string normalization without probabilistic
or generative alteration of user-signed content.
"""

import re


def normalize_whitespace(text: str, trim: bool = True) -> str:
    """Collapses consecutive whitespace into a single space and optionally trims edges.

    Args:
        text: Input string.
        trim: Whether to strip leading and trailing whitespace.

    Returns:
        Cleaned, single-spaced string.
    """
    if not text:
        return ""
    # Collapse multiple consecutive whitespace characters to a single space
    collapsed = re.sub(r"\s+", " ", text)
    return collapsed.strip() if trim else collapsed


def format_sentence_text(words: list[str], active_word: str = "", auto_capitalize: bool = False) -> str:
    """Assembles completed words and any active in-progress word into a clean sentence string.

    Args:
        words: List of completed word strings.
        active_word: Current in-progress word string.
        auto_capitalize: Whether to capitalize the first character of the sentence.

    Returns:
        Formatted space-delimited sentence.
    """
    parts = [w.strip() for w in words if w.strip()]
    if active_word.strip():
        parts.append(active_word.strip())

    if not parts:
        return ""

    raw_sentence = " ".join(parts)
    if auto_capitalize and raw_sentence:
        return raw_sentence[0].upper() + raw_sentence[1:]
    return raw_sentence
