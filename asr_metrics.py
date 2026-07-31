"""Text normalization and error-rate primitives for ASR evaluation."""

from __future__ import annotations

import re
import unicodedata
from typing import Sequence


WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_words(text: str) -> list[str]:
    """Normalize case, width, punctuation, symbols, and whitespace."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    cleaned_characters = [
        "" if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    ]
    collapsed = WHITESPACE_PATTERN.sub(" ", "".join(cleaned_characters)).strip()
    return collapsed.split() if collapsed else []


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Calculate Levenshtein distance with linear auxiliary memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous_row = list(range(len(hypothesis) + 1))
    for reference_index, reference_unit in enumerate(reference, start=1):
        current_row = [reference_index]
        for hypothesis_index, hypothesis_unit in enumerate(hypothesis, start=1):
            substitution_cost = reference_unit != hypothesis_unit
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[hypothesis_index] + 1,
                    previous_row[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]
