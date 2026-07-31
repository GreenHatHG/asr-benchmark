"""Summary data and terminal table rendering for ASR benchmarks."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Sequence


TABLE_COLUMNS = (
    "模型",
    "样本",
    "成功",
    "失败",
    "字符准确率",
    "CER",
    "WER",
    "平均耗时(s)",
    "RTF",
    "音频速度",
)


@dataclass(frozen=True)
class ModelSummary:
    name: str
    total_samples: int
    successful_samples: int
    failed_samples: int
    character_error_rate: float
    word_error_rate: float
    average_latency_seconds: float
    real_time_factor: float

    @property
    def character_accuracy(self) -> float:
        if math.isnan(self.character_error_rate):
            return math.nan
        return max(0.0, 1.0 - self.character_error_rate)

    @property
    def audio_speed(self) -> float:
        if self.real_time_factor == 0:
            return math.inf
        return 1.0 / self.real_time_factor


def render_summary_table(summaries: Sequence[ModelSummary]) -> str:
    rows = [summary_to_row(summary) for summary in summaries]
    widths = [
        max(display_width(str(row[column_index])) for row in (TABLE_COLUMNS, *rows))
        for column_index in range(len(TABLE_COLUMNS))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    rendered_rows = [separator, render_row(TABLE_COLUMNS, widths), separator]
    rendered_rows.extend(render_row(row, widths) for row in rows)
    rendered_rows.append(separator)
    return "\n".join(rendered_rows)


def summary_to_row(summary: ModelSummary) -> tuple[str, ...]:
    return (
        summary.name,
        str(summary.total_samples),
        str(summary.successful_samples),
        str(summary.failed_samples),
        format_percentage(summary.character_accuracy),
        format_ratio(summary.character_error_rate),
        format_ratio(summary.word_error_rate),
        format_seconds(summary.average_latency_seconds),
        format_ratio(summary.real_time_factor),
        format_speed(summary.audio_speed),
    )


def render_row(values: Sequence[str], widths: Sequence[int]) -> str:
    cells = [pad_cell(str(value), width) for value, width in zip(values, widths)]
    return "| " + " | ".join(cells) + " |"


def pad_cell(value: str, width: int) -> str:
    return value + " " * (width - display_width(value))


def display_width(value: str) -> int:
    return sum(
        0
        if unicodedata.combining(character)
        else 2
        if unicodedata.east_asian_width(character) in {"W", "F"}
        else 1
        for character in value
    )


def format_percentage(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.2%}"


def format_ratio(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.4f}"


def format_seconds(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.3f}"


def format_speed(value: float) -> str:
    if math.isnan(value):
        return "-"
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}x"
