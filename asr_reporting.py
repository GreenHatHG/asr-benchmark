"""Benchmark result data and terminal rendering for ASR benchmarks."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

TABLE_COLUMNS = (
    "模型",
    "设备",
    "样本",
    "成功",
    "失败",
    "加载耗时(s)",
    "平均耗时(s)",
    "RTF",
    "音频速度",
)


@dataclass(frozen=True)
class ModelSummary:
    name: str
    device: str
    total_samples: int
    successful_samples: int
    failed_samples: int
    load_latency_seconds: float
    average_latency_seconds: float
    real_time_factor: float

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
        summary.device,
        str(summary.total_samples),
        str(summary.successful_samples),
        str(summary.failed_samples),
        format_seconds(summary.load_latency_seconds),
        format_seconds(summary.average_latency_seconds),
        format_ratio(summary.real_time_factor),
        format_speed(summary.audio_speed),
    )


@dataclass(frozen=True)
class RecognitionResult:
    model_name: str
    sample_id: str
    reference: str
    hypothesis: str
    raw_hypothesis: str | None = None


def render_recognition_results(results: Sequence[RecognitionResult]) -> str:
    lines = ["识别结果："]
    if not results:
        lines.append("没有成功识别的样本。")
        return "\n".join(lines)

    for result in results:
        lines.extend(
            (
                "",
                f"模型：{result.model_name}",
                f"样本：{result.sample_id}",
                f"参考文本：{result.reference!r}",
            )
        )
        if result.raw_hypothesis is not None:
            lines.append(f"标点前原文：{result.raw_hypothesis!r}")
        lines.append(f"识别文本：{result.hypothesis!r}")
    return "\n".join(lines)


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
