"""Benchmark result data and terminal rendering for ASR benchmarks."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
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


def render_summary_table(summaries: Sequence[ModelSummary]) -> str:  # noqa
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
    elapsed_seconds: float | None = None
    runs: int | None = None
    uses_previous_timing: bool = False


def calculate_timing_metrics(
    results: Sequence[RecognitionResult],
    duration_by_sample: Mapping[str, float],
) -> tuple[float, float]:
    """按每条结果实际运行的次数计算平均耗时和实时率。"""

    total_elapsed_seconds = 0.0
    total_runs = 0
    processed_audio_seconds = 0.0
    for result in results:
        if result.elapsed_seconds is None or result.runs is None:
            return math.nan, math.nan
        total_elapsed_seconds += result.elapsed_seconds
        total_runs += result.runs
        processed_audio_seconds += duration_by_sample[result.sample_id] * result.runs
    return (
        total_elapsed_seconds / total_runs,
        total_elapsed_seconds / processed_audio_seconds,
    )


def render_recognition_results(results: Sequence[RecognitionResult]) -> str:  # noqa
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


@dataclass(frozen=True)
class ModelAccuracy:
    name: str
    successful_samples: int
    total_samples: int
    compared_samples: int
    exact_matches: int
    error_rate: float | None


def calculate_model_accuracy(
    summary: ModelSummary,
    results: Sequence[RecognitionResult],
) -> ModelAccuracy:
    compared_samples = 0
    exact_matches = 0
    total_reference_characters = 0
    total_errors = 0
    for result in results:
        score = calculate_character_errors(result.reference, result.hypothesis)
        if score is None:
            continue
        errors, reference_characters = score
        compared_samples += 1
        exact_matches += errors == 0
        total_errors += errors
        total_reference_characters += reference_characters

    error_rate = (
        total_errors / total_reference_characters
        if total_reference_characters
        else None
    )
    return ModelAccuracy(
        summary.name,
        summary.successful_samples,
        summary.total_samples,
        compared_samples,
        exact_matches,
        error_rate,
    )


def group_results_by_sample(
    results: Sequence[RecognitionResult],
) -> dict[str, list[RecognitionResult]]:
    grouped_results: dict[str, list[RecognitionResult]] = {}
    for result in results:
        grouped_results.setdefault(result.sample_id, []).append(result)
    return grouped_results


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    """计算忽略大小写、空白和标点后的字符错误率。"""

    score = calculate_character_errors(reference, hypothesis)
    if score is None:
        return None
    errors, reference_characters = score
    return errors / reference_characters


def calculate_character_errors(
    reference: str,
    hypothesis: str,
) -> tuple[int, int] | None:
    normalized_reference = normalize_for_comparison(reference)
    if not normalized_reference:
        return None
    normalized_hypothesis = normalize_for_comparison(hypothesis)
    return (
        edit_distance(normalized_reference, normalized_hypothesis),
        len(normalized_reference),
    )


def normalize_for_comparison(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def edit_distance(reference: str, hypothesis: str) -> int:
    """返回把参考文本变成识别文本所需的最少字符编辑次数。"""

    previous_row = list(range(len(hypothesis) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current_row = [reference_index]
        for hypothesis_index, hypothesis_character in enumerate(hypothesis, start=1):
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[hypothesis_index] + 1,
                    previous_row[hypothesis_index - 1]
                    + (reference_character != hypothesis_character),
                )
            )
        previous_row = current_row
    return previous_row[-1]


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


def format_error_rate(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2%}"


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
