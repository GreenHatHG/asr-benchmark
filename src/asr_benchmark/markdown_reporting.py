"""将 ASR 识别效果对比写入 Markdown 报告。"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .reporting import (
    ModelAccuracy,
    ModelSummary,
    RecognitionResult,
    calculate_model_accuracy,
    character_error_rate,
    format_error_rate,
    format_seconds,
    group_results_by_sample,
)


class MarkdownReportError(Exception):
    """表示 Markdown 识别对比报告无法保存。"""


@dataclass(frozen=True)
class ModelIdentity:
    series: str
    version: str


@dataclass(frozen=True)
class ReportSample:
    sample_id: str
    reference: str
    audio_path: Path | None = None
    duration_seconds: float | None = None


class ReportSampleLike(Protocol):
    @property
    def sample_id(self) -> str: ...

    @property
    def reference(self) -> str: ...

    @property
    def audio_path(self) -> Path | None: ...

    @property
    def duration_seconds(self) -> float | None: ...


class ReportFailureLike(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def sample_id(self) -> str: ...

    @property
    def message(self) -> str: ...


def report_markdown_path_for_config(config_path: Path) -> Path:
    resolved_config = config_path.expanduser().resolve()
    return resolved_config.with_name(f"{resolved_config.stem}.report.md")


def write_markdown_report(
    summaries: Sequence[ModelSummary],
    results: Sequence[RecognitionResult],
    output_path: Path,
    *,
    samples: Sequence[ReportSampleLike] | None = None,
    failures: Sequence[ReportFailureLike] = (),
) -> None:
    report = render_markdown_report(
        summaries,
        results,
        samples=samples,
        failures=failures,
    )
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(report)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    except OSError as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise MarkdownReportError(
            f"保存识别对比报告失败：{output_path}：{error}"
        ) from error


def render_markdown_report(
    summaries: Sequence[ModelSummary],
    results: Sequence[RecognitionResult],
    *,
    samples: Sequence[ReportSampleLike] | None = None,
    failures: Sequence[ReportFailureLike] = (),
) -> str:
    lines = [
        "# ASR 识别效果对比",
        "",
        "字符错误率越低越好；计算时忽略大小写、空白和标点。",
        "",
        "## 模型汇总",
        "",
    ]
    lines.extend(render_summary_table(summaries, results))
    report_samples = tuple(samples) if samples is not None else derive_samples(results)
    if not report_samples:
        lines.extend(("", "没有成功识别的样本。", ""))
        return "\n".join(lines)

    for sample in report_samples:
        lines.extend(render_sample_section(sample, summaries, results, failures))
    lines.append("")
    return "\n".join(lines)


def render_summary_table(
    summaries: Sequence[ModelSummary],
    results: Sequence[RecognitionResult],
) -> list[str]:
    lines = [
        "| 系列 | 模型版本 | 成功/样本 | 完全一致 | 字符错误率 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    grouped_accuracies = group_accuracies_by_series(
        calculate_accuracies(summaries, results)
    )
    for series, accuracies in grouped_accuracies.items():
        for accuracy_index, accuracy in enumerate(accuracies):
            identity = model_identity(accuracy.name)
            lines.append(
                "| "
                + " | ".join(
                    (
                        escape_markdown_cell(series) if accuracy_index == 0 else "",
                        escape_markdown_cell(identity.version),
                        f"{accuracy.successful_samples}/{accuracy.total_samples}",
                        format_exact_matches(accuracy),
                        format_error_rate(accuracy.error_rate),
                    )
                )
                + " |"
            )
    return lines


def render_sample_section(
    sample: ReportSampleLike,
    summaries: Sequence[ModelSummary],
    results: Sequence[RecognitionResult],
    failures: Sequence[ReportFailureLike],
) -> list[str]:
    lines = [
        "",
        f"## 样本：{escape_markdown_text(sample.sample_id)}",
        "",
        render_sample_metadata(sample),
        "",
        "| 系列 | 模型版本 | 字符错误率 | 运行次数 | 总耗时(s) | 耗时说明 | 识别文本 |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
        f"| 参考 | 参考文本 | - | - | - | - | {escape_markdown_cell(display_text(sample.reference))} |",
    ]
    results_by_model = {
        result.model_name: result
        for result in results
        if result.sample_id == sample.sample_id
    }
    failures_by_model = sample_failures_by_model(sample.sample_id, failures)
    for series, series_summaries in group_summaries_by_series(summaries).items():
        for summary_index, summary in enumerate(series_summaries):
            identity = model_identity(summary.name)
            result = results_by_model.get(summary.name)
            failure = failures_by_model.get(summary.name)
            error_rate = (
                character_error_rate(sample.reference, result.hypothesis)
                if result is not None
                else None
            )
            recognized_text = (
                result.hypothesis
                if result is not None
                else f"识别失败：{failure.message}"
                if failure is not None
                else "没有识别结果"
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        escape_markdown_cell(series) if summary_index == 0 else "",
                        escape_markdown_cell(identity.version),
                        format_error_rate(error_rate),
                        str(result.runs)
                        if result is not None and result.runs is not None
                        else "-",
                        format_seconds(result.elapsed_seconds)
                        if result is not None and result.elapsed_seconds is not None
                        else "-",
                        "上次耗时"
                        if result is not None and result.uses_previous_timing
                        else "-",
                        escape_markdown_cell(display_text(recognized_text)),
                    )
                )
                + " |"
            )
    return lines


def render_sample_metadata(sample: ReportSampleLike) -> str:
    metadata: list[str] = []
    if sample.audio_path is not None:
        metadata.append(f"音频文件：{escape_markdown_text(sample.audio_path.name)}")
    if sample.duration_seconds is not None:
        metadata.append(f"时长：{sample.duration_seconds:.2f} 秒")
    metadata.append(f"参考文本长度：{len(sample.reference)} 个字符")
    return " · ".join(metadata)


def derive_samples(results: Sequence[RecognitionResult]) -> tuple[ReportSample, ...]:
    return tuple(
        ReportSample(sample_id, sample_results[0].reference)
        for sample_id, sample_results in group_results_by_sample(results).items()
    )


def calculate_accuracies(
    summaries: Sequence[ModelSummary],
    results: Sequence[RecognitionResult],
) -> list[ModelAccuracy]:
    results_by_model: dict[str, list[RecognitionResult]] = {}
    for result in results:
        results_by_model.setdefault(result.model_name, []).append(result)
    return [
        calculate_model_accuracy(summary, results_by_model.get(summary.name, ()))
        for summary in summaries
    ]


def group_accuracies_by_series(
    accuracies: Sequence[ModelAccuracy],
) -> dict[str, list[ModelAccuracy]]:
    grouped_accuracies: dict[str, list[ModelAccuracy]] = {}
    for accuracy in accuracies:
        series = model_identity(accuracy.name).series
        grouped_accuracies.setdefault(series, []).append(accuracy)
    return grouped_accuracies


def group_summaries_by_series(
    summaries: Sequence[ModelSummary],
) -> dict[str, list[ModelSummary]]:
    grouped_summaries: dict[str, list[ModelSummary]] = {}
    for summary in summaries:
        series = model_identity(summary.name).series
        grouped_summaries.setdefault(series, []).append(summary)
    return grouped_summaries


def sample_failures_by_model(
    sample_id: str,
    failures: Sequence[ReportFailureLike],
) -> dict[str, ReportFailureLike]:
    load_failures = {
        failure.model_name: failure
        for failure in failures
        if failure.sample_id == "<load>"
    }
    sample_failures = {
        failure.model_name: failure
        for failure in failures
        if failure.sample_id == sample_id
    }
    return load_failures | sample_failures


def model_identity(model_name: str) -> ModelIdentity:
    if "Qwen3-ASR-" in model_name:
        version = model_name.split("Qwen3-ASR-", maxsplit=1)[1]
        version = version.replace("-hf (", " · ").removesuffix(")")
        return ModelIdentity("Qwen3-ASR", version)
    if "FireRedASR2-" in model_name:
        version = model_name.split("FireRedASR2-", maxsplit=1)[1]
        version = version.replace(") + ", " · ").replace(" (", " · ")
        return ModelIdentity("FireRedASR2", version.removesuffix(")"))
    if "Fun-ASR-Nano" in model_name:
        if "官方原始权重" in model_name:
            version = "官方原始权重"
        elif "(MLX " in model_name:
            version = "MLX " + model_name.rsplit("(MLX ", maxsplit=1)[1].removesuffix(
                ")"
            )
        else:
            version = model_name.split("Fun-ASR-Nano", maxsplit=1)[1].strip(" -()")
        return ModelIdentity("Fun-ASR-Nano", version)
    if model_name.startswith("LLM-ASR"):
        version = model_name.removeprefix("LLM-ASR").strip(" ()") or "远程模型"
        return ModelIdentity("LLM-ASR", version)
    if model_name == "Doubao-IME-ASR":
        return ModelIdentity("Doubao", "IME-ASR")
    return ModelIdentity(model_name, "默认版本")


def format_exact_matches(accuracy: ModelAccuracy) -> str:
    if not accuracy.compared_samples:
        return "-"
    return f"{accuracy.exact_matches}/{accuracy.compared_samples}"


def display_text(value: str) -> str:
    return value if value else "（空文本）"


def escape_markdown_cell(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r\n", "<br>")
        .replace("\r", "<br>")
        .replace("\n", "<br>")
    )


def escape_markdown_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("#", "\\#").replace("\n", " ")
