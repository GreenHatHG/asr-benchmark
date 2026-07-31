#!/usr/bin/env python3
"""Benchmark multiple ASR adapters with a shared audio manifest."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import sys
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from asr_metrics import edit_distance, normalize_words
from asr_reporting import ModelSummary, render_summary_table


DEFAULT_RUNS = 1
DEFAULT_WARMUP_RUNS = 0
MODEL_SPEC_SEPARATOR = "="
FACTORY_SPEC_SEPARATOR = ":"


class BenchmarkError(Exception):
    """Raised for invalid benchmark input or configuration."""


@dataclass(frozen=True)
class AudioSample:
    sample_id: str
    audio_path: Path
    reference: str
    duration_seconds: float


@dataclass(frozen=True)
class ModelAdapter:
    name: str
    transcribe: Callable[[Path], str]
    close: Callable[[], Any] | None


@dataclass(frozen=True)
class SampleFailure:
    model_name: str
    sample_id: str
    message: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对多个 ASR 适配器执行准确性和速度基准测试。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "模型格式：名称=模块:工厂函数\n"
            "工厂函数无需参数，返回可调用对象或带 transcribe(audio_path) 方法的对象。\n\n"
            "清单格式（每行一个 JSON 对象）：\n"
            '{"id":"sample-1","audio":"audio/1.wav",'
            '"reference":"参考文本","duration_seconds":3.2}\n'
            "duration_seconds 可省略；省略时仅支持自动读取 PCM WAV 时长。\n\n"
            "CER 按去空格字符计算，WER 按空格分词计算；"
            "字符准确率按 max(0, 1-CER) 计算；"
            "计算前会统一字符宽度和大小写、删除标点符号。"
        ),
    )
    parser.add_argument("manifest", type=Path, help="JSONL 音频清单路径")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="名称=模块:工厂函数",
        help="待测试的 ASR 模型，可重复传入",
    )
    parser.add_argument(
        "--runs",
        type=positive_integer,
        default=DEFAULT_RUNS,
        help=f"每条音频的计时次数，默认 {DEFAULT_RUNS}",
    )
    parser.add_argument(
        "--warmup-runs",
        type=non_negative_integer,
        default=DEFAULT_WARMUP_RUNS,
        help=f"每个模型的预热次数，默认 {DEFAULT_WARMUP_RUNS}",
    )
    return parser.parse_args()


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return parsed


def non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须是大于或等于 0 的整数")
    return parsed


def load_manifest(manifest_path: Path) -> list[AudioSample]:
    resolved_manifest = manifest_path.expanduser().resolve()
    if not resolved_manifest.is_file():
        raise BenchmarkError(f"清单文件不存在：{resolved_manifest}")

    try:
        samples: list[AudioSample] = []
        with resolved_manifest.open(encoding="utf-8") as manifest_file:
            for line_number, raw_line in enumerate(manifest_file, start=1):
                stripped_line = raw_line.strip()
                if not stripped_line:
                    continue
                samples.append(
                    parse_manifest_line(
                        stripped_line,
                        line_number,
                        resolved_manifest.parent,
                    )
                )
    except BenchmarkError:
        raise
    except (OSError, UnicodeError) as error:
        raise BenchmarkError(f"读取清单失败：{resolved_manifest}：{error}") from error

    if not samples:
        raise BenchmarkError("清单中没有可测试的音频样本")
    return samples


def parse_manifest_line(
    raw_line: str,
    line_number: int,
    manifest_directory: Path,
) -> AudioSample:
    try:
        payload = json.loads(raw_line)
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"清单第 {line_number} 行不是有效 JSON：{error}") from error

    if not isinstance(payload, dict):
        raise BenchmarkError(f"清单第 {line_number} 行必须是 JSON 对象")

    audio_value = require_string(payload, "audio", line_number)
    reference = require_string(payload, "reference", line_number, allow_empty=True)
    sample_id = payload.get("id") or f"sample-{line_number}"
    if not isinstance(sample_id, str):
        raise BenchmarkError(f"清单第 {line_number} 行的 id 必须是字符串")

    audio_path = Path(audio_value).expanduser()
    if not audio_path.is_absolute():
        audio_path = manifest_directory / audio_path
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise BenchmarkError(f"清单第 {line_number} 行的音频不存在：{audio_path}")

    duration_value = payload.get("duration_seconds")
    duration_seconds = (
        parse_duration(duration_value, line_number)
        if duration_value is not None
        else read_wav_duration(audio_path, line_number)
    )
    return AudioSample(sample_id, audio_path, reference, duration_seconds)


def require_string(
    payload: dict[str, Any],
    field_name: str,
    line_number: int,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise BenchmarkError(
            f"清单第 {line_number} 行的 {field_name} 必须是"
            f"{'字符串' if allow_empty else '非空字符串'}"
        )
    return value


def parse_duration(value: Any, line_number: int) -> float:
    is_valid_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not is_valid_number or not math.isfinite(value) or value <= 0:
        raise BenchmarkError(
            f"清单第 {line_number} 行的 duration_seconds 必须是大于 0 的数字"
        )
    return float(value)


def read_wav_duration(audio_path: Path, line_number: int) -> float:
    try:
        with wave.open(str(audio_path), "rb") as audio_file:
            frame_rate = audio_file.getframerate()
            frame_count = audio_file.getnframes()
    except (wave.Error, EOFError, OSError) as error:
        raise BenchmarkError(
            f"清单第 {line_number} 行无法自动读取音频时长；"
            "请提供 duration_seconds，或使用 PCM WAV 文件："
            f"{audio_path}"
        ) from error

    if frame_rate <= 0 or frame_count <= 0:
        raise BenchmarkError(f"清单第 {line_number} 行的 WAV 文件时长无效：{audio_path}")
    return frame_count / frame_rate


def validate_model_specs(model_specs: Sequence[str]) -> None:
    names = [parse_model_spec(spec)[0] for spec in model_specs]
    if len(names) != len(set(names)):
        raise BenchmarkError("模型名称必须唯一")


def parse_model_spec(model_spec: str) -> tuple[str, str, str]:
    if MODEL_SPEC_SEPARATOR not in model_spec:
        raise BenchmarkError(f"模型格式错误：{model_spec}")
    name, factory_spec = model_spec.split(MODEL_SPEC_SEPARATOR, maxsplit=1)
    if not name.strip() or FACTORY_SPEC_SEPARATOR not in factory_spec:
        raise BenchmarkError(f"模型格式错误：{model_spec}")
    clean_name = name.strip()
    if any(unicodedata.category(character).startswith("C") for character in clean_name):
        raise BenchmarkError(f"模型名称包含控制字符：{model_spec}")
    module_name, factory_name = factory_spec.rsplit(FACTORY_SPEC_SEPARATOR, maxsplit=1)
    if not module_name or not factory_name:
        raise BenchmarkError(f"模型格式错误：{model_spec}")
    return clean_name, module_name, factory_name


def load_adapter(model_spec: str) -> ModelAdapter:
    name, module_name, factory_name = parse_model_spec(model_spec)
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        adapter_object = factory()
    except Exception as error:
        raise BenchmarkError(f"加载模型 {name} 失败：{error}") from error

    transcribe = getattr(adapter_object, "transcribe", adapter_object)
    if not callable(transcribe):
        raise BenchmarkError(
            f"模型 {name} 的工厂函数必须返回可调用对象或带 transcribe 方法的对象"
        )
    close = getattr(adapter_object, "close", None)
    if close is not None and not callable(close):
        raise BenchmarkError(f"模型 {name} 的 close 属性必须可调用")
    return ModelAdapter(name, transcribe, close)


def benchmark_models(
    model_specs: Sequence[str],
    samples: Sequence[AudioSample],
    runs: int,
    warmup_runs: int,
) -> tuple[list[ModelSummary], list[SampleFailure]]:
    summaries: list[ModelSummary] = []
    failures: list[SampleFailure] = []
    validate_model_specs(model_specs)
    for model_spec in model_specs:
        adapter = load_adapter(model_spec)
        try:
            warm_up_adapter(adapter, samples[0], warmup_runs)
            summary, model_failures = benchmark_model(adapter, samples, runs)
            summaries.append(summary)
            failures.extend(model_failures)
        finally:
            try:
                close_adapter(adapter)
            finally:
                del adapter
                gc.collect()
    return summaries, failures


def close_adapter(adapter: ModelAdapter) -> None:
    if adapter.close is None:
        return
    try:
        adapter.close()
    except Exception as error:
        raise BenchmarkError(f"释放模型 {adapter.name} 失败：{error}") from error


def warm_up_adapter(
    adapter: ModelAdapter,
    sample: AudioSample,
    warmup_runs: int,
) -> None:
    for run_number in range(1, warmup_runs + 1):
        try:
            transcribe(adapter, sample.audio_path)
        except Exception as error:
            raise BenchmarkError(
                f"模型 {adapter.name} 第 {run_number} 次预热失败：{error}"
            ) from error


def benchmark_model(
    adapter: ModelAdapter,
    samples: Sequence[AudioSample],
    runs: int,
) -> tuple[ModelSummary, list[SampleFailure]]:
    character_edits = 0
    character_reference_units = 0
    word_edits = 0
    word_reference_units = 0
    elapsed_seconds = 0.0
    processed_audio_seconds = 0.0
    successful_samples = 0
    failures: list[SampleFailure] = []

    for sample in samples:
        try:
            hypothesis, sample_elapsed = run_sample(adapter, sample, runs)
        except Exception as error:
            failures.append(SampleFailure(adapter.name, sample.sample_id, str(error)))
            continue

        reference_words = normalize_words(sample.reference)
        hypothesis_words = normalize_words(hypothesis)
        reference_characters = list("".join(reference_words))
        hypothesis_characters = list("".join(hypothesis_words))
        character_edits += edit_distance(reference_characters, hypothesis_characters)
        character_reference_units += len(reference_characters)
        word_edits += edit_distance(reference_words, hypothesis_words)
        word_reference_units += len(reference_words)
        elapsed_seconds += sample_elapsed
        processed_audio_seconds += sample.duration_seconds * runs
        successful_samples += 1

    if successful_samples == 0:
        return empty_summary(adapter.name, len(samples)), failures

    invocation_count = successful_samples * runs
    return (
        ModelSummary(
            name=adapter.name,
            total_samples=len(samples),
            successful_samples=successful_samples,
            failed_samples=len(samples) - successful_samples,
            character_error_rate=calculate_error_rate(
                character_edits, character_reference_units
            ),
            word_error_rate=calculate_error_rate(word_edits, word_reference_units),
            average_latency_seconds=elapsed_seconds / invocation_count,
            real_time_factor=elapsed_seconds / processed_audio_seconds,
        ),
        failures,
    )


def calculate_error_rate(edit_count: int, reference_units: int) -> float:
    if reference_units > 0:
        return edit_count / reference_units
    return 0.0 if edit_count == 0 else math.inf


def run_sample(
    adapter: ModelAdapter,
    sample: AudioSample,
    runs: int,
) -> tuple[str, float]:
    first_hypothesis = ""
    total_elapsed = 0.0
    for run_index in range(runs):
        started_at = time.perf_counter()
        hypothesis = transcribe(adapter, sample.audio_path)
        total_elapsed += time.perf_counter() - started_at
        if run_index == 0:
            first_hypothesis = hypothesis
    return first_hypothesis, total_elapsed


def transcribe(adapter: ModelAdapter, audio_path: Path) -> str:
    result = adapter.transcribe(audio_path)
    if not isinstance(result, str):
        raise TypeError(f"识别结果必须是字符串，实际类型为 {type(result).__name__}")
    return result


def empty_summary(model_name: str, total_samples: int) -> ModelSummary:
    return ModelSummary(
        name=model_name,
        total_samples=total_samples,
        successful_samples=0,
        failed_samples=total_samples,
        character_error_rate=math.nan,
        word_error_rate=math.nan,
        average_latency_seconds=math.nan,
        real_time_factor=math.nan,
    )


def print_failures(failures: Sequence[SampleFailure]) -> None:
    if not failures:
        return
    print("\n失败详情：", file=sys.stderr)
    for failure in failures:
        print(
            f"- 模型={failure.model_name} 样本={failure.sample_id} 错误={failure.message}",
            file=sys.stderr,
        )


def main() -> int:
    arguments = parse_arguments()
    try:
        samples = load_manifest(arguments.manifest)
        summaries, failures = benchmark_models(
            arguments.model,
            samples,
            arguments.runs,
            arguments.warmup_runs,
        )
    except BenchmarkError as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2

    print(render_summary_table(summaries))
    print_failures(failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
