#!/usr/bin/env python3
"""按照一个可复用的 TOML 配置比较多个 ASR（自动语音识别）模型。

配置文件保存模型、运行次数和音频样本。脚本会依次加载模型、执行预热和
正式识别，然后输出参考文本、识别文本、模型加载耗时和处理音频的速度。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import math
import sys
import time
import tomllib
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

from .audio_normalization import AudioNormalizationError, normalize_audio_files
from .adapters.doubao import DOUBAO_ASR_MODEL_NAME, create_doubao_asr
from .adapters.firered_onnx import (
    create_fireredasr2_aed_int8,
    create_fireredasr2_ctc_int8,
)
from .adapters.fireredasr2 import create_fireredasr2_aed
from .adapters.fireredpunc import create_punctuated_adapter
from .adapters.fun_asr import (
    create_fun_asr_nano_mlx_4bit,
    create_fun_asr_nano_mlx_8bit,
    create_fun_asr_nano_official,
)
from .adapters.llm_asr import LLM_ASR_MODEL_NAME, LlmAsrConfig, create_llm_asr
from .adapters.qwen3_asr import create_qwen3_asr_0_6b, create_qwen3_asr_1_7b
from .reporting import (
    ModelSummary,
    RecognitionResult,
    render_recognition_results,
    render_summary_table,
)
from .transcription_cache import (
    CachedModel,
    CachedTranscription,
    TranscriptionCache,
    TranscriptionCacheError,
    cache_path_for_config,
)

DEFAULT_RUNS = 1
DEFAULT_WARMUP_RUNS = 0
logger = logging.getLogger(__name__)
MODEL_FACTORIES: dict[str, Callable[[], Any]] = {
    "Qwen3-ASR-0.6B": create_qwen3_asr_0_6b,
    "Qwen3-ASR-1.7B": create_qwen3_asr_1_7b,
    "FireRedASR2-AED": partial(create_punctuated_adapter, create_fireredasr2_aed),
    "FireRedASR2-CTC-INT8": partial(
        create_punctuated_adapter,
        create_fireredasr2_ctc_int8,
    ),
    "FireRedASR2-AED-INT8": partial(
        create_punctuated_adapter,
        create_fireredasr2_aed_int8,
    ),
    "Fun-ASR-Nano-2512": create_fun_asr_nano_official,
    "Fun-ASR-Nano-2512-MLX-8bit": create_fun_asr_nano_mlx_8bit,
    "Fun-ASR-Nano-2512-MLX-4bit": create_fun_asr_nano_mlx_4bit,
    DOUBAO_ASR_MODEL_NAME: create_doubao_asr,
}
SUPPORTED_MODEL_NAMES = (*MODEL_FACTORIES, LLM_ASR_MODEL_NAME)


class BenchmarkError(Exception):
    """表示配置内容或模型适配器不符合测试要求。"""


@dataclass(frozen=True)
class AudioSample:
    """保存一条待识别音频及其参考文本和实际时长。"""

    sample_id: str
    audio_path: Path
    reference: str
    duration_seconds: float


@dataclass(frozen=True)
class BenchmarkConfig:
    """保存一次完整基准测试需要的模型、音频样本和运行次数。"""

    model_names: tuple[str, ...]
    samples: tuple[AudioSample, ...]
    runs: int
    warmup_runs: int
    llm_config: LlmAsrConfig | None = None


@dataclass(frozen=True)
class ModelAdapter:
    """统一不同模型的调用方式，并记录释放资源所需的函数。"""

    name: str
    transcribe: Callable[[Path], str]
    close: Callable[[], Any] | None
    device: str = "unknown"
    get_last_raw_text: Callable[[], str | None] | None = None


@dataclass(frozen=True)
class SampleFailure:
    """记录某个模型处理某条音频时发生的错误。"""

    model_name: str
    sample_id: str
    message: str


@dataclass(frozen=True)
class ActiveModelCache:
    """把转写缓存和当前使用的配置模型名称放在一起。"""

    storage: TranscriptionCache
    model_name: str


def parse_arguments() -> argparse.Namespace:
    """读取命令行参数，并在参数格式不正确时由 argparse 显示错误。"""

    parser = argparse.ArgumentParser(
        description="按照 TOML 配置执行 ASR 识别文本和速度基准测试。",
    )
    parser.add_argument("config", type=Path, help="TOML 基准测试配置文件路径")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用缓存中已成功的转写，只运行尚未完成的模型和样本",
    )
    parser.add_argument(
        "--normalize-audio",
        action="store_true",
        help="将配置中的 WAV 原地替换为 16kHz、16 位、单声道 PCM WAV",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> BenchmarkConfig:
    """读取并校验一个 TOML 配置文件，返回完整的基准测试配置。

    ``runs`` 和 ``warmup_runs`` 可以省略，此时分别使用 1 和 0。配置无效
    或文件无法读取时会抛出 ``BenchmarkError``，由入口函数统一显示错误。
    """

    resolved_config = config_path.expanduser().resolve()
    if not resolved_config.is_file():
        raise BenchmarkError(f"配置文件不存在：{resolved_config}")

    try:
        with resolved_config.open("rb") as config_file:
            payload = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as error:
        raise BenchmarkError(f"配置文件不是有效 TOML：{error}") from error
    except OSError as error:
        raise BenchmarkError(f"读取配置文件失败：{resolved_config}：{error}") from error

    model_names = parse_model_names(payload.get("models"))
    runs = parse_run_count(payload.get("runs", DEFAULT_RUNS), "runs", minimum=1)
    warmup_runs = parse_run_count(
        payload.get("warmup_runs", DEFAULT_WARMUP_RUNS),
        "warmup_runs",
        minimum=0,
    )
    llm_config = parse_llm_config(
        payload.get("llm"),
        required=LLM_ASR_MODEL_NAME in model_names,
    )
    samples_value = payload.get("samples")
    if not isinstance(samples_value, list) or not samples_value:
        raise BenchmarkError("配置中的 samples 必须是非空数组")
    samples = tuple(
        parse_audio_sample(sample_value, sample_number, resolved_config.parent)
        for sample_number, sample_value in enumerate(samples_value, start=1)
    )
    if len(samples) != len({sample.sample_id for sample in samples}):
        raise BenchmarkError("配置中的 samples id 必须唯一")
    return BenchmarkConfig(model_names, samples, runs, warmup_runs, llm_config)


def build_configuration_fingerprint(
    config_path: Path,
    samples: Sequence[AudioSample],
) -> str:
    """计算配置和音频内容的摘要，用于拒绝不匹配的恢复缓存。"""

    fingerprint = hashlib.sha256()
    paths = (
        config_path.expanduser().resolve(),
        *(sample.audio_path for sample in samples),
    )
    for path in paths:
        file_fingerprint = hashlib.sha256()
        try:
            with path.open("rb") as input_file:
                while chunk := input_file.read(1024 * 1024):
                    file_fingerprint.update(chunk)
        except OSError as error:
            raise BenchmarkError(
                f"计算转写缓存标识时无法读取文件：{path}：{error}"
            ) from error
        # 每个文件先生成固定 32 字节的摘要，避免二进制内容与分隔符混淆。
        fingerprint.update(file_fingerprint.digest())
    return fingerprint.hexdigest()


def parse_model_names(value: Any) -> tuple[str, ...]:
    """读取配置中的模型名称数组，并确认所有名称已在程序中注册。"""

    if not isinstance(value, list) or not value:
        raise BenchmarkError("配置中的 models 必须是非空字符串数组")
    if any(not isinstance(model_name, str) or not model_name for model_name in value):
        raise BenchmarkError("配置中的 models 必须是非空字符串数组")
    model_names = tuple(value)
    validate_model_names(model_names)
    return model_names


def parse_run_count(value: Any, field_name: str, *, minimum: int) -> int:
    """读取配置中的运行次数，并拒绝布尔值和小于允许范围的整数。"""

    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise BenchmarkError(f"配置中的 {field_name} 必须是大于或等于 {minimum} 的整数")
    return value


def parse_llm_config(value: Any, *, required: bool) -> LlmAsrConfig | None:
    """仅在选择 ``LLM-ASR`` 时读取并校验对应的远程接口配置。"""

    if not required:
        return None
    if not isinstance(value, dict):
        raise BenchmarkError("选择 LLM-ASR 时，配置中必须提供 [llm] 表")
    model = require_llm_config_string(value, "model")
    base_url = require_llm_config_string(value, "base_url")
    return LlmAsrConfig(model=model, base_url=base_url)


def require_llm_config_string(payload: dict[str, Any], field_name: str) -> str:
    """读取 ``[llm]`` 中的非空字符串，并去掉两端空白。"""

    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"配置中的 llm.{field_name} 必须是非空字符串")
    return value.strip()


def parse_audio_sample(
    payload: Any,
    sample_number: int,
    config_directory: Path,
) -> AudioSample:
    """解析配置中的一条音频样本，并补全音频的绝对路径和时长。

    相对音频路径以配置文件所在目录为起点。未提供
    ``duration_seconds`` 时，会尝试直接读取 PCM WAV 文件的时长。
    """

    if not isinstance(payload, dict):
        raise BenchmarkError(f"配置中的第 {sample_number} 条音频样本必须是 TOML 表")

    audio_value = require_string(payload, "audio", sample_number)
    reference = require_string(payload, "reference", sample_number, allow_empty=True)
    sample_id = payload.get("id") or f"sample-{sample_number}"
    if not isinstance(sample_id, str):
        raise BenchmarkError(f"配置中第 {sample_number} 条音频样本的 id 必须是字符串")

    # 以配置文件所在目录解析相对路径，使脚本可以从任意工作目录启动。
    audio_path = Path(audio_value).expanduser()
    if not audio_path.is_absolute():
        audio_path = config_directory / audio_path
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise BenchmarkError(
            f"配置中第 {sample_number} 条音频样本的音频不存在：{audio_path}"
        )

    duration_value = payload.get("duration_seconds")
    duration_seconds = (
        parse_duration(duration_value, sample_number)
        if duration_value is not None
        else read_wav_duration(audio_path, sample_number)
    )
    return AudioSample(sample_id, audio_path, reference, duration_seconds)


def require_string(
    payload: dict[str, Any],
    field_name: str,
    sample_number: int,
    *,
    allow_empty: bool = False,
) -> str:
    """读取配置中的字符串字段，并按 ``allow_empty`` 决定是否允许空文本。"""

    value = payload.get(field_name)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise BenchmarkError(
            f"配置中第 {sample_number} 条音频样本的 {field_name} 必须是"
            f"{'字符串' if allow_empty else '非空字符串'}"
        )
    return value


def parse_duration(value: Any, sample_number: int) -> float:
    """校验配置提供的音频时长，并统一转换成浮点秒数。"""

    # Python 中 bool 是 int 的子类，但 true/false 显然不是合法的音频时长。
    is_valid_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not is_valid_number or not math.isfinite(value) or value <= 0:
        raise BenchmarkError(
            f"配置中第 {sample_number} 条音频样本的 duration_seconds "
            "必须是大于 0 的数字"
        )
    return float(value)


def read_wav_duration(audio_path: Path, sample_number: int) -> float:
    """根据 PCM WAV 的采样帧数量和采样率计算音频时长。

    文件格式不受 ``wave`` 支持或文件内容损坏时，会提示调用方在配置中
    明确填写 ``duration_seconds``。
    """

    try:
        with wave.open(str(audio_path), "rb") as audio_file:
            frame_rate = audio_file.getframerate()
            frame_count = audio_file.getnframes()
    except (wave.Error, EOFError, OSError) as error:
        raise BenchmarkError(
            f"配置中第 {sample_number} 条音频样本无法自动读取音频时长；"
            "请提供 duration_seconds，或使用 PCM WAV 文件："
            f"{audio_path}"
        ) from error

    if frame_rate <= 0 or frame_count <= 0:
        raise BenchmarkError(
            f"配置中第 {sample_number} 条音频样本的 WAV 文件时长无效：{audio_path}"
        )
    return frame_count / frame_rate


def validate_model_names(model_names: Sequence[str]) -> None:
    """确认模型名称都已注册且没有重复，避免产生含义相同的汇总结果。"""

    unsupported_names = [
        model_name
        for model_name in model_names
        if model_name not in SUPPORTED_MODEL_NAMES
    ]
    if unsupported_names:
        raise BenchmarkError(
            f"不支持的模型：{', '.join(unsupported_names)}；"
            f"可选模型：{', '.join(SUPPORTED_MODEL_NAMES)}"
        )
    if len(model_names) != len(set(model_names)):
        raise BenchmarkError("模型名称必须唯一")


def load_adapter(
    model_name: str,
    llm_config: LlmAsrConfig | None = None,
) -> ModelAdapter:
    """根据已注册的模型名称创建一个统一的 ``ModelAdapter``。

    模型对象本身可以直接调用，也可以提供 ``transcribe`` 方法；可选的
    ``close`` 方法用于释放资源，``name`` 和 ``device`` 字符串分别用于说明
    报告中的完整模型名称和实际运行设备。
    """

    if model_name == LLM_ASR_MODEL_NAME:
        if llm_config is None:
            raise BenchmarkError("加载 LLM-ASR 时缺少 [llm] 配置")
        factory: Callable[[], Any] = partial(create_llm_asr, llm_config)
    else:
        model_factory = MODEL_FACTORIES.get(model_name)
        if model_factory is None:
            raise BenchmarkError(f"不支持的模型：{model_name}")
        factory = model_factory

    try:
        adapter_object = factory()
    except Exception as error:
        raise BenchmarkError(f"加载模型 {model_name} 失败：{error}") from error

    # 优先采用明确的 transcribe 方法；普通可调用对象也能作为最简适配器。
    transcribe_value = getattr(adapter_object, "transcribe", adapter_object)
    if not callable(transcribe_value):
        raise BenchmarkError(
            f"模型 {model_name} 的工厂函数必须返回可调用对象或带 transcribe 方法的对象"
        )
    transcribe_callable = cast(Callable[[Path], str], transcribe_value)
    close = getattr(adapter_object, "close", None)
    if close is not None and not callable(close):
        raise BenchmarkError(f"模型 {model_name} 的 close 属性必须可调用")
    report_name = getattr(adapter_object, "name", model_name)
    if not isinstance(report_name, str) or not report_name.strip():
        raise BenchmarkError(f"模型 {model_name} 的 name 属性必须是非空字符串")
    device = getattr(adapter_object, "device", "unknown")
    if not isinstance(device, str) or not device.strip():
        raise BenchmarkError(f"模型 {model_name} 的 device 属性必须是非空字符串")
    get_last_raw_text_value = getattr(adapter_object, "get_last_raw_text", None)
    if get_last_raw_text_value is not None and not callable(get_last_raw_text_value):
        raise BenchmarkError(f"模型 {model_name} 的 get_last_raw_text 属性必须可调用")
    get_last_raw_text_callable = cast(
        Callable[[], str | None] | None,
        get_last_raw_text_value,
    )
    return ModelAdapter(
        report_name.strip(),
        transcribe_callable,
        close,
        device.strip(),
        get_last_raw_text_callable,
    )


def benchmark_models(  # pylint: disable=too-many-arguments
    model_names: Sequence[str],
    samples: Sequence[AudioSample],
    runs: int,
    warmup_runs: int,
    llm_config: LlmAsrConfig | None = None,
    *,
    cache: TranscriptionCache | None = None,
) -> tuple[list[ModelSummary], list[RecognitionResult], list[SampleFailure]]:
    """依次测试所有模型，返回识别结果、速度汇总和失败样本记录。

    每个模型只加载一次。完成预热和正式测试后，无论中途是否出错都会
    调用其 ``close`` 方法，并要求 Python 回收不再使用的模型对象。
    """

    summaries: list[ModelSummary] = []
    recognition_results: list[RecognitionResult] = []
    failures: list[SampleFailure] = []
    validate_model_names(model_names)
    logger.info(
        "开始基准测试：模型数=%d，样本数=%d，正式运行次数=%d，预热次数=%d",
        len(model_names),
        len(samples),
        runs,
        warmup_runs,
    )
    for model_index, model_name in enumerate(model_names, start=1):
        logger.info(
            "开始处理模型 [%d/%d]：%s",
            model_index,
            len(model_names),
            model_name,
        )
        summary, model_results, model_failures = benchmark_registered_model(
            model_name,
            samples,
            runs,
            warmup_runs,
            llm_config,
            cache=cache,
        )
        summaries.append(summary)
        recognition_results.extend(model_results)
        failures.extend(model_failures)
        logger.info(
            "模型处理完成 [%d/%d]：%s，成功样本=%d，失败样本=%d",
            model_index,
            len(model_names),
            model_name,
            summary.successful_samples,
            summary.failed_samples,
        )
    return summaries, recognition_results, failures


def benchmark_registered_model(  # pylint: disable=too-many-arguments
    model_name: str,
    samples: Sequence[AudioSample],
    runs: int,
    warmup_runs: int,
    llm_config: LlmAsrConfig | None,
    *,
    cache: TranscriptionCache | None,
) -> tuple[ModelSummary, list[RecognitionResult], list[SampleFailure]]:
    """读取缓存或加载一个配置模型，并完成该模型的所有样本。"""

    if cache is not None:
        cached_outcome = restore_completed_model(
            cache.get_model(model_name),
            samples,
            runs,
        )
        if cached_outcome is not None:
            summary, recognition_results = cached_outcome
            logger.info("模型全部样本命中缓存，跳过加载和识别：%s", model_name)
            return summary, recognition_results, []

    # 加载耗时单独统计，不会混入后面的单次识别耗时。
    logger.info("正在加载模型：%s", model_name)
    load_started_at = time.perf_counter()
    try:
        adapter = load_adapter(model_name, llm_config)
    except Exception as error:  # noqa: BLE001 - one model failure must not stop the run
        load_latency_seconds = time.perf_counter() - load_started_at
        logger.exception(
            "模型加载失败：%s，耗时=%.2f 秒",
            model_name,
            load_latency_seconds,
        )
        return (
            empty_summary(
                model_name,
                "unavailable",
                len(samples),
                load_latency_seconds,
            ),
            [],
            [SampleFailure(model_name, "<load>", str(error))],
        )

    load_latency_seconds = time.perf_counter() - load_started_at
    logger.info(
        "模型加载完成：%s，报告名称=%s，设备=%s，耗时=%.2f 秒",
        model_name,
        adapter.name,
        adapter.device,
        load_latency_seconds,
    )
    try:
        active_cache = None
        if cache is not None:
            cache.set_model_metadata(
                model_name,
                adapter.name,
                adapter.device,
                load_latency_seconds,
            )
            active_cache = ActiveModelCache(cache, model_name)
        if warmup_runs:
            logger.info(
                "开始模型预热：%s，样本=%s，次数=%d",
                adapter.name,
                samples[0].sample_id,
                warmup_runs,
            )
            warm_up_adapter(adapter, samples[0], warmup_runs)
            logger.info("模型预热完成：%s", adapter.name)
        return benchmark_model(
            adapter,
            samples,
            runs,
            load_latency_seconds,
            active_cache=active_cache,
        )
    finally:
        # 即使 close 自身失败，也要删除当前引用并触发垃圾回收，避免
        # 前一个模型占用的内存影响下一个模型。
        adapter_name = adapter.name
        logger.info("正在释放模型资源：%s", adapter_name)
        try:
            close_adapter(adapter)
        finally:
            del adapter
            gc.collect()
        logger.info("模型资源释放完成：%s", adapter_name)


def restore_completed_model(
    cached_model: CachedModel | None,
    samples: Sequence[AudioSample],
    runs: int,
) -> tuple[ModelSummary, list[RecognitionResult]] | None:
    """在模型的所有样本都有缓存时，直接重建报告结果。"""

    if cached_model is None:
        return None
    if any(sample.sample_id not in cached_model.transcriptions for sample in samples):
        return None

    elapsed_seconds = 0.0
    recognition_results: list[RecognitionResult] = []
    for sample in samples:
        cached_transcription = cached_model.transcriptions[sample.sample_id]
        elapsed_seconds += cached_transcription.elapsed_seconds
        recognition_results.append(
            RecognitionResult(
                model_name=cached_model.report_name,
                sample_id=sample.sample_id,
                reference=sample.reference,
                raw_hypothesis=cached_transcription.raw_hypothesis,
                hypothesis=cached_transcription.hypothesis,
            )
        )
    invocation_count = len(samples) * runs
    processed_audio_seconds = sum(sample.duration_seconds for sample in samples) * runs
    return (
        ModelSummary(
            name=cached_model.report_name,
            device=cached_model.device,
            total_samples=len(samples),
            successful_samples=len(samples),
            failed_samples=0,
            load_latency_seconds=cached_model.load_latency_seconds,
            average_latency_seconds=elapsed_seconds / invocation_count,
            real_time_factor=elapsed_seconds / processed_audio_seconds,
        ),
        recognition_results,
    )


def close_adapter(adapter: ModelAdapter) -> None:
    """调用模型可选的资源释放方法，并为释放失败补充模型名称。"""

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
    """在正式计时前用第一条音频运行指定次数的预热识别。

    预热结果不会参与识别结果展示或速度统计。任意一次预热失败都会停止当前
    测试，因为此时无法确认模型能够正常执行后续识别。
    """

    for run_number in range(1, warmup_runs + 1):
        try:
            logger.info(
                "正在预热：模型=%s，进度=%d/%d",
                adapter.name,
                run_number,
                warmup_runs,
            )
            transcribe(adapter, sample.audio_path)
        except Exception as error:
            logger.exception(
                "模型预热失败：模型=%s，进度=%d/%d",
                adapter.name,
                run_number,
                warmup_runs,
            )
            raise BenchmarkError(
                f"模型 {adapter.name} 第 {run_number} 次预热失败：{error}"
            ) from error


def benchmark_model(
    adapter: ModelAdapter,
    samples: Sequence[AudioSample],
    runs: int,
    load_latency_seconds: float = 0.0,
    *,
    active_cache: ActiveModelCache | None = None,
) -> tuple[ModelSummary, list[RecognitionResult], list[SampleFailure]]:
    """测试一个模型，并收集识别文本和处理速度。

    每条成功音频的参考文本、识别文本和耗时都会保留。单条音频失败时
    会记录错误并继续处理其他音频；如果全部失败，则返回不含速度数值的
    空汇总结果。
    """

    elapsed_seconds = 0.0
    processed_audio_seconds = 0.0
    recognition_results: list[RecognitionResult] = []
    failures: list[SampleFailure] = []

    for sample_index, sample in enumerate(samples, start=1):
        cached_transcription = get_cached_transcription(
            active_cache,
            sample.sample_id,
        )
        if cached_transcription is None:
            logger.info(
                "开始识别样本 [%d/%d]：模型=%s，样本=%s，音频=%s",
                sample_index,
                len(samples),
                adapter.name,
                sample.sample_id,
                sample.audio_path,
            )
            try:
                raw_hypothesis, hypothesis, sample_elapsed = run_sample(
                    adapter, sample, runs
                )
            except Exception as error:  # noqa: BLE001 - continue with remaining samples
                # 一条音频失败不应阻止同一模型继续测试配置中的其他音频。
                logger.exception(
                    "样本识别失败 [%d/%d]：模型=%s，样本=%s",
                    sample_index,
                    len(samples),
                    adapter.name,
                    sample.sample_id,
                )
                failures.append(
                    SampleFailure(adapter.name, sample.sample_id, str(error))
                )
                continue
            if active_cache is not None:
                active_cache.storage.set_transcription(
                    active_cache.model_name,
                    sample.sample_id,
                    CachedTranscription(
                        raw_hypothesis,
                        hypothesis,
                        sample_elapsed,
                    ),
                )
                logger.info(
                    "样本结果已保存到缓存：模型=%s，样本=%s",
                    adapter.name,
                    sample.sample_id,
                )
        else:
            raw_hypothesis = cached_transcription.raw_hypothesis
            hypothesis = cached_transcription.hypothesis
            sample_elapsed = cached_transcription.elapsed_seconds
            logger.info(
                "样本命中缓存 [%d/%d]：模型=%s，样本=%s",
                sample_index,
                len(samples),
                adapter.name,
                sample.sample_id,
            )

        recognition_results.append(
            RecognitionResult(
                model_name=adapter.name,
                sample_id=sample.sample_id,
                reference=sample.reference,
                raw_hypothesis=raw_hypothesis,
                hypothesis=hypothesis,
            )
        )
        elapsed_seconds += sample_elapsed
        processed_audio_seconds += sample.duration_seconds * runs
        logger.info(
            "样本处理完成 [%d/%d]：模型=%s，样本=%s，正式运行总耗时=%.2f 秒",
            sample_index,
            len(samples),
            adapter.name,
            sample.sample_id,
            sample_elapsed,
        )

    if not recognition_results:
        return (
            empty_summary(
                adapter.name,
                adapter.device,
                len(samples),
                load_latency_seconds,
            ),
            recognition_results,
            failures,
        )

    return (
        ModelSummary(
            name=adapter.name,
            device=adapter.device,
            total_samples=len(samples),
            successful_samples=len(recognition_results),
            failed_samples=len(samples) - len(recognition_results),
            load_latency_seconds=load_latency_seconds,
            average_latency_seconds=elapsed_seconds / (len(recognition_results) * runs),
            real_time_factor=elapsed_seconds / processed_audio_seconds,
        ),
        recognition_results,
        failures,
    )


def get_cached_transcription(
    active_cache: ActiveModelCache | None,
    sample_id: str,
) -> CachedTranscription | None:
    """读取一条已成功的转写；未启用缓存或没有命中时返回 ``None``。"""

    if active_cache is None:
        return None
    cached_model = active_cache.storage.get_model(active_cache.model_name)
    if cached_model is None:
        return None
    return cached_model.transcriptions.get(sample_id)


def run_sample(
    adapter: ModelAdapter,
    sample: AudioSample,
    runs: int,
) -> tuple[str | None, str, float]:
    """重复识别音频，返回首次原文、首次最终文本和所有识别的总耗时。

    支持标点前原文的适配器会同时返回首次 ASR 原文。首次文本用于结果展示，
    全部运行的耗时用于计算平均速度。这样增加 ``runs`` 可以让计时更稳定，
    又不会重复展示同一条音频的识别结果。
    """

    first_raw_hypothesis: str | None = None
    first_hypothesis = ""
    total_elapsed = 0.0
    for run_index in range(runs):
        run_number = run_index + 1
        logger.info(
            "正在识别：模型=%s，样本=%s，运行进度=%d/%d",
            adapter.name,
            sample.sample_id,
            run_number,
            runs,
        )
        started_at = time.perf_counter()
        hypothesis = transcribe(adapter, sample.audio_path)
        run_elapsed = time.perf_counter() - started_at
        total_elapsed += run_elapsed
        logger.info(
            "单次识别完成：模型=%s，样本=%s，运行进度=%d/%d，耗时=%.2f 秒",
            adapter.name,
            sample.sample_id,
            run_number,
            runs,
            run_elapsed,
        )
        if run_index == 0:
            first_raw_hypothesis = get_last_raw_text(adapter)
            first_hypothesis = hypothesis
    return first_raw_hypothesis, first_hypothesis, total_elapsed


def get_last_raw_text(adapter: ModelAdapter) -> str | None:
    """读取适配器最近一次识别的标点前原文，并校验返回类型。"""

    if adapter.get_last_raw_text is None:
        return None
    raw_text = adapter.get_last_raw_text()
    if raw_text is not None and not isinstance(raw_text, str):
        raise TypeError(
            f"标点前原文必须是字符串或 None，实际类型为 {type(raw_text).__name__}"
        )
    return raw_text


def transcribe(adapter: ModelAdapter, audio_path: Path) -> str:
    """调用模型识别音频，并保证返回值是可以展示的字符串。"""

    result = adapter.transcribe(audio_path)
    if not isinstance(result, str):
        raise TypeError(f"识别结果必须是字符串，实际类型为 {type(result).__name__}")
    return result


def empty_summary(
    model_name: str,
    device: str,
    total_samples: int,
    load_latency_seconds: float,
) -> ModelSummary:
    """为没有成功样本的模型创建汇总结果。

    无法计算的速度使用 ``NaN``，输出表格会将其显示为 ``-``。
    """

    return ModelSummary(
        name=model_name,
        device=device,
        total_samples=total_samples,
        successful_samples=0,
        failed_samples=total_samples,
        load_latency_seconds=load_latency_seconds,
        average_latency_seconds=math.nan,
        real_time_factor=math.nan,
    )


def print_failures(failures: Sequence[SampleFailure]) -> None:
    """把失败样本写到标准错误流，正常汇总表仍保留在标准输出流。"""

    if not failures:
        return
    print("\n失败详情：", file=sys.stderr)
    for failure in failures:
        print(
            f"- 模型={failure.model_name} 样本={failure.sample_id} 错误={failure.message}",
            file=sys.stderr,
        )


def configure_progress_logging() -> None:
    """只为当前项目启用写入标准错误流的进度日志。"""

    project_logger = logging.getLogger("asr_benchmark")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    project_logger.addHandler(handler)
    project_logger.setLevel(logging.INFO)
    project_logger.propagate = False


def main() -> int:
    """组织配置读取、模型测试和结果输出，并返回进程退出码。

    配置或输入错误返回 2；测试完成但存在失败样本返回 1；所有样本都成功
    时返回 0，便于脚本或持续集成环境判断本次测试结果。
    """

    configure_progress_logging()
    arguments = parse_arguments()
    logger.info("正在读取配置：%s", arguments.config.expanduser().resolve())
    try:
        config = load_config(arguments.config)
        logger.info(
            "配置读取完成：模型数=%d，样本数=%d，正式运行次数=%d，预热次数=%d",
            len(config.model_names),
            len(config.samples),
            config.runs,
            config.warmup_runs,
        )
        if arguments.normalize_audio:
            logger.info("正在原地标准化音频：样本数=%d", len(config.samples))
            normalize_audio_files(tuple(sample.audio_path for sample in config.samples))
            # 重新读取配置，以标准化后 WAV 的帧数计算实际音频时长。
            config = load_config(arguments.config)
            logger.info("音频标准化完成：样本数=%d", len(config.samples))
        cache_path = cache_path_for_config(arguments.config)
        cache_action = "读取" if arguments.resume and cache_path.exists() else "初始化"
        logger.info(
            "正在%s转写缓存：%s",
            cache_action,
            cache_path,
        )
        cache = TranscriptionCache.prepare(
            cache_path,
            build_configuration_fingerprint(arguments.config, config.samples),
            resume=arguments.resume,
        )
        logger.info(
            "%s转写缓存完成：%s",
            cache_action,
            cache_path,
        )
        summaries, recognition_results, failures = benchmark_models(
            config.model_names,
            config.samples,
            config.runs,
            config.warmup_runs,
            config.llm_config,
            cache=cache,
        )
    except (
        AudioNormalizationError,
        BenchmarkError,
        TranscriptionCacheError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2

    logger.info(
        "基准测试完成：模型数=%d，成功结果=%d，失败数=%d",
        len(summaries),
        len(recognition_results),
        len(failures),
    )
    print(render_recognition_results(recognition_results))
    print()
    print(render_summary_table(summaries))
    print(f"\n转写缓存：{cache_path}")
    print_failures(failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
