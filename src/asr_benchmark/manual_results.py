"""读取人工保存的 ASR 转写，并转换为报告使用的数据。"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .reporting import ModelSummary, RecognitionResult

MANUAL_REPORT_NAME_SUFFIX = "（手动导入）"
ROOT_FIELDS = frozenset({"results"})
RESULT_FIELDS = frozenset({"model", "sample_id", "hypothesis"})


class ManualResultsError(Exception):
    """表示手动转写文件无法读取或内容不符合要求。"""


@dataclass(frozen=True)
class ManualTranscription:
    """保存网页模型的一条人工导入转写。"""

    model_name: str
    sample_id: str
    hypothesis: str


class ManualResultSampleLike(Protocol):
    @property
    def sample_id(self) -> str: ...

    @property
    def reference(self) -> str: ...


def manual_results_path_for_config(config_path: Path) -> Path:
    """返回配置文件旁边对应的手动转写文件路径。"""

    resolved_config = config_path.expanduser().resolve()
    return resolved_config.with_name(f"{resolved_config.stem}.manual-results.toml")


def is_manual_report_name(model_name: str) -> bool:
    """判断报告模型名是否表示人工导入的转写。"""

    return model_name.endswith(MANUAL_REPORT_NAME_SUFFIX)


def load_manual_results(
    path: Path,
    sample_ids: Sequence[str],
) -> tuple[ManualTranscription, ...]:
    """读取手动转写；文件不存在时返回空结果。"""

    if not path.exists():
        return ()
    if not path.is_file():
        raise ManualResultsError(f"手动转写路径不是文件：{path}")

    try:
        with path.open("rb") as results_file:
            payload = tomllib.load(results_file)
    except (tomllib.TOMLDecodeError, UnicodeError) as error:
        raise ManualResultsError(
            f"手动转写文件不是有效 TOML：{path}：{error}"
        ) from error
    except OSError as error:
        raise ManualResultsError(f"读取手动转写文件失败：{path}：{error}") from error

    _reject_unknown_fields(payload, ROOT_FIELDS, "根节点", path)
    results_value = payload.get("results", [])
    if not isinstance(results_value, list):
        raise ManualResultsError(f"手动转写文件中的 results 必须是数组：{path}")

    known_sample_ids = set(sample_ids)
    seen_keys: set[tuple[str, str]] = set()
    results: list[ManualTranscription] = []
    for result_number, result_value in enumerate(results_value, start=1):
        result = _parse_manual_result(result_value, result_number, path)
        if result.sample_id not in known_sample_ids:
            raise ManualResultsError(
                f"手动转写文件中第 {result_number} 条结果使用了配置中不存在的样本 "
                f"id：{result.sample_id}"
            )
        result_key = (result.model_name, result.sample_id)
        if result_key in seen_keys:
            raise ManualResultsError(
                f"手动转写文件中模型 {result.model_name} 的样本 {result.sample_id} 重复"
            )
        seen_keys.add(result_key)
        results.append(result)
    return tuple(results)


def build_manual_report_data(
    transcriptions: Sequence[ManualTranscription],
    samples: Sequence[ManualResultSampleLike],
    existing_model_names: Sequence[str],
) -> tuple[list[ModelSummary], list[RecognitionResult]]:
    """把手动转写转换成模型汇总和逐样本报告结果。"""

    transcriptions_by_model: dict[str, dict[str, str]] = {}
    for transcription in transcriptions:
        transcriptions_by_model.setdefault(transcription.model_name, {})[
            transcription.sample_id
        ] = transcription.hypothesis

    existing_names = set(existing_model_names)
    summaries: list[ModelSummary] = []
    results: list[RecognitionResult] = []
    for model_name, hypotheses_by_sample in transcriptions_by_model.items():
        report_name = f"{model_name}{MANUAL_REPORT_NAME_SUFFIX}"
        if report_name in existing_names:
            raise ManualResultsError(f"手动转写模型与已有报告模型重名：{report_name}")
        summaries.append(
            ModelSummary(
                name=report_name,
                device="web",
                total_samples=len(samples),
                successful_samples=len(hypotheses_by_sample),
                failed_samples=len(samples) - len(hypotheses_by_sample),
                load_latency_seconds=math.nan,
                average_latency_seconds=math.nan,
                real_time_factor=math.nan,
            )
        )
        for sample in samples:
            hypothesis = hypotheses_by_sample.get(sample.sample_id)
            if hypothesis is None:
                continue
            results.append(
                RecognitionResult(
                    model_name=report_name,
                    sample_id=sample.sample_id,
                    reference=sample.reference,
                    hypothesis=hypothesis,
                )
            )
    return summaries, results


def _parse_manual_result(
    value: Any,
    result_number: int,
    path: Path,
) -> ManualTranscription:
    if not isinstance(value, dict):
        raise ManualResultsError(
            f"手动转写文件中第 {result_number} 条结果必须是 TOML 表：{path}"
        )
    _reject_unknown_fields(
        value,
        RESULT_FIELDS,
        f"第 {result_number} 条结果",
        path,
    )
    model_name = _require_non_empty_string(value, "model", result_number, path)
    sample_id = _require_non_empty_string(value, "sample_id", result_number, path)
    hypothesis = value.get("hypothesis")
    if not isinstance(hypothesis, str):
        raise ManualResultsError(
            f"手动转写文件中第 {result_number} 条结果的 hypothesis 必须是字符串：{path}"
        )
    return ManualTranscription(model_name, sample_id, hypothesis)


def _reject_unknown_fields(
    payload: dict[str, Any],
    allowed_fields: frozenset[str],
    location: str,
    path: Path,
) -> None:
    unknown_fields = sorted(payload.keys() - allowed_fields)
    if unknown_fields:
        raise ManualResultsError(
            f"手动转写文件的{location}包含不支持的字段："
            f"{', '.join(unknown_fields)}：{path}"
        )


def _require_non_empty_string(
    payload: dict[str, Any],
    field_name: str,
    result_number: int,
    path: Path,
) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ManualResultsError(
            f"手动转写文件中第 {result_number} 条结果的 {field_name} "
            f"必须是非空字符串：{path}"
        )
    return value.strip()
