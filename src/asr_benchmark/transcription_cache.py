"""持久化 ASR 基准测试中已经成功完成的转写结果。"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TranscriptionCacheError(Exception):
    """表示转写缓存无法读取、校验或保存。"""


@dataclass(frozen=True)
class CachedTranscription:
    """保存一条成功转写、正式运行总耗时及其运行次数。"""

    raw_hypothesis: str | None
    hypothesis: str
    elapsed_seconds: float
    runs: int | None = None


@dataclass(frozen=True)
class CachedLlmConfig:
    """保存判断远程 LLM 缓存是否仍适用所需的配置。"""

    model: str
    base_url: str


@dataclass(frozen=True)
class CachedModel:
    """保存一个模型的报告信息和按样本编号索引的转写。"""

    report_name: str
    device: str
    load_latency_seconds: float
    transcriptions: dict[str, CachedTranscription]
    llm_config: CachedLlmConfig | None = None


class TranscriptionCache:
    """读取和原子更新一个 JSON 转写缓存。"""

    def __init__(
        self,
        path: Path,
        models: dict[str, CachedModel] | None = None,
    ) -> None:
        self.path = path
        self._models = models or {}

    @classmethod
    def prepare(
        cls,
        path: Path,
        *,
        resume: bool,
    ) -> TranscriptionCache:
        """按 ``resume`` 决定读取旧缓存，或立即用空缓存覆盖它。"""

        if resume and path.exists():
            return cls._load(path)

        cache = cls(path)
        cache._save()
        return cache

    @classmethod
    def _load(cls, path: Path) -> TranscriptionCache:
        try:
            with path.open(encoding="utf-8") as cache_file:
                payload = json.load(cache_file)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TranscriptionCacheError(
                f"读取转写缓存失败：{path}：{error}"
            ) from error

        if not isinstance(payload, dict):
            raise TranscriptionCacheError(f"转写缓存的根节点必须是对象：{path}")
        models_value = payload.get("models")
        if not isinstance(models_value, dict):
            raise TranscriptionCacheError(f"转写缓存中的 models 必须是对象：{path}")
        models: dict[str, CachedModel] = {}
        for model_name_value, model_value in models_value.items():
            model_name = _require_non_empty_string(
                model_name_value,
                "模型键",
                path,
            )
            models[model_name] = _parse_cached_model(model_name, model_value, path)
        return cls(path, models)

    def get_model(self, model_name: str) -> CachedModel | None:
        """返回指定配置模型的缓存；不存在时返回 ``None``。"""

        return self._models.get(model_name)

    def set_model_metadata(
        self,
        model_name: str,
        report_name: str,
        device: str,
        load_latency_seconds: float,
        llm_config: CachedLlmConfig | None = None,
    ) -> None:
        """保存模型信息；配置未改变时保留已经完成的样本。"""

        existing = self._models.get(model_name)
        transcriptions = (
            dict(existing.transcriptions)
            if existing is not None and existing.llm_config == llm_config
            else {}
        )
        self._models[model_name] = CachedModel(
            report_name,
            device,
            load_latency_seconds,
            transcriptions,
            llm_config,
        )
        self._save()

    def set_transcription(
        self,
        model_name: str,
        sample_id: str,
        transcription: CachedTranscription,
    ) -> None:
        """保存一条成功转写，并立即把完整缓存原子写入磁盘。"""

        model = self._models.get(model_name)
        if model is None:
            raise TranscriptionCacheError(
                f"保存样本 {sample_id} 前缺少模型 {model_name} 的缓存信息"
            )
        updated_transcriptions = dict(model.transcriptions)
        updated_transcriptions[sample_id] = transcription
        self._models[model_name] = CachedModel(
            model.report_name,
            model.device,
            model.load_latency_seconds,
            updated_transcriptions,
            model.llm_config,
        )
        self._save()

    def _save(self) -> None:
        payload = {
            "models": {
                model_name: _serialize_cached_model(model)
                for model_name, model in self._models.items()
            },
        }

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(
                    payload,
                    temporary_file,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
        except (OSError, TypeError, ValueError) as error:
            cleanup_message = ""
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    cleanup_message = f"；清理临时文件失败：{cleanup_error}"
            raise TranscriptionCacheError(
                f"保存转写缓存失败：{self.path}：{error}{cleanup_message}"
            ) from error


def cache_path_for_config(config_path: Path) -> Path:
    """返回配置文件旁边唯一对应的转写缓存路径。"""

    resolved_config = config_path.expanduser().resolve()
    return resolved_config.with_name(f"{resolved_config.name}.cache.json")


def _serialize_cached_model(model: CachedModel) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "report_name": model.report_name,
        "device": model.device,
        "load_latency_seconds": model.load_latency_seconds,
        "transcriptions": {
            sample_id: {
                "raw_hypothesis": transcription.raw_hypothesis,
                "hypothesis": transcription.hypothesis,
                "elapsed_seconds": transcription.elapsed_seconds,
                "runs": transcription.runs,
            }
            for sample_id, transcription in model.transcriptions.items()
        },
    }
    if model.llm_config is not None:
        payload["llm"] = {
            "model": model.llm_config.model,
            "base_url": model.llm_config.base_url,
        }
    return payload


def _parse_cached_model(
    model_name: str,
    value: Any,
    path: Path,
) -> CachedModel:
    if not isinstance(value, dict):
        raise TranscriptionCacheError(
            f"转写缓存中模型 {model_name} 的内容必须是对象：{path}"
        )
    report_name = _read_non_empty_string(value, "report_name", path)
    device = _read_non_empty_string(value, "device", path)
    load_latency_seconds = _read_non_negative_number(
        value,
        "load_latency_seconds",
        path,
    )
    llm_config = _parse_cached_llm_config(value.get("llm"), model_name, path)
    transcriptions_value = value.get("transcriptions")
    if not isinstance(transcriptions_value, dict):
        raise TranscriptionCacheError(
            f"转写缓存中模型 {model_name} 的 transcriptions 必须是对象：{path}"
        )
    transcriptions: dict[str, CachedTranscription] = {}
    for sample_id_value, sample_value in transcriptions_value.items():
        sample_id = _require_non_empty_string(sample_id_value, "样本键", path)
        transcriptions[sample_id] = _parse_cached_transcription(
            model_name,
            sample_id,
            sample_value,
            path,
        )
    return CachedModel(
        report_name,
        device,
        load_latency_seconds,
        transcriptions,
        llm_config,
    )


def _parse_cached_llm_config(
    value: Any,
    model_name: str,
    path: Path,
) -> CachedLlmConfig | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TranscriptionCacheError(
            f"转写缓存中模型 {model_name} 的 llm 必须是对象：{path}"
        )
    return CachedLlmConfig(
        _read_non_empty_string(value, "model", path),
        _read_non_empty_string(value, "base_url", path),
    )


def _parse_cached_transcription(
    model_name: str,
    sample_id: str,
    value: Any,
    path: Path,
) -> CachedTranscription:
    if not isinstance(value, dict):
        raise TranscriptionCacheError(
            f"转写缓存中模型 {model_name} 的样本 {sample_id} 必须是对象：{path}"
        )
    raw_hypothesis = value.get("raw_hypothesis")
    if raw_hypothesis is not None and not isinstance(raw_hypothesis, str):
        raise TranscriptionCacheError(
            f"转写缓存中的 raw_hypothesis 必须是字符串或 null：{path}"
        )
    hypothesis = value.get("hypothesis")
    if not isinstance(hypothesis, str):
        raise TranscriptionCacheError(f"转写缓存中的 hypothesis 必须是字符串：{path}")
    elapsed_seconds = _read_non_negative_number(value, "elapsed_seconds", path)
    runs = _read_optional_positive_integer(value, "runs", path)
    return CachedTranscription(raw_hypothesis, hypothesis, elapsed_seconds, runs)


def _read_non_empty_string(payload: dict[str, Any], field_name: str, path: Path) -> str:
    return _require_non_empty_string(payload.get(field_name), field_name, path)


def _require_non_empty_string(value: Any, field_name: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TranscriptionCacheError(
            f"转写缓存中的 {field_name} 必须是非空字符串：{path}"
        )
    return value


def _read_non_negative_number(
    payload: dict[str, Any],
    field_name: str,
    path: Path,
) -> float:
    error_message = f"转写缓存中的 {field_name} 必须是大于或等于 0 的数字：{path}"
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TranscriptionCacheError(error_message)
    try:
        number = float(value)
    except OverflowError as error:
        raise TranscriptionCacheError(error_message) from error
    if not math.isfinite(number) or number < 0:
        raise TranscriptionCacheError(error_message)
    return number


def _read_optional_positive_integer(
    payload: dict[str, Any],
    field_name: str,
    path: Path,
) -> int | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TranscriptionCacheError(
            f"转写缓存中的 {field_name} 必须是大于或等于 1 的整数：{path}"
        )
    return value
