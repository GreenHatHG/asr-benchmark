"""OpenAI 兼容的多模态 LLM 音频识别适配器。"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

LLM_ASR_MODEL_NAME: Final = "LLM-ASR"
LLM_API_KEY_ENV: Final = "LLM_AUDIO_TRANSCRIPTION_API_KEY"
SUPPORTED_AUDIO_FORMATS: Final = frozenset({"wav", "mp3"})
MAX_ATTEMPTS: Final = 3
RETRY_BASE_DELAY_SECONDS: Final = 1.0
RETRY_MAX_DELAY_SECONDS: Final = 8.0
RETRYABLE_STATUS_CODES: Final = frozenset({408, 409, 429})
logger = logging.getLogger(__name__)
TRANSCRIPTION_PROMPT: Final = """你是专业的音频转写助手。

请准确转写这段音频，保持音频使用的原语言，并保留人名、公司名、产品名和英文词的原始表达。

要求：
1. 只输出音频对应的转写正文。
2. 使用自然标点，不添加解释、标题或 Markdown。
3. 无法确认的内容保留原始发音，不要编造。
"""


@dataclass(frozen=True)
class LlmAsrConfig:
    """保存创建一个远程 LLM 音频识别客户端所需的非密钥配置。"""

    model: str
    base_url: str
    api_key_env: str = LLM_API_KEY_ENV


class _EmptyLlmResponseError(RuntimeError):
    """表示远程接口成功响应，但没有返回可用文字。"""


class LlmAsrAdapter:
    """通过 OpenAI 兼容的 ``chat.completions`` 接口识别音频。"""

    def __init__(self, config: LlmAsrConfig) -> None:
        api_key = os.environ.get(config.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"缺少环境变量：{config.api_key_env}")

        dependencies = _import_dependencies()
        self._client = dependencies["OpenAI"](
            api_key=api_key,
            base_url=config.base_url,
            max_retries=0,
        )
        self._openai_error = dependencies["OpenAIError"]
        self._always_retryable_errors = (
            dependencies["APIConnectionError"],
            dependencies["APITimeoutError"],
            dependencies["RateLimitError"],
            dependencies["InternalServerError"],
        )
        self._api_status_error = dependencies["APIStatusError"]
        self._model = config.model
        self.name = f"{LLM_ASR_MODEL_NAME} ({config.model})"
        self.device = "remote"

    def transcribe(self, audio_path: Path) -> str:
        audio_format = _get_audio_format(audio_path)
        audio_data = _read_audio(audio_path)
        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self._request_transcription(
                    audio_data,
                    audio_format=audio_format,
                )
            except Exception as error:
                if not self._is_retryable_error(error):
                    raise RuntimeError(f"LLM ASR 请求失败：{error}") from error
                last_error = error
                if attempt < MAX_ATTEMPTS:
                    retry_delay_seconds = _retry_delay_seconds(attempt)
                    logger.warning(
                        "LLM ASR 请求失败，将在 %.1f 秒后重试：音频=%s，"
                        "请求进度=%d/%d，错误=%s",
                        retry_delay_seconds,
                        audio_path,
                        attempt,
                        MAX_ATTEMPTS,
                        error,
                    )
                    time.sleep(retry_delay_seconds)

        if last_error is None:
            raise AssertionError("LLM ASR 重试结束后没有记录错误")
        raise RuntimeError(
            f"LLM ASR 连续 {MAX_ATTEMPTS} 次识别失败，最后一次错误：{last_error}"
        ) from last_error

    def close(self) -> None:
        self._client.close()

    def _request_transcription(
        self,
        audio_data: bytes,
        *,
        audio_format: str,
    ) -> str:
        encoded_audio = base64.b64encode(audio_data).decode("ascii")
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TRANSCRIPTION_PROMPT},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_audio,
                                "format": audio_format,
                            },
                        },
                    ],
                }
            ],
        )
        choices = getattr(response, "choices", None)
        content = choices[0].message.content if choices else None
        normalized_text = content.strip() if isinstance(content, str) else ""
        if not normalized_text:
            raise _EmptyLlmResponseError("LLM 返回了空文本")
        return normalized_text

    def _is_retryable_error(self, error: Exception) -> bool:
        if isinstance(error, _EmptyLlmResponseError):
            return True
        if not isinstance(error, self._openai_error):
            return False
        if isinstance(error, self._always_retryable_errors):
            return True
        if not isinstance(error, self._api_status_error):
            return False

        status_code = getattr(error, "status_code", None)
        if status_code in RETRYABLE_STATUS_CODES:
            return True
        return isinstance(status_code, int) and 500 <= status_code <= 599


def _get_audio_format(audio_path: Path) -> str:
    audio_format = audio_path.suffix.lower().removeprefix(".")
    if audio_format not in SUPPORTED_AUDIO_FORMATS:
        supported = "、".join(sorted(SUPPORTED_AUDIO_FORMATS))
        raise ValueError(f"LLM ASR 仅支持 {supported} 音频：{audio_path}")
    return audio_format


def _read_audio(audio_path: Path) -> bytes:
    try:
        audio_data = audio_path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"读取待识别音频失败：{audio_path}：{error}") from error
    if not audio_data:
        raise ValueError(f"待识别音频不能为空：{audio_path}")
    return audio_data


def _retry_delay_seconds(attempt: int) -> float:
    return min(
        RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
        RETRY_MAX_DELAY_SECONDS,
    )


def _import_dependencies() -> dict[str, Any]:
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            InternalServerError,
            OpenAI,
            OpenAIError,
            RateLimitError,
        )
    except ImportError as error:
        raise RuntimeError("LLM ASR 依赖尚未安装，请执行 uv sync --locked") from error
    return {
        "OpenAI": OpenAI,
        "OpenAIError": OpenAIError,
        "APIConnectionError": APIConnectionError,
        "APIStatusError": APIStatusError,
        "APITimeoutError": APITimeoutError,
        "InternalServerError": InternalServerError,
        "RateLimitError": RateLimitError,
    }


def create_llm_asr(config: LlmAsrConfig) -> LlmAsrAdapter:
    return LlmAsrAdapter(config)
