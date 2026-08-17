"""豆包输入法非官方 ASR 协议适配器。"""

from __future__ import annotations

import asyncio
import ctypes.util
import os
import sys
import time
from pathlib import Path
from typing import Any, Final

DOUBAO_ASR_MODEL_NAME: Final = "Doubao-IME-ASR"
MAX_RESPONSE_ATTEMPTS: Final = 3
RETRY_BASE_DELAY_SECONDS: Final = 1.0
MACOS_OPUS_LIBRARY_PATHS: Final = (
    Path("/opt/homebrew/lib/libopus.dylib"),
    Path("/usr/local/lib/libopus.dylib"),
)


class _IncompleteDoubaoResponseError(RuntimeError):
    """表示豆包没有返回完整的最终结果和会话结束消息。"""


class DoubaoAsrAdapter:
    """通过 ``doubaoime-asr`` 识别一份完整音频文件。"""

    def __init__(self) -> None:
        dependencies = _import_dependencies()
        _ensure_default_ca_file(dependencies["certifi"])
        config = dependencies["ASRConfig"](enable_speech_rejection=False)
        self._client = dependencies["DoubaoASR"](config)
        self._response_type = dependencies["ResponseType"]
        self.name = DOUBAO_ASR_MODEL_NAME
        self.device = "remote"

    def transcribe(self, audio_path: Path) -> str:
        completed_empty_response = False
        last_incomplete_error: _IncompleteDoubaoResponseError | None = None

        for attempt in range(1, MAX_RESPONSE_ATTEMPTS + 1):
            try:
                text = asyncio.run(self._transcribe_once(audio_path))
            except _IncompleteDoubaoResponseError as error:
                last_incomplete_error = error
            else:
                if text:
                    return text
                completed_empty_response = True

            if attempt < MAX_RESPONSE_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))

        if completed_empty_response:
            return ""
        if last_incomplete_error is None:
            raise AssertionError("豆包 ASR 重试结束后没有结果或错误")
        raise last_incomplete_error

    async def _transcribe_once(self, audio_path: Path) -> str:
        responses = [
            response
            async for response in self._client.transcribe_stream(
                audio_path,
                realtime=True,
            )
        ]
        error_response = next(
            (
                response
                for response in responses
                if response.type is self._response_type.ERROR
            ),
            None,
        )
        if error_response is not None:
            raise RuntimeError(f"豆包 ASR 返回错误：{error_response.error_msg}")

        final_responses = [
            response
            for response in responses
            if response.type is self._response_type.FINAL_RESULT
        ]
        has_terminal_response = any(
            response.type is self._response_type.SESSION_FINISHED
            for response in responses
        )
        has_non_final_text = any(
            response.type is not self._response_type.FINAL_RESULT
            and response.text.strip()
            for response in responses
        )
        if not has_terminal_response or (not final_responses and has_non_final_text):
            raise _IncompleteDoubaoResponseError(
                f"豆包 ASR 未返回完整终止响应：{audio_path}"
            )
        return final_responses[-1].text.strip() if final_responses else ""


def _ensure_default_ca_file(certifi_module: Any) -> None:
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return
    os.environ["SSL_CERT_FILE"] = certifi_module.where()


def _import_dependencies() -> dict[str, Any]:
    _ensure_opus_library_path()
    try:
        import certifi
        from doubaoime_asr import ASRConfig, DoubaoASR, ResponseType
    except Exception as error:
        raise RuntimeError(
            "豆包 ASR 依赖加载失败，请执行 uv sync --locked，并确认系统已安装 "
            f"libopus。原始错误：{error}"
        ) from error
    return {
        "certifi": certifi,
        "ASRConfig": ASRConfig,
        "DoubaoASR": DoubaoASR,
        "ResponseType": ResponseType,
    }


def _ensure_opus_library_path() -> None:
    """让 Python 3.11/3.12 能找到 Homebrew 安装的 Opus 动态库。"""

    if sys.platform != "darwin" or ctypes.util.find_library("opus") is not None:
        return
    opus_library = next(
        (path for path in MACOS_OPUS_LIBRARY_PATHS if path.is_file()),
        None,
    )
    if opus_library is None:
        return

    library_directory = str(opus_library.parent)
    current_paths = os.environ.get("DYLD_LIBRARY_PATH", "").split(os.pathsep)
    if library_directory not in current_paths:
        os.environ["DYLD_LIBRARY_PATH"] = os.pathsep.join(
            path for path in (library_directory, *current_paths) if path
        )


def create_doubao_asr() -> DoubaoAsrAdapter:
    return DoubaoAsrAdapter()
