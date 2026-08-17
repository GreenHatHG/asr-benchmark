"""把基准测试音频原地转换为模型共同支持的 PCM WAV 格式。"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
import tempfile
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Final

TARGET_SAMPLE_RATE: Final = 16000
TARGET_CHANNELS: Final = 1
TARGET_SAMPLE_WIDTH: Final = 2
logger = logging.getLogger(__name__)


class AudioNormalizationError(Exception):
    """表示音频无法安全地原地转换为目标格式。"""


def normalize_audio_files(audio_paths: Sequence[Path]) -> None:
    """原地标准化多份音频，同一路径只处理一次。"""

    processed_paths: set[Path] = set()
    for audio_path in audio_paths:
        resolved_path = audio_path.resolve()
        if resolved_path in processed_paths:
            continue
        normalize_audio_file(resolved_path)
        processed_paths.add(resolved_path)


def normalize_audio_file(audio_path: Path) -> None:
    """通过同目录临时文件转换音频，验证成功后替换原 WAV 文件。"""

    if audio_path.suffix.lower() != ".wav":
        raise AudioNormalizationError(
            f"原地标准化只支持 .wav 文件，当前文件为：{audio_path}"
        )
    if _is_target_pcm_wav(audio_path):
        return

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise AudioNormalizationError("找不到 ffmpeg，无法标准化音频；请先安装 ffmpeg")

    try:
        original_mode = stat.S_IMODE(audio_path.stat().st_mode)
    except OSError as error:
        raise AudioNormalizationError(
            f"读取音频文件信息失败：{audio_path}：{error}"
        ) from error

    temporary_path = _create_temporary_wav(audio_path)
    try:
        _run_ffmpeg(ffmpeg_path, audio_path, temporary_path)
        if not _is_target_pcm_wav(temporary_path):
            raise AudioNormalizationError(
                f"ffmpeg 没有生成有效的 16kHz、16 位、单声道 PCM WAV：{audio_path}"
            )
        os.chmod(temporary_path, original_mode)
        os.replace(temporary_path, audio_path)
    except OSError as error:
        raise AudioNormalizationError(
            f"替换标准化音频失败，原文件未修改：{audio_path}：{error}"
        ) from error
    finally:
        _remove_temporary_file(temporary_path)


def _create_temporary_wav(audio_path: Path) -> Path:
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{audio_path.stem}-normalize-",
            suffix=".wav",
            dir=audio_path.parent,
            delete=False,
        ) as temporary_file:
            return Path(temporary_file.name)
    except OSError as error:
        raise AudioNormalizationError(
            f"无法在音频目录中创建临时文件：{audio_path.parent}：{error}"
        ) from error


def _run_ffmpeg(ffmpeg_path: str, audio_path: Path, output_path: Path) -> None:
    try:
        subprocess.run(
            [
                ffmpeg_path,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio_path),
                "-vn",
                "-ar",
                str(TARGET_SAMPLE_RATE),
                "-ac",
                str(TARGET_CHANNELS),
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        error_message = error.stderr.strip() if error.stderr else str(error)
        raise AudioNormalizationError(
            f"音频标准化失败，原文件未修改：{audio_path}：{error_message}"
        ) from error
    except OSError as error:
        raise AudioNormalizationError(
            f"启动 ffmpeg 失败，原文件未修改：{audio_path}：{error}"
        ) from error


def _remove_temporary_file(temporary_path: Path) -> None:
    try:
        temporary_path.unlink(missing_ok=True)
    except OSError as error:
        logger.warning("删除音频标准化临时文件失败：%s：%s", temporary_path, error)


def _is_target_pcm_wav(audio_path: Path) -> bool:
    try:
        with wave.open(str(audio_path), "rb") as audio_file:
            return (
                audio_file.getframerate() == TARGET_SAMPLE_RATE
                and audio_file.getnchannels() == TARGET_CHANNELS
                and audio_file.getsampwidth() == TARGET_SAMPLE_WIDTH
                and audio_file.getcomptype() == "NONE"
                and audio_file.getnframes() > 0
            )
    except (wave.Error, EOFError, OSError):
        return False
