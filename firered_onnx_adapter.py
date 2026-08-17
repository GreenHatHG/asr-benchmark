"""基于 sherpa-onnx 的 FireRedASR2 CTC/AED 适配器。"""

from __future__ import annotations

import array
import os
import wave
from pathlib import Path
from typing import Any

SHERPA_NUM_THREADS = 2
DEFAULT_CTC_MODEL_DIRECTORY = Path(
    "pretrained_models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25"
)
DEFAULT_AED_INT8_MODEL_DIRECTORY = Path(
    "pretrained_models/sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26"
)


class SherpaFireRedAsrAdapter:
    """使用 sherpa-onnx 运行一个 FireRedASR2 ONNX 模型。"""

    def __init__(
        self,
        model_directory: Path,
        *,
        model_kind: str,
        name: str,
        required_files: tuple[str, ...],
    ) -> None:
        self._model_directory = resolve_directory(model_directory)
        validate_model_directory(self._model_directory, required_files, name)
        try:
            import sherpa_onnx
        except ImportError as error:
            raise RuntimeError(
                "缺少 sherpa-onnx 依赖，请执行 uv sync --extra fireredasr2s --locked"
            ) from error

        self.name = name
        self.device = "cpu"
        self._recognizer = create_recognizer(
            sherpa_onnx,
            self._model_directory,
            model_kind,
        )

    def transcribe(self, audio_path: Path) -> str:
        samples = read_audio_samples(audio_path)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        self._recognizer.decode_stream(stream)
        result = stream.result
        text = getattr(result, "text", None)
        if not isinstance(text, str):
            raise TypeError(f"{self.name} 没有返回有效识别文本")
        return text

    def close(self) -> None:
        self._recognizer = None


def create_recognizer(
    sherpa_onnx: Any,
    model_directory: Path,
    model_kind: str,
) -> Any:
    """调用 sherpa-onnx 当前版本提供的 FireRed 工厂方法。"""

    if model_kind == "ctc":
        factory = sherpa_onnx.OfflineRecognizer.from_fire_red_asr_ctc
        return factory(
            model=str(model_directory / "model.int8.onnx"),
            tokens=str(model_directory / "tokens.txt"),
            num_threads=SHERPA_NUM_THREADS,
            provider="cpu",
        )

    if model_kind == "aed":
        factory = sherpa_onnx.OfflineRecognizer.from_fire_red_asr
        return factory(
            encoder=str(model_directory / "encoder.int8.onnx"),
            decoder=str(model_directory / "decoder.int8.onnx"),
            tokens=str(model_directory / "tokens.txt"),
            num_threads=SHERPA_NUM_THREADS,
            provider="cpu",
        )

    raise ValueError(f"未知的 FireRedASR2 ONNX 类型：{model_kind}")


def read_audio_samples(audio_path: Path) -> list[float]:
    """读取 16 kHz、16 位、单声道 PCM WAV，转换为 sherpa-onnx 所需浮点样本。"""

    try:
        with wave.open(str(audio_path), "rb") as audio_file:
            if (
                audio_file.getframerate() != 16000
                or audio_file.getnchannels() != 1
                or audio_file.getsampwidth() != 2
            ):
                raise ValueError("音频必须是 16kHz、16 位、单声道 PCM WAV")
            frames = audio_file.readframes(audio_file.getnframes())
    except (wave.Error, EOFError, OSError) as error:
        raise RuntimeError(f"读取音频失败：{audio_path}：{error}") from error

    pcm_samples = array.array("h")
    pcm_samples.frombytes(frames)
    if os.sys.byteorder != "little":
        pcm_samples.byteswap()
    return [sample / 32768.0 for sample in pcm_samples]


def resolve_directory(model_directory: Path) -> Path:
    if not model_directory.is_absolute():
        model_directory = Path(__file__).resolve().parent / model_directory
    return model_directory.expanduser().resolve()


def validate_model_directory(
    model_directory: Path,
    required_files: tuple[str, ...],
    model_name: str,
) -> None:
    missing_files = [
        filename
        for filename in required_files
        if not (model_directory / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{model_name} 模型目录不完整：{model_directory}；"
            f"缺少 {', '.join(missing_files)}。"
        )


def create_fireredasr2_ctc_int8() -> SherpaFireRedAsrAdapter:
    return SherpaFireRedAsrAdapter(
        Path(os.environ.get("FIREREDASR2_CTC_MODEL_DIR", DEFAULT_CTC_MODEL_DIRECTORY)),
        model_kind="ctc",
        name="FireRedASR2-CTC (sherpa-onnx INT8)",
        required_files=("model.int8.onnx", "tokens.txt"),
    )


def create_fireredasr2_aed_int8() -> SherpaFireRedAsrAdapter:
    return SherpaFireRedAsrAdapter(
        Path(
            os.environ.get(
                "FIREREDASR2_AED_INT8_MODEL_DIR", DEFAULT_AED_INT8_MODEL_DIRECTORY
            )
        ),
        model_kind="aed",
        name="FireRedASR2-AED (sherpa-onnx INT8)",
        required_files=("encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"),
    )
