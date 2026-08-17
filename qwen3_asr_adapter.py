"""Qwen3-ASR 官方 Transformers 模型适配器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

QWEN3_ASR_0_6B_MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"
QWEN3_ASR_1_7B_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
EXPECTED_SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 256


class Qwen3AsrAdapter:
    """使用 Transformers 原生接口运行一个官方 Qwen3-ASR 模型。"""

    def __init__(self, model_id: str) -> None:
        torch, soundfile, auto_model, auto_processor = import_dependencies()
        self._torch = torch
        self._soundfile = soundfile
        self._processor: Any | None = auto_processor.from_pretrained(model_id)
        self._model: Any | None = auto_model.from_pretrained(
            model_id,
            dtype="auto",
            low_cpu_mem_usage=True,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self._model.to(self.device)
        dtype_name = str(self._model.dtype).removeprefix("torch.")
        self.name = f"{model_id} ({dtype_name})"

    def transcribe(self, audio_path: Path) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("Qwen3-ASR 适配器已经释放")

        audio, sample_rate = self._soundfile.read(audio_path, dtype="float32")
        if sample_rate != EXPECTED_SAMPLE_RATE or audio.ndim != 1:
            raise ValueError("Qwen3-ASR 音频必须是 16kHz 单声道")

        inputs = self._processor.apply_transcription_request(audio=audio).to(
            self._model.device,
            self._model.dtype,
        )
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        parsed_outputs = self._processor.decode(
            generated_ids,
            return_format="parsed",
        )
        if not parsed_outputs or not isinstance(parsed_outputs[0], dict):
            raise RuntimeError("Qwen3-ASR 没有返回有效识别结果")
        transcription = parsed_outputs[0].get("transcription")
        if not isinstance(transcription, str):
            raise TypeError("Qwen3-ASR 识别结果缺少 transcription 字段")
        return transcription

    def close(self) -> None:
        self._model = None
        self._processor = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def import_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import soundfile
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as error:
        raise RuntimeError("Qwen3-ASR 依赖尚未安装，请执行 uv sync --locked") from error
    return torch, soundfile, AutoModelForMultimodalLM, AutoProcessor


def create_qwen3_asr_0_6b() -> Qwen3AsrAdapter:
    return Qwen3AsrAdapter(QWEN3_ASR_0_6B_MODEL_ID)


def create_qwen3_asr_1_7b() -> Qwen3AsrAdapter:
    return Qwen3AsrAdapter(QWEN3_ASR_1_7B_MODEL_ID)
