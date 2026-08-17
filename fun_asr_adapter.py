"""Fun-ASR-Nano 官方模型与 MLX 量化模型适配器。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

OFFICIAL_MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
OFFICIAL_MODEL_DIRECTORY_ENV = "FUN_ASR_NANO_MODEL_DIR"
MLX_8BIT_MODEL_ID = "mlx-community/Fun-ASR-Nano-2512-8bit"
MLX_4BIT_MODEL_ID = "mlx-community/Fun-ASR-Nano-2512-4bit"
MAX_NEW_TOKENS = 512


class OfficialFunAsrNanoAdapter:
    """使用 FunASR 官方 PyTorch 实现运行原始 Nano-2512 权重。"""

    def __init__(self) -> None:
        torch, auto_model = import_official_dependencies()
        self._torch = torch
        self.device = select_torch_device(torch)
        self.name = f"{OFFICIAL_MODEL_ID} (官方原始权重)"
        self._model: Any | None = auto_model(
            model=resolve_official_model_reference(),
            hub="hf",
            trust_remote_code=True,
            device=self.device,
            disable_update=True,
        )

    def transcribe(self, audio_path: Path) -> str:
        if self._model is None:
            raise RuntimeError("Fun-ASR-Nano 官方适配器已经释放")

        results = self._model.generate(
            input=[str(audio_path)],
            cache={},
            batch_size=1,
            itn=True,
            max_length=MAX_NEW_TOKENS,
            llm_kwargs={"do_sample": False},
        )
        return extract_official_text(results)

    def close(self) -> None:
        self._model = None
        if self.device == "mps":
            self._torch.mps.empty_cache()
        elif self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()


class MlxFunAsrNanoAdapter:
    """使用 mlx-audio-plus 运行一个 Fun-ASR-Nano 量化权重。"""

    def __init__(self, model_id: str, quantization: str) -> None:
        mlx, model_class = import_mlx_dependencies()
        self._mlx = mlx
        self._model: Any | None = model_class.from_pretrained(model_id)
        self.name = f"{model_id} (MLX {quantization})"
        self.device = "metal"

    def transcribe(self, audio_path: Path) -> str:
        if self._model is None:
            raise RuntimeError("Fun-ASR-Nano MLX 适配器已经释放")

        result = self._model.generate(
            str(audio_path),
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
        )
        text = getattr(result, "text", None)
        if not isinstance(text, str):
            raise TypeError("Fun-ASR-Nano MLX 识别结果缺少 text 字段")
        return text

    def close(self) -> None:
        self._model = None
        self._mlx.clear_cache()


def select_torch_device(torch: Any) -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_official_model_reference() -> str:
    configured_directory = os.environ.get(OFFICIAL_MODEL_DIRECTORY_ENV)
    if configured_directory is None:
        return OFFICIAL_MODEL_ID

    model_directory = Path(configured_directory).expanduser()
    if not model_directory.is_absolute():
        model_directory = Path(__file__).resolve().parent / model_directory
    return str(model_directory.resolve())


def extract_official_text(results: Any) -> str:
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        raise RuntimeError("Fun-ASR-Nano 官方模型没有返回有效识别结果")
    text = results[0].get("text")
    if not isinstance(text, str):
        raise TypeError("Fun-ASR-Nano 官方识别结果缺少 text 字段")
    return text


def import_official_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        from funasr import AutoModel
    except ImportError as error:
        raise RuntimeError(
            "Fun-ASR 官方依赖尚未安装，请执行 uv sync --extra funasr --locked"
        ) from error
    return torch, AutoModel


def import_mlx_dependencies() -> tuple[Any, Any]:
    try:
        import mlx.core as mlx
        from mlx_audio.stt.models.funasr import Model
    except ImportError as error:
        raise RuntimeError(
            "Fun-ASR MLX 依赖尚未安装，请执行 uv sync --extra funasr --locked"
        ) from error
    return mlx, Model


def create_fun_asr_nano_official() -> OfficialFunAsrNanoAdapter:
    return OfficialFunAsrNanoAdapter()


def create_fun_asr_nano_mlx_8bit() -> MlxFunAsrNanoAdapter:
    return MlxFunAsrNanoAdapter(MLX_8BIT_MODEL_ID, "8bit")


def create_fun_asr_nano_mlx_4bit() -> MlxFunAsrNanoAdapter:
    return MlxFunAsrNanoAdapter(MLX_4BIT_MODEL_ID, "4bit")
