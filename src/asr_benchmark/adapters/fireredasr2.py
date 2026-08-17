"""FireRedASR2-AED adapter for the shared ASR benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import ConfiguredDirectory, find_missing_files

MODEL_DIRECTORY_ENV = "FIREREDASR2_MODEL_DIR"
DEFAULT_MODEL_DIRECTORY = Path("pretrained_models/FireRedASR2-AED")
MODEL_DIRECTORY_CONFIG = ConfiguredDirectory(
    environment_variable=MODEL_DIRECTORY_ENV,
    default_directory=DEFAULT_MODEL_DIRECTORY,
)
MODEL_REPORT_NAME = "FireRedTeam/FireRedASR2-AED (FP32, 未量化)"
MODEL_DOWNLOAD_COMMAND = (
    "HF_HUB_DISABLE_XET=1 uv run --extra fireredasr2s hf download "
    "FireRedTeam/FireRedASR2-AED --local-dir pretrained_models/FireRedASR2-AED"
)
REQUIRED_MODEL_FILES = (
    "cmvn.ark",
    "dict.txt",
    "model.pth.tar",
    "train_bpe1000.model",
)
DEFAULT_BEAM_SIZE = 3


class FireRedAsr2AedAdapter:
    """Run FireRedASR2-AED on CUDA when available, otherwise on CPU."""

    def __init__(self, model_directory: Path | None = None) -> None:
        self._model_directory = resolve_model_directory(model_directory)
        validate_model_directory(self._model_directory)
        torch, asr_class, config_class = import_fireredasr2()
        self._torch = torch
        self.name = MODEL_REPORT_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = config_class(
            use_gpu=self.device == "cuda",
            use_half=False,
            beam_size=DEFAULT_BEAM_SIZE,
        )
        self._asr: Any | None = asr_class.from_pretrained(
            "aed",
            str(self._model_directory),
            config,
        )

    def transcribe(self, audio_path: Path) -> str:
        if self._asr is None:
            raise RuntimeError("FireRedASR2-AED 适配器已经释放")
        results = self._asr.transcribe(
            [audio_path.stem],
            [str(audio_path)],
        )
        if not results or not isinstance(results[0], dict):
            raise RuntimeError("FireRedASR2-AED 没有返回有效识别结果")
        text = results[0].get("text")
        if not isinstance(text, str):
            raise TypeError("FireRedASR2-AED 识别结果缺少 text 字段")
        return text

    def close(self) -> None:
        self._asr = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def resolve_model_directory(model_directory: Path | None) -> Path:
    return MODEL_DIRECTORY_CONFIG.resolve(model_directory)


def validate_model_directory(model_directory: Path) -> None:
    missing_files = find_missing_files(model_directory, REQUIRED_MODEL_FILES)
    if missing_files:
        missing_text = ", ".join(missing_files)
        raise FileNotFoundError(
            f"FireRedASR2-AED 模型目录不完整：{model_directory}；"
            f"缺少 {missing_text}。下载命令：{MODEL_DOWNLOAD_COMMAND}"
        )


def import_fireredasr2() -> tuple[Any, Any, Any]:
    try:
        import torch
        from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
    except ImportError as error:
        raise RuntimeError(
            "FireRedASR2S 依赖尚未安装，请执行 uv sync --extra fireredasr2s --locked"
        ) from error
    return torch, FireRedAsr2, FireRedAsr2Config


def create_fireredasr2_aed() -> FireRedAsr2AedAdapter:
    return FireRedAsr2AedAdapter()
