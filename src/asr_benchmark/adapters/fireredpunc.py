"""使用官方 FireRedPunc 为 ASR 识别文本补充标点。"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT

MODEL_DIRECTORY_ENV = "FIREREDPUNC_MODEL_DIR"
DEFAULT_MODEL_DIRECTORY = Path("pretrained_models/FireRedPunc")
REQUIRED_MODEL_FILES = (
    "chinese-bert-wwm-ext_vocab.txt",
    "chinese-lert-base/config.json",
    "chinese-lert-base/pytorch_model.bin",
    "chinese-lert-base/vocab.txt",
    "model.pth.tar",
    "out_dict",
)


class FireRedPuncAdapter:
    """在 CPU 上运行 FireRedPunc。"""

    def __init__(self, model_directory: Path | None = None) -> None:
        self._model_directory = resolve_model_directory(model_directory)
        validate_model_directory(self._model_directory)
        try:
            import torch
            from fireredasr2s.fireredpunc.punc import FireRedPunc, FireRedPuncConfig
        except ImportError as error:
            raise RuntimeError(
                "FireRedPunc 依赖尚未安装，请执行 uv sync --extra fireredasr2s --locked"
            ) from error

        self._torch = torch
        self._punc: Any | None = FireRedPunc.from_pretrained(
            str(self._model_directory),
            FireRedPuncConfig(use_gpu=False),
        )

    def restore(self, text: str) -> str:
        if not text:
            return text
        if self._punc is None:
            raise RuntimeError("FireRedPunc 适配器已经释放")
        results = self._punc.process([text])
        if not results or not isinstance(results[0], dict):
            raise RuntimeError("FireRedPunc 没有返回有效处理结果")
        punctuated_text = results[0].get("punc_text")
        if not isinstance(punctuated_text, str):
            raise TypeError("FireRedPunc 处理结果缺少 punc_text 字段")
        return punctuated_text

    def close(self) -> None:
        self._punc = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


class PunctuatedAsrAdapter:
    """组合一个 ASR 适配器和 FireRedPunc。"""

    def __init__(self, asr_factory: Callable[[], Any]) -> None:
        self._asr = asr_factory()
        try:
            self._punc = FireRedPuncAdapter()
        except Exception:
            close = getattr(self._asr, "close", None)
            if callable(close):
                close()
            raise
        self.name = f"{self._asr.name} + FireRedPunc"
        self.device = self._asr.device
        self._last_raw_text: str | None = None

    def transcribe(self, audio_path: Path) -> str:
        raw_text = self._asr.transcribe(audio_path)
        self._last_raw_text = raw_text
        return self._punc.restore(raw_text)

    def get_last_raw_text(self) -> str | None:
        """返回最近一次送入 FireRedPunc 前的 ASR 原始文本。"""

        return self._last_raw_text

    def close(self) -> None:
        try:
            self._punc.close()
        finally:
            self._asr.close()


def resolve_model_directory(model_directory: Path | None) -> Path:
    configured_directory = model_directory or Path(
        os.environ.get(MODEL_DIRECTORY_ENV, DEFAULT_MODEL_DIRECTORY)
    )
    if not configured_directory.is_absolute():
        configured_directory = PROJECT_ROOT / configured_directory
    return configured_directory.expanduser().resolve()


def validate_model_directory(model_directory: Path) -> None:
    missing_files = [
        filename
        for filename in REQUIRED_MODEL_FILES
        if not (model_directory / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"FireRedPunc 模型目录不完整：{model_directory}；"
            f"缺少 {', '.join(missing_files)}。"
        )


def create_punctuated_adapter(asr_factory: Callable[[], Any]) -> PunctuatedAsrAdapter:
    return PunctuatedAsrAdapter(asr_factory)
