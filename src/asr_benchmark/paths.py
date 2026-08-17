"""项目内共享的文件系统路径。"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ConfiguredDirectory:
    """保存一个可由环境变量覆盖的项目目录配置。"""

    environment_variable: str
    default_directory: Path

    def resolve(self, explicit_directory: Path | None = None) -> Path:
        """优先使用显式目录，其次读取环境变量和默认值。"""

        configured_directory = explicit_directory or Path(
            os.environ.get(self.environment_variable, self.default_directory)
        )
        if not configured_directory.is_absolute():
            configured_directory = PROJECT_ROOT / configured_directory
        return configured_directory.expanduser().resolve()


def find_missing_files(
    directory: Path,
    required_filenames: Sequence[str],
) -> tuple[str, ...]:
    """返回指定目录中不存在的必需文件名。"""

    return tuple(
        filename
        for filename in required_filenames
        if not (directory / filename).is_file()
    )
