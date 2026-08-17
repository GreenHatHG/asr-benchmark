from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from asr_benchmark.benchmark import SampleFailure
from asr_benchmark.markdown_reporting import (
    ReportSample,
    render_markdown_report,
    report_markdown_path_for_config,
    write_markdown_report,
)
from asr_benchmark.reporting import ModelSummary, RecognitionResult


class MarkdownReportingTest(unittest.TestCase):
    def test_report_groups_models_and_escapes_table_content(self) -> None:
        summaries = (
            ModelSummary("Qwen3-ASR-0.6B", "cpu", 1, 1, 0, 0.1, 1.0, 0.5),
            ModelSummary("Doubao-IME-ASR", "remote", 1, 1, 0, 0.1, 1.0, 0.5),
            ModelSummary("Qwen3-ASR-1.7B", "cpu", 1, 1, 0, 0.1, 1.0, 0.5),
        )
        results = (
            RecognitionResult("Qwen3-ASR-0.6B", "sample|1", "你好，世界", "你好世界"),
            RecognitionResult("Doubao-IME-ASR", "sample|1", "你好，世界", "结果二"),
            RecognitionResult("Qwen3-ASR-1.7B", "sample|1", "你好，世界", "结果|三"),
        )

        report = render_markdown_report(summaries, results)
        summary = report.split("## 样本", maxsplit=1)[0]

        self.assertLess(summary.index("|  | 1.7B |"), summary.index("| Doubao |"))
        self.assertEqual(
            report.count("| 系列 | 模型版本 | 字符错误率 | 识别文本 |"),
            1,
        )
        self.assertIn("| 参考 | 参考文本 | - | 你好，世界 |", report)
        self.assertIn("| Qwen3-ASR | 0.6B | 0.00% | 你好世界 |", report)
        self.assertIn("结果\\|三", report)
        self.assertIn("## 样本：sample|1", report)

    def test_failed_model_is_included_in_sample_table(self) -> None:
        summaries = (
            ModelSummary("Qwen3-ASR-0.6B", "cpu", 1, 1, 0, 0.1, 1.0, 0.5),
            ModelSummary("Doubao-IME-ASR", "remote", 1, 0, 1, 0.1, 1.0, 0.5),
        )
        results = (RecognitionResult("Qwen3-ASR-0.6B", "sample-1", "参考", "结果"),)

        report = render_markdown_report(
            summaries,
            results,
            samples=(
                ReportSample(
                    "sample-1",
                    "参考",
                    Path("audio/sample.wav"),
                    12.345,
                ),
            ),
            failures=(SampleFailure("Doubao-IME-ASR", "sample-1", "请求超时"),),
        )

        self.assertIn(
            "音频文件：sample.wav · 时长：12.35 秒 · 参考文本长度：2 个字符",
            report,
        )
        self.assertIn("| Doubao | IME-ASR | - | 识别失败：请求超时 |", report)

    def test_report_is_saved_next_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            config_path = directory / "benchmark.toml"
            output_path = report_markdown_path_for_config(config_path)

            write_markdown_report((), (), output_path)

            self.assertEqual(
                output_path,
                directory.resolve() / "benchmark.report.md",
            )
            self.assertTrue(output_path.read_text(encoding="utf-8").endswith("\n"))


if __name__ == "__main__":
    unittest.main()
