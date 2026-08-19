from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_benchmark.benchmark import (
    AudioSample,
    ModelAdapter,
    benchmark_models,
)
from asr_benchmark.transcription_cache import (
    CachedTranscription,
    TranscriptionCache,
    TranscriptionCacheError,
)

MODEL_NAME = "Doubao-IME-ASR"


class TranscriptionCacheTest(unittest.TestCase):
    def test_default_overwrites_cache_and_resume_loads_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "benchmark.toml.cache.json"
            cache = TranscriptionCache.prepare(cache_path, resume=False)
            cache.set_model_metadata(MODEL_NAME, "报告模型", "cpu", 0.5)
            cache.set_transcription(
                MODEL_NAME,
                "sample-1",
                CachedTranscription(None, "旧结果", 1.25),
            )

            resumed = TranscriptionCache.prepare(
                cache_path,
                resume=True,
            )
            resumed_model = resumed.get_model(MODEL_NAME)
            self.assertIsNotNone(resumed_model)
            assert resumed_model is not None
            self.assertEqual(
                resumed_model.transcriptions["sample-1"].hypothesis,
                "旧结果",
            )

            overwritten = TranscriptionCache.prepare(
                cache_path,
                resume=False,
            )
            self.assertIsNone(overwritten.get_model(MODEL_NAME))
            self.assertNotIn(
                "version",
                json.loads(cache_path.read_text(encoding="utf-8")),
            )

    def test_resume_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "benchmark.toml.cache.json"
            cache_path.write_text("不是 JSON", encoding="utf-8")

            with self.assertRaisesRegex(
                TranscriptionCacheError,
                "读取转写缓存失败",
            ):
                TranscriptionCache.prepare(cache_path, resume=True)

    def test_resume_rejects_invalid_cache_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "benchmark.toml.cache.json"
            cache_path.write_text(
                '{"models": []}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                TranscriptionCacheError,
                "models 必须是对象",
            ):
                TranscriptionCache.prepare(
                    cache_path,
                    resume=True,
                )

    def test_resume_rejects_number_too_large_for_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "benchmark.toml.cache.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "models": {
                            MODEL_NAME: {
                                "report_name": "缓存模型",
                                "device": "cpu",
                                "load_latency_seconds": 10**1000,
                                "transcriptions": {},
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                TranscriptionCacheError,
                "load_latency_seconds 必须是大于或等于 0 的数字",
            ):
                TranscriptionCache.prepare(cache_path, resume=True)

    def test_resume_ignores_legacy_and_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "benchmark.toml.cache.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "configuration_fingerprint": "旧摘要",
                        "unknown": "忽略",
                        "models": {
                            MODEL_NAME: {
                                "report_name": "缓存模型",
                                "device": "cpu",
                                "load_latency_seconds": 0.5,
                                "transcriptions": {
                                    "sample-1": {
                                        "raw_hypothesis": None,
                                        "hypothesis": "缓存结果",
                                        "elapsed_seconds": 2.0,
                                        "runs": 2,
                                        "unknown": "忽略",
                                    }
                                },
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            cache = TranscriptionCache.prepare(cache_path, resume=True)

            cached_model = cache.get_model(MODEL_NAME)
            self.assertIsNotNone(cached_model)
            assert cached_model is not None
            self.assertEqual(
                cached_model.transcriptions["sample-1"],
                CachedTranscription(None, "缓存结果", 2.0, 2),
            )

    def test_resume_rejects_invalid_transcription_fields(self) -> None:
        invalid_fields = (
            ("raw_hypothesis", 1, "raw_hypothesis 必须是字符串或 null"),
            ("hypothesis", None, "hypothesis 必须是字符串"),
            ("elapsed_seconds", -1, "elapsed_seconds 必须是大于或等于 0 的数字"),
            ("runs", 0, "runs 必须是大于或等于 1 的整数"),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "benchmark.toml.cache.json"
            for field_name, invalid_value, expected_message in invalid_fields:
                with self.subTest(field_name=field_name):
                    transcription = {
                        "raw_hypothesis": None,
                        "hypothesis": "缓存结果",
                        "elapsed_seconds": 2.0,
                        "runs": 2,
                    }
                    transcription[field_name] = invalid_value
                    cache_path.write_text(
                        json.dumps(
                            {
                                "models": {
                                    MODEL_NAME: {
                                        "report_name": "缓存模型",
                                        "device": "cpu",
                                        "load_latency_seconds": 0.5,
                                        "transcriptions": {
                                            "sample-1": transcription,
                                        },
                                    }
                                }
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(
                        TranscriptionCacheError,
                        expected_message,
                    ):
                        TranscriptionCache.prepare(cache_path, resume=True)

    def test_complete_cache_skips_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            cache = TranscriptionCache.prepare(
                directory / "benchmark.toml.cache.json",
                resume=False,
            )
            samples = self._samples(directory)
            cache.set_model_metadata(MODEL_NAME, "缓存模型", "cpu", 0.5)
            cache.set_transcription(
                MODEL_NAME,
                "sample-1",
                CachedTranscription("原文一", "结果一", 1.0, 1),
            )
            cache.set_transcription(
                MODEL_NAME,
                "sample-2",
                CachedTranscription(None, "结果二", 2.0, 1),
            )

            with patch(
                "asr_benchmark.benchmark.load_adapter",
                side_effect=AssertionError("完整缓存不应加载模型"),
            ):
                summaries, results, failures = benchmark_models(
                    (MODEL_NAME,),
                    samples,
                    runs=1,
                    warmup_runs=0,
                    cache=cache,
                )

            self.assertEqual(
                [result.hypothesis for result in results], ["结果一", "结果二"]
            )
            self.assertEqual(summaries[0].successful_samples, 2)
            self.assertEqual(summaries[0].average_latency_seconds, 1.5)
            self.assertEqual(failures, [])

    def test_partial_cache_only_transcribes_missing_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            cache_path = directory / "benchmark.toml.cache.json"
            cache = TranscriptionCache.prepare(
                cache_path,
                resume=False,
            )
            samples = self._samples(directory)
            cache.set_model_metadata(MODEL_NAME, "缓存模型", "cpu", 0.5)
            cache.set_transcription(
                MODEL_NAME,
                "sample-1",
                CachedTranscription(None, "缓存结果", 1.0, 1),
            )
            transcribed_paths: list[Path] = []

            def transcribe(audio_path: Path) -> str:
                transcribed_paths.append(audio_path)
                return "新结果"

            adapter = ModelAdapter("当前模型", transcribe, None, "cpu")
            with patch("asr_benchmark.benchmark.load_adapter", return_value=adapter):
                _, results, failures = benchmark_models(
                    (MODEL_NAME,),
                    samples,
                    runs=1,
                    warmup_runs=0,
                    cache=cache,
                )

            self.assertEqual(transcribed_paths, [samples[1].audio_path])
            self.assertEqual(
                [result.hypothesis for result in results], ["缓存结果", "新结果"]
            )
            self.assertEqual(failures, [])
            reloaded = TranscriptionCache.prepare(
                cache_path,
                resume=True,
            )
            reloaded_model = reloaded.get_model(MODEL_NAME)
            self.assertIsNotNone(reloaded_model)
            assert reloaded_model is not None
            self.assertEqual(
                reloaded_model.transcriptions["sample-2"].hypothesis,
                "新结果",
            )

    @staticmethod
    def _samples(directory: Path) -> tuple[AudioSample, AudioSample]:
        return (
            AudioSample("sample-1", directory / "one.wav", "参考一", 2.0),
            AudioSample("sample-2", directory / "two.wav", "参考二", 4.0),
        )


if __name__ == "__main__":
    unittest.main()
