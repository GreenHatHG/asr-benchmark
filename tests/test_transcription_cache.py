from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_benchmark.benchmark import (
    AudioSample,
    ModelAdapter,
    benchmark_models,
    build_configuration_fingerprint,
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
            cache = TranscriptionCache.prepare(cache_path, "fingerprint", resume=False)
            cache.set_model_metadata(MODEL_NAME, "报告模型", "cpu", 0.5)
            cache.set_transcription(
                MODEL_NAME,
                "sample-1",
                CachedTranscription(None, "旧结果", 1.25),
            )

            resumed = TranscriptionCache.prepare(
                cache_path,
                "fingerprint",
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
                "fingerprint",
                resume=False,
            )
            self.assertIsNone(overwritten.get_model(MODEL_NAME))

    def test_resume_rejects_cache_from_different_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "benchmark.toml.cache.json"
            TranscriptionCache.prepare(cache_path, "old", resume=False)

            with self.assertRaisesRegex(
                TranscriptionCacheError,
                "与当前配置或音频不匹配",
            ):
                TranscriptionCache.prepare(cache_path, "new", resume=True)

    def test_resume_rejects_boolean_cache_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "benchmark.toml.cache.json"
            cache_path.write_text(
                '{"version": true, "configuration_fingerprint": "fingerprint", '
                '"models": {}}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                TranscriptionCacheError,
                "版本不受支持",
            ):
                TranscriptionCache.prepare(
                    cache_path,
                    "fingerprint",
                    resume=True,
                )

    def test_audio_content_change_changes_configuration_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            config_path = directory / "benchmark.toml"
            audio_path = directory / "sample.wav"
            config_path.write_text('models = ["Doubao-IME-ASR"]', encoding="utf-8")
            audio_path.write_bytes(b"first audio content")
            samples = (AudioSample("sample-1", audio_path, "参考", 1.0),)

            original = build_configuration_fingerprint(config_path, samples)
            audio_path.write_bytes(b"changed audio content")

            self.assertNotEqual(
                build_configuration_fingerprint(config_path, samples),
                original,
            )

    def test_complete_cache_skips_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            cache = TranscriptionCache.prepare(
                directory / "benchmark.toml.cache.json",
                "fingerprint",
                resume=False,
            )
            samples = self._samples(directory)
            cache.set_model_metadata(MODEL_NAME, "缓存模型", "cpu", 0.5)
            cache.set_transcription(
                MODEL_NAME,
                "sample-1",
                CachedTranscription("原文一", "结果一", 1.0),
            )
            cache.set_transcription(
                MODEL_NAME,
                "sample-2",
                CachedTranscription(None, "结果二", 2.0),
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
                "fingerprint",
                resume=False,
            )
            samples = self._samples(directory)
            cache.set_model_metadata(MODEL_NAME, "缓存模型", "cpu", 0.5)
            cache.set_transcription(
                MODEL_NAME,
                "sample-1",
                CachedTranscription(None, "缓存结果", 1.0),
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
                "fingerprint",
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
