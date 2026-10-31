import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_eval_harness.providers import PROVIDERS, make_judge, resolve  # noqa: E402


class TestProviders(unittest.TestCase):
    def test_known_providers_resolve(self):
        for name, provider in PROVIDERS.items():
            self.assertEqual(resolve(name), provider)

    def test_unknown_model_falls_back_to_openai(self):
        provider = resolve("some-unknown-model")
        self.assertEqual(provider.model, "some-unknown-model")
        self.assertEqual(provider.key_env, "OPENAI_API_KEY")

    def test_kimi_forces_temperature_one(self):
        self.assertEqual(resolve("kimi-k2.6").temperature, 1.0)

    def test_gemini_model_keeps_prefix(self):
        self.assertTrue(resolve("models/gemini-3.1-flash-lite").model.startswith("models/"))

    def test_missing_key_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                make_judge("deepseek-flash")

    def test_make_judge_uses_provider_settings(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            judge = make_judge("deepseek-flash")
        self.assertEqual(judge.model, "deepseek-flash")
        self.assertEqual(judge.base_url, "https://api.deepseek.com/v1")
        self.assertEqual(judge.temperature, 0.0)

    def test_make_judge_nvidia_hosted_deepseek(self):
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=True):
            judge = make_judge("nvidia/deepseek-v4.1-flash")
        self.assertEqual(judge.model, "deepseek-ai/deepseek-v4.1-flash")
        self.assertEqual(judge.base_url, "https://integrate.api.nvidia.com/v1")


if __name__ == "__main__":
    unittest.main()
