"""Noisy prompts are scored against clean intent, never decoded as queries."""
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import evals
from config import Config, ConfigError, EvalConfig, ModelConfig, TestNoiseConfig, parse_config
from model import build_model
from train import save_checkpoint
from synthdata.codebooks import Codebooks
from synthdata.config import Config as DataConfig, LanguageConfig, CodeConfig, TaskConfig, TrainNoiseConfig
from synthdata.dataset import Splits
from synthdata.graphs import make_grid
from synthdata.language import Language
from synthdata.qa import LatentQASample, encode_qa
from synthdata.qa_noise import apply_query_noise, _bounded_count
from synthdata.storage import save_dataset
from synthdata.tokenizer import BitTokenizer


class QueryNoiseTests(unittest.TestCase):
    def test_recoverable_noise_keeps_clean_intent_in_both_answer_formats(self):
        self.language = Language(make_grid(2), Codebooks((("00000",), ("11100",),
                                 ("10011",), ("01111",))), (2, 10))
        noise = TestNoiseConfig(mode="recoverable", sample_rate=.5)
        self.assertEqual(noise.codeword_occurrences, (1, 1))
        for cot in ("none", "segments"):
            samples = self.samples(cot=cot)
            def responses(*args, **kwargs):
                return [prefix + s.clean_bits[len(s.query_bits) + 1:]
                        for prefix, s in zip(kwargs["prefix_bits"], samples)]
            with patch.object(evals, "generate", side_effect=responses) as generate:
                report = evals.eval_path_qa(self.model, self.language, self.tok, samples,
                    n_samples=10, test_noise=noise,
                    task_config={"query_len": [2, 4], "cot_mode": cot})
            self.assertEqual(report["semantic_success_pct"], 100.)
            self.assertEqual(report["test_noise"]["corrupted_samples"], 5)
            self.assertEqual(report["test_noise"]["requested_samples"], 5)
            self.assertEqual(report["test_noise"]["skipped_no_safe_flip"], 0)
            prompts = generate.call_args.kwargs["prefix_bits"]
            self.assertEqual(sum(prefix != sample.query_bits + "_"
                                 for prefix, sample in zip(prompts, samples)), 5)

    def setUp(self):
        self.language = Language(make_grid(2), Codebooks((("0",), ("10",), ("110",), ("111",))), (2, 10))
        self.tok = BitTokenizer(True)
        self.model = SimpleNamespace(cfg=SimpleNamespace(context_len=128))

    def samples(self, n=10, cot="none"):
        task = TaskConfig(type="path-qa", query_len=(2, 2), segment_len=(2, 4), cot_mode=cot)
        return tuple(encode_qa(LatentQASample((0, 3), (0, 1, 3), (3,)), self.language,
                     task, random.Random(i), f"test:{i}") for i in range(n))

    def evaluate(self, samples, noise, cot="none", fail_noisy=False, n=None):
        prompts = []
        def generate(*args, **kwargs):
            prompts.extend(kwargs["prefix_bits"])
            return [prefix + ("" if fail_noisy and prefix != s.query_bits + "_" else
                    s.clean_bits[len(s.query_bits) + 1:])
                    for i, prefix in enumerate(prompts)
                    for s in [samples[i % len(samples)]]]
        with patch.object(evals, "generate", side_effect=generate):
            result = evals.eval_path_qa(self.model, self.language, self.tok, samples,
                n_samples=n or len(samples), test_noise=noise, task_config={"cot_mode": cot})
        return result, prompts

    def test_bounds_are_capped_before_uniform_draw(self):
        rng = Mock()
        _bounded_count((1, 3), 2, rng)
        rng.randint.assert_called_once_with(1, 2)
        rng.reset_mock()
        _bounded_count((1, 2), 1, rng)
        rng.randint.assert_called_once_with(1, 1)
        samples = self.samples(100)
        noisy, _ = apply_query_noise(samples, TrainNoiseConfig(sample_rate=1, scope="query"))
        self.assertEqual({len(s.noise_edits) for s in noisy}, {1, 2})
        for original, s in zip(samples, noisy):
            self.assertEqual(s.answer_bits, original.answer_bits)
            self.assertTrue(all(e["scope"] == "query" and e["codeword_index"] < 2 for e in s.noise_edits))
            for e in s.noise_edits:
                if e["codeword_index"] == 0:
                    self.assertEqual(e["bit_indices"], (0,))
        with self.assertRaisesRegex(ValueError, "scope=query"):
            apply_query_noise(samples, TrainNoiseConfig())

    def test_only_answer_is_decoded_and_clean_intent_is_used(self):
        samples = self.samples()
        original = Language.is_decodable
        def check(language, bits):
            self.assertIn(bits, {s.answer_bits for s in samples})
            return original(language, bits)
        with patch.object(Language, "is_decodable", new=check):
            result, prompts = self.evaluate(samples, TestNoiseConfig())
        self.assertEqual(result["semantic_success_pct"], 100.)
        self.assertEqual(result["test_noise"]["by_group"]["noisy"]["n"], 10)
        self.assertEqual(result["test_noise"]["by_group"]["clean"]["n"], 0)
        self.assertIsNone(result["test_noise"]["by_group"]["clean"]["semantic_success_pct"])
        text = evals.format_report({"dataset_dir": "test", "model": {"size": "small"},
                                  "num_params": 1, "generation": {"path_qa": result}})
        self.assertIn("clean queries n=0", text)
        self.assertTrue(all(prefix != s.query_bits + "_" for prefix, s in zip(prompts, samples)))
        self.assertTrue(all(s.noised_bits is None for s in samples))
        self.assertEqual(result, self.evaluate(samples, TestNoiseConfig())[0])
        self.assertNotEqual(prompts, self.evaluate(samples, TestNoiseConfig(seed=3))[1])

    def test_partial_noise_subgroup_denominators_and_cot(self):
        for cot in ("none", "segments"):
            samples = self.samples(10, cot)
            result, _ = self.evaluate(samples, TestNoiseConfig(sample_rate=.3), cot, fail_noisy=True)
            self.assertEqual(result["semantic_success_pct"], 70.)
            groups = result["test_noise"]["by_group"]
            self.assertEqual(groups["noisy"]["n"], 3)
            self.assertEqual(groups["noisy"]["semantic_success_pct"], 0.)
            self.assertEqual(groups["clean"]["n"], 7)
            self.assertEqual(groups["clean"]["semantic_success_pct"], 100.)
            perfect, _ = self.evaluate(samples, TestNoiseConfig(), cot)
            if cot == "segments":
                self.assertEqual(perfect["cot"]["cot_and_semantic_success_pct"], 100.)

    def test_repeated_selection_keeps_the_same_prompt_and_zero_rate_is_clean(self):
        samples = self.samples(2)
        result, prompts = self.evaluate(samples, TestNoiseConfig(), n=6)
        self.assertEqual(prompts[:2], prompts[2:4])
        self.assertEqual(prompts[:2], prompts[4:])
        self.assertEqual(result["test_noise"]["test_samples"], 2)
        self.assertEqual(result["test_noise"]["by_group"]["noisy"]["n"], 6)
        result, prompts = self.evaluate(samples, TestNoiseConfig(sample_rate=0))
        self.assertEqual(prompts, [s.query_bits + "_" for s in samples])
        self.assertEqual(result["test_noise"]["by_group"]["clean"]["n"], 2)
        self.assertEqual(self.evaluate(samples, None)[0]["semantic_success_pct"], 100.)

    def test_config_roundtrip_and_invalid_parameters(self):
        cfg = Config(eval=EvalConfig(test_noise=TestNoiseConfig(sample_rate=.5, seed=7)))
        raw = cfg.to_dict()
        raw.pop("path")
        self.assertEqual(parse_config(raw).eval, cfg.eval)
        self.assertNotIn("test_noise", EvalConfig().to_dict())
        for noise in ({"sample_rate": 2}, {"codeword_occurrences": [0, 2]},
                      {"bit_flips_per_occurrence": "bad"}, {"scope": "answer"}):
            with self.subTest(noise=noise), self.assertRaises(ConfigError):
                parse_config({"eval": {"test_noise": noise}})

    def test_evaluate_cli_uses_separate_output_and_can_disable_noise(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            samples = self.samples(4)
            cfg = DataConfig(language=LanguageConfig(graph="grid-2x2", k=1,
                code=CodeConfig(depth_range=(1, 4))), task=TaskConfig(type="path-qa",
                query_len=(2, 2), segment_len=(2, 4)))
            save_dataset(root / "dataset", cfg, self.language, Splits(samples, samples, samples))
            run_cfg = Config(model=ModelConfig(vocab_size=6, d_model=8, n_heads=2, n_layers=1, context_len=64))
            checkpoint = root / "best.pt"
            save_checkpoint(checkpoint, build_model(run_cfg.model), run_cfg, 1, .1)
            clean_report = root / "eval_report.json"
            clean_report.write_text("clean result must not be overwritten")
            script = Path(__file__).resolve().parents[1] / "scripts/evaluate.py"
            command = [sys.executable, "-B", str(script), str(checkpoint), "--dataset", str(root / "dataset"),
                       "--device", "cpu", "--n-samples", "2", "--max-len", "32"]
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            subprocess.run(command + ["--test-noise"], check=True, capture_output=True, text=True, env=env)
            self.assertEqual(clean_report.read_text(), "clean result must not be overwritten")
            noisy = json.loads((root / "eval_noisy/eval_report.json").read_text())
            self.assertEqual(noisy["generation"]["path_qa"]["test_noise"]["corrupted_samples"], 2)
            subprocess.run(command + ["--no-test-noise"], check=True, capture_output=True, text=True, env=env)
            clean = json.loads(clean_report.read_text())
            self.assertNotIn("test_noise", clean["generation"]["path_qa"])


if __name__ == "__main__":
    unittest.main()
