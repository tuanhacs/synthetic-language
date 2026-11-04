"""Train-only noise edits local occurrences while preserving clean references."""
import json
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthdata.codebooks import Codebooks
from synthdata.config import (Config, ConfigError, CodeConfig, DataConfig, LanguageConfig,
    TaskConfig, TrainNoiseConfig, parse_config)
from synthdata.dataset import Splits
from synthdata.graphs import make_grid
from synthdata.language import Language
from synthdata.qa import LatentQASample, QASample, encode_qa
from synthdata.qa_noise import apply_train_noise, clean_sample, noise_report
from synthdata.qa_reversal import reverse_sample
from synthdata.storage import load_dataset, save_dataset


class TrainNoiseTests(unittest.TestCase):
    def setUp(self):
        self.language = Language(make_grid(2), Codebooks((("0",), ("10",), ("110",), ("111",))), (2, 10))
        self.task = TaskConfig(type="path-qa", query_len=(2, 3), segment_len=(2, 4))
        self.cfg = Config(language=LanguageConfig(graph="grid-2x2", k=1,
                            code=CodeConfig(depth_range=(1, 4), power_x=3)),
            data=DataConfig(walk_len=(2, 10), context_len=None), task=self.task)

    def samples(self, cot="none", n=80):
        latent = LatentQASample((0, 3, 0), (0, 1, 3, 2, 0), (3, 5))
        return tuple(encode_qa(latent, self.language, replace(self.task, cot_mode=cot),
            random.Random(i), f"sample:{i}") for i in range(n))

    def test_exact_rate_reproducibility_and_local_edits(self):
        samples = self.samples()
        cfg = TrainNoiseConfig()
        result, report = apply_train_noise(samples, cfg)
        self.assertEqual(result, apply_train_noise(samples, cfg)[0])
        self.assertEqual(report["corrupted_samples"], 4)
        self.assertEqual(report["actual_sample_rate"], .05)
        self.assertNotEqual(result, apply_train_noise(samples, replace(cfg, seed=7))[0])
        self.assertEqual(tuple(map(clean_sample, result)), samples)
        for clean, sample in zip(samples, result):
            self.assertEqual(sample.query_bits, clean.query_bits)
            self.assertEqual(sample.answer_bits, clean.answer_bits)
            self.assertEqual(len(sample.bits), len(clean.bits))
            if sample.noised_bits is None:
                self.assertIs(sample, clean)
                continue
            self.assertTrue(1 <= len(sample.noise_edits) <= 3)
            changed = {i for i, (a, b) in enumerate(zip(sample.bits, clean.bits)) if a != b}
            expected = set()
            seen = set()
            for edit in sample.noise_edits:
                key = edit["scope"], edit["codeword_index"]
                self.assertNotIn(key, seen)
                seen.add(key)
                cuts = sample.query_cuts if edit["scope"] == "query" else sample.answer_cuts
                i = edit["codeword_index"]
                start = cuts[i - 1] if i else 0
                offset = 0 if edit["scope"] == "query" else sample.answer_start
                positions = edit["bit_indices"]
                self.assertTrue(1 <= len(positions) <= 2)
                self.assertEqual(len(positions), len(set(positions)))
                expected.update(offset + start + p for p in positions)
            self.assertEqual(changed, expected)
            self.assertTrue(all(clean.bits[i] in "01" for i in changed))
            self.assertEqual(QASample.from_dict(json.loads(json.dumps(sample.to_dict()))), sample)
        self.assertEqual(report["flipped_bits"], sum(sum(a != b for a, b in zip(s.bits, s.clean_bits)) for s in result))
        self.assertEqual(apply_train_noise(samples, replace(cfg, sample_rate=0))[0], samples)
        self.assertEqual(apply_train_noise(samples[:3], cfg)[1]["corrupted_samples"], 0)

    def test_scope_cot_and_reversal(self):
        for cot in ("none", "segments"):
            for scope in ("query", "answer", "both"):
                with self.subTest(cot=cot, scope=scope):
                    samples = self.samples(cot, 10)
                    result, _ = apply_train_noise(samples, TrainNoiseConfig(sample_rate=1, scope=scope))
                    for original, s in zip(samples, result):
                        fields = s.bits.split("_")
                        self.assertEqual(fields[1:-1], list(original.cot_segments_bits))
                        if scope == "query":
                            self.assertEqual(fields[-1], original.answer_bits)
                        if scope == "answer":
                            self.assertEqual(fields[0], original.query_bits)
                        self.assertEqual(reverse_sample(reverse_sample(s)), s)
                        self.assertEqual(reverse_sample(s).bits,
                            "_".join((fields[0][::-1], *(p[::-1] for p in reversed(fields[1:-1])), fields[-1][::-1])))

    def test_serialized_clean_train_and_dataset_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clean = self.samples()
            noise_cfg = TrainNoiseConfig()
            noisy, report = apply_train_noise(clean, noise_cfg)
            cfg = replace(self.cfg, data=replace(self.cfg.data, train_noise=noise_cfg))
            save_dataset(root / "clean", self.cfg, self.language, Splits(clean, clean[:3], clean[3:6]))
            save_dataset(root / "noisy", cfg, self.language, Splits(noisy, clean[:3], clean[3:6]), extra={"train_noise": report})
            for name in ("codebook.json", "valid.jsonl", "test.jsonl"):
                self.assertEqual((root / "clean" / name).read_bytes(), (root / "noisy" / name).read_bytes())
            self.assertEqual((root / "clean/train.jsonl").read_bytes(), (root / "noisy/train.clean.jsonl").read_bytes())
            loaded = load_dataset(root / "noisy")
            self.assertEqual(loaded.splits.train, noisy)
            self.assertEqual(loaded.manifest["stats"]["train_noise"], report)
            self.assertEqual(loaded.manifest["files"]["train_clean"], "train.clean.jsonl")

    def test_config_validation_and_default_compatibility(self):
        self.assertNotIn("train_noise", self.cfg.to_dict()["data"])
        cfg = replace(self.cfg, data=replace(self.cfg.data, train_noise=TrainNoiseConfig()))
        self.assertEqual(parse_config(cfg.to_dict()), cfg)
        for kwargs in ({"sample_rate": -1}, {"sample_rate": float("nan")}, {"sample_rate": True},
                       {"scope": "cot"}, {"seed": True}, {"codeword_occurrences": (0, 3)},
                       {"bit_flips_per_occurrence": (2, 1)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigError):
                TrainNoiseConfig(**kwargs)
        with self.assertRaises(ConfigError):
            parse_config(replace(cfg, task=TaskConfig()).to_dict())
        with self.assertRaises(ValueError):
            apply_train_noise(apply_train_noise(self.samples(), TrainNoiseConfig())[0], TrainNoiseConfig())

    def test_generator_cli_preserves_clean_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = replace(self.cfg, data=replace(self.cfg.data, pool_tokens=None,
                pool_query_pairs=200, split=(60., 20., 20.)))
            script = Path(__file__).resolve().parents[1] / "scripts/gen_dataset.py"
            for name, config in (("clean", cfg), ("noisy", replace(cfg,
                    data=replace(cfg.data, train_noise=TrainNoiseConfig(sample_rate=.05))))):
                path = root / f"{name}.yaml"
                path.write_text(yaml.safe_dump(config.to_dict()))
                subprocess.run([sys.executable, "-B", str(script), str(path), "--out", str(root / name),
                                "--entropy-samples", "0"], check=True, capture_output=True, text=True)
            clean = load_dataset(root / "clean")
            noisy = load_dataset(root / "noisy")
            self.assertEqual(tuple(map(clean_sample, noisy.splits.train)), clean.splits.train)
            report = noisy.manifest["stats"]["train_noise"]
            self.assertEqual(report["corrupted_samples"], int(len(clean.splits.train) * .05))
            self.assertGreater(report["corrupted_samples"], 0)
            for name in ("codebook.json", "valid.jsonl", "test.jsonl"):
                self.assertEqual((root / "clean" / name).read_bytes(), (root / "noisy" / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
