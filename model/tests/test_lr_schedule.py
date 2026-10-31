"""Independent cosine horizon, floor holding, and checkpoint compatibility."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config, ConfigError, TrainConfig, parse_config
from train import lr_at, _resume_signature


class LRScheduleTests(unittest.TestCase):
    def test_independent_decay_and_floor(self):
        cfg = TrainConfig(max_steps=50000, lr_decay_steps=20000, warmup=100)
        self.assertAlmostEqual(lr_at(0, cfg), cfg.lr / 100)
        self.assertAlmostEqual(lr_at(100, cfg), cfg.lr)
        self.assertAlmostEqual(lr_at(10050, cfg), cfg.lr * 0.55)
        for step in (20000, 25000, 49999, 50000, 60000):
            self.assertAlmostEqual(lr_at(step, cfg), cfg.lr * cfg.min_lr_frac)
        self.assertAlmostEqual(lr_at(15000, cfg),
                               lr_at(15000, replace(cfg, max_steps=100000)))

    def test_default_matches_legacy_schedule(self):
        cfg = TrainConfig(max_steps=50000)
        explicit = replace(cfg, lr_decay_steps=50000)
        for step in (0, 99, 100, 20000, 49999, 50000, 60000):
            self.assertEqual(lr_at(step, cfg), lr_at(step, explicit))
        self.assertNotIn("lr_decay_steps", _resume_signature(Config(train=cfg))["train"])
        self.assertEqual(_resume_signature(Config(train=explicit))["train"]["lr_decay_steps"], 50000)

    def test_config_validation_and_roundtrip(self):
        cfg = Config(train=TrainConfig(max_steps=50000, lr_decay_steps=20000))
        raw = cfg.to_dict()
        raw.pop("path", None)
        self.assertEqual(parse_config(raw).train, cfg.train)
        for value in (0, -1, True, 1.5, "20000", 100):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                TrainConfig(lr_decay_steps=value)


if __name__ == "__main__":
    unittest.main()
