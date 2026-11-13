"""Independent cosine horizon, floor holding, and checkpoint compatibility."""

import sys
import runpy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

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

    def test_plateau_config_roundtrip_and_signature(self):
        cfg = Config(train=TrainConfig(lr_scheduler="plateau", lr_plateau_factor=.5,
                     lr_plateau_patience=4, lr_plateau_threshold=.0001))
        raw = cfg.to_dict()
        raw.pop("path")
        self.assertEqual(parse_config(raw).train, cfg.train)
        self.assertNotEqual(_resume_signature(cfg), _resume_signature(Config()))
        self.assertNotEqual(_resume_signature(cfg),
                            _resume_signature(replace(cfg, train=replace(cfg.train, eval_batches=99))))
        self.assertNotIn("lr_scheduler", _resume_signature(Config())["train"])
        for kwargs in ({"lr_scheduler": "unknown"}, {"lr_plateau_factor": 1.},
                       {"lr_plateau_factor": 0.}, {"lr_plateau_factor": float("nan")},
                       {"lr_plateau_patience": -1}, {"lr_plateau_patience": True},
                       {"lr_plateau_threshold": float("nan")},
                       {"lr_scheduler": "plateau", "min_lr_frac": 0.},
                       {"lr_scheduler": "plateau", "lr_decay_steps": 20000}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigError):
                TrainConfig(**kwargs)

    def test_train_cli_passes_plateau_options_to_training(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            path.write_text("train:\n  max_steps: 50000\n")
            command = ["train.py", str(path), "--lr-scheduler", "plateau",
                       "--lr-plateau-factor", "0.4", "--lr-plateau-patience", "6",
                       "--lr-plateau-threshold", "0.0002"]
            with patch.object(sys, "argv", command), patch.object(sys, "path", list(sys.path)), \
                 patch("train.train") as run:
                with self.assertRaises(SystemExit) as exit:
                    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/train.py"),
                                   run_name="__main__")
                self.assertEqual(exit.exception.code, 0)
            cfg = run.call_args.args[0]
            self.assertEqual(cfg.train.lr_scheduler, "plateau")
            self.assertEqual(cfg.train.lr_plateau_factor, .4)
            self.assertEqual(cfg.train.lr_plateau_patience, 6)
            self.assertEqual(cfg.train.lr_plateau_threshold, .0002)


if __name__ == "__main__":
    unittest.main()
