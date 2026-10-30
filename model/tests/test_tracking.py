"""No-network W&B tests using a fake SDK."""

import builtins
import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config, ConfigError, WandbConfig, parse_config
from tracking import RunTracker, log_evaluation


class FakeConfig(dict):
    def update(self, values, **kwargs):
        super().update(values)


class FakeRun:
    def __init__(self):
        self.summary = {}
        self.config = FakeConfig()
        self.records = []
        self.metrics = []
        self.exit_codes = []

    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def log(self, values):
        self.records.append(values)

    def finish(self, exit_code=0):
        self.exit_codes.append(exit_code)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        cfg = Config()
        self.cfg = replace(cfg, train=replace(cfg.train, out_dir=str(self.root)),
                           wandb=WandbConfig(mode="online", project="test-project", group="sweep", tags=("ud",)))
        self.sdk = types.ModuleType("wandb")
        self.created = []
        self.arguments = []

        def init(**kwargs):
            self.arguments.append(kwargs)
            run = FakeRun()
            self.created.append(run)
            return run

        self.sdk.init = init

    def test_disabled_does_not_import_sdk_or_create_files(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "wandb":
                raise AssertionError("disabled tracking must not import W&B")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            tracker = RunTracker(replace(self.cfg, wandb=WandbConfig()))
            tracker.summary({"steps": 1})
            tracker.finish()
        self.assertFalse(self.root.exists())

    def test_online_reuses_persisted_id_and_logs_custom_step(self):
        with patch.dict(sys.modules, {"wandb": self.sdk}):
            first = RunTracker(self.cfg)
            first.log_training({"step": 50, "lr": .001, "train_bits_per_token": .2,
                                "elapsed_s": 2., "early_stopping_bad_checks": 1,
                                "valid_bits_per_token": .3})
            first.summary({"steps": 50, "stopped_early": True})
            first.finish()
            second = RunTracker(self.cfg)
            second.finish()
        self.assertEqual(self.arguments[0]["id"], self.arguments[1]["id"])
        self.assertEqual(self.arguments[1]["resume"], "allow")
        record = self.created[0].records[0]
        self.assertEqual(record["trainer/step"], 50)
        self.assertEqual(record["valid/loss_bits_per_token"], .3)
        self.assertTrue(self.created[0].summary["train/stopped_early"])
        self.assertEqual(self.created[0].exit_codes, [0])
        self.assertEqual(json.loads((self.root / "wandb_run.json").read_text())["id"], self.arguments[0]["id"])

    def test_offline_avoids_unsupported_cloud_resume_and_new_train_gets_new_id(self):
        cfg = replace(self.cfg, wandb=replace(self.cfg.wandb, mode="offline"))
        with patch.dict(sys.modules, {"wandb": self.sdk}):
            RunTracker(cfg).finish()
            RunTracker(cfg).finish()
            RunTracker(cfg, resume=False).finish()
        self.assertNotIn("resume", self.arguments[0])
        self.assertEqual(self.arguments[0]["id"], self.arguments[1]["id"])
        self.assertNotEqual(self.arguments[1]["id"], self.arguments[2]["id"])

    def test_evaluation_reuses_training_run_and_logs_success(self):
        with patch.dict(sys.modules, {"wandb": self.sdk}):
            RunTracker(self.cfg).finish()
            log_evaluation(self.cfg, self.root / "best.pt", {
                "generation": {"path_qa": {"n_samples": 1000, "temperature": 1.,
                                           "answer_decodable_pct": 99., "semantic_success_pct": 85.}}
            }, 100)
        self.assertEqual(self.arguments[0]["id"], self.arguments[1]["id"])
        self.assertEqual(self.created[1].summary["test/decodable_pct"], 99.)
        self.assertEqual(self.created[1].summary["test/success_pct"], 85.)
        self.assertEqual(self.created[1].records[0]["trainer/step"], 100)

    def test_config_roundtrip_and_invalid_settings(self):
        raw = self.cfg.to_dict()
        raw.pop("path")
        self.assertEqual(parse_config(raw).wandb, self.cfg.wandb)
        with self.assertRaises(ConfigError):
            WandbConfig(mode="invalid")
        with self.assertRaises(ConfigError):
            parse_config({"wandb": {"tags": "not-a-list"}})

    def test_training_failure_finishes_tracking_as_failed(self):
        import train
        tracker = Mock()
        with patch.object(train, "RunTracker", return_value=tracker), \
             patch.object(train, "_train", side_effect=RuntimeError("training failed")):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                train.train(self.cfg)
        tracker.finish.assert_called_once_with(exit_code=1)


if __name__ == "__main__":
    unittest.main()
