"""CPU integration tests for step-based validation stopping and exact resume."""

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import train as training
from config import Config, ConfigError, ModelConfig, TrainConfig, parse_config
from data import BatchSampler
from synthdata.tokenizer import BitTokenizer


class EarlyStoppingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.windows = torch.tensor([[2, 0, 1, 0, 3], [2, 1, 0, 0, 3], [2, 1, 1, 0, 3],
                                     [2, 0, 0, 1, 3], [2, 1, 0, 1, 3]])
        self.data = SimpleNamespace(
            train=self.windows, valid=self.windows[:2], tokenizer=BitTokenizer(),
            dataset_dir=self.root, entropy_floor=None,
            manifest={"config_hash": "test-dataset", "config": {"task": {"type": "language-modeling"}}},
            token_counts=lambda: {"train": 25, "valid": 10},
        )

    def cfg(self, name, **kwargs):
        train = TrainConfig(max_steps=10, batch_size=2, device="cpu", warmup=0,
                            eval_batches=1, eval_every=3, checkpoint_every=4, seed=7,
                            out_dir=str(self.root / name), early_stopping_patience=2,
                            early_stopping_min_delta=0.1)
        return Config(model=ModelConfig(d_model=8, n_heads=2, n_layers=1, context_len=4),
                      train=replace(train, **kwargs))

    def run_train(self, cfg, losses):
        with patch.object(training, "load_frozen_train_valid", return_value=self.data), \
             patch.object(training, "bits_per_token", side_effect=losses):
            return training.train(cfg, verbose=False)

    def checkpoint(self, name):
        return torch.load(self.root / name / "latest.pt", map_location="cpu", weights_only=False)

    def test_random_sampler_behavior_and_state_unchanged(self):
        sampler = BatchSampler(self.windows, 2, seed=7)
        generator = torch.Generator().manual_seed(7)
        for _ in range(5):
            expected = torch.randint(0, len(self.windows), (2,), generator=generator)
            inputs, _ = sampler.batch()
            self.assertTrue(torch.equal(inputs, self.windows[expected, :-1]))
        self.assertEqual(set(sampler.state_dict()), {"generator_state"})

    def test_disabled_early_stopping_keeps_step_budget_and_validation_cadence(self):
        result = self.run_train(self.cfg("steps", max_steps=5,
                                        early_stopping_patience=0, eval_every=2), [1., .9, .8, .7])
        self.assertEqual(result["steps"], 5)
        self.assertEqual(result["stop_reason"], "max_steps")
        self.assertFalse(result["stopped_early"])
        self.assertEqual([r["step"] for r in result["history"] if r["valid_bits_per_token"] is not None], [1, 2, 4, 5])
        self.assertNotIn("stopping", training._resume_signature(self.cfg("sig", early_stopping_patience=0)))

    def test_early_stop_keeps_raw_best_and_does_not_restart(self):
        cfg = self.cfg("early")
        result = self.run_train(cfg, [1., .95, .94])
        self.assertEqual(result["steps"], 6)
        self.assertEqual(result["max_steps"], 10)
        self.assertTrue(result["stopped_early"])
        best = torch.load(self.root / "early" / "best.pt", weights_only=False)
        self.assertEqual(best["step"], 6)
        self.assertEqual(best["valid_bits_per_token"], .94)
        checkpoint = self.checkpoint("early")
        self.assertEqual(checkpoint["step"], 6)  # forced save outside checkpoint interval
        self.assertEqual(checkpoint["early_stopping"]["best"], 1.)
        self.assertEqual(checkpoint["early_stopping"]["bad_checks"], 2)
        with patch.object(training, "load_frozen_train_valid", return_value=self.data), \
             patch.object(training, "bits_per_token") as validation:
            resumed = training.train(cfg, verbose=False)
            self.assertTrue(resumed["resumed_complete"])
            self.assertTrue(resumed["stopped_early"])
            validation.assert_not_called()

    def test_interrupted_resume_matches_uninterrupted_weights_and_patience(self):
        self.run_train(self.cfg("full"), [1., .95, .94])
        save = training.save_training_checkpoint

        def interrupt_after_save(*args, **kwargs):
            save(*args, **kwargs)
            if args[5] == 4:
                raise RuntimeError("simulated interruption")

        cfg = self.cfg("resumed")
        with patch.object(training, "save_training_checkpoint", side_effect=interrupt_after_save):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.run_train(cfg, [1., .95])
        self.assertEqual(self.checkpoint("resumed")["early_stopping"]["bad_checks"], 1)
        result = self.run_train(cfg, [.94])
        self.assertEqual(result["steps"], 6)
        full, resumed = self.checkpoint("full"), self.checkpoint("resumed")
        for name in full["model"]:
            self.assertTrue(torch.equal(full["model"][name], resumed["model"][name]), name)
        self.assertEqual(full["early_stopping"], resumed["early_stopping"])
        self.assertTrue(torch.equal(full["train_sampler"]["generator_state"],
                                    resumed["train_sampler"]["generator_state"]))

    def test_legacy_checkpoint_without_new_state_can_resume(self):
        cfg = self.cfg("legacy", max_steps=5, early_stopping_patience=0,
                       eval_every=2, checkpoint_every=2)
        save = training.save_training_checkpoint

        def interrupt_after_save(*args, **kwargs):
            save(*args, **kwargs)
            if args[5] == 2:
                checkpoint = self.checkpoint("legacy")
                checkpoint.pop("early_stopping")
                torch.save(checkpoint, self.root / "legacy" / "latest.pt")
                raise RuntimeError("legacy interruption")

        with patch.object(training, "save_training_checkpoint", side_effect=interrupt_after_save):
            with self.assertRaisesRegex(RuntimeError, "legacy interruption"):
                self.run_train(cfg, [1., .9])
        result = self.run_train(cfg, [.8, .7])
        self.assertEqual(result["steps"], 5)

    def test_delta_reference_and_patience_reset(self):
        cfg = TrainConfig(early_stopping_patience=2, early_stopping_min_delta=.1)
        state = training.EarlyStopping()
        self.assertFalse(state.update(1., cfg))
        self.assertFalse(state.update(.95, cfg))
        self.assertEqual(state.bad_checks, 1)
        self.assertFalse(state.update(.89, cfg))
        self.assertEqual(state.bad_checks, 0)
        self.assertFalse(state.update(.88, cfg))
        self.assertTrue(state.update(.88, cfg))
        with self.assertRaises(ValueError):
            state.update(float("nan"), cfg)

    def test_requires_validation_and_valid_config(self):
        self.data.valid = torch.empty(0)
        with self.assertRaisesRegex(ValueError, "validation"):
            self.run_train(self.cfg("invalid"), [])
        for kwargs in ({"early_stopping_patience": -1},
                       {"early_stopping_min_delta": float("nan")}, {"eval_every": 0}):
            with self.assertRaises(ConfigError):
                TrainConfig(**kwargs)
        with self.assertRaises(ConfigError):
            parse_config({"train": {"max_epochs": 2}})


if __name__ == "__main__":
    unittest.main()
