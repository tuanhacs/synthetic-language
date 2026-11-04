"""QA examples stay complete and attention-independent in train/validation."""

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import train as training
import data as batching
from config import Config, ModelConfig, TrainConfig
from data import (BatchSampler, IGNORE_INDEX, QA_PACKING_VERSION, make_targets,
                  pack_windows, sentence_batches)
from evals import masked_ce
from model import build_model
from synthdata.codebooks import Codebooks
from synthdata.config import TaskConfig, TrainNoiseConfig
from synthdata.qa_noise import apply_train_noise
from synthdata.graphs import make_grid
from synthdata.language import Language
from synthdata.qa import LatentQASample, encode_qa
from synthdata.tokenizer import BitTokenizer
import random


class QAPackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tok = BitTokenizer(include_separator=True)
        self.language = Language(make_grid(2), Codebooks(tuple(
            ("0" + format(v, "02b"), "1" + format(v, "02b") + "0")
            for v in range(4))), (2, 5))

    def samples(self, cot="none"):
        task = TaskConfig(type="path-qa", cot_mode=cot)
        paths = [LatentQASample((0, 3, 0), (0, 1, 3, 2, 0), (3, 5)),
                 LatentQASample((1, 2), (1, 0, 2), (3,)),
                 LatentQASample((2, 1), (2, 3, 1), (3,))]
        return tuple(encode_qa(s, self.language, task, random.Random(i)) for i, s in enumerate(paths))

    def test_no_splitting_padding_and_answer_masks_both_cot_modes(self):
        for cot in ("none", "segments"):
            samples = self.samples(cot)
            # Bigger than every example, but not an integer multiple of lengths.
            context = max(len(s.bits) + 2 for s in samples) + 9
            windows = pack_windows(samples, context, self.tok)
            inputs, targets = make_targets(windows, self.tok)
            self.assertEqual(len(inputs), len(samples))
            self.assertTrue(bool((inputs[:, 0] == self.tok.BOS).all()))
            for i, sample in enumerate(samples):
                seq = self.tok.encode(sample.bits)
                torch.testing.assert_close(inputs[i, :len(seq) - 1], torch.tensor(seq[:-1]))
                scored = targets[i][targets[i] != IGNORE_INDEX].tolist()
                completion = sample.bits[len(sample.query_bits) + 1:]
                self.assertEqual(scored, self.tok.encode(completion, bos=False))
                self.assertEqual(int((windows.remainder(self.tok.vocab_size) == self.tok.EOS).sum()), len(samples))
            # Short/final packs are retained, including a single-example dataset.
            self.assertEqual(len(pack_windows(samples[:1], context * 3, self.tok)), 1)
            self.assertEqual(pack_windows((), context, self.tok).shape, (0, context + 1))

    def test_context_too_short_and_split_windows_fail_loudly(self):
        samples = self.samples("segments")
        with self.assertRaisesRegex(ValueError, "increase model.context_len"):
            pack_windows(samples, 5, self.tok)
        encoded = torch.tensor([self.tok.encode(samples[0].bits)])
        with self.assertRaises(ValueError):
            make_targets(encoded[:, :10], self.tok)
        with self.assertRaises(ValueError):
            make_targets(encoded[:, 10:], self.tok)
        ds = SimpleNamespace(config=SimpleNamespace(task=SimpleNamespace(type="path-qa")),
            splits=SimpleNamespace(train=samples[1:], valid=samples[:1]), language=self.language, manifest={})
        with patch.object(batching, "load_dataset", return_value=ds):
            with self.assertRaisesRegex(ValueError, "increase model.context_len"):
                batching.load_frozen_train_valid("unused", 5)

    def test_packed_and_standalone_logits_loss_gradients_match(self):
        samples = self.samples("segments")
        context = sum(len(s.bits) + 2 for s in samples) + 10
        windows = pack_windows(samples, context, self.tok)
        self.assertEqual(len(windows), 1)  # several samples stored in one pack
        inputs, targets = make_targets(windows, self.tok)
        model = build_model(ModelConfig(d_model=16, n_heads=2, n_layers=2,
                                        vocab_size=6, context_len=context)).eval()
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                               ignore_index=IGNORE_INDEX, reduction="sum")
        loss.backward()
        gradients = {k: p.grad.clone() for k, p in model.named_parameters()}
        model.zero_grad()
        separate_loss = 0
        for i, sample in enumerate(samples):
            seq = torch.tensor([self.tok.encode(sample.bits)])
            x, y = make_targets(seq, self.tok)
            individual = model(x)
            torch.testing.assert_close(logits[i, :x.shape[1]], individual[0], atol=1e-6, rtol=1e-5)
            separate_loss = separate_loss + F.cross_entropy(individual.flatten(0, 1), y.flatten(),
                ignore_index=IGNORE_INDEX, reduction="sum")
        torch.testing.assert_close(loss, separate_loss)
        separate_loss.backward()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(gradients[name], parameter.grad, atol=2e-5, rtol=1e-4)
        packed_total, count = masked_ce(model, [(inputs, targets)])
        sentence_total, sentence_count = masked_ce(model, sentence_batches(samples, 2, self.tok))
        self.assertEqual(count, sentence_count)
        self.assertAlmostEqual(packed_total, sentence_total, places=4)

    def test_lm_packing_is_unchanged(self):
        tok = BitTokenizer()
        samples = ["0101", "101", "00"]
        tokens = [t for sample in samples for t in tok.encode(sample)]
        self.assertEqual(list(tok.pack(samples, 6)),
                         [tokens[i:i + 6] for i in range(0, len(tokens) - 5, 6)])

    def test_noise_is_used_by_inputs_and_targets_with_query_mask_preserved(self):
        for cot in ("none", "segments"):
            for scope in ("query", "answer", "both"):
                with self.subTest(cot=cot, scope=scope):
                    noisy, _ = apply_train_noise(self.samples(cot),
                        TrainNoiseConfig(sample_rate=1, scope=scope))
                    context = sum(len(s.bits) + 2 for s in noisy)
                    inputs, targets = make_targets(pack_windows(noisy, context, self.tok), self.tok)
                    for i, s in enumerate(noisy):
                        seq = self.tok.encode(s.noised_bits)
                        self.assertEqual(inputs[i, :len(seq) - 1].tolist(), seq[:-1])
                        completion = s.noised_bits[len(s.query_bits) + 1:]
                        scored = targets[i][targets[i] != IGNORE_INDEX].tolist()
                        self.assertEqual(scored, self.tok.encode(completion, bos=False))

    def test_sampler_and_training_resume_exactly_and_reject_legacy_packing(self):
        with tempfile.TemporaryDirectory() as temp:
            samples = self.samples("segments")
            context = max(len(s.bits) + 2 for s in samples) + 10
            windows = pack_windows(samples, context, self.tok)
            data = SimpleNamespace(train=windows, valid=windows, tokenizer=self.tok,
                entropy_floor=None, dataset_dir=Path(temp),
                manifest={"config_hash": "qa-test", "config": {"task": {"type": "path-qa"}}},
                token_counts=lambda: {"train": windows.numel(), "valid": windows.numel()})
            cfg = Config(model=ModelConfig(d_model=8, n_heads=2, n_layers=1,
                    vocab_size=6, context_len=context),
                train=TrainConfig(max_steps=4, batch_size=2, device="cpu", warmup=0,
                    eval_every=2, eval_batches=1, checkpoint_every=2, seed=7,
                    out_dir=str(Path(temp) / "full")))
            def run(config):
                with patch.object(training, "load_frozen_train_valid", return_value=data):
                    return training.train(config, verbose=False)
            run(cfg)
            interrupted_cfg = replace(cfg, train=replace(cfg.train, out_dir=str(Path(temp) / "resumed")))
            save = training.save_training_checkpoint
            def interrupt(*args, **kwargs):
                save(*args, **kwargs)
                if args[5] == 2:
                    raise RuntimeError("interrupt")
            with patch.object(training, "save_training_checkpoint", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "interrupt"):
                    run(interrupted_cfg)
            run(interrupted_cfg)
            full = torch.load(Path(cfg.train.out_dir) / "latest.pt", weights_only=False)
            resumed = torch.load(Path(interrupted_cfg.train.out_dir) / "latest.pt", weights_only=False)
            self.assertEqual(resumed["qa_packing"], QA_PACKING_VERSION)
            for key in full["model"]:
                self.assertTrue(torch.equal(full["model"][key], resumed["model"][key]), key)
            resumed.pop("qa_packing")
            with self.assertRaisesRegex(ValueError, "old split-window QA packing"):
                training._validate_resume_checkpoint(resumed, interrupted_cfg, data)
            sampler = BatchSampler(windows, 2, seed=7, tokenizer=self.tok)
            state = sampler.state_dict()
            x, y = sampler.batch()
            sampler.load_state_dict(state)
            xx, yy = sampler.batch()
            self.assertTrue(torch.equal(x, xx) and torch.equal(y, yy))
            with self.assertRaisesRegex(ValueError, "packing changed"):
                sampler.load_state_dict({"generator_state": state["generator_state"]})


if __name__ == "__main__":
    unittest.main()
