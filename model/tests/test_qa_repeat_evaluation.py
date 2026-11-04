"""Evaluation must use the dataset's repeat policy for final and CoT success."""
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evals
from cross_region import _walk_events
from synthdata.codebooks import Codebooks
from synthdata.config import TaskConfig
from synthdata.graphs import make_grid
from synthdata.language import Language
from synthdata.qa import LatentQASample, encode_qa
from synthdata.tokenizer import BitTokenizer


class RepeatEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.language = Language(make_grid(4), Codebooks(tuple((format(v, "04b"),)
            for v in range(16))), (2, 14))
        self.model = SimpleNamespace(cfg=SimpleNamespace(context_len=512))

    def test_final_and_cot_success_follow_dataset_policy(self):
        for cot in ("none", "segments"):
            task = TaskConfig(type="path-qa", cot_mode=cot, allow_repeats=True)
            sample = encode_qa(LatentQASample((0, 5), (0, 1, 0, 1, 5), (5,)),
                               self.language, task, random.Random(0))
            reports = []
            for allow in (False, True):
                with patch.object(evals, "generate", return_value=[sample.bits]):
                    report = evals.eval_path_qa(self.model, self.language, BitTokenizer(True), [sample],
                        n_samples=1, task_config={"cot_mode": cot, "allow_repeats": allow})
                reports.append(report)
            strict, loose = reports
            self.assertEqual(strict["semantic_success_pct"], 0.)
            self.assertEqual(loose["semantic_success_pct"], 100.)
            self.assertEqual(loose["full_success_pct"], 100.)
            self.assertEqual(loose["piecewise_simple_pct"], 0.)
            self.assertTrue(loose["allow_repeats"])
            if cot == "segments":
                self.assertEqual(strict["cot"]["cot_success_pct"], 0.)
                self.assertEqual(loose["cot"]["cot_success_pct"], 100.)
                self.assertEqual(loose["cot"]["cot_and_semantic_success_pct"], 100.)

    def test_cross_region_analysis_uses_the_same_policy(self):
        task = TaskConfig(type="path-qa", split_mode="cross-region", allow_repeats=True)
        sample = encode_qa(LatentQASample((0, 15), (0, 1, 0, 4, 8, 12, 13, 14, 15), (9,)),
                           self.language, task, random.Random(0))
        strict = _walk_events(self.language, sample, sample.answer_bits, True, 2)
        loose = _walk_events(self.language, sample, sample.answer_bits, True, 2, allow_repeats=True)
        self.assertFalse(strict["semantic_success"])
        self.assertTrue(loose["semantic_success"])
        self.assertFalse(loose["piecewise_simple"])


if __name__ == "__main__":
    unittest.main()
