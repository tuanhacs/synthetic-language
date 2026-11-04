"""Optional bounded walks: repeat permission, not an exact length or forced loop."""
import contextlib
import io
import random
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthdata.codebooks import Codebooks
from synthdata.config import Config, CodeConfig, ConfigError, LanguageConfig, TaskConfig, parse_config
from synthdata.graphs import make_grid
from synthdata.language import Language
from synthdata.qa import (LatentQASample, random_bounded_walk, encode_qa,
    validate_qa_answer, validate_segment_cot, sample_latent_qa, query_contains_held_out,
    cross_region_layout, cross_region_segment_count)
from synthdata.qa_latent import create_latent_pool, encode_latent_pool, semantic_config
from synthdata.qa_reversal import reverse_sample


class RepeatTests(unittest.TestCase):
    def setUp(self):
        self.graph = make_grid(4)
        self.language = Language(self.graph, Codebooks(tuple((format(v, "04b"),)
            for v in range(16))), (2, 14))
        self.task = TaskConfig(type="path-qa", query_len=(2, 4), segment_len=(2, 14), allow_repeats=True)

    def test_sampler_allows_both_types_and_variable_lengths(self):
        paths = [random_bounded_walk(self.graph, 0, 5, (2, 14), random.Random(s), 1000)
                 for s in range(150)]
        self.assertTrue(all(p is not None for p in paths))
        self.assertTrue(any(len(set(p)) < len(p) for p in paths))
        self.assertTrue(any(len(set(p)) == len(p) for p in paths))
        self.assertGreater(len({len(p) for p in paths}), 1)
        for path in paths:
            self.assertEqual((path[0], path[-1]), (0, 5))
            self.assertTrue(2 <= len(path) <= 14)
            self.assertNotIn(5, path[:-1])  # first permitted target hit
            self.assertTrue(all(self.graph.is_edge(u, v) for u, v in zip(path, path[1:])))

    def test_minimum_maximum_and_region_bounds(self):
        allowed = frozenset(range(8))
        for s in range(20):
            path = random_bounded_walk(self.graph, 0, 1, (6, 14), random.Random(s), 1000, allowed)
            self.assertIsNotNone(path)
            self.assertTrue(6 <= len(path) <= 14)
            self.assertTrue(set(path) <= allowed)
            self.assertNotIn(1, path[:-1])
        self.assertIsNone(random_bounded_walk(self.graph, 0, 15, (2, 3), random.Random(0), 2))
        self.assertIsNone(random_bounded_walk(self.graph, 0, 15, (2, 14), random.Random(0), 2, allowed))
        self.assertIsNone(random_bounded_walk(self.graph, 0, 5, (2, 14), random.Random(0), 2, {0, 5}))

    def test_semantic_accepts_loops_but_not_wrong_order_or_endpoints(self):
        sample = encode_qa(LatentQASample((0, 5), (0, 1, 0, 1, 5), (5,)),
                           self.language, self.task, random.Random(0))
        strict = validate_qa_answer(self.language, sample.answer_bits, sample.query_vertices)
        loose = validate_qa_answer(self.language, sample.answer_bits, sample.query_vertices, allow_repeats=True)
        self.assertTrue(strict["valid_walk"] and strict["waypoints_in_order"])
        self.assertFalse(strict["piecewise_simple"] or strict["semantic_success"])
        self.assertTrue(loose["semantic_success"])
        for query in ((0, 5, 1), (2, 5), (0, 2, 5)):
            self.assertFalse(validate_qa_answer(self.language, sample.answer_bits, query,
                                               allow_repeats=True)["semantic_success"])
        invalid = self.language.codebooks[0][0] + self.language.codebooks[15][0]
        self.assertFalse(validate_qa_answer(self.language, invalid, (0, 15),
                                           allow_repeats=True)["semantic_success"])

    def test_cot_loops_accepted_without_losing_consistency_checks(self):
        task = replace(self.task, cot_mode="segments")
        latent = LatentQASample((0, 5, 2), (0, 1, 0, 1, 5, 1, 2), (5, 7))
        sample = encode_qa(latent, self.language, task, random.Random(2))
        args = (self.language, sample.cot_segments_bits, sample.query_vertices,
                sample.answer_walk, sample.answer_bits)
        self.assertFalse(validate_segment_cot(*args)["cot_success"])
        result = validate_segment_cot(*args, allow_repeats=True)
        self.assertTrue(result["cot_success"] and result["final_consistent"])
        self.assertFalse(result["segments_simple"])
        wrong = validate_segment_cot(self.language, sample.cot_segments_bits[::-1],
            sample.query_vertices, sample.answer_walk, sample.answer_bits, allow_repeats=True)
        self.assertFalse(wrong["cot_success"])

    def test_config_roundtrip_legacy_identity_and_limits(self):
        cfg = Config(language=LanguageConfig(code=CodeConfig(depth_range=(1, 4))), task=self.task)
        self.assertEqual(parse_config(cfg.to_dict()).task, self.task)
        legacy = replace(self.task, allow_repeats=False)
        self.assertNotIn("allow_repeats", legacy.to_dict())
        self.assertNotEqual(semantic_config(self.graph, legacy), semantic_config(self.graph, self.task))
        raw = cfg.to_dict()
        for value in ("true", 1, None):
            raw["task"]["allow_repeats"] = value
            with self.assertRaises(ConfigError):
                parse_config(raw)
        cfg = replace(cfg, task=replace(self.task, segment_len=(2, 40)))
        parse_config(cfg.to_dict())  # walks may be longer than the vertex count
        with self.assertRaises(ConfigError):
            parse_config(replace(cfg, task=replace(cfg.task, allow_repeats=False)).to_dict())

    def test_reversal_preserves_repeat_enabled_answers_and_cot(self):
        suffix = Language(self.graph, Codebooks(tuple(tuple(w[::-1] for w in b)
            for b in self.language.codebooks.books)), self.language.walk_len)
        for cot in ("none", "segments"):
            task = replace(self.task, cot_mode=cot)
            sample = encode_qa(LatentQASample((0, 5, 2), (0, 1, 0, 1, 5, 1, 2), (5, 7)),
                               self.language, task, random.Random(0))
            reversed_sample = reverse_sample(sample)
            self.assertEqual(reverse_sample(reversed_sample), sample)
            self.assertTrue(validate_qa_answer(suffix, reversed_sample.answer_bits,
                reversed_sample.query_vertices, allow_repeats=True)["semantic_success"])
            if cot == "segments":
                self.assertTrue(validate_segment_cot(suffix, reversed_sample.cot_segments_bits,
                    reversed_sample.query_vertices, reversed_sample.answer_walk,
                    reversed_sample.answer_bits, allow_repeats=True)["cot_success"])

    def test_split_constraints_and_latent_contract(self):
        for mode in ("iid", "held-out-pairs", "cross-region"):
            task = replace(self.task, split_mode=mode,
                held_out_pairs=((0, 15),) if mode == "held-out-pairs" else ())
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "pool"
                with contextlib.redirect_stdout(io.StringIO()):
                    create_latent_pool(path, self.graph, task, 0, 80, 20, 20)
                splits, _ = encode_latent_pool(path, self.language, replace(task, cot_mode="segments"), 60, 0)
                with self.assertRaisesRegex(ValueError, "differs"):
                    encode_latent_pool(path, self.language, replace(task, allow_repeats=False))
                for name, samples in splits.items():
                    for s in samples:
                        self.assertTrue(validate_qa_answer(self.language, s.answer_bits, s.query_vertices,
                                                           allow_repeats=True)["semantic_success"])
                        self.assertTrue(validate_segment_cot(self.language, s.cot_segments_bits, s.query_vertices,
                            s.answer_walk, s.answer_bits, allow_repeats=True)["cot_success"])
                        if mode == "held-out-pairs":
                            self.assertEqual(query_contains_held_out(s.query_vertices, task), name == "test")
                        if mode == "cross-region" and name != "test":
                            layout = cross_region_layout(4, 2)
                            self.assertTrue(set(s.answer_walk) <= layout.upper or set(s.answer_walk) <= layout.lower)
                        if mode == "cross-region" and name == "test":
                            self.assertGreater(cross_region_segment_count(s.query_vertices, 4, 2), 0)


if __name__ == "__main__":
    unittest.main()
