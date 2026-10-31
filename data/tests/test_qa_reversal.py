"""Bit-exact involution, CoT boundaries, split safety and persisted datasets."""
import contextlib
import hashlib
import io
import json
import random
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthdata.codebooks import Codebooks
from synthdata.config import Config, CodeConfig, DataConfig, LanguageConfig, OverlapConfig, TaskConfig
from synthdata.graphs import make_grid
from synthdata.language import Language
from synthdata.qa import LatentQASample, encode_qa, validate_qa_answer, validate_segment_cot, query_contains_held_out, cross_region_segment_count
from synthdata.qa_latent import create_latent_pool, encode_latent_pool, sample_key
from synthdata.qa_reversal import reverse_sample, build_reversed_dataset
from synthdata.storage import save_dataset, load_dataset


class ReversalTests(unittest.TestCase):
    def language(self, overlap=False):
        books = [("0" + format(v, "04b"), "1" + format(v, "04b") + "0") for v in range(16)]
        if overlap:
            books[15] = (books[0][0], books[15][1])
        return Language(make_grid(4), Codebooks(tuple(books)), (2, 10))

    def test_involution_and_cot_with_repeated_waypoints(self):
        language = self.language()
        reverse_language = Language(language.graph, Codebooks(tuple(
            tuple(w[::-1] for w in b) for b in language.codebooks.books)), language.walk_len)
        latent = LatentQASample((0, 5, 0, 6), (0, 1, 5, 4, 0, 1, 2, 6), (3, 5, 8))
        for cot in ("none", "segments"):
            task = TaskConfig(type="path-qa", cot_mode=cot)
            sample = encode_qa(latent, language, task, random.Random(7), "source-id")
            reversed_sample = reverse_sample(sample)
            self.assertEqual(reverse_sample(reversed_sample), sample)
            self.assertEqual(reversed_sample.segment_cuts, (4, 6, 8))
            self.assertEqual(reversed_sample.query_bits, sample.query_bits[::-1])
            self.assertTrue(validate_qa_answer(reverse_language, reversed_sample.answer_bits,
                                               reversed_sample.query_vertices)["semantic_success"])
            if cot == "segments":
                result = validate_segment_cot(reverse_language, reversed_sample.cot_segments_bits,
                                              reversed_sample.query_vertices, reversed_sample.answer_walk,
                                              reversed_sample.answer_bits)
                self.assertTrue(result["cot_success"])
                self.assertEqual(len(sample.bits), len(reversed_sample.bits))

    def test_saved_splits_all_tasks_and_cot_modes(self):
        for mode in ("iid", "held-out-pairs", "cross-region"):
            for cot in ("none", "segments"):
                with self.subTest(mode=mode, cot=cot), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    overlap = mode == "cross-region"
                    language = self.language(overlap)
                    cfg = Config(
                        language=LanguageConfig(code=CodeConfig(depth_range=(1, 6)), k=2,
                            assignment="arbitrary-overlap" if overlap else "disjoint-random",
                            overlap=OverlapConfig(shared_codewords=1) if overlap else None),
                        data=DataConfig(walk_len=(2, 10), seed=17, context_len=None),
                        task=TaskConfig(type="path-qa", query_len=(2, 4), segment_len=(2, 10),
                            split_mode=mode, held_out_pairs=((0, 15),) if mode == "held-out-pairs" else (),
                            cot_mode=cot))
                    pool = root / "pool"
                    create_latent_pool(pool, language.graph, cfg.task, 17, 80, 20, 20)
                    forced_collision = mode == "iid" and cot == "none"
                    if forced_collision:
                        # Plant a reversed test sample in the full train pool.
                        # Natural suffix test must exclude it even for subsets.
                        test_row = json.loads((pool / "test.jsonl").read_text().splitlines()[0])
                        n = len(test_row["answer_walk"])
                        planted = LatentQASample(tuple(test_row["query_vertices"][::-1]),
                            tuple(test_row["answer_walk"][::-1]),
                            tuple(n + 1 - c for c in reversed(test_row["segment_cuts"][:-1])) + (n,))
                        lines = (pool / "train.jsonl").read_text().splitlines()
                        row = planted.to_dict()
                        row["sample_id"] = "train:" + sample_key(planted)
                        lines[0] = json.dumps(row)
                        (pool / "train.jsonl").write_text("\n".join(lines) + "\n")
                        manifest = json.loads((pool / "manifest.json").read_text())
                        manifest["sha256"]["train"] = hashlib.sha256((pool / "train.jsonl").read_bytes()).hexdigest()
                        (pool / "manifest.json").write_text(json.dumps(manifest))
                    splits, provenance = encode_latent_pool(pool, language, cfg.task, 40, 7)
                    save_dataset(root / "prefix", cfg, language, splits, extra={"latent_pool": provenance})
                    suffix_cfg = replace(cfg, language=replace(cfg.language,
                        code=replace(cfg.language.code, type="suffix-free")))
                    with contextlib.redirect_stdout(io.StringIO()):
                        build_reversed_dataset(suffix_cfg, root / "prefix", pool, root / "suffix", 7)
                    target = load_dataset(root / "suffix")
                    if forced_collision:
                        self.assertGreaterEqual(target.manifest["stats"]["reversal"]["natural_test_collisions_removed"], 1)
                    paired = load_dataset(root / "suffix" / "test_reversed")
                    for name in ("train", "valid"):
                        self.assertEqual(getattr(target.splits, name),
                                         tuple(map(reverse_sample, getattr(splits, name))))
                    self.assertEqual(paired.splits.test, tuple(map(reverse_sample, splits.test)))
                    self.assertNotEqual(target.manifest["dataset_fingerprint"],
                                        paired.manifest["dataset_fingerprint"])
                    forbidden = {sample_key(s) for s in (*target.splits.train, *target.splits.valid)}
                    self.assertTrue(all(sample_key(s) not in forbidden for s in target.splits.test))
                    for name, samples in target.splits.items():
                        for s in samples:
                            self.assertTrue(validate_qa_answer(target.language, s.answer_bits,
                                                              s.query_vertices)["semantic_success"])
                            if mode == "held-out-pairs":
                                self.assertEqual(query_contains_held_out(s.query_vertices, cfg.task), name == "test")
                            if mode == "cross-region":
                                self.assertEqual(cross_region_segment_count(s.query_vertices, 4, 2) > 0,
                                                 name == "test")
                            if cot == "segments":
                                self.assertTrue(validate_segment_cot(target.language, s.cot_segments_bits,
                                    s.query_vertices, s.answer_walk, s.answer_bits)["cot_success"])


if __name__ == "__main__":
    unittest.main()
