"""Contract tests for shared semantic datasets and nested train prefixes."""

import contextlib
import io
import json
import random
import sys
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthdata.codebooks import Codebooks, assign, remap_assignment
from synthdata.codes import Code
from synthdata.config import LanguageConfig, TaskConfig, OverlapConfig
from synthdata.graphs import make_grid
from synthdata.language import Language
from synthdata.qa import LatentQASample, query_contains_held_out, validate_segment_cot
from synthdata.qa_latent import create_latent_pool, encode_latent_pool, semantic_stats


class LatentPoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.graph = make_grid(4)
        self.task = TaskConfig(type="path-qa", query_len=(2, 4), segment_len=(2, 10))
        self.language = Language(
            graph=self.graph,
            codebooks=Codebooks(tuple((format(v, "05b"), format(v + 16, "05b"))
                                     for v in self.graph.vertices)),
            walk_len=(2, 10),
        )

    def pool(self, name, pairs, task=None, graph=None):
        path = self.root / name
        with contextlib.redirect_stdout(io.StringIO()):
            create_latent_pool(path, graph or self.graph, task or self.task, 17,
                               pairs, 35, 35, [50, 100])
        return path

    def rows(self, pool, split):
        return [json.loads(line) for line in (pool / f"{split}.jsonl").read_text().splitlines()]

    def test_direct_small_equals_large_prefix_and_fixed_evaluation(self):
        small, large = self.pool("small", 50), self.pool("large", 150)
        n = len(self.rows(small, "train"))
        self.assertEqual(self.rows(small, "train"), self.rows(large, "train")[:n])
        for split in ("valid", "test"):
            self.assertEqual((small / f"{split}.jsonl").read_bytes(),
                             (large / f"{split}.jsonl").read_bytes())
        ids = [{row["sample_id"].rsplit(":", 1)[-1] for row in self.rows(large, split)}
               for split in ("train", "valid", "test")]
        for i in range(3):
            for j in range(i):
                self.assertFalse(ids[i] & ids[j])

    def test_encoding_nested_stable_and_cot_independent(self):
        path = self.pool("pool", 150)
        small, _ = encode_latent_pool(path, self.language, self.task, 50, 9)
        large, _ = encode_latent_pool(path, self.language, self.task, 150, 9)
        self.assertEqual(small.train, large.train[:len(small.train)])
        self.assertEqual(small.valid, large.valid)
        self.assertEqual(small.test, large.test)
        cot_task = replace(self.task, cot_mode="segments")
        cot, _ = encode_latent_pool(path, self.language, cot_task, 50, 9)
        for plain, rationale in zip(small.train, cot.train):
            self.assertEqual(plain.sample_id, rationale.sample_id)
            self.assertEqual(plain.query_bits, rationale.query_bits)
            self.assertEqual(plain.answer_bits, rationale.answer_bits)
            result = validate_segment_cot(
                self.language, rationale.cot_segments_bits, rationale.query_vertices,
                rationale.answer_walk, rationale.answer_bits,
            )
            self.assertTrue(result["cot_success"])
            self.assertTrue(result["final_consistent"])
        changed = Language(
            self.graph,
            Codebooks(tuple(tuple(word[::-1] for word in book)
                            for book in self.language.codebooks.books)),
            self.language.walk_len,
        )
        other, _ = encode_latent_pool(path, changed, self.task, 50, 9)
        self.assertEqual([s.sample_id for s in small.train], [s.sample_id for s in other.train])
        self.assertEqual([s.answer_walk for s in small.train], [s.answer_walk for s in other.train])

    def test_reject_wrong_task_and_budget_and_corruption(self):
        path = self.pool("pool", 50)
        with self.assertRaises(ValueError):
            encode_latent_pool(path, self.language, replace(self.task, query_len=(2, 5)))
        with self.assertRaises(ValueError):
            encode_latent_pool(path, self.language, self.task, 999)
        with (path / "train.jsonl").open("ab") as handle:
            handle.write(b"\n")
        with self.assertRaises(ValueError):
            encode_latent_pool(path, self.language, self.task)

    def test_heldout_and_cross_region_contracts(self):
        held = replace(self.task, split_mode="held-out-pairs", held_out_pairs=((0, 15),))
        path = self.pool("held", 70, held)
        for split in ("train", "valid", "test"):
            for row in self.rows(path, split):
                self.assertEqual(query_contains_held_out(row["query_vertices"], held), split == "test")
        graph = make_grid(6)
        task = replace(self.task, split_mode="cross-region", segment_len=(2, 14))
        path = self.pool("cross", 70, task, graph)
        report = json.loads((path / "manifest.json").read_text())
        regions = report["stats"]["train"]["histograms"]["sampled_region"]
        self.assertLessEqual(abs(regions["upper"] - regions["lower"]), 1)
        for row in self.rows(path, "train"):
            walk = row["answer_walk"]
            self.assertTrue(all(v // 6 <= 3 for v in walk) or all(v // 6 >= 2 for v in walk))

    def test_histograms_count_edges_once_and_include_zero_frequency(self):
        sample = LatentQASample((0, 5, 0), (0, 1, 5, 4, 0), (3, 5))
        stats = semantic_stats([sample], self.graph, self.task, "train")
        self.assertEqual(stats["histograms"]["segment_vertices"], {3: 2})
        self.assertEqual(stats["histograms"]["query_vertices"], {3: 1})
        self.assertEqual(stats["query_pair_occurrences"], 2)
        self.assertEqual(stats["query_pairs_directed"]["eligible"], 16 * 15)
        self.assertEqual(stats["query_pairs_directed"]["frequency_histogram_including_zero"][0], 238)
        edges = stats["graph_edges_directed"]
        self.assertEqual(sum(edges["frequencies"].values()), 4)
        self.assertEqual(edges["seen"], 4)
        pairs = stats["query_pairs_directed"]
        self.assertEqual(pairs["frequencies"]["0,5"], 1)
        self.assertEqual(pairs["relative_frequencies"]["0,5"], 0.5)
        self.assertAlmostEqual(sum(pairs["relative_frequencies"].values()), 1.0)
        self.assertEqual(pairs["frequency_summary_including_zero"]["min"], 0)
        self.assertEqual(pairs["frequency_summary_including_zero"]["median"], 0)
        self.assertEqual(pairs["frequency_summary_observed_only"]["min"], 1)
        self.assertEqual(pairs["frequency_summary_observed_only"]["median"], 1)
        self.assertEqual(sum(b["num_items"] for b in pairs["binned_frequency_histogram"]), 240)
        self.assertEqual(edges["total_occurrences"], 4)
        self.assertEqual(edges["relative_frequencies"]["0,1"], 0.25)

    def test_empty_frequency_report(self):
        stats = semantic_stats([], self.graph, self.task, "train")
        pairs = stats["query_pairs_directed"]
        self.assertEqual(pairs["total_occurrences"], 0)
        self.assertEqual(sum(pairs["relative_frequencies"].values()), 0)
        self.assertEqual(pairs["frequency_summary_including_zero"]["mean"], 0)
        self.assertIsNone(pairs["frequency_summary_observed_only"]["mean"])
        self.assertIsNone(pairs["frequency_summary_including_zero"]["cv"])

    def test_assignment_remap_preserves_shared_slots(self):
        graph = make_grid(2)
        template = Codebooks((("000", "001"), ("010",), ("011",), ("100",)))
        cfg = LanguageConfig(graph="grid-2x2", k=(1, 2))
        pool = Code(("00", "010", "011", "10", "11"))
        mapped = remap_assignment(template, cfg, graph, pool, random.Random(1))
        self.assertEqual(mapped.k_per_vertex, template.k_per_vertex)
        self.assertTrue(mapped.certify(graph).theorem_d1)

    def test_overlap_template_preserves_each_word_support_and_slot(self):
        graph = make_grid(6)
        cfg = LanguageConfig(graph="grid-6x6", k=(1, 4),
                             assignment="arbitrary-overlap", overlap=OverlapConfig())
        template = assign(cfg, graph, Code(tuple(format(i, "08b") for i in range(256))), random.Random(2))
        mapped = remap_assignment(template, cfg, graph,
                                  Code(tuple("0" + format(i, "08b") for i in range(256))), random.Random(3))
        self.assertTrue(mapped.certify(graph).theorem_d1)
        pairs = list(zip(template.all_words(), mapped.all_words()))
        for left_old, left_new in pairs:
            for right_old, right_new in pairs:
                self.assertEqual(left_old == right_old, left_new == right_new)
        self.assertEqual(template.k_per_vertex, mapped.k_per_vertex)

    def test_full_cli_prefix_suffix_ud_shared_pool(self):
        import yaml

        data_dir = Path(__file__).resolve().parents[1]
        pool = self.pool("cli-pool", 90)
        prior = None
        template = None
        for kind in ("prefix-free", "suffix-free", "ud"):
            config = {
                "language": {
                    "graph": "grid-4x4", "k": [1, 2],
                    "assignment": "disjoint-random",
                    "code": {"type": kind, "base_size": 3, "power_x": 5, "depth_range": [1, 4]},
                },
                "data": {"seed": 17, "pool_query_pairs": 90, "context_len": "auto"},
                "task": self.task.to_dict(),
            }
            if kind == "ud":
                config["language"]["codeword_pool_file"] = str(
                    data_dir / "codeword_pools" / "hard_ud_codebook_200.json"
                )
            path = self.root / f"{kind}.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            out = self.root / kind
            cmd = [sys.executable, "-B", str(data_dir / "scripts" / "gen_dataset.py"),
                   str(path), "--out", str(out), "--latent-pool", str(pool),
                   "--train-query-pairs", "50", "--entropy-samples", "0"]
            if template:
                cmd += ["--assignment-template", str(template)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
            latent_fields = [(r["sample_id"], r["query_vertices"], r["answer_walk"]) for r in rows]
            if prior:
                self.assertEqual(latent_fields, prior)
            prior = latent_fields
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertIn("semantic_splits", manifest["stats"])
            self.assertGreaterEqual(manifest["stats"]["splits"]["train"]["query_pair_occurrences"], 50)
            if template is None:
                template = out / "codebook.json"


if __name__ == "__main__":
    unittest.main()
