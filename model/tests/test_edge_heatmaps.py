"""Edge occurrence accounting and local, shared-scale plots."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths
from synthdata.codebooks import Codebooks
from synthdata.graphs import make_grid
from synthdata.language import Language
from synthdata.qa import validate_qa_answer
from qa_errors import QAErrorStats


class EdgeHeatmapTests(unittest.TestCase):
    def test_five_distinct_hues_and_opacity_increases_within_each_band(self):
        import edge_heatmaps
        self.assertTrue(hasattr(edge_heatmaps, 'level_colormap'), 'Frequency-level palette is missing')
        cmap = edge_heatmaps.level_colormap()
        colors = []
        for level in range(5):
            faint = cmap((level + .1) / 5)
            opaque = cmap((level + .9) / 5)
            self.assertEqual(faint[:3], opaque[:3])
            self.assertLess(faint[3], opaque[3])
            colors.append(faint[:3])
        self.assertEqual(len(set(colors)), 5)
        self.assertEqual(cmap(1.)[3], 1.)

    def test_counts_all_occurrences_not_only_successes_or_first_ten(self):
        language = Language(make_grid(2), Codebooks((("00",), ("01",), ("10",), ("11",))))
        stats = QAErrorStats(language)
        for walk in [(0, 1, 0)] * 12 + [(0, 1, 2, 3, 0)]:
            bits = "".join(language.codebooks[v][0] for v in walk)
            stats.observe(bits, validate_qa_answer(language, bits, (0, 3), allow_repeats=True))
        stats.observe("x", validate_qa_answer(language, "x", (0, 3)))
        result = stats.edge_usage_report()
        self.assertEqual(result["included_samples"], 13)
        self.assertEqual(result["excluded_undecodable"], 1)
        self.assertEqual(result["counts"]["0,1"], 13)
        self.assertEqual(result["counts"]["1,0"], 12)
        self.assertEqual(result["total_occurrences"], 26)
        self.assertEqual(result["invalid_transitions"], {"1,2": 1, "3,0": 1})
        self.assertAlmostEqual(sum(result["frequencies"].values()), 1.)

    def test_ambiguous_ownership_is_excluded_not_arbitrarily_assigned(self):
        language = Language(make_grid(2), Codebooks((("00",), ("01",), ("10",), ("00", "11"))))
        stats = QAErrorStats(language)
        bits = "0000"
        stats.observe(bits, validate_qa_answer(language, bits, (0, 3)))
        result = stats.edge_usage_report()
        self.assertEqual(result["excluded_ambiguous"], 1)
        self.assertEqual(result["included_samples"], 0)
        self.assertEqual(result["total_occurrences"], 0)

    def test_manifest_fast_path_fallback_and_shared_scale(self):
        from edge_heatmaps import training_edge_usage, save_edge_heatmaps
        graph = make_grid(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = {"stats": {"semantic_splits": {"train": {
                "graph_edges_directed": {"frequencies": {"0,1": 3, "1,0": 1}}}}}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            train = training_edge_usage(root, graph)
            self.assertEqual(train["total_occurrences"], 4)
            self.assertEqual(train["source"], "manifest")  # No train JSONL needed.
            generated = {**train, "counts": {"0,1": 1}, "frequencies": {"0,1": 1.}}
            plots = save_edge_heatmaps(graph, train, generated, root / "plots")
            self.assertEqual(plots["vmax"], 1.)
            for path in plots["plots"].values():
                self.assertEqual(Path(path).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            shared = save_edge_heatmaps(graph, train, generated, root / "shared", vmax=2.)
            self.assertEqual(shared['vmax'], 2.)
            self.assertEqual(len(shared['levels']), 5)
            self.assertEqual(shared['levels'][-1]['upper_frequency'], 2.)
            with self.assertRaises(ValueError):
                save_edge_heatmaps(graph, train, generated, root / "clipped", vmax=.5)
            (root / "manifest.json").write_text(json.dumps({"files": {"train": "train.jsonl"}}))
            (root / "train.jsonl").write_text(json.dumps({"answer_walk": [0, 1, 0]}) + "\n")
            self.assertEqual(training_edge_usage(root, graph)["counts"]["1,0"], 1)
            missing = training_edge_usage(root / "absent", graph)
            self.assertFalse(missing["available"])
            save_edge_heatmaps(graph, missing, {"frequencies": {}}, root / "empty")
            (root / "manifest.json").write_text(json.dumps({"config": {
                "task": {"type": "language-modeling"}}}))
            self.assertFalse(training_edge_usage(root, graph)["available"])


if __name__ == "__main__":
    unittest.main()
