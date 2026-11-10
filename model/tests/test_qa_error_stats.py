"""Conditional QA errors and the first non-edge, independent of reference paths."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evals
from synthdata.codebooks import Codebooks
from synthdata.graphs import make_grid
from synthdata.language import Language
from synthdata.qa import QASample, validate_qa_answer
from synthdata.tokenizer import BitTokenizer


class QAErrorStatsTests(unittest.TestCase):
    def test_conditionals_denominators_first_edges_and_log(self):
        self.assertTrue(hasattr(evals, "QAErrorStats"), "QA error analysis is missing")
        language = Language(make_grid(6), Codebooks(tuple((f"{v:06b}",) for v in range(36))))
        query = (0, 2, 8, 14)
        def encode(walk):
            return "".join(language.codebooks[v][0] for v in walk)
        walks = [(0, 1, 2, 8, 14), (1, 2, 8, 14), (0, 1, 2, 8, 14, 15),
                 (0, 1, 2, 3, 9, 15, 14), (1, 2, 8, 14, 15), (0, 2, 8, 14), (0, 14, 0)]
        answers = [encode(w) for w in walks] + ["1"]
        sample = QASample(encode(query), encode(walks[0]), query, walks[0],
                          (6, 12, 18, 24), (6, 12, 18, 24, 30), (3, 4, 5))
        with patch.object(evals, "generate", return_value=[sample.query_bits + "_" + a for a in answers]):
            report = evals.eval_path_qa(SimpleNamespace(cfg=SimpleNamespace(context_len=256)),
                language, BitTokenizer(True), [sample] * 8, n_samples=8,
                task_config={"allow_repeats": True, "query_len": [4, 4]})
        errors = report["error_analysis"]
        self.assertEqual(errors["undecodable"]["n"], 1)
        valid = errors["valid_walk"]
        self.assertEqual(valid["n"], 5)
        self.assertEqual(valid["pct_of_total"], 62.5)
        self.assertEqual(valid["wrong_start_pct"], 40.)
        self.assertEqual(valid["wrong_end_pct"], 40.)
        self.assertEqual(valid["correct_endpoints_n"], 2)
        self.assertEqual(valid["middle_waypoints_failed_pct"], 20.)
        self.assertEqual(valid["semantic_success_pct"], 20.)
        invalid = errors["invalid_walk"]
        self.assertEqual(invalid["n"], 2)
        self.assertEqual(invalid["localized_n"], 2)
        self.assertEqual(invalid["unlocalized_n"], 0)
        pairs = invalid["first_invalid_edge_by_rows"]
        self.assertEqual(len(pairs), 36)
        self.assertEqual(pairs["1->1"], {"count": 1, "pct": 50.})
        self.assertEqual(pairs["1->3"], {"count": 1, "pct": 50.})
        self.assertEqual(pairs["3->1"]["count"], 0)  # Do not count the second bad edge.
        self.assertEqual(sum(v["count"] for v in pairs.values()), 2)
        self.assertEqual(sum(v["pct"] for v in pairs.values()), 100.)
        self.assertEqual(report["semantic_success_pct"], 12.5)  # Old all-sample metric unchanged.
        self.assertEqual(json.loads(json.dumps(report))["error_analysis"], errors)
        text = evals.format_report({"dataset_dir": "demo", "model": {"size": "base"},
            "num_params": 1, "generation": {"path_qa": report}})
        self.assertIn("QA errors", text)
        self.assertIn("1->1", text)
        self.assertIn("6->6", text)
        self.assertIn("wrong-start", text)

    def test_graph_free_variable_codewords_and_ambiguous_ownership(self):
        self.assertTrue(hasattr(evals, "QAErrorStats"), "QA error analysis is missing")
        for words in (("0", "01", "011", "0111"), ("0", "10", "110", "1110"),
                      ("0", "01", "110", "11110")):  # UD, neither prefix- nor suffix-free.
            language = Language(make_grid(2), Codebooks(tuple((w,) for w in words)))
            stats = evals.QAErrorStats(language)
            bits = words[0] + words[3]  # Diagonal jump, not a graph edge.
            stats.observe(bits, validate_qa_answer(language, bits, (0, 3), allow_repeats=True))
            self.assertEqual(stats.report()["invalid_walk"]["first_invalid_edge_by_rows"]["1->2"]["count"], 1)
        language = Language(make_grid(2), Codebooks((("00",), ("01",), ("10",), ("00", "11"))))
        stats = evals.QAErrorStats(language)
        bits = "0000"  # Every ownership assignment gives a self-loop or diagonal jump.
        stats.observe(bits, validate_qa_answer(language, bits, (0, 3), allow_repeats=True))
        invalid = stats.report()["invalid_walk"]
        self.assertEqual(invalid["unlocalized_n"], 1)
        self.assertEqual(invalid["localized_n"], 0)
        self.assertTrue(all(cell["pct"] is None for cell in invalid["first_invalid_edge_by_rows"].values()))

    def test_empty_groups_have_null_percentages(self):
        self.assertTrue(hasattr(evals, "QAErrorStats"), "QA error analysis is missing")
        language = Language(make_grid(2), Codebooks((("00",), ("01",), ("10",), ("11",))))
        stats = evals.QAErrorStats(language)
        report = stats.report()
        self.assertEqual(report["total_n"], 0)
        self.assertIsNone(report["valid_walk"]["wrong_start_pct"])
        stats.observe("x", validate_qa_answer(language, "x", (0, 1)))
        self.assertEqual(stats.report()["undecodable"]["pct_of_total"], 100.)


if __name__ == "__main__":
    unittest.main()
