"""Recovery certificates do not assume word cuts or use the target answer."""
import itertools
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthdata.config import (TrainNoiseConfig, Config, DataConfig, TaskConfig,
                              LanguageConfig, CodeConfig, parse_config)
from synthdata.qa import QASample
from synthdata.qa_noise import QueryRecoveryChecker, apply_train_noise, apply_query_noise


def language(books):
    return SimpleNamespace(codebooks=SimpleNamespace(books=books))


class RecoverableNoiseTests(unittest.TestCase):
    def test_multibit_checker_matches_exhaustive_oracle(self):
        books = (("0", "01"), ("110",), ("10",))
        checker = QueryRecoveryChecker(language(books), (1, 3))
        queries = []
        for length in range(1, 4):
            for vertices in itertools.product(range(3), repeat=length):
                for words in itertools.product(*(books[v] for v in vertices)):
                    queries.append(("".join(words), vertices))
        for bits, expected in queries[:70]:
            for radius in (0, 1, 2, 3):
                noisy = bits.translate(str.maketrans("01", "10"))[:radius] + bits[radius:]
                candidates = {vertices for clean, vertices in queries if len(clean) == len(noisy)
                              and sum(a != b for a, b in zip(clean, noisy)) <= radius}
                self.assertEqual(checker.uniquely_recovers(noisy, expected, radius), candidates == {expected})

    def test_multiple_occurrences_bits_budget_and_histograms(self):
        lang = language((("00000000",), ("11111111",)))
        samples = tuple(QASample("0000000011111111", "0011", (0, 1), (0, 1),
            (8, 16), (2, 4), (2,), ("0011",), f"multi:{i}") for i in range(30))
        cfg = TrainNoiseConfig(mode="recoverable", sample_rate=1, codeword_occurrences=(2, 2),
                               bit_flips_per_occurrence=(1, 2), max_total_flips=3, max_attempts=20)
        result, report = apply_train_noise(samples, cfg, language=lang, query_len=(2, 4))
        self.assertEqual(result, apply_train_noise(samples, cfg, language=lang, query_len=(2, 4))[0])
        self.assertEqual(report["corrupted_samples"], 30)
        self.assertEqual(report["noised_occurrences_histogram"], {"2": 30})
        self.assertEqual(set(report["total_flips_histogram"]), {"2", "3"})
        checker = QueryRecoveryChecker(lang, (2, 4))
        for s in result:
            total = sum(len(e["bit_indices"]) for e in s.noise_edits)
            self.assertEqual(sum(a != b for a, b in zip(s.query_bits, s.bits[:16])), total)
            self.assertTrue(checker.uniquely_recovers(s.bits[:16], s.query_vertices, total))
            self.assertEqual(s.bits[16:], s.clean_bits[16:])
            self.assertEqual(len({e["codeword_index"] for e in s.noise_edits}), 2)
        _, test_report = apply_query_noise(samples, cfg, language=lang, query_len=(2, 4))
        self.assertEqual(test_report["corrupted_samples"], 30)

    def test_multibit_one_occurrence_and_ambiguous_budget_not_relaxed(self):
        safe = language((("00000000",), ("11111111",)))
        s = QASample("0000000011111111", "01", (0, 1), (0, 1), (8, 16), (1, 2), (2,))
        cfg = TrainNoiseConfig(mode="recoverable", sample_rate=1, codeword_occurrences=(1, 1),
                               bit_flips_per_occurrence=(2, 2), max_attempts=5)
        result, report = apply_train_noise((s,), cfg, language=safe, query_len=(2, 4))
        self.assertEqual(report["total_flips_histogram"], {"2": 1})
        self.assertEqual(len(result[0].noise_edits[0]["bit_indices"]), 2)
        ambiguous = language((("0000",), ("1111",)))
        s = QASample("00001111", "01", (0, 1), (0, 1), (4, 8), (1, 2), (2,))
        result, report = apply_train_noise((s,), cfg, language=ambiguous, query_len=(2, 4))
        self.assertEqual(result, (s,))
        self.assertEqual(report["skipped_attempt_limit"], 1)
        self.assertEqual(report["requested_total_flips_histogram"], {"2": 1})
        self.assertEqual(report["corrupted_samples"], 0)
        self.assertLessEqual(report["candidate_attempts"], 5)
        with self.assertRaises(ValueError):
            TrainNoiseConfig(mode="recoverable", codeword_occurrences=(2, 2),
                             bit_flips_per_occurrence=(2, 2), max_total_flips=3)

    def test_config_roundtrip_and_old_default_serialization(self):
        cfg = Config(language=LanguageConfig(code=CodeConfig(depth_range=(1, 4), power_x=6)),
                     data=DataConfig(train_noise=TrainNoiseConfig(mode="recoverable")),
                     task=TaskConfig(type="path-qa"))
        self.assertEqual(parse_config(cfg.to_dict()), cfg)
        self.assertNotIn("mode", TrainNoiseConfig().to_dict())

    def test_trie_matches_exhaustive_oracle_with_unknown_boundaries(self):
        books = (("0", "01"), ("110",), ("10",))
        checker = QueryRecoveryChecker(language(books), (1, 3))
        queries = []
        for length in range(1, 4):
            for vertices in itertools.product(range(3), repeat=length):
                for words in itertools.product(*(books[v] for v in vertices)):
                    queries.append(("".join(words), vertices))
        for bits, expected in queries:
            for index in range(len(bits)):
                noisy = bits[:index] + str(1 - int(bits[index])) + bits[index + 1:]
                candidates = {vertices for clean, vertices in queries if len(clean) == len(noisy)
                              and sum(a != b for a, b in zip(clean, noisy)) <= 1}
                self.assertEqual(checker.uniquely_recovers(noisy, expected), candidates == {expected})

    def test_private_dictionary_and_different_representations(self):
        checker = QueryRecoveryChecker(language((("000", "001", "100"), ("111", "100"))), (1, 1))
        # 100 is shared: not used by the query encoder, but still forbidden as a mutated word.
        self.assertIn("100", checker.all_words)
        self.assertTrue(checker.uniquely_recovers("000", (0,)))
        self.assertFalse(checker.uniquely_recovers("011", (0,)))

    def test_actual_mutations_one_bit_clean_answer_cot_and_determinism(self):
        lang = language((("00000",), ("11100",), ("10011",), ("01111",)))
        samples = tuple(QASample("0000011100", "answer", (0, 1), (0, 1),
                        (5, 10), (3, 6), (2,), ("cot",), f"id:{i}") for i in range(20))
        cfg = TrainNoiseConfig(mode="recoverable", sample_rate=.5)
        self.assertEqual(cfg.scope, "query")
        self.assertEqual(cfg.codeword_occurrences, (1, 1))
        result, report = apply_train_noise(samples, cfg, language=lang, query_len=(2, 4))
        self.assertEqual(report["corrupted_samples"], 10)
        self.assertEqual(report["skipped_no_safe_flip"], 0)
        self.assertEqual(result, apply_train_noise(samples, cfg, language=lang, query_len=(2, 4))[0])
        checker = QueryRecoveryChecker(lang, (2, 4))
        for sample in result:
            self.assertEqual(sample.bits[10:], samples[0].clean_bits[10:])
            if sample.noised_bits is not None:
                self.assertEqual(sum(a != b for a, b in zip(sample.query_bits, sample.bits[:10])), 1)
                self.assertTrue(checker.uniquely_recovers(sample.bits[:10], sample.query_vertices))
                edit = sample.noise_edits[0]
                start = 5 * edit["codeword_index"]
                self.assertNotIn(sample.bits[start:start + 5], checker.all_words)
        test_result, test_report = apply_query_noise(samples, cfg, language=lang, query_len=(2, 4))
        self.assertEqual(test_report["test_samples"], 20)
        self.assertEqual(test_report["corrupted_samples"], 10)
        self.assertEqual(len(test_result), 20)

    def test_no_safe_mutation_is_reported_not_relaxed(self):
        # Every flipped bit is an assigned codeword.
        sample = QASample("01", "01", (0, 1), (0, 1), (1, 2), (1, 2), (2,))
        cfg = TrainNoiseConfig(mode="recoverable", sample_rate=1)
        result, report = apply_train_noise((sample,), cfg, language=language((("0",), ("1",))), query_len=(2, 4))
        self.assertEqual(result, (sample,))
        self.assertEqual(report["requested_samples"], 1)
        self.assertEqual(report["skipped_no_safe_flip"], 1)
        self.assertEqual(report["actual_sample_rate"], 0)
        # Mutated words are outside the codebook, but ambiguity still forbids them.
        sample = QASample("0011", "0011", (0, 1), (0, 1), (2, 4), (2, 4), (2,))
        _, report = apply_train_noise((sample,), cfg, language=language((("00",), ("11",))), query_len=(2, 2))
        self.assertEqual(report["skipped_no_safe_flip"], 1)
        self.assertEqual(report["rejected_ambiguous_query"], 4)

    def test_missing_context_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "requires language"):
            apply_train_noise((), TrainNoiseConfig(mode="recoverable"))


if __name__ == "__main__":
    unittest.main()
