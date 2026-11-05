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
