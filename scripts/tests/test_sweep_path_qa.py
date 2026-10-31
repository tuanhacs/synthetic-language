import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

SPEC = importlib.util.spec_from_file_location("sweep_path_qa", Path(__file__).resolve().parents[1] / "sweep_path_qa.py")
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


class SweepTests(unittest.TestCase):
    def test_default_dry_run_has_sixteen_runs_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "not-created"
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                self.assertEqual(sweep.main(["--dry-run", "--output-root", str(output)]), 0)
            text = log.getvalue().replace("\\", "/")
            self.assertIn("Runs: 8 x 2 = 16", text)
            self.assertEqual(text.count("model/scripts/train.py"), 16)
            self.assertEqual(text.count("model/scripts/evaluate.py"), 16)
            self.assertFalse(output.exists())

    def test_small_real_data_sweep_with_mock_gpu_stages_and_rerun(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "sweep"
            arguments = ["--output-root", str(output), "--pool-train-pairs", "30",
                         "--train-min", "12", "--train-max", "20", "--train-step", "4",
                         "--valid-query-pairs", "10", "--test-query-pairs", "10",
                         "--qa-mode", "iid", "--assignment", "disjoint", "--cot-mode", "none",
                         "--max-steps", "10", "--lr-decay-steps", "20", "--early-stopping-patience", "2", "--eval-samples", "1",
                         "--code-types", "prefix", "suffix", "ud",
                         "--wandb-mode", "offline", "--wandb-project", "qa-tests",
                         "--train-seed", "7"]
            real_run = sweep.run
            calls = []

            def run(script, args, log, dry_run):
                calls.append(script)
                if script == "model/scripts/train.py":
                    cfg = yaml.safe_load(Path(args[0]).read_text())
                    self.assertEqual(cfg["train"]["max_steps"], 10)
                    self.assertEqual(cfg["train"]["seed"], 7)
                    self.assertEqual(cfg["eval"]["seed"], 0)
                    self.assertEqual(cfg["train"]["lr_decay_steps"], 20)
                    self.assertEqual(cfg["train"]["early_stopping_patience"], 2)
                    self.assertEqual(cfg["wandb"]["mode"], "offline")
                    self.assertEqual(cfg["wandb"]["project"], "qa-tests")
                    run_dir = Path(cfg["train"]["out_dir"])
                    run_dir.mkdir(parents=True)
                    (run_dir / "best.pt").write_bytes(b"mock GPU checkpoint")
                    (run_dir / "train_summary.json").write_text(json.dumps({
                        "steps": 2, "max_steps": 10,
                        "stopped_early": True, "stop_reason": "early_stopping",
                    }))
                elif script == "model/scripts/evaluate.py":
                    run_dir = Path(args[0]).parent
                    (run_dir / "eval_report.json").write_text(json.dumps({"generation": {"path_qa": {
                        "n_samples": 1, "answer_decodable_pct": 100.0, "semantic_success_pct": 90.0,
                    }}}))
                else:
                    real_run(script, args, log, dry_run)

            with patch.object(sweep, "run", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sweep.main(arguments), 0)
                summary = json.loads((output / "results.json").read_text())
                self.assertEqual(len(summary), 9)
                self.assertTrue(all(row["stopped_early"] for row in summary))
                self.assertTrue(all(row["max_steps"] == 10 for row in summary))
                self.assertTrue(all(row["train_seed"] == 7 and row["data_seed"] == 0 for row in summary))
                count = len(calls)
                self.assertEqual(sweep.main(arguments), 0)
                self.assertEqual(len(calls), count)
            reference_ids = None
            for kind in ("prefix", "suffix", "ud"):
                rows = [json.loads(line) for line in (output / "datasets" / kind / "12" / "train.jsonl").read_text().splitlines()]
                larger = [json.loads(line) for line in (output / "datasets" / kind / "20" / "train.jsonl").read_text().splitlines()]
                self.assertEqual(rows, larger[:len(rows)])
                ids = [row["sample_id"] for row in rows]
                if reference_ids is not None:
                    self.assertEqual(reference_ids, ids)
                reference_ids = ids

    def test_ud_only_dry_run_prepares_reference_and_runs_eight_models(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()) as log:
            sweep.main(["--dry-run", "--code-types", "ud", "--output-root", str(Path(temp) / "out")])
            text = log.getvalue().replace("\\", "/")
            self.assertIn("Runs: 8 x 1 = 8", text)
            self.assertEqual(text.count("model/scripts/train.py"), 8)
            self.assertIn("--assignment-template", text)

    def test_paired_reversal_sweep_two_cot_modes(self):
        for cot in ("none", "segments"):
            with self.subTest(cot=cot), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "sweep"
                arguments = ["--output-root", str(output), "--pool-train-pairs", "30",
                    "--train-min", "12", "--train-max", "20", "--train-step", "4",
                    "--valid-query-pairs", "10", "--test-query-pairs", "10",
                    "--qa-mode", "iid", "--assignment", "disjoint", "--cot-mode", cot,
                    "--max-steps", "10", "--eval-samples", "1", "--paired-reversal"]
                real_run = sweep.run
                calls = []

                def run(script, args, log, dry_run):
                    calls.append(script)
                    if script == "model/scripts/train.py":
                        cfg = yaml.safe_load(Path(args[0]).read_text())
                        directory = Path(cfg["train"]["out_dir"])
                        directory.mkdir(parents=True)
                        (directory / "best.pt").write_bytes(b"mock GPU")
                    elif script == "model/scripts/evaluate.py":
                        self.assertIn("--eval-label", args)
                        directory = Path(args[args.index("--out-dir") + 1]) if "--out-dir" in args else Path(args[0]).parent
                        directory.mkdir(parents=True, exist_ok=True)
                        (directory / "eval_report.json").write_text(json.dumps({"generation": {"path_qa": {
                            "n_samples": 1, "answer_decodable_pct": 100., "semantic_success_pct": 90.,
                        }}}))
                    else:
                        real_run(script, args, log, dry_run)

                with patch.object(sweep, "run", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                    sweep.main(arguments)
                    rows = json.loads((output / "results.json").read_text())
                    self.assertEqual(len(rows), 6)
                    self.assertEqual(sum("reversed_test" in row for row in rows), 3)
                    self.assertEqual(calls.count("model/scripts/evaluate.py"), 9)
                    before = len(calls)
                    sweep.main(arguments)
                    self.assertEqual(len(calls), before)
                reference = None
                for size in (12, 16, 20):
                    prefix = output / "datasets" / "prefix" / str(size)
                    suffix = output / "datasets" / "suffix" / str(size)
                    pbooks = json.loads((prefix / "codebook.json").read_text())["codebooks"]["books"]
                    sbooks = json.loads((suffix / "codebook.json").read_text())["codebooks"]["books"]
                    self.assertEqual(sbooks, [[w[::-1] for w in b] for b in pbooks])
                    identity = tuple(sweep.digest(suffix / filename) for filename in
                        ("valid.jsonl", "test.jsonl", "test_reversed/test.jsonl"))
                    if reference:
                        self.assertEqual(identity, reference)
                    reference = identity

    def test_paired_reversal_rejects_ud(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            sweep.main(["--paired-reversal", "--code-types", "ud", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
