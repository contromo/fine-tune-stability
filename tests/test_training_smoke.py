from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TrainingSmokeTest(unittest.TestCase):
    def _require_smoke_environment(self) -> None:
        if os.environ.get("FINE_TUNE_STABILITY_RUN_TRAINING_SMOKE") != "1":
            self.skipTest("Training smoke test is disabled by default.")
        if importlib.util.find_spec("jax") is None:
            self.skipTest("Training dependencies are not installed in this interpreter.")

    def test_vertical_slice_smoke(self) -> None:
        self._require_smoke_environment()

        pretrain_dir = ROOT / "results" / "runs" / "unittest_pretrain"
        finetune_dir = ROOT / "results" / "runs" / "unittest_finetune"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_pretrain.py"),
                "--output-dir",
                str(pretrain_dir),
                "--train-steps",
                "16",
                "--eval-interval",
                "16",
                "--num-envs",
                "2",
                "--eval-episodes",
                "2",
                "--batch-size",
                "8",
                "--min-replay-size",
                "8",
                "--episode-length",
                "16",
            ],
            check=True,
            cwd=ROOT,
        )

    def test_pilot_smoke(self) -> None:
        self._require_smoke_environment()

        pilot_dir = ROOT / "results" / "runs" / "unittest_pilot"
        finetune_dir = ROOT / "results" / "runs" / "unittest_finetune"
        pretrain_dir = pilot_dir / "shared_pretrain"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_pilot.py"),
                "--profile",
                "production",
                "--output-dir",
                str(pilot_dir),
                "--allow-cpu",
                "--force",
                "--pretrain-steps",
                "16",
                "--fine-tune-steps",
                "160",
                "--seeds",
                "0,1",
                "--eval-interval",
                "32",
                "--num-envs",
                "2",
                "--eval-episodes",
                "2",
                "--baseline-eval-episodes",
                "2",
                "--batch-size",
                "8",
                "--min-replay-size",
                "8",
                "--diagnostic-min-transitions",
                "16",
                "--diagnostic-minibatches",
                "2",
                "--diagnostic-batch-size",
                "8",
                "--episode-length",
                "16",
                "--throughput-probe-updates",
                "2",
            ],
            check=True,
            cwd=ROOT,
        )
        self.assertTrue((pilot_dir / "preflight.json").exists())
        self.assertTrue((pilot_dir / "pilot.log").exists())
        self.assertTrue((pilot_dir / "pilot_report.json").exists())
        self.assertTrue((pilot_dir / "shared_pretrain" / "checkpoint" / "checkpoint.msgpack").exists())
        self.assertTrue((pilot_dir / "seed_0" / "eval_log.jsonl").exists())
        self.assertTrue((pilot_dir / "seed_0" / "diagnostic_summary.json").exists())
        self.assertTrue((pilot_dir / "seed_1" / "eval_log.jsonl").exists())
        self.assertTrue((pilot_dir / "extreme_probe" / "summary.json").exists())
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_finetune.py"),
                "--checkpoint",
                str(pretrain_dir / "checkpoint"),
                "--output-dir",
                str(finetune_dir),
                "--train-steps",
                "96",
                "--eval-interval",
                "32",
                "--num-envs",
                "2",
                "--eval-episodes",
                "2",
                "--batch-size",
                "8",
                "--min-replay-size",
                "8",
                "--diagnostic-min-transitions",
                "16",
                "--diagnostic-minibatches",
                "2",
                "--diagnostic-batch-size",
                "8",
                "--episode-length",
                "16",
            ],
            check=True,
            cwd=ROOT,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_diagnostic.py"),
                "--eval-log",
                str(finetune_dir / "eval_log.jsonl"),
                "--output",
                str(ROOT / "results" / "unittest_diagnostic_summary.json"),
            ],
            check=True,
            cwd=ROOT,
        )

        summary = json.loads((finetune_dir / "summary.json").read_text(encoding="utf-8"))
        probe_size = summary.get("probe_size")
        self.assertIsInstance(probe_size, int)
        self.assertGreater(probe_size, 0)

        rows = [
            json.loads(line)
            for line in (finetune_dir / "eval_log.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertGreater(len(rows), 0, "fine-tune produced no eval rows")
        rows_with_drift = [
            row for row in rows if "actor_kl_drift" in row and "q_magnitude_drift" in row
        ]
        self.assertGreater(
            len(rows_with_drift), 0, "no eval rows carry actor_kl_drift / q_magnitude_drift"
        )
        for row in rows_with_drift:
            self.assertTrue(math.isfinite(float(row["actor_kl_drift"])))
            self.assertTrue(math.isfinite(float(row["q_magnitude_drift"])))


if __name__ == "__main__":
    unittest.main()
