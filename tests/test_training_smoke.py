from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TrainingSmokeTest(unittest.TestCase):
    def test_vertical_slice_smoke(self) -> None:
        if os.environ.get("FINE_TUNE_STABILITY_RUN_TRAINING_SMOKE") != "1":
            self.skipTest("Training smoke test is disabled by default.")
        if importlib.util.find_spec("jax") is None:
            self.skipTest("Training dependencies are not installed in this interpreter.")

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


if __name__ == "__main__":
    unittest.main()
