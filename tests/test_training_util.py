from __future__ import annotations

import json
import math
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from atlas_training.util import hours_per_100m, json_ready, summarize_throughput_rates, write_json


class TrainingUtilTest(unittest.TestCase):
    def test_summarize_throughput_rates_uses_mean_of_window_hour_estimates(self) -> None:
        summary = summarize_throughput_rates([100.0, 200.0])
        self.assertEqual(summary["steps_per_second_mean"], 150.0)
        self.assertAlmostEqual(summary["hours_per_100m_mean"], (hours_per_100m(100.0) + hours_per_100m(200.0)) / 2.0)
        self.assertNotAlmostEqual(summary["hours_per_100m_mean"], hours_per_100m(summary["steps_per_second_mean"]))
        self.assertAlmostEqual(summary["hours_per_100m_min"], hours_per_100m(200.0))
        self.assertAlmostEqual(summary["hours_per_100m_max"], hours_per_100m(100.0))

    def test_hours_per_100m_handles_zero_and_nonfinite_rates(self) -> None:
        self.assertEqual(hours_per_100m(0.0), math.inf)
        self.assertEqual(hours_per_100m(float("inf")), math.inf)
        self.assertEqual(hours_per_100m(float("nan")), math.inf)
        self.assertAlmostEqual(hours_per_100m(10_000.0), 2.7777777777777777)

    def test_summarize_throughput_rates_handles_zero_window_rate(self) -> None:
        summary = summarize_throughput_rates([0.0, 100.0])
        self.assertEqual(summary["hours_per_100m_mean"], math.inf)
        self.assertEqual(summary["hours_per_100m_max"], math.inf)

    def test_json_ready_converts_paths_recursively(self) -> None:
        payload = {
            "path": Path("results/example"),
            "nested": {"items": [Path("foo"), ("bar", Path("baz"))]},
        }
        ready = json_ready(payload)
        self.assertEqual(ready["path"], "results/example")
        self.assertEqual(ready["nested"]["items"][0], "foo")
        self.assertEqual(ready["nested"]["items"][1], ["bar", "baz"])

    def test_write_json_writes_normalized_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "payload.json"
            write_json(output, {"path": Path("results/example")})
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(persisted["path"], "results/example")

    def test_write_json_cleans_temporary_artifacts_on_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output = output_dir / "payload.json"
            with mock.patch("pathlib.Path.replace", side_effect=RuntimeError("replace failed")):
                with self.assertRaises(RuntimeError):
                    write_json(output, {"path": Path("results/example")})
            self.assertFalse(output.exists())
            self.assertEqual(list(output_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
