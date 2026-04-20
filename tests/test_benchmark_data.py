from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "benchmark" / "data"
MANIFEST = DATA_DIR / "manifest.sha256"


def _parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, filename = line.split(None, 1)
        entries[filename] = digest
    return entries


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BenchmarkDataTest(unittest.TestCase):
    def test_manifest_exists_and_covers_all_data_files(self) -> None:
        self.assertTrue(MANIFEST.exists(), f"missing {MANIFEST}")
        entries = _parse_manifest(MANIFEST)
        on_disk = {p.name for p in DATA_DIR.iterdir() if p.is_file() and p.name != "manifest.sha256"}
        self.assertEqual(set(entries.keys()), on_disk)

    def test_manifest_hashes_match_file_contents(self) -> None:
        entries = _parse_manifest(MANIFEST)
        for filename, expected in entries.items():
            with self.subTest(file=filename):
                actual = _sha256(DATA_DIR / filename)
                self.assertEqual(actual, expected)

    def test_horizon_points_schema_stable(self) -> None:
        path = DATA_DIR / "horizon_final_return_points.csv"
        with path.open(encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
        self.assertEqual(
            header,
            [
                "horizon",
                "seed",
                "collapsed",
                "warning_triggered",
                "final_return_mean",
                "final_return_std",
                "steps_per_second",
                "summary_path",
            ],
        )

    def test_warning_pilot_runs_schema_stable(self) -> None:
        path = DATA_DIR / "warning_pilot_runs.csv"
        with path.open(encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
        self.assertEqual(
            header,
            ["pilot_id", "run_id", "evals", "first_warning_eval", "first_collapse_eval", "lead_time_evals"],
        )

    def test_horizon_summary_json_contains_all_three_horizons(self) -> None:
        payload = json.loads((DATA_DIR / "horizon_final_return_summary.json").read_text(encoding="utf-8"))
        horizons = {int(entry["horizon"]) for entry in payload}
        self.assertEqual(horizons, {1, 3, 10})

    def test_warning_pilot_summary_json_contains_both_pilots(self) -> None:
        payload = json.loads((DATA_DIR / "warning_pilot_summary.json").read_text(encoding="utf-8"))
        self.assertIn("pilot_summaries", payload)
        self.assertIn("pilot_runs", payload)
        pilot_ids = {entry["pilot_id"] for entry in payload["pilot_summaries"]}
        self.assertIn("pilot_gate_1m_v3", pilot_ids)
        self.assertIn("pilot_gate_1m_v4", pilot_ids)


if __name__ == "__main__":
    unittest.main()
