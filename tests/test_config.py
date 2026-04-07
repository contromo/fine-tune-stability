from __future__ import annotations

import unittest

from atlas.config import build_budget_table, generate_sweep


class ConfigTest(unittest.TestCase):
    def test_generate_sweep_matches_preregistered_run_count(self) -> None:
        sweep = generate_sweep()
        self.assertEqual(len(sweep), 48)

    def test_budget_table_includes_total_row(self) -> None:
        sweep = generate_sweep()
        rows = build_budget_table(1.5, sweep)
        self.assertEqual(rows[-1]["runs"], 48)
        self.assertEqual(rows[-1]["estimated_gpu_hours"], 72.0)


if __name__ == "__main__":
    unittest.main()
