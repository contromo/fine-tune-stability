from __future__ import annotations

import unittest

from atlas.config import (
    ATLAS_HYPERPARAMETERS_COMPAT_FINE_TUNE_STEPS,
    DEFAULT_SWEEP_FINE_TUNE_STEPS,
    HOURS_PER_100M_NORMALIZATION_STEPS,
    REPRESENTATIVE_PRETRAIN_SENSITIVITY_CRITIC_WIDTH,
    REPRESENTATIVE_PRETRAIN_SENSITIVITY_N_STEP,
    AtlasHyperparameters,
    build_budget_table,
    default_hyperparameters,
    default_pretrain_sensitivity_pretrain_seeds,
    estimate_run_hours,
    generate_pretrain_sensitivity_sweep,
    generate_sweep,
)
from atlas.diagnostics import DEFAULT_TRIGGER_THRESHOLD


class ConfigTest(unittest.TestCase):
    def test_default_hyperparameters_reuse_trigger_threshold_constant(self) -> None:
        self.assertEqual(default_hyperparameters().trigger_threshold, DEFAULT_TRIGGER_THRESHOLD)
        self.assertEqual(default_hyperparameters().total_fine_tune_steps, DEFAULT_SWEEP_FINE_TUNE_STEPS)

    def test_atlas_hyperparameters_constructor_keeps_compatibility_default(self) -> None:
        self.assertEqual(AtlasHyperparameters().total_fine_tune_steps, ATLAS_HYPERPARAMETERS_COMPAT_FINE_TUNE_STEPS)

    def test_generate_sweep_matches_preregistered_run_count(self) -> None:
        sweep = generate_sweep()
        self.assertEqual(len(sweep), 48)

    def test_generate_pretrain_sensitivity_sweep_uses_representative_cell_defaults(self) -> None:
        sweep = generate_pretrain_sensitivity_sweep()
        self.assertEqual(len(sweep), 24)
        self.assertEqual({cell.n_step for cell in sweep}, {REPRESENTATIVE_PRETRAIN_SENSITIVITY_N_STEP})
        self.assertEqual({cell.critic.width for cell in sweep}, {REPRESENTATIVE_PRETRAIN_SENSITIVITY_CRITIC_WIDTH})
        self.assertEqual({cell.pretrain_seed for cell in sweep}, set(default_pretrain_sensitivity_pretrain_seeds()))
        self.assertEqual({cell.finetune_seed for cell in sweep}, set(range(default_hyperparameters().seeds_per_cell)))

    def test_estimate_run_hours_scales_from_100m_normalization(self) -> None:
        self.assertEqual(HOURS_PER_100M_NORMALIZATION_STEPS, 100_000_000)
        self.assertAlmostEqual(estimate_run_hours(12.5, 2_000_000), 0.25)

    def test_budget_table_includes_total_row(self) -> None:
        sweep = generate_sweep()
        rows = build_budget_table(1.5, sweep)
        self.assertEqual(rows[-1]["runs"], 48)
        self.assertEqual(rows[-1]["estimated_gpu_hours"], 72.0)


if __name__ == "__main__":
    unittest.main()
