from __future__ import annotations

import unittest
from pathlib import Path

from atlas_training.config import VerticalSliceConfig, build_run_id, checkpoint_signature, validate_checkpoint_compatibility


class TrainingConfigTest(unittest.TestCase):
    def test_build_run_id(self) -> None:
        self.assertEqual(build_run_id("finetune", 3, 1024, 7), "finetune_n3_c1024_seed7")

    def test_with_run_id_preserves_shift_dataclass(self) -> None:
        config = VerticalSliceConfig(stage="pretrain", output_dir=Path("results"))
        updated = config.with_run_id()
        self.assertTrue(updated.run_id.startswith("pretrain_n1_c256_seed0"))
        self.assertEqual(updated.shift.fine_tune_friction, config.shift.fine_tune_friction)

    def test_validate_checkpoint_compatibility(self) -> None:
        config = VerticalSliceConfig(stage="finetune", output_dir=Path("results"), n_step=3, critic_width=1024, seed=2)
        metadata = {"signature": checkpoint_signature(config)}
        validate_checkpoint_compatibility(config, metadata)

    def test_validate_checkpoint_compatibility_rejects_mismatch(self) -> None:
        config = VerticalSliceConfig(stage="finetune", output_dir=Path("results"), n_step=3, critic_width=1024, seed=2)
        with self.assertRaises(ValueError):
            validate_checkpoint_compatibility(config, {"signature": {"n_step": 1, "critic_width": 256}})


if __name__ == "__main__":
    unittest.main()
