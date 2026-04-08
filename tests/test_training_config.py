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
        metadata = {
            "signature": checkpoint_signature(
                config,
                observation_spec={"state": (48,), "privileged_state": (123,)},
                observation_dtype="float32",
            )
        }
        validate_checkpoint_compatibility(
            config,
            metadata,
            observation_spec={"state": (48,), "privileged_state": (123,)},
            observation_dtype="float32",
        )

    def test_validate_checkpoint_compatibility_rejects_mismatch(self) -> None:
        config = VerticalSliceConfig(stage="finetune", output_dir=Path("results"), n_step=3, critic_width=1024, seed=2)
        with self.assertRaises(ValueError):
            validate_checkpoint_compatibility(config, {"signature": {"n_step": 1, "critic_width": 256}})

    def test_checkpoint_signature_includes_replay_layout_fields(self) -> None:
        config = VerticalSliceConfig(
            stage="finetune",
            output_dir=Path("results"),
            batch_size=64,
            replay_capacity=12345,
        )
        signature = checkpoint_signature(
            config,
            observation_spec={"state": (48,), "privileged_state": (123,)},
            observation_dtype="float32",
        )
        self.assertEqual(signature["batch_size"], 64)
        self.assertEqual(signature["replay_capacity"], 12345)
        self.assertEqual(signature["observation_spec"], {"state": [48], "privileged_state": [123]})
        self.assertEqual(signature["observation_dtype"], "float32")

    def test_validate_checkpoint_compatibility_rejects_replay_layout_mismatch(self) -> None:
        config = VerticalSliceConfig(
            stage="finetune",
            output_dir=Path("results"),
            batch_size=64,
            replay_capacity=1000,
        )
        metadata = {
            "signature": checkpoint_signature(
                config,
                observation_spec={"state": (48,), "privileged_state": (123,)},
                observation_dtype="float32",
            )
        }
        with self.assertRaises(ValueError):
            validate_checkpoint_compatibility(
                VerticalSliceConfig(
                    stage="finetune",
                    output_dir=Path("results"),
                    batch_size=32,
                    replay_capacity=1000,
                ),
                metadata,
                observation_spec={"state": (48,), "privileged_state": (123,)},
                observation_dtype="float32",
            )
        with self.assertRaises(ValueError):
            validate_checkpoint_compatibility(
                config,
                metadata,
                observation_spec={"state": (64,), "privileged_state": (123,)},
                observation_dtype="float32",
            )


if __name__ == "__main__":
    unittest.main()
