from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping

from atlas.config import ShiftSpec, default_shift_spec


@dataclass(frozen=True)
class VerticalSliceConfig:
    stage: str
    output_dir: Path
    checkpoint: Path | None = None
    run_id: str = ""
    env_name: str = "Go1JoystickFlatTerrain"
    n_step: int = 1
    critic_width: int = 256
    actor_width: int = 256
    critic_depth: int = 3
    actor_depth: int = 3
    seed: int = 0
    train_steps: int = 100_000
    eval_interval: int = 100_000
    num_envs: int = 32
    eval_episodes: int = 10
    baseline_eval_episodes: int | None = None
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 100_000
    min_replay_size: int = 1024
    grad_updates_per_step: int = 1
    reward_scaling: float = 1.0
    tau: float = 0.005
    normalize_observations: bool = False
    recent_buffer_capacity: int = 10_000
    diagnostic_min_transitions: int = 1024
    diagnostic_minibatches: int = 100
    diagnostic_batch_size: int = 256
    stop_on_collapse: bool = True
    eval_deterministic: bool = True
    shift: ShiftSpec = field(default_factory=default_shift_spec)
    episode_length: int = 1000
    action_repeat: int = 1

    def with_run_id(self) -> "VerticalSliceConfig":
        if self.run_id:
            return self
        return replace(self, run_id=build_run_id(self.stage, self.n_step, self.critic_width, self.seed))

    def actor_layers(self) -> tuple[int, ...]:
        return tuple([self.actor_width] * self.actor_depth)

    def critic_layers(self) -> tuple[int, ...]:
        return tuple([self.critic_width] * self.critic_depth)

    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoint"

    def config_path(self) -> Path:
        return self.output_dir / "config.json"

    def summary_path(self) -> Path:
        return self.output_dir / "summary.json"

    def pretrain_baseline_path(self) -> Path:
        return self.output_dir / "pretrain_baseline.json"

    def eval_log_path(self) -> Path:
        return self.output_dir / "eval_log.jsonl"

    def effective_baseline_eval_episodes(self) -> int:
        return self.eval_episodes if self.baseline_eval_episodes is None else self.baseline_eval_episodes

    def uses_separate_baseline_evaluator(self) -> bool:
        return self.effective_baseline_eval_episodes() != self.eval_episodes

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["checkpoint"] = str(self.checkpoint) if self.checkpoint is not None else None
        return data


def build_run_id(stage: str, n_step: int, critic_width: int, seed: int) -> str:
    return f"{stage}_n{n_step}_c{critic_width}_seed{seed}"


def add_common_cli_args(
    parser: argparse.ArgumentParser,
    *,
    output_dir_default: Path,
) -> argparse.ArgumentParser:
    parser.add_argument("--output-dir", type=Path, default=output_dir_default)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--env-name", type=str, default="Go1JoystickFlatTerrain")
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--critic-width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=4096)
    parser.add_argument("--eval-interval", type=int, default=1024)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("--min-replay-size", type=int, default=1024)
    parser.add_argument("--diagnostic-min-transitions", type=int, default=1024)
    parser.add_argument("--diagnostic-minibatches", type=int, default=100)
    parser.add_argument("--diagnostic-batch-size", type=int, default=256)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--action-repeat", type=int, default=1)
    return parser


def add_shift_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    default_shift = default_shift_spec()
    parser.add_argument("--train-friction-min", type=float, default=default_shift.train_friction_range[0])
    parser.add_argument("--train-friction-max", type=float, default=default_shift.train_friction_range[1])
    parser.add_argument("--train-payload-min", type=float, default=default_shift.train_payload_range[0])
    parser.add_argument("--train-payload-max", type=float, default=default_shift.train_payload_range[1])
    parser.add_argument("--fine-tune-friction", type=float, default=default_shift.fine_tune_friction)
    parser.add_argument("--fine-tune-payload", type=float, default=default_shift.fine_tune_payload)
    return parser


def shift_from_args(args: argparse.Namespace) -> ShiftSpec:
    return ShiftSpec(
        train_friction_range=(float(args.train_friction_min), float(args.train_friction_max)),
        train_payload_range=(float(args.train_payload_min), float(args.train_payload_max)),
        fine_tune_friction=float(args.fine_tune_friction),
        fine_tune_payload=float(args.fine_tune_payload),
    )


def _normalize_nested_spec(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_nested_spec(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_normalize_nested_spec(item) for item in value]
    if isinstance(value, list):
        return [_normalize_nested_spec(item) for item in value]
    return value


def checkpoint_signature(
    config: VerticalSliceConfig,
    *,
    observation_spec: Any | None = None,
    observation_dtype: str | None = None,
) -> Dict[str, Any]:
    signature = {
        "env_name": config.env_name,
        "n_step": config.n_step,
        "critic_width": config.critic_width,
        "critic_depth": config.critic_depth,
        "actor_width": config.actor_width,
        "actor_depth": config.actor_depth,
        "gamma": config.gamma,
        "batch_size": config.batch_size,
        "replay_capacity": config.replay_capacity,
    }
    if observation_spec is not None:
        signature["observation_spec"] = _normalize_nested_spec(observation_spec)
    if observation_dtype is not None:
        signature["observation_dtype"] = observation_dtype
    return signature


def validate_checkpoint_compatibility(
    config: VerticalSliceConfig,
    metadata: Mapping[str, Any],
    *,
    observation_spec: Any | None = None,
    observation_dtype: str | None = None,
) -> None:
    expected = checkpoint_signature(
        config,
        observation_spec=observation_spec,
        observation_dtype=observation_dtype,
    )
    actual = dict(metadata.get("signature", {}))
    mismatches = []
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, found {actual.get(key)!r}")
    if mismatches:
        raise ValueError("Checkpoint is incompatible with the requested fine-tune config: " + "; ".join(mismatches))
