from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from atlas.diagnostics import DEFAULT_TRIGGER_THRESHOLD


@dataclass(frozen=True)
class NetworkShape:
    depth: int
    width: int

    def layers(self) -> List[int]:
        return [self.width] * self.depth


@dataclass(frozen=True)
class ShiftSpec:
    train_friction_range: Tuple[float, float]
    train_payload_range: Tuple[float, float]
    fine_tune_friction: float
    fine_tune_payload: float


@dataclass(frozen=True)
class AtlasHyperparameters:
    learning_rate: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    target_tau: float = 0.005
    actor_update_interval: int = 1
    critic_update_interval: int = 1
    minibatches_per_step: int = 1
    eval_episodes: int = 10
    eval_interval_steps: int = 100_000
    gamma: float = 0.99
    n_steps: Tuple[int, ...] = (1, 3, 10)
    critic_widths: Tuple[int, ...] = (256, 1024)
    critic_depth: int = 3
    actor_width: int = 256
    actor_depth: int = 3
    seeds_per_cell: int = 8
    collapse_c: float = 2.0
    collapse_rho: float = 0.2
    diagnostics_window: int = 100
    trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD
    trigger_hold_evals: int = 2
    prediction_horizon_evals: int = 10
    total_fine_tune_steps: int = 100_000_000

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SweepCell:
    run_id: str
    n_step: int
    critic: NetworkShape
    actor: NetworkShape
    seed: int
    total_steps: int
    eval_interval_steps: int
    shift: ShiftSpec

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["critic_layers"] = self.critic.layers()
        data["actor_layers"] = self.actor.layers()
        return data


def default_shift_spec() -> ShiftSpec:
    return ShiftSpec(
        train_friction_range=(0.8, 1.2),
        train_payload_range=(0.8, 1.2),
        fine_tune_friction=0.3,
        fine_tune_payload=1.5,
    )


def default_hyperparameters() -> AtlasHyperparameters:
    return AtlasHyperparameters()


def generate_sweep(
    hyperparameters: AtlasHyperparameters | None = None,
    seed_values: Sequence[int] | None = None,
    shift: ShiftSpec | None = None,
) -> List[SweepCell]:
    hyperparameters = hyperparameters or default_hyperparameters()
    shift = shift or default_shift_spec()
    seeds = list(seed_values or range(hyperparameters.seeds_per_cell))

    actor = NetworkShape(depth=hyperparameters.actor_depth, width=hyperparameters.actor_width)
    cells: List[SweepCell] = []

    for n_step in hyperparameters.n_steps:
        for critic_width in hyperparameters.critic_widths:
            critic = NetworkShape(depth=hyperparameters.critic_depth, width=critic_width)
            size_label = "small" if critic_width == min(hyperparameters.critic_widths) else "large"
            for seed in seeds:
                run_id = f"n{n_step}_{size_label}_seed{seed}"
                cells.append(
                    SweepCell(
                        run_id=run_id,
                        n_step=n_step,
                        critic=critic,
                        actor=actor,
                        seed=seed,
                        total_steps=hyperparameters.total_fine_tune_steps,
                        eval_interval_steps=hyperparameters.eval_interval_steps,
                        shift=shift,
                    )
                )
    return cells


def build_budget_table(pilot_hours_per_run: float, sweep: Iterable[SweepCell]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sweep_list = list(sweep)

    grouped: Dict[Tuple[int, int], int] = {}
    for cell in sweep_list:
        key = (cell.n_step, cell.critic.width)
        grouped[key] = grouped.get(key, 0) + 1

    for (n_step, critic_width), run_count in sorted(grouped.items()):
        rows.append(
            {
                "n_step": n_step,
                "critic_width": critic_width,
                "runs": run_count,
                "pilot_hours_per_run": pilot_hours_per_run,
                "estimated_gpu_hours": round(run_count * pilot_hours_per_run, 3),
            }
        )

    rows.append(
        {
            "n_step": "total",
            "critic_width": "-",
            "runs": len(sweep_list),
            "pilot_hours_per_run": pilot_hours_per_run,
            "estimated_gpu_hours": round(len(sweep_list) * pilot_hours_per_run, 3),
        }
    )
    return rows
