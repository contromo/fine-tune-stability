#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained Go1 SAC checkpoint under the shifted domain.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs/finetune_go1"))
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
    parser.add_argument("--stop-on-collapse", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from atlas_training.runtime import run_finetune_cli

    run_finetune_cli(args)


if __name__ == "__main__":
    main()
