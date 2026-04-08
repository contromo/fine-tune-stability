#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    from atlas_training.config import add_common_cli_args

    parser = argparse.ArgumentParser(description="Fine-tune a pretrained Go1 SAC checkpoint under the shifted domain.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    add_common_cli_args(parser, output_dir_default=Path("results/runs/finetune_go1"))
    parser.add_argument("--stop-on-collapse", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from atlas_training.runtime import run_finetune_cli

    run_finetune_cli(args)


if __name__ == "__main__":
    main()
