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

    parser = argparse.ArgumentParser(description="Pretrain a Go1 SAC checkpoint for the Stability Atlas vertical slice.")
    add_common_cli_args(parser, output_dir_default=Path("results/runs/pretrain_go1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from atlas_training.runtime import run_pretrain_cli

    run_pretrain_cli(args)


if __name__ == "__main__":
    main()
