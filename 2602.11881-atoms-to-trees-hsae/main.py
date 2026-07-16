"""CLI entry point for the HSAE reference implementation.

Usage:
    uv run python main.py train [--baseline] [--steps N]
    uv run python main.py eval [--steps N]
    uv run python main.py ablate [--steps N]
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "eval", "ablate"}:
        print(__doc__)
        raise SystemExit(1)

    command, rest = sys.argv[1], sys.argv[2:]
    sys.argv = [sys.argv[0], *rest]

    if command == "train":
        from experiments.run_train import main as run
    elif command == "eval":
        from experiments.run_eval import main as run
    else:
        from experiments.run_ablation import main as run

    run()


if __name__ == "__main__":
    main()
