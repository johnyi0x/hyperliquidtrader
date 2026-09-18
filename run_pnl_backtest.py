"""CLI: Numba rank-strategy backtest on the local PnL-ranked collector backup."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bagrank.backtest import main as backtest_main
from bagrank.dsn import default_pnl_sqlite_path, load_env


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = list(sys.argv[1:] if argv is None else argv)
    if "--db" not in args:
        args = ["--db", str(default_pnl_sqlite_path()), *args]
    return backtest_main(args, kind="pnl")


if __name__ == "__main__":
    raise SystemExit(main())
