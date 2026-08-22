"""Offline parameter backtest from saved telemetry.

  python profit-meta-follower/run_backtest.py
  python profit-meta-follower/run_backtest.py --days 3 --profile local
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

import config as cfg
from pmf.backtest import BACKTEST_PARAM_KEYS, grid_search


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay telemetry to score strategy params")
    ap.add_argument("--days", type=int, default=7, help="How many recent day folders to use")
    ap.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Override data dir (default: data-{PMF_PROFILE})",
    )
    ap.add_argument("--profile", type=str, default="", help="local or cloud (sets PMF_DATA_DIR hint)")
    args = ap.parse_args()

    if args.profile:
        import os

        os.environ["PMF_PROFILE"] = args.profile.strip().lower()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        import os

        prof = os.environ.get("PMF_PROFILE", getattr(cfg, "PMF_PROFILE", "local"))
        data_dir = ROOT / f"data-{prof}"

    telemetry = data_dir / "telemetry"
    if not telemetry.exists():
        print(f"No telemetry at {telemetry} — run the bot for a few hours first.")
        sys.exit(1)

    grid = {
        "FLOW_EMA_ALPHA": [0.22, 0.26, 0.30],
        "EXIT_FLOW": [-0.010, -0.011, -0.013],
        "EXIT_RAW_FLOW": [-0.018, -0.020, -0.024],
        "EXIT_AGREEMENT_GIVEBACK": [0.20, 0.25, 0.30],
        "OPEN_CONFIRM_S": [90.0, 120.0, 150.0],
    }
    for k in grid:
        if k not in BACKTEST_PARAM_KEYS:
            raise RuntimeError(f"grid key {k} not in BACKTEST_PARAM_KEYS")

    print(f"Backtest telemetry={telemetry} days={args.days} profile={getattr(cfg, 'PMF_PROFILE', '?')}")
    top = grid_search(cfg, telemetry, grid, days=args.days, top_n=8)
    if not top:
        print("No tick files found.")
        sys.exit(1)

    print("\nTop parameter sets (higher score = better hold/churn balance):\n")
    for i, r in enumerate(top, 1):
        p = ", ".join(f"{k}={v}" for k, v in sorted(r.params.items()))
        print(
            f"  #{i} score={r.score:.2f} entries={r.entries} exits={r.exits} "
            f"avg_hold_ticks={r.avg_hold_ticks:.0f} in_market={r.time_in_market_pct:.0f}%"
        )
        print(f"      {p}\n")


if __name__ == "__main__":
    main()
