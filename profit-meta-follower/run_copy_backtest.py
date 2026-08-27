"""Copy-trade backtest — separate from crowd run_backtest.py.

Scores wallets from research books (synthetic fill stats from position changes),
re-selects leaders on a rolling schedule, mirrors their books, simulates PnL.

  python profit-meta-follower/run_copy_backtest.py
  python profit-meta-follower/run_copy_backtest.py --days 7 --data-dir data-local
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import config as cfg
from pmf.copy_backtest import run_copy_backtest
from pmf.research_load import build_dataset


def _data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir:
        p = Path(args.data_dir)
        return p if p.is_absolute() else ROOT / p
    import os

    prof = os.environ.get("PMF_PROFILE", getattr(cfg, "PMF_PROFILE", "local"))
    return ROOT / f"data-{prof}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy-trade wallet selection backtest")
    ap.add_argument("--days", type=int, default=7, help="Max research days to load")
    ap.add_argument("--data-dir", type=str, default="", help="Data folder (default data-<profile>)")
    ap.add_argument("--margin-frac", type=float, default=0.30, help="Per-position margin fraction")
    ap.add_argument("--leverage", type=float, default=10.0)
    args = ap.parse_args()

    data_dir = _data_dir(args)
    research_dir = data_dir / "research"
    print(f"Copy backtest | profile={getattr(cfg, 'PMF_PROFILE', 'local')} data={data_dir}")
    print(
        f"Config: COPY_TOP_N={getattr(cfg, 'COPY_TOP_N', 3)} "
        f"scan={getattr(cfg, 'COPY_CANDIDATE_SCAN', 120)} "
        f"reselect={getattr(cfg, 'COPY_RESELECT_HOURS', 24)}h"
    )

    ds = build_dataset(research_dir, max_days=args.days, progress=True)
    if ds is None:
        print("No research data — run local gather first (books.jsonl under research/).")
        sys.exit(1)

    print(
        f"Dataset: ticks={ds.n_ticks} coins={ds.n_coins} span={ds.span_days:.2f}d "
        f"gather_pool={ds.pool_size} trade_listed={ds.live_basket_target}"
    )

    result = run_copy_backtest(
        ds,
        cfg,
        margin_frac=args.margin_frac,
        leverage=args.leverage,
    )

    print("\n=== Copy-trade backtest result ===")
    print(f"  return:      {result.return_pct:+.2f}%")
    print(f"  max_dd:      {result.max_dd_pct:.1f}%")
    print(f"  round_trips: {result.round_trips}")
    print(f"  win_rate:    {result.win_rate_pct:.1f}%")
    print(f"  open_legs:   {result.open_legs}")
    print(f"  reselects:   {result.reselects}")
    print(f"  final_leaders: {', '.join(a[:10] for a in result.leaders_picked) or '-'}")


if __name__ == "__main__":
    main()
