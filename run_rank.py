"""Live rank trader. Reads rank_live.csv (header + one copied backtest row).

Rebuilds the hourly bag-rank board from Hyperliquid (leaderboard + wallet
snapshots), same tally as the collector. At boot only, copies enough lookback
hours from Neon (`NEON_DATABASE` for ROI, `NEON_DATABASE_PNL` for PnL) then
never queries Neon again.

`board=pnl` in the CSV row snapshots top-week PnL wallets (like the PnL
collector). `board=roi` (default) snapshots top-week ROI wallets.

Paper (local default):  python run_rank.py
Live (real $):          python run_rank.py --live
Railway deploy:         python run_rank.py --live
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "profit-meta-follower"))

from bagrank.live import main

if __name__ == "__main__":
    raise SystemExit(main())
