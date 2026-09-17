"""Live rank trader. Reads rank_live.csv (header + one copied backtest row).

Rebuilds the hourly bag-rank board from Hyperliquid (leaderboard + wallet
snapshots), same tally as the collector. Neon is not used.

Paper (default):  python run_rank.py
Live (real $):    python run_rank.py --live
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
