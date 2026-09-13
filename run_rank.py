"""Live rank trader. Reads rank_live.csv (header + one copied backtest row).

Rebuilds the hourly bag-rank board from Hyperliquid (leaderboard + wallet
snapshots), same tally as the collector. Neon is not used.

Local paper:  python run_rank.py
Local live:   python run_rank.py --live
Railway:      python run_rank.py
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
