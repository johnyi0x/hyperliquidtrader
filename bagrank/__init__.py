"""Bag-rank backtests (local sqlite from Neon) and live trading (Hyperliquid API).

Neon is for `backup_bagrank.py` / `run_rank_backtest.py` (ROI board)
and `backup_pnl.py` / `run_pnl_backtest.py` (PnL board).
`run_rank.py` seeds lookback once from `NEON_DATABASE` / `NEON_DATABASE_PNL`,
then rebuilds new hours from Hyperliquid.
A CSV row with board=pnl snapshots the PnL-ranked wallet list; board=roi uses ROI.
"""
