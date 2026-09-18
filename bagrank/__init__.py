"""Bag-rank backtests (local sqlite from Neon) and live trading (Hyperliquid API).

Neon is for `backup_bagrank.py` / `run_rank_backtest.py` (ROI board)
and `backup_pnl.py` / `run_pnl_backtest.py` (PnL board) only.
`run_rank.py` rebuilds the hourly majority board from Hyperliquid.
A CSV row with board=pnl snapshots the PnL-ranked wallet list; board=roi uses ROI.
"""
