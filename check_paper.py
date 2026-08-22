#!/usr/bin/env python3
"""Show paper account balance / open position for this project."""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
ACCT = DATA / "paper_account.json"
TRADES = DATA / "paper_trades.jsonl"


def main() -> None:
    if not ACCT.exists():
        print("No paper_account.json yet — run bot_live.py with PAPER_TRADING=True")
        return
    acct = json.loads(ACCT.read_text(encoding="utf-8"))
    print("Paper account:")
    print(json.dumps(acct, indent=2))
    if TRADES.exists():
        lines = TRADES.read_text(encoding="utf-8").strip().splitlines()
        print(f"\nTrades logged: {len(lines)}")
        if lines:
            print("Last trade:")
            print(lines[-1])


if __name__ == "__main__":
    main()
