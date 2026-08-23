"""Compact research logs for offline filter on/off + PnL backtests (local only)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import MarketCtx, WalletSnapshot


def _utc_day(ts: float | None = None) -> str:
    t = ts if ts is not None else time.time()
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")


def compact_book(snap: WalletSnapshot) -> list[list[Any]]:
    """[coin, side_sign (+1/-1), conviction, lev] — only open positions."""
    out: list[list[Any]] = []
    for p in snap.positions:
        if abs(p.conviction) < 1e-6:
            continue
        sign = 1 if p.side == "long" else -1
        out.append([p.coin, sign, round(p.conviction, 4), int(p.leverage)])
    return out


class ResearchWriter:
    """
    data-{profile}/research/YYYY-MM-DD/
      pool-HHMMSS.json  — ROI shortlist + holder labels (at basket build)
      books.jsonl       — sparse wallet books over time
      (marks embedded in books rows as px)
    """

    def __init__(self, data_dir: Path, *, enabled: bool = False, instance: str = "local") -> None:
        self.base = data_dir / "research"
        self.enabled = bool(enabled)
        self.instance = instance
        self._day = ""
        self._books_path: Path | None = None
        self._last_record_at = 0.0

    def _rotate(self, day: str) -> None:
        if day == self._day:
            return
        self._day = day
        folder = self.base / day
        folder.mkdir(parents=True, exist_ok=True)
        self._books_path = folder / "books.jsonl"

    def save_pool(self, *, ts: float, wallets: list[dict[str, Any]]) -> Path | None:
        if not self.enabled:
            return None
        day = _utc_day(ts)
        folder = self.base / day
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H%M%S")
        path = folder / f"pool-{stamp}.json"
        path.write_text(
            json.dumps(
                {"ts": ts, "inst": self.instance, "n": len(wallets), "wallets": wallets},
                separators=(",", ":"),
                default=str,
            ),
            encoding="utf-8",
        )
        return path

    def maybe_record_books(
        self,
        *,
        ts: float,
        interval_s: float,
        snaps_by_addr: dict[str, WalletSnapshot],
        research_addrs: list[str],
        markets: dict[str, MarketCtx],
    ) -> bool:
        if not self.enabled or not research_addrs:
            return False
        if ts - self._last_record_at < float(interval_s):
            return False
        self._rotate(_utc_day(ts))
        books: list[list[Any]] = []
        coins: set[str] = set()
        for addr in research_addrs:
            snap = snaps_by_addr.get(addr)
            if snap is None or snap.error:
                continue
            pos = compact_book(snap)
            for row in pos:
                coins.add(str(row[0]))
            books.append(
                [
                    addr,
                    round(float(snap.account_value), 2),
                    pos,
                ]
            )
        if not books:
            return False
        px: dict[str, float] = {}
        for c in coins:
            ctx = markets.get(c)
            if ctx is not None and ctx.mark > 0:
                px[c] = round(ctx.mark, 6)
        row = {"ts": ts, "inst": self.instance, "w": books, "px": px}
        assert self._books_path is not None
        with self._books_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
        self._last_record_at = ts
        return True
