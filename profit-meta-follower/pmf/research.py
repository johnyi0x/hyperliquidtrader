"""Research gather logs for offline crowd + indicator backtests.

Order of work (must not distort live-parity crowd samples):
  1) wallet book snapshots (clearinghouse — same as live)
  2) books.jsonl + crowd_ticks.jsonl (all / holders / cloud-like trade)
  3) marks.jsonl
  4) candles (after books; round-robin, never blocks step 1–2)

Cloud keeps RESEARCH_DATA_ENABLED=False.
Canonical for strategy backtests: books.jsonl (+ pool holder labels).
crowd_ticks are convenience / validation vs live BookEngine.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.candles import fetch_closed_candles

from .types import MarketCtx, WalletSnapshot

SCHEMA_VERSION = 2


def _utc_day(ts: float | None = None) -> str:
    t = ts if ts is not None else time.time()
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")


def _coin_dir_name(coin: str) -> str:
    return str(coin).replace(":", "_").replace("/", "_")


def compact_book(snap: WalletSnapshot) -> list[list[Any]]:
    """[coin, side_sign (+1/-1), conviction, lev] — only open positions."""
    out: list[list[Any]] = []
    for p in snap.positions:
        if abs(p.conviction) < 1e-6:
            continue
        sign = 1 if p.side == "long" else -1
        out.append([p.coin, sign, round(p.conviction, 4), int(p.leverage)])
    return out


def compact_mkt(ctx: MarketCtx) -> list[float]:
    """[mark, funding, oi, basis, day_vol]."""
    return [
        round(float(ctx.mark or 0.0), 6),
        round(float(ctx.funding or 0.0), 8),
        round(float(ctx.open_interest or 0.0), 2),
        round(float(ctx.basis or 0.0), 6),
        round(float(ctx.day_volume or 0.0), 0),
    ]


def compact_candle(c: dict[str, Any]) -> list[Any]:
    """[t_open_ms, o, h, l, c, v]."""
    return [
        int(c["t"]),
        round(float(c.get("o") or 0), 6),
        round(float(c.get("h") or 0), 6),
        round(float(c.get("l") or 0), 6),
        round(float(c.get("c") or 0), 6),
        round(float(c.get("v") or 0), 4),
    ]


def compact_research_votes(votes: list[Any]) -> list[dict[str, Any]]:
    """Live-parity vote dump including flow lanes used by cloud exits."""
    out: list[dict[str, Any]] = []
    for v in votes:
        out.append(
            {
                "c": v.coin,
                "s": v.side,
                "agr": round(float(v.agreement), 4),
                "conv": round(float(getattr(v, "raw_conviction", v.avg_conviction) or v.avg_conviction), 4),
                "ema": round(float(getattr(v, "ema", v.avg_conviction) or v.avg_conviction), 4),
                "flow": round(float(getattr(v, "flow", 0.0) or 0.0), 5),
                "rflow": round(float(getattr(v, "raw_flow", 0.0) or 0.0), 5),
                "pers": round(float(getattr(v, "persist_s", 0.0) or 0.0), 1),
                "wl": int(v.wallets_long),
                "ws": int(v.wallets_short),
                "lev": int(getattr(v, "median_leverage", 0) or 0),
            }
        )
    return out


class ResearchWriter:
    """
    data-{profile}/research/
      SCHEMA.json
      YYYY-MM-DD/
        pool-HHMMSS.json
        books.jsonl
        crowd_ticks.jsonl   — all / holders raw + trade (refined, cloud knobs)
        marks.jsonl
        candles/<coin>/<interval>.jsonl
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        enabled: bool = False,
        instance: str = "local",
        logger: logging.Logger | None = None,
    ) -> None:
        self.base = data_dir / "research"
        self.enabled = bool(enabled)
        self.instance = instance
        self.log = logger or logging.getLogger("pmf-research")
        self._day = ""
        self._books_path: Path | None = None
        self._marks_path: Path | None = None
        self._ticks_path: Path | None = None
        self._last_books_at = 0.0
        self._last_marks_at = 0.0
        self._last_candle_at = 0.0
        self.seen_coins: set[str] = set()
        self._candle_cursor = 0
        self._candle_last_t: dict[str, int] = {}  # "coin|interval" -> last open ms written
        self._schema_written = False

    def load_resume(self, raw: dict[str, Any] | None) -> None:
        data = raw or {}
        self.seen_coins = {str(c) for c in (data.get("seen_coins") or []) if str(c).strip()}
        self._last_books_at = float(data.get("last_books_at") or 0.0)
        self._last_marks_at = float(data.get("last_marks_at") or 0.0)
        self._last_candle_at = float(data.get("last_candle_at") or 0.0)
        self._candle_cursor = int(data.get("candle_cursor") or 0)
        self._candle_last_t = {
            str(k): int(v) for k, v in (data.get("candle_last_t") or {}).items()
        }

    def dump_resume(self) -> dict[str, Any]:
        return {
            "seen_coins": sorted(self.seen_coins),
            "last_books_at": self._last_books_at,
            "last_marks_at": self._last_marks_at,
            "last_candle_at": self._last_candle_at,
            "candle_cursor": self._candle_cursor,
            "candle_last_t": dict(self._candle_last_t),
            "schema": SCHEMA_VERSION,
        }

    def ensure_schema(self) -> None:
        if not self.enabled or self._schema_written:
            return
        self.base.mkdir(parents=True, exist_ok=True)
        path = self.base / "SCHEMA.json"
        path.write_text(
            json.dumps(
                {
                    "version": SCHEMA_VERSION,
                    "inst": self.instance,
                    "canonical": "books.jsonl + pool-*.json holder labels — rebuild any strategy offline",
                    "order": [
                        "wallet clearinghouse snapshots (crowd base, live-parity)",
                        "books.jsonl + crowd_ticks.jsonl (only when coverage OK)",
                        "marks.jsonl",
                        "candles/* (after books; never delay book samples)",
                    ],
                    "books": {
                        "w": "[addr, equity, [[coin, side(+1/-1), conv, lev], ...], fetched_at]",
                        "mkt": "coin -> [mark, funding, oi, basis, day_vol] at book ts",
                        "cov": "{fresh, listed, holders_fresh, holders_listed, frac}",
                    },
                    "crowd_ticks": {
                        "all": "raw build_votes over full research pool (filter-off)",
                        "holders": "raw build_votes over holder-labeled wallets",
                        "trade": "BookEngine.refine on holders with cloud consensus knobs (flow/rflow)",
                    },
                    "marks": "coin -> [mark, funding, oi, basis, day_vol]",
                    "candles": "[t_open_ms, o, h, l, c, v] closed bars only",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._schema_written = True

    def _rotate(self, day: str) -> None:
        if day == self._day:
            return
        self._day = day
        folder = self.base / day
        folder.mkdir(parents=True, exist_ok=True)
        self._books_path = folder / "books.jsonl"
        self._marks_path = folder / "marks.jsonl"
        self._ticks_path = folder / "crowd_ticks.jsonl"

    def save_pool(self, *, ts: float, wallets: list[dict[str, Any]]) -> Path | None:
        if not self.enabled:
            return None
        self.ensure_schema()
        day = _utc_day(ts)
        folder = self.base / day
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H%M%S")
        path = folder / f"pool-{stamp}.json"
        path.write_text(
            json.dumps(
                {
                    "ts": ts,
                    "inst": self.instance,
                    "schema": SCHEMA_VERSION,
                    "n": len(wallets),
                    "wallets": wallets,
                },
                separators=(",", ":"),
                default=str,
            ),
            encoding="utf-8",
        )
        return path

    def _mkt_map(self, coins: set[str], markets: dict[str, MarketCtx]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for c in coins:
            ctx = markets.get(c)
            if ctx is None or not (ctx.mark > 0):
                continue
            out[c] = compact_mkt(ctx)
        return out

    def maybe_record_crowd(
        self,
        *,
        ts: float,
        books_interval_s: float,
        marks_interval_s: float,
        snaps_by_addr: dict[str, WalletSnapshot],
        research_addrs: list[str],
        markets: dict[str, MarketCtx],
        coverage: dict[str, Any] | None = None,
        min_coverage: float = 0.0,
        min_fresh_wallets: int = 20,
        crowd_all: list[dict[str, Any]] | None = None,
        crowd_holders: list[dict[str, Any]] | None = None,
        crowd_trade: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, bool]:
        """Steps 2–3: books (+ optional live-like ticks) then marks. No candles."""
        if not self.enabled or not research_addrs:
            return False, False
        self.ensure_schema()
        self._rotate(_utc_day(ts))
        wrote_books = False
        wrote_marks = False
        cov = coverage or {}
        frac = float(cov.get("frac") or 0.0)
        fresh_n = int(cov.get("fresh") or 0)
        min_fresh = max(1, int(min_fresh_wallets or 20))
        pct_gate = float(min_coverage or 0.0)
        warm_ok = fresh_n >= min_fresh or (pct_gate > 0 and frac + 1e-9 >= pct_gate)

        if ts - self._last_books_at >= float(books_interval_s):
            if not warm_ok:
                self.log.info(
                    "Research warm-up — %s/%s wallets snapped (need %s before first books row)",
                    fresh_n,
                    int(cov.get("listed") or 0),
                    min_fresh,
                )
                self._last_books_at = ts
            else:
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
                            round(float(snap.fetched_at), 3),
                        ]
                    )
                if books:
                    self.seen_coins |= coins
                    row: dict[str, Any] = {
                        "ts": ts,
                        "inst": self.instance,
                        "schema": SCHEMA_VERSION,
                        "cov": cov,
                        "w": books,
                        "mkt": self._mkt_map(coins, markets),
                    }
                    assert self._books_path is not None
                    with self._books_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
                    if (
                        crowd_all is not None
                        or crowd_holders is not None
                        or crowd_trade is not None
                    ):
                        tick = {
                            "ts": ts,
                            "inst": self.instance,
                            "schema": SCHEMA_VERSION,
                            "cov": cov,
                            "all": crowd_all or [],
                            "holders": crowd_holders or [],
                            "trade": crowd_trade or [],
                        }
                        assert self._ticks_path is not None
                        with self._ticks_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(tick, separators=(",", ":"), default=str) + "\n")
                    self._last_books_at = ts
                    wrote_books = True

        mark_coins = set(self.seen_coins)
        if ts - self._last_marks_at >= float(marks_interval_s) and mark_coins:
            mkt = self._mkt_map(mark_coins, markets)
            if mkt:
                assert self._marks_path is not None
                with self._marks_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "ts": ts,
                                "inst": self.instance,
                                "schema": SCHEMA_VERSION,
                                "mkt": mkt,
                            },
                            separators=(",", ":"),
                            default=str,
                        )
                        + "\n"
                    )
                self._last_marks_at = ts
                wrote_marks = True

        return wrote_books, wrote_marks

    def maybe_fetch_candles(
        self,
        *,
        ts: float,
        info: Any,
        data_dir: Path,
        intervals: tuple[str, ...] | list[str],
        bars: int,
        per_tick: int,
        cooldown_s: float,
    ) -> int:
        """
        Step 4: after crowd samples. Round-robin 1 coin × intervals.
        Returns number of candle API calls made.
        """
        if not self.enabled or not self.seen_coins:
            return 0
        if ts - self._last_candle_at < float(cooldown_s):
            return 0
        coins = sorted(self.seen_coins)
        if not coins:
            return 0
        n = max(1, int(per_tick))
        calls = 0
        day = _utc_day(ts)
        for _ in range(n):
            coin = coins[self._candle_cursor % len(coins)]
            self._candle_cursor = (self._candle_cursor + 1) % len(coins)
            for iv in intervals:
                key = f"{coin}|{iv}"
                last_t = int(self._candle_last_t.get(key) or 0)
                need = int(bars) if last_t <= 0 else min(int(bars), 60)
                try:
                    rows = fetch_closed_candles(
                        info,
                        coin,
                        str(iv),
                        need,
                        data_dir=data_dir,
                        logger=self.log,
                    )
                except Exception as exc:
                    self.log.debug("Candle %s %s failed: %s", coin, iv, exc)
                    continue
                calls += 1
                new_rows = [c for c in rows if int(c.get("t") or 0) > last_t]
                if not new_rows:
                    continue
                folder = self.base / day / "candles" / _coin_dir_name(coin)
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{iv}.jsonl"
                with path.open("a", encoding="utf-8") as fh:
                    for c in new_rows:
                        fh.write(
                            json.dumps(
                                {
                                    "ts": ts,
                                    "coin": coin,
                                    "iv": iv,
                                    "bar": compact_candle(c),
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                self._candle_last_t[key] = int(new_rows[-1]["t"])
        self._last_candle_at = ts
        return calls
