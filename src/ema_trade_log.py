"""Compact EMA-dev trade journal for later filter work.

One JSONL row per entry or exit (data/ema_trades.jsonl) plus one log line.
Scan dumps stay out of the journal.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_files import append_jsonl


def tape_from_candles(candles: list[dict]) -> dict[str, float]:
    """Closed-bar tape stats used to filter entries later. Empty if too short."""
    if len(candles) < 8:
        return {}
    closes = [float(c["c"]) for c in candles]
    highs = [float(c["h"]) for c in candles]
    lows = [float(c["l"]) for c in candles]
    close = closes[-1]
    if close <= 0:
        return {}
    n = min(14, len(closes) - 1)
    trs: list[float] = []
    for i in range(-n, 0):
        prev = closes[i - 1]
        trs.append(max(highs[i] - lows[i], abs(highs[i] - prev), abs(lows[i] - prev)))
    atr = sum(trs) / max(1, len(trs))
    atr_pct = atr / close * 100.0
    last_bps = 0.0
    if closes[-2] > 0:
        last_bps = (closes[-1] - closes[-2]) / closes[-2] * 10_000.0
    ret5 = 0.0
    if len(closes) >= 6 and closes[-6] > 0:
        ret5 = (closes[-1] - closes[-6]) / closes[-6] * 10_000.0
    look = min(45, len(highs))
    rng = (max(highs[-look:]) - min(lows[-look:])) / close * 10_000.0
    er_n = min(20, len(closes) - 1)
    path = 0.0
    up = 0
    dn = 0
    for i in range(-er_n, 0):
        d = closes[i] - closes[i - 1]
        path += abs(d)
        if d > 0:
            up += 1
        elif d < 0:
            dn += 1
    net = abs(closes[-1] - closes[-1 - er_n])
    er = 1.0 if path <= 1e-12 else min(1.0, net / path)
    sides = up + dn
    return {
        "atr_pct": round(atr_pct, 4),
        "last_bar_bps": round(last_bps, 2),
        "ret_5_bps": round(ret5, 2),
        "range_45_bps": round(rng, 1),
        "er_20": round(er, 3),
        "up_frac": round((up / sides) if sides else 0.5, 3),
    }


def scan_top(snaps: list, *, n: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(snaps, key=lambda s: (-float(s.abs_dev_pct), s.coin))
    out: list[dict[str, Any]] = []
    for s in ranked[: max(1, n)]:
        out.append(
            {
                "coin": s.coin,
                "d_pct": round(float(s.abs_dev_pct), 4),
                "signed_pct": round(float(s.signed_dev_pct), 4),
                "xbars": int(s.cross_bars),
                "side": "below" if int(s.signal_side) > 0 else "above",
            }
        )
    return out


def reason_kind(reason: str) -> str:
    r = str(reason or "").lower()
    if r.startswith("tp_") or "take_profit" in r or r.startswith("exit_ema"):
        return "tp"
    if "stop_loss" in r or r.startswith("sl_") or "stop loss" in r:
        return "sl"
    if "max_hold" in r:
        return "max_hold"
    if "one_pair" in r or "outside" in r:
        return "forced"
    if "protect" in r or "tpsl" in r:
        return "protect"
    return "other"


def compact_line(event: str, row: dict[str, Any]) -> str:
    """One line you can paste; the JSONL file has the full record."""
    if event == "entry":
        return (
            "EMA_TRADE entry %(coin)s %(side)s fill=%(fill)s D=%(d_pct).2f%% "
            "ema=%(ema)s xbars=%(xbars)s atr=%(atr_pct).3f%% er=%(er_20).2f "
            "last=%(last_bar_bps)+.0fb rng=%(range_45_bps).0fb spr=%(spread_bps).1fb "
            "tp=%(tp_pct).2f sl=%(sl_pct).2f ntl=$%(notional).1f eq=$%(equity).1f "
            "lev=%(lev)sx bucket=%(bucket)s"
            % {
                "coin": row.get("coin"),
                "side": str(row.get("side") or "").upper(),
                "fill": row.get("fill"),
                "d_pct": float(row.get("d_pct") or 0),
                "ema": row.get("ema"),
                "xbars": int(row.get("xbars") or 0),
                "atr_pct": float(row.get("atr_pct") or 0),
                "er_20": float(row.get("er_20") or 0),
                "last_bar_bps": float(row.get("last_bar_bps") or 0),
                "range_45_bps": float(row.get("range_45_bps") or 0),
                "spread_bps": float(row.get("spread_bps") or 0),
                "tp_pct": float(row.get("tp_pct") or 0),
                "sl_pct": float(row.get("sl_pct") or 0),
                "notional": float(row.get("notional") or 0),
                "equity": float(row.get("equity") or 0),
                "lev": int(row.get("lev") or 0),
                "bucket": row.get("bucket") or "-",
            }
        )
    return (
        "EMA_TRADE exit %(coin)s %(side)s %(kind)s fill=%(fill)s->%(exit_px)s "
        "pnl=%(pnl_pct)+.2f%% $%(pnl_usd)+.3f hold=%(hold_s).0fs "
        "mfe=%(mfe_pct).2f mae=%(mae_pct).2f D=%(d_pct).2f%% "
        "exit_ema=%(exit_ema)s eq=$%(equity).1f | %(reason)s"
        % {
            "coin": row.get("coin"),
            "side": str(row.get("side") or "").upper(),
            "kind": row.get("reason_kind") or "other",
            "fill": row.get("fill"),
            "exit_px": row.get("exit_px"),
            "pnl_pct": float(row.get("pnl_pct") or 0),
            "pnl_usd": float(row.get("pnl_usd") or 0),
            "hold_s": float(row.get("hold_s") or 0),
            "mfe_pct": float(row.get("mfe_pct") or 0),
            "mae_pct": float(row.get("mae_pct") or 0),
            "d_pct": float(row.get("d_pct") or 0),
            "exit_ema": row.get("exit_ema"),
            "equity": float(row.get("equity") or 0),
            "reason": row.get("reason") or "",
        }
    )


class EmaTradeJournal:
    def __init__(self, path: Path, logger: logging.Logger) -> None:
        self.path = Path(path)
        self.logger = logger

    def record(self, event: str, row: dict[str, Any]) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **row,
        }
        try:
            append_jsonl(self.path, payload, logger=self.logger)
        except OSError as exc:
            self.logger.warning("EMA_TRADE jsonl write failed: %s", exc)
        try:
            self.logger.info(compact_line(event, payload))
        except (KeyError, TypeError, ValueError):
            self.logger.info("EMA_TRADE %s %s", event, payload.get("coin"))
