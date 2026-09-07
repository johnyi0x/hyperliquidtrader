"""One-line EMA-race journal. JSONL has the full record for later filters."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_files import append_jsonl


def compact_board(rows: list[dict], *, n: int = 8) -> str:
    parts = []
    for r in rows[:n]:
        d = float(r.get("d") or 0)
        parts.append(
            "%s %+0.2f%% z=%+.1f %s/%s sc=%.2f w=%.2f %s"
            % (
                r.get("coin"),
                d,
                float(r.get("z") or 0),
                r.get("pol"),
                str(r.get("side") or "").upper(),
                float(r.get("sc") or 0),
                float(r.get("w") or 0),
                r.get("bkt") or "-",
            )
        )
    return " | ".join(parts)


def compact_learn(bins: list[dict], *, trades: int, best: str) -> str:
    bits = []
    for b in bins[:8]:
        bits.append(
            "%s %s/%s wr=%.0f%% pnl=%+.0f w=%.2f"
            % (
                b.get("ctx"),
                b.get("wins"),
                b.get("n"),
                float(b.get("wr") or 0),
                float(b.get("pnl") or 0),
                float(b.get("w") or 0),
            )
        )
    body = " | ".join(bits) if bits else "no closed trades yet"
    return "RACE_LEARN n=%s best=%s | %s" % (trades, best, body)


def compact_line(event: str, row: dict[str, Any]) -> str:
    if event == "entry":
        return (
            "RACE_ENTRY %(coin)s %(side)s pol=%(policy)s ctx=%(ctx)s "
            "fill=%(fill)s ema=%(ema)s D=%(d_signed)+.2f%% z=%(z)+.2f rel=%(rel).2f "
            "rank=%(rank)s/%(n_watch)s bkt=%(bucket)s xbars=%(xbars)s "
            "tp=sl=%(tp_sl).2f%% lev=%(lev)sx ntl=$%(notional).1f eq=$%(equity).1f "
            "risk=%(risk_pct).1f%% w=%(weight).2f cm=%(coin_mult).2f "
            "board=%(board)s"
            % {
                "coin": row.get("coin"),
                "side": str(row.get("side") or "").upper(),
                "policy": row.get("policy") or "-",
                "ctx": row.get("ctx") or "-",
                "fill": row.get("fill"),
                "ema": row.get("ema"),
                "d_signed": float(row.get("signed_pct") or 0),
                "z": float(row.get("z") or 0),
                "rel": float(row.get("rel") or 0),
                "rank": int(row.get("rank") or 0),
                "n_watch": int(row.get("n_watch") or 0),
                "bucket": row.get("bucket") or "-",
                "xbars": int(row.get("xbars") or 0),
                "tp_sl": float(row.get("tp_sl_pct") or 0),
                "lev": int(row.get("lev") or 0),
                "notional": float(row.get("notional") or 0),
                "equity": float(row.get("equity") or 0),
                "risk_pct": float(row.get("risk_pct") or 0),
                "weight": float(row.get("weight") or 0),
                "coin_mult": float(row.get("coin_mult") or 0),
                "board": row.get("board_txt") or "-",
            }
        )
    if event == "exit":
        return (
            "RACE_EXIT %(coin)s %(side)s %(kind)s pol=%(policy)s ctx=%(ctx)s "
            "fill=%(fill)s->%(exit_px)s pnl=%(pnl_pct)+.2f%% $%(pnl_usd)+.3f "
            "hold=%(hold_s).0fs mfe=%(mfe_pct).2f mae=%(mae_pct).2f "
            "eq=$%(equity).1f | %(reason)s"
            % {
                "coin": row.get("coin"),
                "side": str(row.get("side") or "").upper(),
                "kind": row.get("reason_kind") or "other",
                "policy": row.get("policy") or "-",
                "ctx": row.get("ctx") or "-",
                "fill": row.get("fill"),
                "exit_px": row.get("exit_px"),
                "pnl_pct": float(row.get("pnl_pct") or 0),
                "pnl_usd": float(row.get("pnl_usd") or 0),
                "hold_s": float(row.get("hold_s") or 0),
                "mfe_pct": float(row.get("mfe_pct") or 0),
                "mae_pct": float(row.get("mae_pct") or 0),
                "equity": float(row.get("equity") or 0),
                "reason": row.get("reason") or "",
            }
        )
    if event == "learn":
        return str(row.get("line") or "RACE_LEARN")
    if event == "scan":
        return "RACE_SCAN n=%s leader=%s | %s" % (
            row.get("n_watch"),
            row.get("leader") or "-",
            row.get("board_txt") or "-",
        )
    return "RACE %s %s" % (event, row.get("coin") or "")


class RaceJournal:
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
            self.logger.warning("RACE jsonl write failed: %s", exc)
        try:
            self.logger.info(compact_line(event, payload))
        except (KeyError, TypeError, ValueError):
            self.logger.info("RACE %s %s", event, payload.get("coin"))
