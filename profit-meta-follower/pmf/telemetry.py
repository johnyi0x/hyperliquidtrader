"""Daily-split JSONL telemetry for offline parameter backtests."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_day(ts: float | None = None) -> str:
    t = ts if ts is not None else time.time()
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")


class TelemetryWriter:
    """
    Append-only daily files under data/telemetry/YYYY-MM-DD/:
      ticks.jsonl   — one compact row per bot cycle (~5k/day)
      events.jsonl  — rebalances, basket refresh, drops
    """

    def __init__(self, data_dir: Path, *, enabled: bool = True, instance: str = "local") -> None:
        self.base = data_dir / "telemetry"
        self.enabled = bool(enabled)
        self.instance = instance
        self._day = ""
        self._ticks_path: Path | None = None
        self._events_path: Path | None = None

    def _paths(self, day: str) -> tuple[Path, Path]:
        folder = self.base / day
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "ticks.jsonl", folder / "events.jsonl"

    def _rotate(self, day: str) -> None:
        if day == self._day:
            return
        self._day = day
        self._ticks_path, self._events_path = self._paths(day)

    def _append(self, path: Path | None, row: dict[str, Any]) -> None:
        if not self.enabled or path is None:
            return
        line = json.dumps(row, separators=(",", ":"), default=str)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def record_tick(
        self,
        *,
        ts: float,
        voters: int,
        listed: int,
        raw: list[dict[str, Any]],
        trade: list[str],
        managed: list[str],
        equity: float | None = None,
    ) -> None:
        self._rotate(_utc_day(ts))
        row: dict[str, Any] = {
            "ts": ts,
            "inst": self.instance,
            "voters": voters,
            "listed": listed,
            "raw": raw,
            "trade": trade,
            "managed": managed,
        }
        if equity is not None and equity > 0:
            row["eq"] = round(equity, 2)
        self._append(self._ticks_path, row)

    def record_event(self, *, ts: float, kind: str, payload: dict[str, Any]) -> None:
        self._rotate(_utc_day(ts))
        self._append(
            self._events_path,
            {"ts": ts, "inst": self.instance, "kind": kind, **payload},
        )

    def save_basket_snapshot(self, *, ts: float, wallets: list[dict[str, Any]]) -> Path:
        day = _utc_day(ts)
        folder = self.base / day
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H%M%S")
        path = folder / f"basket-{stamp}.json"
        path.write_text(
            json.dumps({"ts": ts, "inst": self.instance, "wallets": wallets}, indent=2),
            encoding="utf-8",
        )
        return path


def compact_votes(votes: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for v in votes:
        out.append(
            {
                "c": v.coin,
                "s": v.side,
                "agr": round(float(v.agreement), 4),
                "conv": round(float(getattr(v, "raw_conviction", v.avg_conviction) or v.avg_conviction), 4),
                "wl": int(v.wallets_long),
                "ws": int(v.wallets_short),
            }
        )
    return out
