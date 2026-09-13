"""Read the latest collector meta_index hour for live trading."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .dsn import database_url, default_sqlite_path, open_neon
from .store import connect, latest_run, meta_for_cycle
from .timeutil import to_iso, to_unix

log = logging.getLogger("bagrank.board")


@dataclass
class CollectorBoard:
    cycle_ts: str
    cycle_unix: int
    status: str
    listed: int
    snapped_ok: int
    coverage: float | None
    rows: list[dict[str, Any]]


def _board_from_sqlite(path, venue: str) -> CollectorBoard | None:
    conn = connect(path)
    try:
        run = latest_run(conn, venue=venue)
        if not run:
            return None
        cycle = str(run["cycle_ts"])
        rows = meta_for_cycle(conn, venue=venue, cycle_ts=cycle)
        cov_raw = run.get("coverage")
        return CollectorBoard(
            cycle_ts=cycle,
            cycle_unix=to_unix(cycle),
            status=str(run.get("status") or ""),
            listed=int(run.get("listed") or 0),
            snapped_ok=int(run.get("snapped_ok") or 0),
            coverage=None if cov_raw is None else float(cov_raw),
            rows=rows,
        )
    finally:
        conn.close()


def _board_from_neon(dsn: str, venue: str) -> CollectorBoard | None:
    try:
        conn = open_neon(dsn)
    except ImportError:
        return None
    try:
        try:
            conn.execute("SET default_transaction_read_only = on")
        except Exception:
            pass
        run = conn.execute(
            """
            SELECT cycle_ts, status, listed, snapped_ok, coverage
            FROM collector_runs
            WHERE venue = %s AND status = ANY(%s)
            ORDER BY cycle_ts DESC
            LIMIT 1
            """,
            (venue, ["ok", "partial"]),
        ).fetchone()
        if not run:
            return None
        cycle = run["cycle_ts"]
        rows = conn.execute(
            """
            SELECT coin, side, wallets, hold_pct, agreement, long_n, short_n,
                   median_leverage, mean_leverage, avg_conviction, notional_usd, rank
            FROM meta_index
            WHERE venue = %s AND cycle_ts = %s
            ORDER BY rank ASC
            """,
            (venue, cycle),
        ).fetchall()
        return CollectorBoard(
            cycle_ts=to_iso(cycle),
            cycle_unix=to_unix(cycle),
            status=str(run.get("status") or ""),
            listed=int(run.get("listed") or 0),
            snapped_ok=int(run.get("snapped_ok") or 0),
            coverage=None if run.get("coverage") is None else float(run.get("coverage")),
            rows=[dict(r) for r in rows],
        )
    finally:
        conn.close()


def fetch_latest_board(
    *,
    venue: str = "hyperliquid",
    dsn: str = "",
    sqlite_path=None,
) -> CollectorBoard | None:
    url = database_url(dsn)
    if url:
        try:
            board = _board_from_neon(url, venue)
            if board is not None:
                return board
        except Exception as exc:
            log.warning("Neon meta_index read failed: %s", exc)
    path = sqlite_path or default_sqlite_path()
    if path.exists():
        try:
            return _board_from_sqlite(path, venue)
        except Exception as exc:
            log.warning("Local bagrank sqlite read failed: %s", exc)
    return None
