"""Local SQLite replica of the collector schema. Trading-only extras live here too."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .timeutil import to_iso, to_unix

SCHEMA = """
CREATE TABLE IF NOT EXISTS collector_runs (
    cycle_ts TEXT NOT NULL,
    venue TEXT NOT NULL DEFAULT 'hyperliquid',
    started_at TEXT,
    finished_at TEXT,
    status TEXT NOT NULL,
    listed INTEGER NOT NULL DEFAULT 0,
    snapped_ok INTEGER NOT NULL DEFAULT 0,
    snapped_err INTEGER NOT NULL DEFAULT 0,
    empty_books INTEGER NOT NULL DEFAULT 0,
    coverage REAL,
    leaderboard_refreshed INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    duration_s REAL,
    cohort TEXT,
    PRIMARY KEY (cycle_ts, venue)
);

CREATE TABLE IF NOT EXISTS accounts (
    venue TEXT NOT NULL DEFAULT 'hyperliquid',
    address TEXT NOT NULL,
    display_name TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    PRIMARY KEY (venue, address)
);

CREATE TABLE IF NOT EXISTS cohort_members (
    cycle_ts TEXT NOT NULL,
    venue TEXT NOT NULL DEFAULT 'hyperliquid',
    cohort TEXT NOT NULL,
    rank INTEGER NOT NULL,
    address TEXT NOT NULL,
    score REAL,
    account_value REAL,
    pnl REAL,
    roi REAL,
    volume REAL,
    PRIMARY KEY (cycle_ts, venue, cohort, address)
);

CREATE TABLE IF NOT EXISTS wallet_books (
    cycle_ts TEXT NOT NULL,
    venue TEXT NOT NULL DEFAULT 'hyperliquid',
    address TEXT NOT NULL,
    account_value REAL NOT NULL,
    fingerprint TEXT,
    n_positions INTEGER NOT NULL DEFAULT 0,
    ok INTEGER NOT NULL,
    error TEXT,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (cycle_ts, venue, address)
);

CREATE TABLE IF NOT EXISTS wallet_positions (
    cycle_ts TEXT NOT NULL,
    venue TEXT NOT NULL DEFAULT 'hyperliquid',
    address TEXT NOT NULL,
    coin TEXT NOT NULL,
    side TEXT NOT NULL,
    size REAL NOT NULL,
    notional REAL NOT NULL,
    entry_px REAL,
    leverage INTEGER NOT NULL,
    isolated INTEGER NOT NULL DEFAULT 0,
    conviction REAL NOT NULL,
    PRIMARY KEY (cycle_ts, venue, address, coin)
);

CREATE TABLE IF NOT EXISTS meta_index (
    cycle_ts TEXT NOT NULL,
    venue TEXT NOT NULL DEFAULT 'hyperliquid',
    coin TEXT NOT NULL,
    side TEXT NOT NULL,
    wallets INTEGER NOT NULL,
    hold_pct REAL NOT NULL,
    agreement REAL NOT NULL,
    long_n INTEGER NOT NULL,
    short_n INTEGER NOT NULL,
    median_leverage INTEGER,
    mean_leverage REAL,
    avg_conviction REAL,
    notional_usd REAL,
    rank INTEGER NOT NULL,
    PRIMARY KEY (cycle_ts, venue, coin)
);

CREATE INDEX IF NOT EXISTS meta_index_coin_idx
    ON meta_index (venue, coin, cycle_ts DESC);

CREATE TABLE IF NOT EXISTS coin_prices (
    cycle_ts TEXT NOT NULL,
    venue TEXT NOT NULL DEFAULT 'hyperliquid',
    coin TEXT NOT NULL,
    mark_px REAL,
    mid_px REAL,
    oracle_px REAL,
    funding REAL,
    open_interest REAL,
    prev_day_px REAL,
    day_ntl_vlm REAL,
    premium REAL,
    ohlc_open REAL,
    ohlc_high REAL,
    ohlc_low REAL,
    ohlc_close REAL,
    ohlc_volume REAL,
    ohlc_trades INTEGER,
    ohlc_start_ts TEXT,
    ohlc_closed INTEGER NOT NULL DEFAULT 0,
    delisted INTEGER NOT NULL DEFAULT 0,
    source TEXT,
    error TEXT,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (cycle_ts, venue, coin)
);

CREATE INDEX IF NOT EXISTS coin_prices_coin_idx
    ON coin_prices (venue, coin, cycle_ts DESC);

CREATE TABLE IF NOT EXISTS price_candles (
    coin TEXT NOT NULL,
    interval TEXT NOT NULL,
    open_ts INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL,
    PRIMARY KEY (coin, interval, open_ts)
);

CREATE TABLE IF NOT EXISTS _sync (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

CYCLE_TABLES = (
    "collector_runs",
    "cohort_members",
    "wallet_books",
    "wallet_positions",
    "meta_index",
    "coin_prices",
)

UPSERT_SQL = {
    "collector_runs": """
        INSERT OR REPLACE INTO collector_runs (
            cycle_ts, venue, started_at, finished_at, status, listed,
            snapped_ok, snapped_err, empty_books, coverage,
            leaderboard_refreshed, error, duration_s, cohort
        ) VALUES (
            :cycle_ts, :venue, :started_at, :finished_at, :status, :listed,
            :snapped_ok, :snapped_err, :empty_books, :coverage,
            :leaderboard_refreshed, :error, :duration_s, :cohort
        )
    """,
    "accounts": """
        INSERT INTO accounts (venue, address, display_name, first_seen_at, last_seen_at)
        VALUES (:venue, :address, :display_name, :first_seen_at, :last_seen_at)
        ON CONFLICT (venue, address) DO UPDATE SET
            display_name = COALESCE(excluded.display_name, accounts.display_name),
            last_seen_at = excluded.last_seen_at
    """,
    "cohort_members": """
        INSERT OR REPLACE INTO cohort_members (
            cycle_ts, venue, cohort, rank, address, score,
            account_value, pnl, roi, volume
        ) VALUES (
            :cycle_ts, :venue, :cohort, :rank, :address, :score,
            :account_value, :pnl, :roi, :volume
        )
    """,
    "wallet_books": """
        INSERT OR REPLACE INTO wallet_books (
            cycle_ts, venue, address, account_value, fingerprint,
            n_positions, ok, error, fetched_at
        ) VALUES (
            :cycle_ts, :venue, :address, :account_value, :fingerprint,
            :n_positions, :ok, :error, :fetched_at
        )
    """,
    "wallet_positions": """
        INSERT OR REPLACE INTO wallet_positions (
            cycle_ts, venue, address, coin, side, size, notional,
            entry_px, leverage, isolated, conviction
        ) VALUES (
            :cycle_ts, :venue, :address, :coin, :side, :size, :notional,
            :entry_px, :leverage, :isolated, :conviction
        )
    """,
    "meta_index": """
        INSERT OR REPLACE INTO meta_index (
            cycle_ts, venue, coin, side, wallets, hold_pct, agreement,
            long_n, short_n, median_leverage, mean_leverage,
            avg_conviction, notional_usd, rank
        ) VALUES (
            :cycle_ts, :venue, :coin, :side, :wallets, :hold_pct, :agreement,
            :long_n, :short_n, :median_leverage, :mean_leverage,
            :avg_conviction, :notional_usd, :rank
        )
    """,
    "coin_prices": """
        INSERT OR REPLACE INTO coin_prices (
            cycle_ts, venue, coin,
            mark_px, mid_px, oracle_px, funding, open_interest,
            prev_day_px, day_ntl_vlm, premium,
            ohlc_open, ohlc_high, ohlc_low, ohlc_close, ohlc_volume, ohlc_trades,
            ohlc_start_ts, ohlc_closed, delisted, source, error, fetched_at
        ) VALUES (
            :cycle_ts, :venue, :coin,
            :mark_px, :mid_px, :oracle_px, :funding, :open_interest,
            :prev_day_px, :day_ntl_vlm, :premium,
            :ohlc_open, :ohlc_high, :ohlc_low, :ohlc_close, :ohlc_volume, :ohlc_trades,
            :ohlc_start_ts, :ohlc_closed, :delisted, :source, :error, :fetched_at
        )
    """,
}


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def get_sync(conn: sqlite3.Connection, name: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM _sync WHERE name = ?", (name,)).fetchone()
    if row is None:
        return default
    return str(row["value"] or default)


def set_sync(conn: sqlite3.Connection, name: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO _sync (name, value) VALUES (?, ?)",
        (name, value),
    )


def watermark_iso(conn: sqlite3.Connection) -> str:
    return get_sync(conn, "cycle_watermark", "")


def local_max_cycle(conn: sqlite3.Connection, venue: str) -> str:
    row = conn.execute(
        "SELECT MAX(cycle_ts) AS m FROM collector_runs WHERE venue = ?",
        (venue,),
    ).fetchone()
    return str(row["m"] or "") if row else ""


def _cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return to_iso(value)
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        from decimal import Decimal

        if isinstance(value, Decimal):
            return float(value)
    except Exception:
        pass
    return value


def normalize_row(row: dict[str, Any], *, venue: str) -> dict[str, Any]:
    out = {k: _cell(v) for k, v in row.items()}
    out.setdefault("venue", venue)
    if "cycle_ts" in out and out["cycle_ts"] is not None:
        out["cycle_ts"] = to_iso(out["cycle_ts"])
    for key in (
        "started_at",
        "finished_at",
        "fetched_at",
        "first_seen_at",
        "last_seen_at",
        "ohlc_start_ts",
    ):
        if key in out and out[key] is not None:
            out[key] = to_iso(out[key])
    if "address" in out and out["address"]:
        out["address"] = str(out["address"]).lower()
    for flag in ("ok", "isolated", "leaderboard_refreshed", "ohlc_closed", "delisted"):
        if flag in out:
            out[flag] = 1 if out[flag] else 0
    return out


def upsert_rows(
    conn: sqlite3.Connection,
    table: str,
    rows: Iterable[dict[str, Any]],
    *,
    venue: str,
) -> int:
    sql = UPSERT_SQL[table]
    n = 0
    for raw in rows:
        row = normalize_row(dict(raw), venue=venue)
        if table == "coin_prices":
            row.setdefault("fetched_at", now_iso())
            for key in (
                "mark_px",
                "mid_px",
                "oracle_px",
                "funding",
                "open_interest",
                "prev_day_px",
                "day_ntl_vlm",
                "premium",
                "ohlc_open",
                "ohlc_high",
                "ohlc_low",
                "ohlc_close",
                "ohlc_volume",
                "ohlc_trades",
                "ohlc_start_ts",
                "source",
                "error",
            ):
                row.setdefault(key, None)
            row.setdefault("ohlc_closed", 0)
            row.setdefault("delisted", 0)
        conn.execute(sql, row)
        n += 1
    return n


def upsert_candles(
    conn: sqlite3.Connection,
    rows: Iterable[dict[str, Any]],
) -> int:
    sql = """
        INSERT OR REPLACE INTO price_candles (
            coin, interval, open_ts, open, high, low, close, volume
        ) VALUES (:coin, :interval, :open_ts, :open, :high, :low, :close, :volume)
    """
    n = 0
    for row in rows:
        conn.execute(sql, row)
        n += 1
    return n


def candle_span(
    conn: sqlite3.Connection, coin: str, interval: str
) -> tuple[int, int] | None:
    row = conn.execute(
        """
        SELECT MIN(open_ts) AS a, MAX(open_ts) AS b
        FROM price_candles
        WHERE coin = ? AND interval = ?
        """,
        (coin, interval),
    ).fetchone()
    if row is None or row["a"] is None:
        return None
    return int(row["a"]), int(row["b"])


def load_meta_rows(
    conn: sqlite3.Connection,
    *,
    venue: str,
    status_ok: tuple[str, ...] = ("ok", "partial"),
) -> list[dict[str, Any]]:
    placeholders = ",".join("?" * len(status_ok))
    cur = conn.execute(
        f"""
        SELECT m.*
        FROM meta_index m
        JOIN collector_runs r
          ON r.cycle_ts = m.cycle_ts AND r.venue = m.venue
        WHERE m.venue = ?
          AND r.status IN ({placeholders})
        ORDER BY m.cycle_ts ASC, m.rank ASC
        """,
        (venue, *status_ok),
    )
    return [dict(row) for row in cur.fetchall()]


def latest_run(
    conn: sqlite3.Connection,
    *,
    venue: str,
    status_ok: tuple[str, ...] = ("ok", "partial"),
) -> dict[str, Any] | None:
    placeholders = ",".join("?" * len(status_ok))
    row = conn.execute(
        f"""
        SELECT * FROM collector_runs
        WHERE venue = ? AND status IN ({placeholders})
        ORDER BY cycle_ts DESC
        LIMIT 1
        """,
        (venue, *status_ok),
    ).fetchone()
    return dict(row) if row else None


def meta_for_cycle(
    conn: sqlite3.Connection, *, venue: str, cycle_ts: str
) -> list[dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT * FROM meta_index
        WHERE venue = ? AND cycle_ts = ?
        ORDER BY rank ASC
        """,
        (venue, cycle_ts),
    )
    return [dict(row) for row in cur.fetchall()]


def load_coin_prices(
    conn: sqlite3.Connection,
    *,
    venue: str,
) -> list[dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT * FROM coin_prices
        WHERE venue = ?
        ORDER BY cycle_ts ASC
        """,
        (venue,),
    )
    return [dict(row) for row in cur.fetchall()]


def max_price_fetched(conn: sqlite3.Connection, venue: str) -> str:
    row = conn.execute(
        "SELECT MAX(fetched_at) AS m FROM coin_prices WHERE venue = ?",
        (venue,),
    ).fetchone()
    return str(row["m"] or "") if row else ""


def count_table(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    return int(row["n"] if row else 0)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cycle_unix(cycle_ts: str) -> int:
    return to_unix(cycle_ts)
