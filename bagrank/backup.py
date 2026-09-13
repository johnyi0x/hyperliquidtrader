"""Read-only incremental copy of collector Neon tables onto this PC."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .dsn import default_sqlite_path, load_env, open_neon, redacted_dsn, resolve_database_url
from .store import (
    CYCLE_TABLES,
    connect,
    count_table,
    get_sync,
    local_max_cycle,
    max_price_fetched,
    set_sync,
    upsert_rows,
    watermark_iso,
)
from .timeutil import as_utc, to_iso

log = logging.getLogger("bagrank.backup")

VENUE_DEFAULT = "hyperliquid"
CHUNK_HOURS = 12
OVERLAP = timedelta(hours=1)
OK_STATUS = ("ok", "partial")

CYCLE_SELECT = {
    "collector_runs": """
        SELECT cycle_ts, venue, started_at, finished_at, status, listed,
               snapped_ok, snapped_err, empty_books, coverage,
               leaderboard_refreshed, error, duration_s, cohort
        FROM collector_runs
        WHERE venue = %s AND cycle_ts = ANY(%s)
    """,
    "cohort_members": """
        SELECT cycle_ts, venue, cohort, rank, address, score,
               account_value, pnl, roi, volume
        FROM cohort_members
        WHERE venue = %s AND cycle_ts = ANY(%s)
    """,
    "wallet_books": """
        SELECT cycle_ts, venue, address, account_value, fingerprint,
               n_positions, ok, error, fetched_at
        FROM wallet_books
        WHERE venue = %s AND cycle_ts = ANY(%s)
    """,
    "wallet_positions": """
        SELECT cycle_ts, venue, address, coin, side, size, notional,
               entry_px, leverage, isolated, conviction
        FROM wallet_positions
        WHERE venue = %s AND cycle_ts = ANY(%s)
    """,
    "meta_index": """
        SELECT cycle_ts, venue, coin, side, wallets, hold_pct, agreement,
               long_n, short_n, median_leverage, mean_leverage,
               avg_conviction, notional_usd, rank
        FROM meta_index
        WHERE venue = %s AND cycle_ts = ANY(%s)
    """,
    "coin_prices": """
        SELECT cycle_ts, venue, coin,
               mark_px, mid_px, oracle_px, funding, open_interest,
               prev_day_px, day_ntl_vlm, premium,
               ohlc_open, ohlc_high, ohlc_low, ohlc_close, ohlc_volume, ohlc_trades,
               ohlc_start_ts, ohlc_closed, delisted, source, error, fetched_at
        FROM coin_prices
        WHERE venue = %s AND cycle_ts = ANY(%s)
    """,
}


class Source:
    """Anything we can SELECT collector tables from. Never writes."""

    def list_cycles(
        self, venue: str, after: datetime | None
    ) -> list[datetime]:
        raise NotImplementedError

    def fetch_table(
        self, table: str, venue: str, cycles: list[datetime]
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def fetch_accounts(
        self, venue: str, after: datetime | None
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def fetch_prices_since(
        self, venue: str, after: datetime | None
    ) -> list[dict[str, Any]]:
        """Rows whose fetched_at is newer than after (collector price backfill)."""
        return []

    def close(self) -> None:
        return None


class NeonSource(Source):
    def __init__(self, dsn: str) -> None:
        last: Exception | None = None
        self.conn = None
        for attempt in range(3):
            try:
                self.conn = open_neon(dsn)
                break
            except ImportError as exc:
                raise RuntimeError(
                    "psycopg is required for Neon backup. pip install 'psycopg[binary]'"
                ) from exc
            except Exception as cop_exc:
                last = cop_exc
                delay = min(20.0, 2.0 ** attempt)
                log.warning(
                    "Neon connect failed (%s/3): %s — retry in %.0fs",
                    attempt + 1,
                    cop_exc,
                    delay,
                )
                time.sleep(delay)
        if self.conn is None:
            raise RuntimeError(
                "Could not connect to Neon. Set NEON_BAGRANK in this repo's .env "
                "to the collector DB URI (direct host, sslmode=require). "
                "Wake the project in the Neon console if it is idle. Last error: %s"
                % last
            ) from last
        try:
            self.conn.execute("SET default_transaction_read_only = on")
        except Exception:
            pass

    def list_cycles(self, venue: str, after: datetime | None) -> list[datetime]:
        if after is None:
            rows = self.conn.execute(
                """
                SELECT cycle_ts FROM collector_runs
                WHERE venue = %s AND status = ANY(%s)
                ORDER BY cycle_ts ASC
                """,
                (venue, list(OK_STATUS)),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT cycle_ts FROM collector_runs
                WHERE venue = %s AND status = ANY(%s) AND cycle_ts > %s
                ORDER BY cycle_ts ASC
                """,
                (venue, list(OK_STATUS), after),
            ).fetchall()
        out: list[datetime] = []
        for row in rows:
            out.append(as_utc(row["cycle_ts"]))
        return out

    def list_recent_cycles(self, venue: str, limit: int) -> list[datetime]:
        rows = self.conn.execute(
            """
            SELECT cycle_ts FROM collector_runs
            WHERE venue = %s AND status = ANY(%s)
            ORDER BY cycle_ts DESC
            LIMIT %s
            """,
            (venue, list(OK_STATUS), max(1, int(limit))),
        ).fetchall()
        out = [as_utc(row["cycle_ts"]) for row in rows]
        out.reverse()
        return out

    def fetch_table(
        self, table: str, venue: str, cycles: list[datetime]
    ) -> list[dict[str, Any]]:
        if not cycles:
            return []
        sql = CYCLE_SELECT[table]
        try:
            rows = self.conn.execute(sql, (venue, cycles)).fetchall()
        except Exception as exc:
            if table == "coin_prices" and "coin_prices" in str(exc).lower():
                log.warning("coin_prices missing on source — skip this table")
                return []
            raise
        return [dict(r) for r in rows]

    def fetch_accounts(
        self, venue: str, after: datetime | None
    ) -> list[dict[str, Any]]:
        if after is None:
            rows = self.conn.execute(
                """
                SELECT venue, address, display_name, first_seen_at, last_seen_at
                FROM accounts WHERE venue = %s
                """,
                (venue,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT venue, address, display_name, first_seen_at, last_seen_at
                FROM accounts WHERE venue = %s AND last_seen_at > %s
                """,
                (venue, after),
            ).fetchall()
        return [dict(r) for r in rows]

    def fetch_prices_since(
        self, venue: str, after: datetime | None
    ) -> list[dict[str, Any]]:
        if after is None:
            return []
        try:
            rows = self.conn.execute(
                """
                SELECT cycle_ts, venue, coin,
                       mark_px, mid_px, oracle_px, funding, open_interest,
                       prev_day_px, day_ntl_vlm, premium,
                       ohlc_open, ohlc_high, ohlc_low, ohlc_close, ohlc_volume,
                       ohlc_trades, ohlc_start_ts, ohlc_closed, delisted,
                       source, error, fetched_at
                FROM coin_prices
                WHERE venue = %s AND fetched_at > %s
                """,
                (venue, after),
            ).fetchall()
        except Exception as exc:
            if "coin_prices" in str(exc).lower():
                log.warning("coin_prices missing on source — skip price refresh")
                return []
            raise
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()


class SqliteSource(Source):
    """Used by tests; same SELECT shape as Neon."""

    def __init__(self, path: Path) -> None:
        from .store import connect as sqlite_connect

        self.conn = sqlite_connect(path)

    def list_cycles(self, venue: str, after: datetime | None) -> list[datetime]:
        if after is None:
            rows = self.conn.execute(
                """
                SELECT cycle_ts FROM collector_runs
                WHERE venue = ? AND status IN ('ok', 'partial')
                ORDER BY cycle_ts ASC
                """,
                (venue,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT cycle_ts FROM collector_runs
                WHERE venue = ? AND status IN ('ok', 'partial') AND cycle_ts > ?
                ORDER BY cycle_ts ASC
                """,
                (venue, to_iso(after)),
            ).fetchall()
        return [as_utc(r["cycle_ts"]) for r in rows]

    def fetch_table(
        self, table: str, venue: str, cycles: list[datetime]
    ) -> list[dict[str, Any]]:
        if not cycles:
            return []
        marks = ",".join("?" * len(cycles))
        iso = [to_iso(c) for c in cycles]
        cols = {
            "collector_runs": "*",
            "cohort_members": "*",
            "wallet_books": "*",
            "wallet_positions": "*",
            "meta_index": "*",
            "coin_prices": "*",
        }[table]
        rows = self.conn.execute(
            f"SELECT {cols} FROM {table} WHERE venue = ? AND cycle_ts IN ({marks})",
            (venue, *iso),
        ).fetchall()
        return [dict(r) for r in rows]

    def fetch_accounts(
        self, venue: str, after: datetime | None
    ) -> list[dict[str, Any]]:
        if after is None:
            rows = self.conn.execute(
                "SELECT * FROM accounts WHERE venue = ?", (venue,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM accounts WHERE venue = ? AND last_seen_at > ?",
                (venue, to_iso(after)),
            ).fetchall()
        return [dict(r) for r in rows]

    def fetch_prices_since(
        self, venue: str, after: datetime | None
    ) -> list[dict[str, Any]]:
        if after is None:
            return []
        rows = self.conn.execute(
            """
            SELECT * FROM coin_prices
            WHERE venue = ? AND fetched_at > ?
            """,
            (venue, to_iso(after)),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()


def _chunks(items: list[datetime], n: int) -> Iterable[list[datetime]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _refresh_late_prices(
    source: Source,
    conn: Any,
    *,
    venue: str,
    stats: dict[str, Any],
) -> None:
    """Pull coin_prices the collector backfilled on hours we already stored."""
    raw = get_sync(conn, "price_fetched_watermark", "") or max_price_fetched(conn, venue)
    after = as_utc(raw) - OVERLAP if raw else None
    extra = source.fetch_prices_since(venue, after)
    if not extra:
        return
    n = upsert_rows(conn, "coin_prices", extra, venue=venue)
    stats["rows"]["coin_prices"] = int(stats["rows"].get("coin_prices") or 0) + n
    stats["price_refresh"] = n
    newest = raw
    for row in extra:
        ts = row.get("fetched_at")
        if ts is None:
            continue
        iso = to_iso(ts)
        if not newest or iso > newest:
            newest = iso
    if newest:
        set_sync(conn, "price_fetched_watermark", newest)
    conn.commit()
    log.info("Price backfill refresh | rows=%s", n)


def sync(source: Source, sqlite_path: Path, *, venue: str = VENUE_DEFAULT) -> dict[str, Any]:
    """Copy new completed hours into local sqlite. First run = full history."""
    conn = connect(sqlite_path)
    try:
        mark_raw = watermark_iso(conn) or local_max_cycle(conn, venue)
        after: datetime | None = None
        overlap_after: datetime | None = None
        if mark_raw:
            after = as_utc(mark_raw)
            overlap_after = after - OVERLAP
        listed_after = overlap_after if overlap_after is not None else after
        cycles = source.list_cycles(venue, listed_after)
        # Drop hours we already have except the overlap hour (in case collector rewrote it).
        if after is not None:
            keep: list[datetime] = []
            for c in cycles:
                if c >= after:
                    keep.append(c)
                elif overlap_after is not None and c >= overlap_after:
                    keep.append(c)
            cycles = keep
        stats = {
            "sqlite": str(sqlite_path),
            "first_run": not bool(mark_raw),
            "watermark_before": mark_raw or "",
            "cycles": len(cycles),
            "rows": {t: 0 for t in CYCLE_TABLES},
            "accounts": 0,
            "price_refresh": 0,
        }
        if not cycles:
            if mark_raw:
                accounts = source.fetch_accounts(venue, after)
                if accounts:
                    stats["accounts"] = upsert_rows(conn, "accounts", accounts, venue=venue)
                _refresh_late_prices(source, conn, venue=venue, stats=stats)
                conn.commit()
                stats["watermark_after"] = mark_raw
                stats["local_counts"] = {t: count_table(conn, t) for t in CYCLE_TABLES}
                log.info(
                    "Backup hours up to date | last=%s | price_refresh=%s | sqlite=%s",
                    mark_raw,
                    stats["price_refresh"],
                    sqlite_path,
                )
                return stats
            log.info("No completed collector hours on the source yet")
            stats["watermark_after"] = ""
            return stats

        log.info(
            "Backup %s | hours=%s | from=%s | to=%s",
            "full" if stats["first_run"] else "incremental",
            len(cycles),
            to_iso(cycles[0]),
            to_iso(cycles[-1]),
        )
        for chunk in _chunks(cycles, CHUNK_HOURS):
            for table in CYCLE_TABLES:
                rows = source.fetch_table(table, venue, chunk)
                n = upsert_rows(conn, table, rows, venue=venue)
                stats["rows"][table] = int(stats["rows"][table]) + n
            accounts = source.fetch_accounts(venue, chunk[0] - OVERLAP)
            stats["accounts"] = int(stats["accounts"]) + upsert_rows(
                conn, "accounts", accounts, venue=venue
            )
            set_sync(conn, "cycle_watermark", to_iso(chunk[-1]))
            set_sync(conn, "updated_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            conn.commit()
            log.info(
                "Saved through %s | meta=%s books=%s positions=%s prices=%s",
                to_iso(chunk[-1]),
                stats["rows"]["meta_index"],
                stats["rows"]["wallet_books"],
                stats["rows"]["wallet_positions"],
                stats["rows"]["coin_prices"],
            )
        price_max = max_price_fetched(conn, venue)
        if price_max:
            set_sync(conn, "price_fetched_watermark", price_max)
        _refresh_late_prices(source, conn, venue=venue, stats=stats)
        stats["watermark_after"] = to_iso(cycles[-1])
        stats["local_max"] = local_max_cycle(conn, venue)
        stats["local_counts"] = {t: count_table(conn, t) for t in CYCLE_TABLES}
        log.info(
            "Backup done | last=%s | meta_index=%s | coin_prices=%s | wallet_positions=%s",
            stats["watermark_after"],
            stats["local_counts"]["meta_index"],
            stats["local_counts"]["coin_prices"],
            stats["local_counts"]["wallet_positions"],
        )
        return stats
    finally:
        conn.close()


def seed_recent_board(
    sqlite_path: Path,
    hours: int,
    *,
    venue: str = VENUE_DEFAULT,
) -> int:
    """One-shot copy of the last N collector hours. Does not stay connected."""
    url, _src = resolve_database_url()
    if not url:
        return 0
    from .hlcycle import prune_old_hours

    keep = max(1, int(hours))
    neon = NeonSource(url)
    try:
        cycles = neon.list_recent_cycles(venue, keep)
        if not cycles:
            return 0
        conn = connect(sqlite_path)
        try:
            for table in ("collector_runs", "meta_index", "coin_prices"):
                rows = neon.fetch_table(table, venue, cycles)
                if rows:
                    upsert_rows(conn, table, rows, venue=venue)
            prune_old_hours(conn, venue=venue, keep_hours=keep)
            conn.commit()
        finally:
            conn.close()
        return len(cycles)
    finally:
        neon.close()


def sync_neon(
    sqlite_path: Path,
    *,
    dsn: str = "",
    venue: str = VENUE_DEFAULT,
) -> dict[str, Any]:
    url, src = resolve_database_url(dsn)
    if not url:
        raise RuntimeError(
            "No database URL. Set NEON_BAGRANK in hl-multi-strategy-bot/.env "
            "(read-only Neon URI for the collector). Prefer the direct host "
            "(no -pooler) with sslmode=require."
        )
    log.info("Connecting to Neon read-only via %s → %s", src, redacted_dsn(url))
    neon = NeonSource(url)
    try:
        return sync(neon, sqlite_path, venue=venue)
    finally:
        neon.close()


def print_status(sqlite_path: Path, *, venue: str = VENUE_DEFAULT) -> None:
    conn = connect(sqlite_path)
    try:
        mark = watermark_iso(conn) or local_max_cycle(conn, venue)
        print(f"sqlite: {sqlite_path}")
        print(f"watermark: {mark or '(empty — next run downloads everything)'}")
        print(f"prices fetched watermark: {get_sync(conn, 'price_fetched_watermark', '') or max_price_fetched(conn, venue) or '(none)'}")
        for table in (*CYCLE_TABLES, "accounts"):
            print(f"  {table}: {count_table(conn, table)}")
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    load_env()
    p = argparse.ArgumentParser(
        description=(
            "Read-only incremental backup of collector Neon tables to this PC. "
            "First run stores everything (including coin_prices). Later runs "
            "pull new hours plus any price backfills on older hours."
        )
    )
    p.add_argument(
        "--db",
        type=Path,
        default=default_sqlite_path(),
        help="Local sqlite path (default data/bagrank/bagrank.sqlite)",
    )
    p.add_argument("--dsn", default="", help="Neon URI (otherwise env)")
    p.add_argument("--venue", default=VENUE_DEFAULT)
    p.add_argument("--status", action="store_true", help="Print local backup stats and exit")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.status:
        print_status(args.db, venue=args.venue)
        return 0
    sync_neon(args.db, dsn=args.dsn, venue=args.venue)
    print_status(args.db, venue=args.venue)
    return 0


if __name__ == "__main__":
    sys.exit(main())
