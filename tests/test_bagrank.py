"""Incremental bagrank backup + Numba rank strategies."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "profit-meta-follower"))

from bagrank.backup import SqliteSource, sync
from bagrank.kernels import SID_RANK_UP_DOWN, SID_TOP_K, build_targets, simulate
from bagrank.panel import panel_from_meta_rows, panel_from_sqlite
from bagrank.prices import fill_panel_from_collector, row_fill_px
from bagrank.store import connect, count_table, upsert_rows


def _run(cycle: str, *, listed: int = 10, status: str = "ok") -> dict:
    return {
        "cycle_ts": cycle,
        "venue": "hyperliquid",
        "started_at": cycle,
        "finished_at": cycle,
        "status": status,
        "listed": listed,
        "snapped_ok": listed,
        "snapped_err": 0,
        "empty_books": 0,
        "coverage": 1.0,
        "leaderboard_refreshed": 1,
        "error": None,
        "duration_s": 1.0,
        "cohort": "top200_week",
    }


def _meta(cycle: str, coin: str, rank: int, wallets: int, side: str = "long") -> dict:
    return {
        "cycle_ts": cycle,
        "venue": "hyperliquid",
        "coin": coin,
        "side": side,
        "wallets": wallets,
        "hold_pct": wallets / 100.0,
        "agreement": 0.8,
        "long_n": wallets if side == "long" else 0,
        "short_n": wallets if side == "short" else 0,
        "median_leverage": 5,
        "mean_leverage": 5.0,
        "avg_conviction": 0.2,
        "notional_usd": 1000.0,
        "rank": rank,
    }


def _price(
    cycle: str,
    coin: str,
    *,
    mark: float | None = 10.0,
    ohlc_open: float | None = 9.5,
    ohlc_close: float | None = None,
    closed: bool = False,
    fetched_at: str | None = None,
) -> dict:
    return {
        "cycle_ts": cycle,
        "venue": "hyperliquid",
        "coin": coin,
        "mark_px": mark,
        "ohlc_open": ohlc_open,
        "ohlc_high": (ohlc_close or mark or 10) + 0.5,
        "ohlc_low": (ohlc_open or 9.0),
        "ohlc_close": ohlc_close,
        "ohlc_volume": 100.0,
        "ohlc_trades": 10,
        "ohlc_start_ts": cycle,
        "ohlc_closed": 1 if closed else 0,
        "delisted": 0,
        "source": "ctx+candle" if mark is not None else "candle",
        "error": None,
        "fetched_at": fetched_at or cycle,
    }


def _seed(path: Path, cycles: list[str], *, with_prices: bool = True) -> None:
    conn = connect(path)
    try:
        for i, cycle in enumerate(cycles):
            upsert_rows(conn, "collector_runs", [_run(cycle)], venue="hyperliquid")
            rows = [
                _meta(cycle, "AAA", 1, 50),
                _meta(cycle, "BBB", 2, 40),
                _meta(cycle, "CCC", 3 + (i % 2), 20 - i),
            ]
            upsert_rows(conn, "meta_index", rows, venue="hyperliquid")
            if with_prices:
                upsert_rows(
                    conn,
                    "coin_prices",
                    [_price(cycle, r["coin"], mark=100.0 + i, ohlc_close=101.0 + i, closed=True) for r in rows],
                    venue="hyperliquid",
                )
            upsert_rows(
                conn,
                "accounts",
                [
                    {
                        "venue": "hyperliquid",
                        "address": "0xabc",
                        "display_name": "x",
                        "first_seen_at": cycle,
                        "last_seen_at": cycle,
                    }
                ],
                venue="hyperliquid",
            )
        conn.commit()
    finally:
        conn.close()


class BagrankBackupTests(unittest.TestCase):
    def test_incremental_does_not_refetch_full_history(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.sqlite"
            dst = Path(tmp) / "dst.sqlite"
            t0 = "2026-01-01T00:00:00Z"
            t1 = "2026-01-01T01:00:00Z"
            t2 = "2026-01-01T02:00:00Z"
            _seed(src, [t0, t1])
            source = SqliteSource(src)
            try:
                first = sync(source, dst)
            finally:
                source.close()
            self.assertEqual(first["cycles"], 2)
            self.assertTrue(first["first_run"])
            conn = connect(dst)
            try:
                self.assertEqual(count_table(conn, "meta_index"), 6)
            finally:
                conn.close()

            _seed(src, [t0, t1, t2])
            source = SqliteSource(src)
            try:
                second = sync(source, dst)
            finally:
                source.close()
            self.assertGreaterEqual(second["cycles"], 1)
            self.assertLess(second["cycles"], 3)
            self.assertFalse(second["first_run"])
            conn = connect(dst)
            try:
                self.assertEqual(count_table(conn, "collector_runs"), 3)
                self.assertEqual(count_table(conn, "meta_index"), 9)
                self.assertEqual(count_table(conn, "coin_prices"), 9)
            finally:
                conn.close()

    def test_late_price_backfill_updates_old_hour(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.sqlite"
            dst = Path(tmp) / "dst.sqlite"
            t0 = "2026-01-01T00:00:00Z"
            t1 = "2026-01-01T01:00:00Z"
            t2 = "2026-01-01T02:00:00Z"
            t3 = "2026-01-01T03:00:00Z"
            _seed(src, [t0, t1, t2])
            source = SqliteSource(src)
            try:
                sync(source, dst)
            finally:
                source.close()
            conn = connect(src)
            try:
                upsert_rows(
                    conn,
                    "coin_prices",
                    [
                        _price(
                            t0,
                            "AAA",
                            mark=50.0,
                            ohlc_open=49.0,
                            ohlc_close=55.0,
                            closed=True,
                            fetched_at=t3,
                        )
                    ],
                    venue="hyperliquid",
                )
                conn.commit()
            finally:
                conn.close()
            source = SqliteSource(src)
            try:
                stats = sync(source, dst)
            finally:
                source.close()
            self.assertGreater(int(stats.get("price_refresh") or 0), 0)
            conn = connect(dst)
            try:
                row = conn.execute(
                    """
                    SELECT mark_px, ohlc_close, ohlc_closed
                    FROM coin_prices
                    WHERE coin='AAA' AND cycle_ts=?
                    """,
                    (t0,),
                ).fetchone()
            finally:
                conn.close()
            self.assertIsNotNone(row)
            self.assertEqual(float(row["mark_px"]), 50.0)
            self.assertEqual(float(row["ohlc_close"]), 55.0)
            self.assertEqual(int(row["ohlc_closed"]), 1)


class RankStrategyTests(unittest.TestCase):
    def test_rank_up_down_enters_on_improve_exits_on_drop(self) -> None:
        hours = [
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
            "2026-01-01T03:00:00Z",
        ]
        rows = []
        boards = [
            [("AAA", 1, 50), ("BBB", 2, 40), ("CCC", 3, 30), ("DDD", 4, 20), ("EEE", 5, 10), ("FFF", 6, 8)],
            [("AAA", 1, 50), ("BBB", 2, 40), ("CCC", 3, 30), ("EEE", 4, 22), ("DDD", 5, 20), ("FFF", 6, 8)],
            [("AAA", 1, 50), ("BBB", 2, 40), ("CCC", 3, 30), ("EEE", 4, 22), ("DDD", 5, 20), ("FFF", 6, 8)],
            [("AAA", 1, 50), ("BBB", 2, 40), ("CCC", 3, 30), ("DDD", 4, 21), ("EEE", 5, 10), ("FFF", 6, 8)],
        ]
        for cycle, board in zip(hours, boards):
            for coin, rank, wallets in board:
                rows.append(_meta(cycle, coin, rank, wallets))
        panel = panel_from_meta_rows(rows, step_hours=1)
        panel.marks[:] = 100.0
        panel.marks[2, :] = 110.0
        panel.marks[3, :] = 90.0
        tc, ts, tw, tl = build_targets(
            panel.rank,
            panel.side,
            panel.wallets,
            panel.hold_pct,
            panel.agreement,
            panel.mean_leverage,
            SID_RANK_UP_DOWN,
            5,
            5,
            1,
            1,
            0.0,
            0.0,
        )
        eee = panel.coins.index("EEE")
        self.assertEqual(int(tc[0, 0]), -1)
        self.assertIn(eee, [int(x) for x in tc[1] if x >= 0])
        self.assertIn(eee, [int(x) for x in tc[2] if x >= 0])
        self.assertNotIn(eee, [int(x) for x in tc[3] if x >= 0])

        ret, dd, trips, wr, fees, final, _eq, _hold = simulate(
            panel.marks, tc, ts, tw, tl, 0.0005, 0.95, 1000.0, 5, 0.7
        )
        self.assertGreater(trips, 0)
        self.assertTrue(np.isfinite(ret))
        self.assertTrue(np.isfinite(dd))

    def test_top_k_holds_current_leaders(self) -> None:
        hours = ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"]
        rows = []
        for cycle in hours:
            rows.extend(
                [
                    _meta(cycle, "AAA", 1, 50),
                    _meta(cycle, "BBB", 2, 40),
                ]
            )
        panel = panel_from_meta_rows(rows)
        panel.marks[:] = 50.0
        tc, ts, tw, tl = build_targets(
            panel.rank,
            panel.side,
            panel.wallets,
            panel.hold_pct,
            panel.agreement,
            panel.mean_leverage,
            SID_TOP_K,
            2,
            2,
            1,
            1,
            0.0,
            0.0,
        )
        names = [panel.coins[int(c)] for c in tc[1] if int(c) >= 0]
        self.assertEqual(set(names), {"AAA", "BBB"})


class CollectorBoardShapeTests(unittest.TestCase):
    def test_hold_rows_follow_collector_rank(self) -> None:
        from pmf.majority import hold_rows_from_meta, rank_map_from_state

        rows = hold_rows_from_meta(
            [
                _meta("2026-01-01T00:00:00Z", "BBB", 2, 40),
                _meta("2026-01-01T00:00:00Z", "AAA", 1, 50),
            ]
        )
        self.assertEqual([r.coin for r in rows], ["AAA", "BBB"])
        mapped = rank_map_from_state({"AAA|long": 1, "BBB": 4})
        self.assertEqual(mapped["AAA"], 1)
        self.assertEqual(mapped["BBB"], 4)

    def test_fill_prefers_collector_mark_over_close(self) -> None:
        self.assertEqual(row_fill_px({"mark_px": 10, "ohlc_open": 8, "ohlc_close": 12}), 10.0)
        self.assertEqual(row_fill_px({"mark_px": None, "ohlc_open": 8, "ohlc_close": 12}), 8.0)
        self.assertEqual(row_fill_px({"ohlc_close": 12}), 12.0)

    def test_panel_joins_collector_prices(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "bag.sqlite"
            t0 = "2026-01-01T00:00:00Z"
            t1 = "2026-01-01T01:00:00Z"
            _seed(db, [t0, t1])
            conn = connect(db)
            try:
                panel = panel_from_sqlite(conn, venue="hyperliquid")
                out = fill_panel_from_collector(conn, panel, venue="hyperliquid")
            finally:
                conn.close()
            self.assertGreater(out["filled"], 0)
            self.assertEqual(out["missing"], 0)
            self.assertTrue(float(panel.marks[0, panel.coins.index("AAA")]) > 0.0)


class UniverseAndDsnTests(unittest.TestCase):
    def test_prepare_dsn_strips_pooler_and_adds_ssl(self) -> None:
        from bagrank.dsn import prepare_dsn

        raw = "postgresql://u:p@ep-x-pooler.c-7.us-east-1.aws.neon.tech/db"
        out = prepare_dsn(raw)
        self.assertNotIn("-pooler.", out)
        self.assertIn("ep-x.c-7.us-east-1.aws.neon.tech", out)
        self.assertIn("sslmode=require", out)
        self.assertIn("gssencmode=disable", out)
        self.assertIn("channel_binding=require", out)
        self.assertIn("options=endpoint", out)
        self.assertIn("ep-x", out)
        self.assertEqual(prepare_dsn(""), "")
        from bagrank.dsn import pooler_dsn

        pooled = pooler_dsn(out)
        self.assertIn("-pooler.", pooled)

    def test_resolve_uses_neon_bagrank_not_database_url(self) -> None:
        import os
        from unittest.mock import patch

        from bagrank.dsn import redacted_dsn, resolve_database_url

        env = {
            "NEON_BAGRANK": "postgresql://u:secret@ep-from-bagrank.c-5.us-east-2.aws.neon.tech/neondb",
            "DATABASE_URL": "postgresql://u:secret@ep-from-database-url.c-7.us-east-1.aws.neon.tech/other",
        }
        with patch("bagrank.dsn.load_env"), patch.dict(os.environ, env, clear=False):
            url, src = resolve_database_url()
        self.assertEqual(src, "NEON_BAGRANK")
        self.assertIn("ep-from-bagrank", url)
        self.assertNotIn("ep-from-database-url", url)
        shown = redacted_dsn(url)
        self.assertIn("ep-from-bagrank", shown)
        self.assertNotIn("secret", shown)

    def test_resolve_uses_neon_database_when_bagrank_missing(self) -> None:
        import os
        from unittest.mock import patch

        from bagrank.dsn import resolve_database_url

        env = {
            "NEON_DATABASE": "postgresql://u:secret@ep-live.c-5.us-east-2.aws.neon.tech/neondb",
            "DATABASE_URL": "postgresql://u:secret@ep-other.c-7.us-east-1.aws.neon.tech/other",
        }
        with patch("bagrank.dsn.load_env"), patch.dict(os.environ, env, clear=True):
            url, src = resolve_database_url()
        self.assertEqual(src, "NEON_DATABASE")
        self.assertIn("ep-live", url)
        self.assertNotIn("ep-other", url)

    def test_railway_does_not_use_database_url(self) -> None:
        import os
        from unittest.mock import patch

        from bagrank.dsn import resolve_database_url

        env = {
            "RAILWAY_ENVIRONMENT": "production",
            "DATABASE_URL": "postgresql://u:secret@ep-rail.c-5.us-east-2.aws.neon.tech/neondb",
        }
        with patch("bagrank.dsn.load_env"), patch.dict(os.environ, env, clear=True):
            url, src = resolve_database_url()
        self.assertEqual(src, "")
        self.assertEqual(url, "")

    def test_resolve_pnl_uses_neon_pnl_not_roi(self) -> None:
        import os
        from unittest.mock import patch

        from bagrank.dsn import default_pnl_sqlite_path, resolve_pnl_database_url

        path = default_pnl_sqlite_path()
        self.assertEqual(path.name, "pnlrank.sqlite")
        self.assertEqual(path.parent.name, "pnlrank")
        env = {
            "NEON_PNL": "postgresql://u:secret@ep-pnl.c-5.us-east-2.aws.neon.tech/neondb",
            "NEON_BAGRANK": "postgresql://u:secret@ep-from-bagrank.c-5.us-east-2.aws.neon.tech/neondb",
            "DATABASE_URL": "postgresql://u:secret@ep-other.c-7.us-east-1.aws.neon.tech/other",
        }
        with patch("bagrank.dsn.load_env"), patch.dict(os.environ, env, clear=True):
            url, src = resolve_pnl_database_url()
        self.assertEqual(src, "NEON_PNL")
        self.assertIn("ep-pnl", url)
        self.assertNotIn("ep-from-bagrank", url)
        self.assertNotIn("ep-other", url)
        roi_only = {"NEON_BAGRANK": env["NEON_BAGRANK"]}
        with patch("bagrank.dsn.load_env"), patch("bagrank.dsn.PNL_COLLECTOR_ENV") as pnl_env, patch.dict(
            os.environ, roi_only, clear=True
        ):
            pnl_env.exists.return_value = False
            empty, empty_src = resolve_pnl_database_url()
        self.assertEqual(empty_src, "")
        self.assertEqual(empty, "")

    def test_hl_overlay_writes_last_bar_mark(self) -> None:
        from bagrank.prices import overlay_hl_market

        hours = ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"]
        rows = [_meta(cycle, "AAA", 1, 50) for cycle in hours]
        panel = panel_from_meta_rows(rows)
        panel.marks[:] = 10.0

        class _Info:
            def meta_and_asset_ctxs(self):
                return (
                    {"universe": [{"name": "AAA"}]},
                    [
                        {
                            "markPx": "77.5",
                            "funding": "0.0001",
                            "openInterest": "9",
                            "premium": "0.01",
                            "dayNtlVlm": "100",
                            "prevDayPx": "70",
                        }
                    ],
                )

            def all_mids(self):
                return {"AAA": "88.25"}

        n = overlay_hl_market(panel, _Info())
        self.assertGreaterEqual(n, 1)
        self.assertAlmostEqual(float(panel.marks[-1, 0]), 88.25)

    def test_enter_top_zero_allows_names_outside_top_five(self) -> None:
        hours = ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"]
        rows = []
        board = [
            ("AAA", 1, 50),
            ("BBB", 2, 40),
            ("CCC", 3, 30),
            ("DDD", 4, 20),
            ("EEE", 5, 10),
            ("FFF", 6, 8),
        ]
        for cycle in hours:
            for coin, rank, wallets in board:
                rows.append(_meta(cycle, coin, rank, wallets))
        panel = panel_from_meta_rows(rows)
        panel.marks[:] = 10.0
        tc, *_rest = build_targets(
            panel.rank,
            panel.side,
            panel.wallets,
            panel.hold_pct,
            panel.agreement,
            panel.mean_leverage,
            SID_TOP_K,
            0,
            8,
            1,
            1,
            0.0,
            0.0,
        )
        names = {panel.coins[int(c)] for c in tc[1] if int(c) >= 0}
        self.assertIn("FFF", names)
        self.assertGreaterEqual(len(names), 6)

    def test_resample_closed_uses_last_hour(self) -> None:
        from bagrank.panel import resample_closed

        hours = [f"2026-01-01T{h:02d}:00:00Z" for h in range(4)]
        rows = []
        for i, cycle in enumerate(hours):
            rows.append(_meta(cycle, "AAA", 9 if i % 2 == 0 else 1, 50))
        panel = panel_from_meta_rows(rows)
        out = resample_closed(panel, 2)
        self.assertEqual(out.n_times, 2)
        self.assertEqual(int(out.rank[0, 0]), 1)
        self.assertEqual(int(out.rank[1, 0]), 1)

    def test_min_hold_blocks_same_bar_exit(self) -> None:
        hours = [f"2026-01-01T{h:02d}:00:00Z" for h in range(6)]
        rows = []
        for cycle in hours:
            rows.append(_meta(cycle, "AAA", 1, 50))
            rows.append(_meta(cycle, "BBB", 2, 40))
        panel = panel_from_meta_rows(rows)
        panel.marks[:] = 10.0
        tc, ts, tw, tl = build_targets(
            panel.rank,
            panel.side,
            panel.wallets,
            panel.hold_pct,
            panel.agreement,
            panel.mean_leverage,
            SID_TOP_K,
            2,
            2,
            1,
            1,
            0.0,
            0.0,
        )
        *_rest, hold0 = simulate(
            panel.marks, tc, ts, tw, tl, 0.0005, 0.95, 1000.0, 2, 0.7, 1, 0, 0, 1.0
        )
        *_rest, hold8 = simulate(
            panel.marks, tc, ts, tw, tl, 0.0005, 0.95, 1000.0, 2, 0.7, 1, 8, 0, 1.0
        )
        self.assertGreaterEqual(float(hold8), float(hold0))

    def test_live_min_hold_does_not_open_second_slot(self) -> None:
        import time as time_mod

        from bagrank.live import PaperBook, _apply_paper

        book = PaperBook(1000.0)
        book.positions["BCH"] = {"side": "long", "size": 1.0, "entry": 100.0}
        state = {"opened": {"BCH": time_mod.time()}}
        desired = [
            {
                "coin": "GOOGL",
                "side": "short",
                "size": 1.0,
                "px": 10.0,
                "notional": 100.0,
            }
        ]
        _apply_paper(
            book,
            desired,
            state,
            min_hold_h=6,
            marks={"BCH": 100.0, "GOOGL": 10.0},
            slots=1,
        )
        self.assertIn("BCH", book.positions)
        self.assertNotIn("GOOGL", book.positions)

    def test_closed_hour_does_not_replace_neon_hour(self) -> None:
        from datetime import datetime, timezone

        from bagrank.hlcycle import closed_cycle_ts, should_append_hour

        now = datetime(2026, 9, 13, 3, 2, 0, tzinfo=timezone.utc)
        self.assertEqual(closed_cycle_ts(now), "2026-09-13T02:00:00Z")
        self.assertFalse(should_append_hour("2026-09-13T02:00:00Z", "2026-09-13T02:00:00Z"))
        self.assertTrue(should_append_hour("2026-09-13T02:00:00Z", "2026-09-13T03:00:00Z"))

    def test_score_wallet_flow_picks_rising_wallets(self) -> None:
        import numpy as np
        from bagrank.kernels import MODE_TOP, N_FEAT, build_score_targets
        from bagrank.search import data_span, range_tag, run_one, sample_spec

        hours = [f"2026-01-01T{h:02d}:00:00Z" for h in range(5)]
        rows = []
        for i, cycle in enumerate(hours):
            rows.append(_meta(cycle, "AAA", 3, 10 + i * 8))
            rows.append(_meta(cycle, "BBB", 1, 40))
            rows.append(_meta(cycle, "CCC", 2, 20))
        panel = panel_from_meta_rows(rows)
        panel.marks[:] = 10.0
        w = np.zeros(N_FEAT, dtype=np.float64)
        w[3] = 1.0
        tc, ts, tw, tl = build_score_targets(
            panel.rank,
            panel.side,
            panel.wallets,
            panel.hold_pct,
            panel.agreement,
            panel.mean_leverage,
            panel.conviction,
            panel.notional,
            panel.long_n,
            panel.short_n,
            panel.funding,
            panel.oi,
            panel.premium,
            panel.volume,
            panel.prev_day,
            panel.marks,
            w,
            5,
            1,
            0.0,
            0.0,
            4,
            MODE_TOP,
            0,
            0.0,
            0.0,
            0.0,
            0,
        )
        aaa = panel.coins.index("AAA")
        picked = [int(tc[t, 0]) for t in range(1, panel.n_times)]
        self.assertIn(aaa, picked)
        span = data_span(panel)
        self.assertTrue(span["data_from"].startswith("2026-01-01T00"))
        self.assertTrue(span["data_until"].startswith("2026-01-01T04"))
        self.assertIn("from_", range_tag(span["data_from_unix"], span["data_until_unix"]))
        self.assertIn("until_", range_tag(span["data_from_unix"], span["data_until_unix"]))
        spec = sample_spec(__import__("random").Random(1))
        spec["step_h"] = 1
        spec["slots"] = 2
        spec["enter_top"] = 5
        spec["min_hold_h"] = 0
        spec["exec_lag"] = 1
        row = run_one(panel, spec, equity=1000.0, gross_pct=95.0, max_pair_share=0.7)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["data_from"], span["data_from"])
        self.assertEqual(row["data_until"], span["data_until"])
        self.assertIn("spec", row)

    def test_each_search_run_gets_its_own_folder(self) -> None:
        import tempfile
        from pathlib import Path

        from bagrank.search import ResultStore, data_span

        hours = [f"2026-01-01T{h:02d}:00:00Z" for h in range(3)]
        rows = [_meta(cycle, "AAA", 1, 10) for cycle in hours]
        panel = panel_from_meta_rows(rows)
        span = data_span(panel)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = ResultStore(root, span)
            b = ResultStore(root, span)
            self.assertNotEqual(a.directory, b.directory)
            self.assertTrue(a.meta_path.exists())
            self.assertTrue(b.meta_path.exists())
            self.assertIn("from_", a.directory.name)
            self.assertIn("until_", a.directory.name)
            self.assertIn("run_", a.directory.name)

    def test_csv_row_roundtrips_to_same_spec(self) -> None:
        import tempfile
        from pathlib import Path

        from bagrank.search import run_one, sample_spec
        from bagrank.specio import flatten_result, load_csv_row, spec_from_row, write_csv

        hours = [f"2026-01-01T{h:02d}:00:00Z" for h in range(8)]
        rows = []
        for cycle in hours:
            rows.append(_meta(cycle, "AAA", 1, 50))
            rows.append(_meta(cycle, "BBB", 2, 40))
        panel = panel_from_meta_rows(rows)
        panel.marks[:] = 10.0
        spec = sample_spec(__import__("random").Random(3))
        spec["step_h"] = 1
        spec["slots"] = 2
        spec["enter_top"] = 5
        spec["min_hold_h"] = 4
        spec["exec_lag"] = 1
        result = run_one(panel, spec, equity=1000.0, gross_pct=95.0, max_pair_share=0.7)
        self.assertIsNotNone(result)
        assert result is not None
        flat = flatten_result(result, run_id="test-run")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaderboard.csv"
            write_csv(path, [flat])
            loaded = load_csv_row(path, 2)
        back = spec_from_row(loaded)
        self.assertEqual(int(back["step_h"]), 1)
        self.assertEqual(int(back["exec_lag"]), 1)
        self.assertEqual(int(back["slots"]), 2)
        self.assertEqual(back["engine"], spec["engine"])
        if spec.get("engine") != "named":
            from bagrank.kernels import N_FEAT

            self.assertEqual(len(back["weights"]), N_FEAT)

    def test_lagged_holdings_use_exec_lag(self) -> None:
        from bagrank.engine import lagged_holdings

        hours = [f"2026-01-01T{h:02d}:00:00Z" for h in range(6)]
        rows = []
        for cycle in hours:
            rows.append(_meta(cycle, "AAA", 1, 50))
            rows.append(_meta(cycle, "BBB", 2, 40))
        panel = panel_from_meta_rows(rows)
        panel.marks[:] = 10.0
        spec = {
            "engine": "named",
            "name": "top_k",
            "step_h": 1,
            "slots": 2,
            "enter_top": 2,
            "exec_lag": 1,
            "k": 1,
            "lookback": 1,
            "a": 0,
            "b": 0,
        }
        holds = lagged_holdings(panel, spec)
        coins = {h["coin"] for h in holds}
        self.assertEqual(coins, {"AAA", "BBB"})
        from bagrank.specio import flatten_result, spec_from_row

        flat = flatten_result(
            {
                "sharpe": 0,
                "return_pct": 0,
                "round_trips": 0,
                "spec": spec,
            }
        )
        replayed = lagged_holdings(panel, spec_from_row({k: str(flat.get(k, "") or "") for k in flat}))
        self.assertEqual(
            [(h["coin"], h["side"]) for h in holds],
            [(h["coin"], h["side"]) for h in replayed],
        )

    def test_csv_cell_skips_nan_and_inf(self) -> None:
        from bagrank.specio import _cell, append_csv

        self.assertEqual(_cell(float("nan")), "")
        self.assertEqual(_cell(float("inf")), "")
        self.assertEqual(_cell(float("-inf")), "")
        self.assertEqual(_cell(3.0), "3")
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.csv"
            append_csv(path, {"sharpe": float("nan"), "return_pct": 1.5, "family": "x"})

    def test_paste_csv_and_tsv_roundtrip(self) -> None:
        from bagrank.search import run_one, sample_spec
        from bagrank.specio import (
            CSV_COLUMNS,
            flatten_result,
            parse_strategy_text,
            spec_from_row,
        )

        hours = [f"2026-01-01T{h:02d}:00:00Z" for h in range(8)]
        rows = []
        for cycle in hours:
            rows.append(_meta(cycle, "AAA", 1, 50))
            rows.append(_meta(cycle, "BBB", 2, 40))
        panel = panel_from_meta_rows(rows)
        panel.marks[:] = 10.0
        spec = sample_spec(__import__("random").Random(4))
        spec["step_h"] = 1
        spec["slots"] = 2
        spec["enter_top"] = 5
        spec["min_hold_h"] = 4
        spec["exec_lag"] = 1
        result = run_one(panel, spec, equity=1000.0, gross_pct=95.0, max_pair_share=0.7)
        self.assertIsNotNone(result)
        assert result is not None
        flat = flatten_result(result, run_id="paste-run")
        values = [str(flat.get(k, "") or "") for k in CSV_COLUMNS]
        csv_line = ",".join(values)
        tsv_line = "\t".join(values)
        from_csv = spec_from_row(parse_strategy_text(csv_line))
        from_tsv = spec_from_row(parse_strategy_text(tsv_line))
        self.assertEqual(from_csv["engine"], spec["engine"])
        self.assertEqual(from_tsv["slots"], spec["slots"])
        self.assertEqual(from_csv["min_hold_h"], spec["min_hold_h"])
        header = ",".join(CSV_COLUMNS)
        with_header = spec_from_row(parse_strategy_text(header + "\n" + csv_line))
        self.assertEqual(with_header["exec_lag"], 1)
        named = {
            "sharpe": 1.2,
            "return_pct": 3.4,
            "spec": {
                "engine": "named",
                "family": "top_k",
                "name": "top_k",
                "step_h": 4,
                "min_hold_h": 8,
                "enter_top": 5,
                "slots": 5,
                "exec_lag": 1,
                "k": 1,
            },
        }
        from_json = spec_from_row(parse_strategy_text(__import__("json").dumps(named)))
        self.assertEqual(from_json["engine"], "named")
        self.assertEqual(from_json["name"], "top_k")
        self.assertEqual(from_json["step_h"], 4)
        mixed = spec_from_row(parse_strategy_text(header + "\n" + tsv_line))
        self.assertEqual(mixed["engine"], spec["engine"])
        self.assertEqual(int(mixed["slots"]), int(spec["slots"]))

    def test_rank_live_csv_is_header_ready_for_one_row_paste(self) -> None:
        from pathlib import Path

        from bagrank.specio import (
            LIVE_CSV_NAME,
            csv_header_line,
            default_live_csv,
            parse_strategy_text,
            require_strategy_row,
        )

        path = default_live_csv()
        self.assertEqual(path.name, LIVE_CSV_NAME)
        self.assertTrue(path.exists())
        first = path.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(first, csv_header_line())
        with self.assertRaises(SystemExit):
            parse_strategy_text(first)
        with self.assertRaises(SystemExit):
            require_strategy_row({"engine": "", "name": "", "family": ""})

    def test_export_jsonl_folder_writes_ranked_csvs(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from bagrank.specio import export_search_dir, read_csv_rows

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "run_old"
            folder.mkdir()
            row = {
                "sharpe": 2.5,
                "return_pct": 10.0,
                "round_trips": 7,
                "spec": {
                    "engine": "named",
                    "name": "top_k",
                    "family": "top_k",
                    "step_h": 1,
                    "min_hold_h": 8,
                    "slots": 3,
                    "exec_lag": 1,
                },
            }
            worse = dict(row)
            worse["sharpe"] = 0.1
            worse["return_pct"] = 50.0
            worse["round_trips"] = 99
            (folder / "results.jsonl").write_text(
                json.dumps(row) + "\n" + json.dumps(worse) + "\n",
                encoding="utf-8",
            )
            csv_path = export_search_dir(folder)
            self.assertTrue(csv_path.exists())
            ranked = read_csv_rows(csv_path)
            self.assertEqual(ranked[0]["sharpe"], "2.5")
            by_ret = read_csv_rows(folder / "by_return.csv")
            self.assertEqual(by_ret[0]["return_pct"], "50")
            by_tr = read_csv_rows(folder / "by_trades.csv")
            self.assertEqual(by_tr[0]["round_trips"], "99")
            self.assertTrue((folder / "by_fitness.csv").exists())
            self.assertTrue((folder / "by_live.csv").exists())
            self.assertEqual(read_csv_rows(folder / "by_live.csv"), [])

    def test_fitness_prefers_real_profit_over_empty_sharpe(self) -> None:
        from bagrank.search import fitness, sample_spec

        weak = {"return_pct": 0.4, "sharpe": 40.0, "max_dd_pct": 0.2, "round_trips": 1, "win_rate_pct": 100.0}
        strong = {"return_pct": 18.0, "sharpe": 3.0, "max_dd_pct": 6.0, "round_trips": 8, "win_rate_pct": 62.0}
        self.assertGreater(fitness(strong), fitness(weak))
        spec = sample_spec(__import__("random").Random(9))
        self.assertIn("gross_pct", spec)
        self.assertIn("size_mode", spec)
        self.assertIn("exposure_mode", spec)
        self.assertEqual(spec.get("lev_x"), 3)
        self.assertGreaterEqual(len(spec.get("weights") or spec.get("name") or ""), 1)

    def test_last_days_keeps_newest_window(self) -> None:
        from bagrank.panel import last_days

        hours = [f"2026-01-01T{h:02d}:00:00Z" for h in range(10)]
        rows = [_meta(cycle, "AAA", 1, 10) for cycle in hours]
        panel = panel_from_meta_rows(rows)
        sliced = last_days(panel, 3.0 / 24.0)
        self.assertEqual(sliced.n_times, 4)
        self.assertEqual(int(sliced.cycle_unix[0]), int(panel.cycle_unix[6]))
        self.assertEqual(last_days(panel, 0).n_times, 10)
        self.assertEqual(last_days(panel, 7).n_times, 10)

    def test_live_ok_requires_1x_and_strict_cuts(self) -> None:
        from bagrank.search import live_ok

        good = {
            "use_lev": 0,
            "round_trips": 18,
            "max_dd_pct": 8.5,
            "slots": 1,
            "return_pct": 47.0,
            "sharpe": 13.0,
            "n_bars": 44,
            "win_rate_pct": 66.0,
            "trips_per_day": 3.3,
            "min_hold_h": 6,
            "exec_lag": 2,
        }
        self.assertTrue(live_ok(good))
        self.assertFalse(live_ok(dict(good, use_lev=1)))
        self.assertFalse(live_ok(dict(good, max_dd_pct=80)))
        self.assertFalse(live_ok(dict(good, slots=2)))
        self.assertFalse(live_ok(dict(good, round_trips=3)))
        self.assertFalse(live_ok(dict(good, return_pct=5)))
        self.assertFalse(live_ok(dict(good, min_hold_h=2)))
        self.assertFalse(live_ok(dict(good, exec_lag=1)))
        self.assertFalse(live_ok(dict(good, n_bars=20)))
        pnl = dict(
            good,
            board="pnl",
            round_trips=3,
            return_pct=4.0,
            sharpe=0.8,
            win_rate_pct=40.0,
            trips_per_day=1.2,
            min_hold_h=0,
            exec_lag=1,
        )
        self.assertTrue(live_ok(pnl))
        self.assertFalse(live_ok(dict(pnl, board="roi")))

    def test_by_live_csv_keeps_only_strict_1x_rows(self) -> None:
        import tempfile
        from pathlib import Path

        from bagrank.search import ResultStore
        from bagrank.specio import read_csv_rows

        span = {
            "data_from": "2026-01-01T00:00:00Z",
            "data_until": "2026-01-03T00:00:00Z",
            "data_from_unix": 1,
            "data_until_unix": 2,
            "n_hours": 48,
            "n_bars": 44,
            "n_coins": 2,
        }

        def trial(*, lev: int, ret: float, dd: float, slots: int, trips: int, family: str) -> dict:
            return {
                "return_pct": ret,
                "sharpe": 13.0,
                "fitness": ret,
                "max_dd_pct": dd,
                "final_equity": 1400,
                "round_trips": trips,
                "trips_per_day": 3.0,
                "avg_hold_h": 6,
                "win_rate_pct": 66,
                "fees": 1,
                "n_hours": 48,
                "n_bars": 44,
                "n_coins": 2,
                "data_from": span["data_from"],
                "data_until": span["data_until"],
                "spec": {
                    "family": family,
                    "engine": "score",
                    "name": family,
                    "step_h": 3,
                    "min_hold_h": 6,
                    "enter_top": 30,
                    "slots": slots,
                    "exec_lag": 2,
                    "use_lev": lev,
                    "gross_pct": 125,
                    "mode": 0,
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp), span)
            store.append(trial(lev=1, ret=3000, dd=80, slots=9, trips=24, family="flip_majority"))
            store.append(trial(lev=0, ret=48, dd=3.0, slots=3, trips=26, family="flip_majority"))
            store.append(trial(lev=0, ret=47, dd=8.5, slots=1, trips=18, family="rank_mom"))
            store.write_csvs()
            live = read_csv_rows(store.csv_live)
            self.assertEqual(len(live), 1)
            self.assertEqual(live[0]["family"], "rank_mom")
            self.assertEqual(live[0]["use_lev"], "0")
            fit = read_csv_rows(store.csv_fit)
            self.assertEqual(len(fit), 3)
            self.assertTrue(store.csv_return.exists())

    def test_run_one_keeps_spec_gross_pct(self) -> None:
        from bagrank.search import run_one, sample_spec

        hours = [f"2026-01-01T{h:02d}:00:00Z" for h in range(8)]
        rows = []
        for cycle in hours:
            rows.append(_meta(cycle, "AAA", 1, 50))
            rows.append(_meta(cycle, "BBB", 2, 40))
        panel = panel_from_meta_rows(rows)
        panel.marks[:] = 10.0
        spec = sample_spec(__import__("random").Random(2))
        spec["step_h"] = 1
        spec["slots"] = 2
        spec["enter_top"] = 5
        spec["gross_pct"] = 40
        spec["size_mode"] = 1
        spec["exposure_mode"] = 1
        spec["exec_lag"] = 1
        spec["min_hold_h"] = 0
        row = run_one(panel, spec, equity=1000.0, gross_pct=95.0, max_pair_share=0.7)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(float(row["gross_pct"]), 40.0)
        self.assertEqual(int(row["size_mode"]), 1)
        self.assertIn("fitness", row)

    def test_hl_tally_ranks_by_wallets_then_hold_pct(self) -> None:
        from bagrank.hlcycle import tally_holds

        snaps = [
            {
                "ok": True,
                "positions": [
                    {"coin": "BBB", "side": "long", "notional": 100, "leverage": 3, "conviction": 0.1},
                    {"coin": "AAA", "side": "long", "notional": 100, "leverage": 5, "conviction": 0.2},
                ],
            },
            {
                "ok": True,
                "positions": [
                    {"coin": "AAA", "side": "long", "notional": 80, "leverage": 4, "conviction": 0.15},
                ],
            },
            {
                "ok": True,
                "positions": [
                    {"coin": "AAA", "side": "short", "notional": 90, "leverage": 2, "conviction": -0.1},
                    {"coin": "CCC", "side": "short", "notional": 200, "leverage": 8, "conviction": -0.4},
                ],
            },
            {"ok": False, "positions": []},
        ]
        rows, stats = tally_holds(snaps)
        self.assertEqual(stats["ok"], 3)
        self.assertEqual(stats["errors"], 1)
        self.assertEqual(rows[0]["coin"], "AAA")
        self.assertEqual(rows[0]["side"], "long")
        self.assertEqual(rows[0]["wallets"], 2)
        self.assertEqual(rows[0]["rank"], 1)
        self.assertAlmostEqual(rows[0]["hold_pct"], 2 / 3)
        self.assertAlmostEqual(rows[0]["agreement"], 2 / 3)

    def test_persist_hl_cycle_loads_panel(self) -> None:
        import tempfile

        from bagrank.hlcycle import persist_cycle

        payload = {
            "cycle_ts": "2026-09-12T21:00:00Z",
            "listed": 3,
            "snapped_ok": 3,
            "snapped_err": 0,
            "empty_books": 0,
            "coverage": 1.0,
            "duration_s": 1.2,
            "status": "ok",
            "meta_index": [
                {
                    "coin": "AAA",
                    "side": "long",
                    "wallets": 10,
                    "hold_pct": 0.5,
                    "agreement": 0.9,
                    "long_n": 10,
                    "short_n": 1,
                    "median_leverage": 5,
                    "mean_leverage": 5.0,
                    "avg_conviction": 0.2,
                    "notional_usd": 1000.0,
                    "rank": 1,
                }
            ],
            "coin_prices": [
                {
                    "cycle_ts": "2026-09-12T21:00:00Z",
                    "coin": "AAA",
                    "mark_px": 12.5,
                    "mid_px": 12.4,
                    "funding": 0.0001,
                    "open_interest": 9,
                    "prev_day_px": 11,
                    "day_ntl_vlm": 100,
                    "premium": 0.01,
                    "source": "hl",
                    "fetched_at": "2026-09-12T21:01:00Z",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bagrank.sqlite"
            persist_cycle(path, payload)
            conn = connect(path)
            try:
                panel = panel_from_sqlite(conn, venue="hyperliquid")
                fill_panel_from_collector(conn, panel, venue="hyperliquid")
            finally:
                conn.close()
        self.assertEqual(panel.n_times, 1)
        self.assertEqual(panel.coins, ["AAA"])
        self.assertAlmostEqual(float(panel.marks[0, 0]), 12.5)

    def test_persist_keeps_rolling_lookback_window(self) -> None:
        import tempfile

        from bagrank.hlcycle import persist_cycle

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "live.sqlite"
            for hour in range(8):
                cycle = f"2026-09-12T{hour:02d}:00:00Z"
                persist_cycle(
                    path,
                    {
                        "cycle_ts": cycle,
                        "listed": 1,
                        "snapped_ok": 1,
                        "snapped_err": 0,
                        "empty_books": 0,
                        "coverage": 1.0,
                        "duration_s": 1.0,
                        "status": "ok",
                        "meta_index": [
                            {
                                "coin": "AAA",
                                "side": "long",
                                "wallets": 10,
                                "hold_pct": 0.5,
                                "agreement": 0.9,
                                "long_n": 10,
                                "short_n": 0,
                                "median_leverage": 5,
                                "mean_leverage": 5.0,
                                "avg_conviction": 0.2,
                                "notional_usd": 1000.0,
                                "rank": 1,
                            }
                        ],
                        "coin_prices": [
                            {
                                "cycle_ts": cycle,
                                "coin": "AAA",
                                "mark_px": 10.0 + hour,
                                "source": "hl",
                                "fetched_at": cycle,
                            }
                        ],
                    },
                    keep_hours=6,
                )
            conn = connect(path)
            try:
                hours = [
                    str(r["cycle_ts"])
                    for r in conn.execute(
                        "SELECT cycle_ts FROM collector_runs ORDER BY cycle_ts"
                    )
                ]
                n_meta = conn.execute("SELECT COUNT(*) AS n FROM meta_index").fetchone()["n"]
                n_px = conn.execute("SELECT COUNT(*) AS n FROM coin_prices").fetchone()["n"]
            finally:
                conn.close()
        self.assertEqual(hours, [f"2026-09-12T{h:02d}:00:00Z" for h in range(2, 8)])
        self.assertEqual(n_meta, 6)
        self.assertEqual(n_px, 6)

    def test_live_loop_uses_hl_board_not_neon(self) -> None:
        import inspect

        from bagrank import live

        src = inspect.getsource(live.run_live)
        self.assertIn("LiveBoard", src)
        self.assertIn("panel = load_panel", src)
        self.assertIn("_mids_from_info", src)
        self.assertIn("seed_local_hours", src)
        self.assertIn("rank_by", src)
        self.assertNotIn("sync_neon", src)
        self.assertIn("fetch_exchange_max_lev", src)
        self.assertIn("apply_exchange_max_lev", src)
        self.assertNotIn("maybe_sync_backup", src)
        self.assertNotIn("NEON_BAGRANK", src)
        main_src = inspect.getsource(live.main)
        self.assertNotIn("on_railway", main_src)
        self.assertIn('mode = "paper"', main_src)
        self.assertIn('mode = "live"', main_src)
        self.assertIn("need_hours", src)
        self.assertIn("rank_by=board_kind", src)
        self.assertIn("Gathering PnL board from Hyperliquid", src)
        self.assertNotIn("NEON_PNL", src)

    def test_pnl_spec_trades_before_72h_retain(self) -> None:
        from bagrank.live import live_hour_windows, spec_board

        spec = {
            "board": "pnl",
            "name": "funding_follow",
            "lookback": 4,
            "step_h": 4,
            "exec_lag": 1,
        }
        self.assertEqual(spec_board(spec), "pnl")
        need, keep = live_hour_windows(spec)
        self.assertEqual(need, 20)
        self.assertEqual(keep, 72)
        roi_need, roi_keep = live_hour_windows({"board": "roi", "lookback": 4, "step_h": 2, "exec_lag": 2})
        self.assertEqual(roi_need, 12)
        self.assertEqual(roi_keep, 12)

    def test_railway_deploys_real_money(self) -> None:
        from pathlib import Path

        text = (Path(__file__).resolve().parent.parent / "railway.toml").read_text(encoding="utf-8")
        self.assertIn("python run_rank.py --live", text)
        self.assertNotIn('startCommand = "python run_rank.py"', text)


class PairSizeLevTests(unittest.TestCase):
    def test_pair_size_lev_uses_x_only_when_max_allows(self) -> None:
        from bagrank.lev import apply_exchange_max_lev, pair_size_lev

        self.assertEqual(pair_size_lev(10, 5), 2.0)
        self.assertEqual(pair_size_lev(20, 5), 4.0)
        self.assertEqual(pair_size_lev(5, 5), 1.0)
        self.assertEqual(pair_size_lev(3, 5), 1.0)
        self.assertEqual(pair_size_lev(10, 3), 3.0)
        self.assertEqual(pair_size_lev(11, 3), 3.0)
        self.assertEqual(pair_size_lev(20, 3), 6.0)
        self.assertEqual(pair_size_lev(50, 3), 16.0)
        self.assertEqual(pair_size_lev(3, 3), 1.0)
        self.assertEqual(pair_size_lev(2, 3), 1.0)
        self.assertEqual(pair_size_lev(0, 3), 1.0)
        self.assertEqual(pair_size_lev(50, 0), 1.0)
        self.assertEqual(pair_size_lev(10, 3) % 1, 0.0)

        hours = [f"2026-01-01T{h:02d}:00:00Z" for h in range(6)]
        rows = [_meta(cycle, "AAA", 1, 50) for cycle in hours]
        panel = panel_from_meta_rows(rows)
        apply_exchange_max_lev(panel, {"AAA": 3})
        self.assertEqual(float(panel.max_leverage[0]), 3.0)

    def test_simulate_lev_x_five_vs_max_below_x(self) -> None:
        hours = [f"2026-01-01T{h:02d}:00:00Z" for h in range(8)]
        rows = [_meta(cycle, "AAA", 1, 50) for cycle in hours]
        panel = panel_from_meta_rows(rows)
        for i in range(panel.n_times):
            panel.marks[i, 0] = 100.0 + 8.0 * i
        tc, ts, tw, tl = build_targets(
            panel.rank,
            panel.side,
            panel.wallets,
            panel.hold_pct,
            panel.agreement,
            panel.mean_leverage,
            SID_TOP_K,
            1,
            1,
            1,
            1,
            0.0,
            0.0,
        )
        kw = dict(
            fee_rate=0.0005,
            gross_frac=1.0,
            initial_equity=1000.0,
            max_slots=1,
            max_pair_share=1.0,
            exec_lag=1,
            min_hold=0,
            use_lev=0,
            bar_hours=1.0,
            exposure_mode=0,
        )
        *_a, final1, _eq1, _h1 = simulate(
            panel.marks, tc, ts, tw, tl, max_lev=np.array([50.0]), lev_x=0.0, **kw
        )
        *_b, final5, _eq5, _h5 = simulate(
            panel.marks, tc, ts, tw, tl, max_lev=np.array([50.0]), lev_x=5.0, **kw
        )
        *_c, final_lo, _eq_lo, _h_lo = simulate(
            panel.marks, tc, ts, tw, tl, max_lev=np.array([3.0]), lev_x=5.0, **kw
        )
        self.assertGreater(final5, final1)
        self.assertAlmostEqual(final_lo, final1, places=4)

    def test_live_weight_targets_follow_pair_max_lev(self) -> None:
        from bagrank.live import _weight_targets

        holds = [{"coin": "ETH", "wallets": 10, "px": 100.0, "side": "short", "lev": 1.0}]
        hi = _weight_targets(
            holds,
            equity=1000.0,
            gross_pct=100.0,
            max_pair_share=1.0,
            use_lev=0,
            slots=1,
            lev_x=3,
            max_lev={"ETH": 50},
        )
        mid = _weight_targets(
            holds,
            equity=1000.0,
            gross_pct=100.0,
            max_pair_share=1.0,
            use_lev=0,
            slots=1,
            lev_x=3,
            max_lev={"ETH": 11},
        )
        lo = _weight_targets(
            holds,
            equity=1000.0,
            gross_pct=100.0,
            max_pair_share=1.0,
            use_lev=0,
            slots=1,
            lev_x=3,
            max_lev={"ETH": 3},
        )
        self.assertEqual(len(hi), 1)
        self.assertEqual(hi[0]["lev"], 16)
        self.assertAlmostEqual(hi[0]["notional"], 16000.0)
        self.assertEqual(mid[0]["lev"], 3)
        self.assertAlmostEqual(mid[0]["notional"], 3000.0)
        self.assertEqual(lo[0]["lev"], 1)
        self.assertAlmostEqual(lo[0]["notional"], 1000.0)

    def test_pnl_backup_and_backtest_clis_are_separate(self) -> None:
        import inspect
        from pathlib import Path

        from bagrank.backup import main as backup_main
        from bagrank.dsn import default_pnl_sqlite_path, default_sqlite_path

        root = Path(__file__).resolve().parent.parent
        self.assertTrue((root / "backup_pnl.py").exists())
        self.assertTrue((root / "run_pnl_backtest.py").exists())
        self.assertTrue((root / "backup_bagrank.py").exists())
        self.assertTrue((root / "run_rank_backtest.py").exists())
        self.assertNotEqual(default_pnl_sqlite_path(), default_sqlite_path())
        src = inspect.getsource(backup_main)
        self.assertIn('kind == "pnl"', src)
        self.assertIn("NEON_PNL", src)


class FollowRank1Tests(unittest.TestCase):
    def test_skips_existing_leader_until_fresh_move(self) -> None:
        from bagrank.engine import compute_targets

        hours = [f"2026-01-01T{h:02d}:00:00Z" for h in range(5)]
        rows: list[dict] = []
        for i, cycle in enumerate(hours):
            if i < 2:
                rows.append(_meta(cycle, "AAA", 1, 50))
                rows.append(_meta(cycle, "BBB", 2, 40))
            else:
                rows.append(_meta(cycle, "BBB", 1, 50))
                rows.append(_meta(cycle, "AAA", 2, 40))
        panel = panel_from_meta_rows(rows)
        panel.marks[:] = 10.0
        tc, ts, _tw, _tl = compute_targets(
            panel,
            {"engine": "named", "name": "follow_rank1", "slots": 3, "enter_top": 10},
        )
        self.assertEqual(int(tc[0, 0]), -1)
        self.assertEqual(int(tc[1, 0]), -1)
        bbb = panel.coins.index("BBB")
        self.assertEqual(int(tc[2, 0]), bbb)
        self.assertEqual(int(tc[3, 0]), bbb)
        self.assertEqual(int(tc[4, 0]), bbb)

    def test_shortlist_pnl_ranks_by_pnl_not_roi(self) -> None:
        from bagrank.hlcycle import shortlist_top_pnl, shortlist_top_roi

        hi_roi = "0x" + "a" * 40
        hi_pnl = "0x" + "b" * 40
        rows = [
            {"address": hi_roi, "windows": {"perpWeek": {"roi": 9.0, "pnl": 10.0}}},
            {"address": hi_pnl, "windows": {"perpWeek": {"roi": 1.0, "pnl": 100.0}}},
        ]
        self.assertEqual(shortlist_top_roi(rows, limit=1), [hi_roi])
        self.assertEqual(shortlist_top_pnl(rows, limit=1), [hi_pnl])

    def test_spec_board_roundtrip(self) -> None:
        from bagrank.specio import flatten_result, spec_from_row

        spec = {
            "engine": "named",
            "name": "follow_rank1",
            "family": "follow_rank1",
            "board": "pnl",
            "slots": 1,
            "enter_top": 1,
            "use_lev": 0,
            "exec_lag": 1,
        }
        flat = flatten_result({"return_pct": 1, "spec": spec})
        self.assertEqual(flat["board"], "pnl")
        back = spec_from_row({k: str(flat.get(k, "") or "") for k in flat})
        self.assertEqual(back["board"], "pnl")
        self.assertEqual(back["name"], "follow_rank1")


if __name__ == "__main__":
    unittest.main()
