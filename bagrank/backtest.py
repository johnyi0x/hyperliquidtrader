"""Run every rank strategy on local collector hours + collector coin_prices."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .dsn import default_sqlite_path, load_env
from .kernels import N_FEAT, NAME_TO_ID, STRATEGY_NAMES, build_targets, simulate
from .panel import panel_from_sqlite, resample_closed
from .prices import fill_panel_from_collector, sync_hl_gaps
from .search import ResultStore, data_span, run_one, search_loop
from .specio import export_search_dir
from .store import connect

log = logging.getLogger("bagrank.backtest")


def _variant(
    name: str,
    *,
    enter_top: int,
    slots: int,
    k: int = 1,
    lookback: int = 1,
    a: float = 0.0,
    b: float = 0.0,
) -> dict[str, Any]:
    return {
        "name": name,
        "enter_top": enter_top,
        "slots": slots,
        "k": k,
        "lookback": lookback,
        "a": a,
        "b": b,
    }


def default_variants() -> list[dict[str, Any]]:
    """Rank rules plus slower lookbacks; indicators on all-board and top-20."""
    rows: list[dict[str, Any]] = []
    for top, slots in ((5, 5), (10, 10), (20, 20), (0, 24)):
        for lookback in (1, 4, 8):
            rows.append(_variant("rank_up_down", enter_top=top, slots=slots, lookback=lookback))
        rows.append(_variant("top_k", enter_top=top, slots=slots))
        for k in (2, 3, 5):
            rows.append(_variant("rank_up_k", enter_top=top, slots=slots, k=k, lookback=max(k, 4)))
            rows.append(_variant("peak_giveback", enter_top=top, slots=slots, k=k))
        rows.append(_variant("agree_mom", enter_top=top, slots=slots, lookback=4))
        rows.append(_variant("wallets_mom", enter_top=top, slots=slots, lookback=4))
    for top, slots in ((0, 24), (20, 20)):
        rows.append(_variant("rank_ema", enter_top=top, slots=slots, k=14, a=8.0, b=21.0))
        rows.append(_variant("rank_ema", enter_top=top, slots=slots, k=14, a=12.0, b=26.0))
        rows.append(_variant("rank_ema", enter_top=top, slots=slots, k=14, a=21.0, b=55.0))
        rows.append(_variant("rank_rsi", enter_top=top, slots=slots, k=14, a=30.0, b=70.0))
        rows.append(_variant("rank_rsi", enter_top=top, slots=slots, k=14, a=25.0, b=75.0))
        rows.append(_variant("rank_breakout", enter_top=top, slots=slots, k=14, lookback=12))
        rows.append(_variant("rank_breakout", enter_top=top, slots=slots, k=14, lookback=24))
        rows.append(_variant("wallets_ema", enter_top=top, slots=slots, a=8.0, b=21.0))
        rows.append(_variant("wallets_ema", enter_top=top, slots=slots, a=21.0, b=55.0))
    return rows


DEFAULT_VARIANTS = default_variants()


def _ints(raw: str, fallback: list[int]) -> list[int]:
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or list(fallback)


def _hold_bars(min_hold_hours: int, step_hours: int) -> int:
    if min_hold_hours <= 0:
        return 0
    return max(1, int(math.ceil(float(min_hold_hours) / float(max(1, step_hours)))))


def _sharpe(equity: np.ndarray, bar_hours: float) -> float:
    if equity.size < 3:
        return 0.0
    prev = equity[:-1]
    nxt = equity[1:]
    ok = (prev > 1e-12) & np.isfinite(prev) & np.isfinite(nxt)
    if not bool(np.any(ok)):
        return 0.0
    rets = nxt[ok] / prev[ok] - 1.0
    sd = float(np.std(rets))
    if sd < 1e-12:
        return 0.0
    bars_year = (365.0 * 24.0) / max(float(bar_hours), 1.0)
    return float(np.mean(rets) / sd * np.sqrt(bars_year))


def run_strategies(
    panel,
    *,
    enter_top: int,
    max_slots: int,
    equity: float,
    gross_pct: float,
    max_pair_share: float,
    variants: list[dict[str, Any]] | None = None,
    exec_lag: int = 1,
    min_hold: int = 0,
    use_lev: int = 0,
    bar_hours: float = 1.0,
    step_hours: int = 1,
) -> list[dict[str, Any]]:
    variants = variants or DEFAULT_VARIANTS
    gross = max(0.01, min(1.0, float(gross_pct) / 100.0))
    rows: list[dict[str, Any]] = []
    for spec in variants:
        name = str(spec["name"])
        sid = int(NAME_TO_ID[name])
        top = int(spec.get("enter_top", enter_top))
        slots = max(1, int(spec.get("slots", max_slots)))
        tc, ts, tw, tl = build_targets(
            panel.rank,
            panel.side,
            panel.wallets,
            panel.hold_pct,
            panel.agreement,
            panel.mean_leverage,
            sid,
            top,
            slots,
            int(spec.get("k") or 1),
            int(spec.get("lookback") or 1),
            float(spec.get("a") or 0),
            float(spec.get("b") or 0),
        )
        ret, dd, trips, wr, fees, final, eq, avg_hold = simulate(
            panel.marks,
            tc,
            ts,
            tw,
            tl,
            0.0005,
            gross,
            float(equity),
            slots,
            float(max_pair_share),
            int(exec_lag),
            int(min_hold),
            int(use_lev),
            float(bar_hours),
        )
        span_h = 0.0
        if panel.n_times >= 2:
            span_h = float(panel.cycle_unix[-1] - panel.cycle_unix[0]) / 3600.0
        trips_day = (float(trips) / (span_h / 24.0)) if span_h >= 24.0 else float(trips)
        rows.append(
            {
                "strategy": name,
                "step_h": int(step_hours),
                "min_hold_h": int(round(float(min_hold) * float(bar_hours))),
                "enter_top": top,
                "slots": slots,
                "k": spec.get("k"),
                "lookback": spec.get("lookback"),
                "a": spec.get("a"),
                "b": spec.get("b"),
                "return_pct": round(float(ret), 4),
                "max_dd_pct": round(float(dd), 4),
                "sharpe": round(_sharpe(eq, bar_hours), 3),
                "round_trips": int(trips),
                "trips_per_day": round(trips_day, 2),
                "avg_hold_h": round(float(avg_hold), 2),
                "win_rate_pct": round(float(wr), 2),
                "fees": round(float(fees), 4),
                "final_equity": round(float(final), 2),
            }
        )
    rows.sort(key=lambda r: (-r["sharpe"], -r["return_pct"], r["max_dd_pct"]))
    return rows


def main(argv: list[str] | None = None) -> int:
    load_env()
    p = argparse.ArgumentParser(
        description=(
            "Search rank/feature strategies on local collector hours until Ctrl+C. "
            "Use --once for a single finite grid."
        )
    )
    p.add_argument("--db", type=Path, default=default_sqlite_path())
    p.add_argument("--venue", default="hyperliquid")
    p.add_argument(
        "--step-hours",
        default="1,4,8,24",
        help="Comma list of closed-bar lengths. 1 = collector hour.",
    )
    p.add_argument(
        "--min-hold-hours",
        default="8,24",
        help="Do not exit before this many hours (comma list).",
    )
    p.add_argument("--exec-lag", type=int, default=1, help="1 = fill next bar (no same-hour trade).")
    p.add_argument("--use-lev", action="store_true", help="Multiply size by collector mean leverage.")
    p.add_argument(
        "--enter-top",
        type=int,
        default=-1,
        help="Override universe depth for all variants. 0 = every coin on the board.",
    )
    p.add_argument("--slots", type=int, default=0, help="Override max concurrent names. 0 = per-variant default.")
    p.add_argument("--equity", type=float, default=1000.0)
    p.add_argument("--gross-pct", type=float, default=95.0)
    p.add_argument("--max-pair-share", type=float, default=0.70)
    p.add_argument(
        "--fetch-hl",
        action="store_true",
        help="Gap-fill missing hours from Hyperliquid 1h candles (collector prices are the default)",
    )
    p.add_argument("--strategy", default="", help="Run one name from: " + ",".join(STRATEGY_NAMES))
    p.add_argument("--json-out", type=Path, default=None, help="Optional extra JSON dump (CSV is the default).")
    p.add_argument("--top", type=int, default=8, help="Log this many top-Sharpe rows.")
    p.add_argument(
        "--once",
        action="store_true",
        help="Run the finite grid once and exit (old behavior).",
    )
    p.add_argument(
        "--search-dir",
        type=Path,
        default=None,
        help="Where to write CSV result folders (default data/bagrank/search).",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Convert an old search folder (results.jsonl / leaderboard.json) into sortable CSVs and exit.",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.export_csv:
        csv_path = export_search_dir(args.export_csv)
        log.info("Wrote sortable CSVs. Open %s (Sharpe), by_return.csv (profit), by_trades.csv.", csv_path)
        log.info("Copy one row into rank_live.csv, then: python run_rank.py")
        return 0
    if not args.db.exists():
        raise SystemExit(f"No local backup at {args.db}. Run: python backup_bagrank.py")
    steps = _ints(args.step_hours, [1, 4, 8, 24])
    holds = _ints(args.min_hold_hours, [8, 24])
    conn = connect(args.db)
    try:
        hourly = panel_from_sqlite(conn, venue=args.venue, step_hours=1)
        if hourly.n_times < 3:
            raise SystemExit(
                f"Need more collector hours (have {hourly.n_times}). Run backup_bagrank.py again later."
            )
        joined = fill_panel_from_collector(conn, hourly, venue=args.venue)
        if args.fetch_hl and int(joined.get("missing") or 0) > 0:
            sync_hl_gaps(conn, hourly)
        log.info(
            "Using collector prices | filled=%s missing_on_board=%s | next-bar=%s lev=%s",
            joined.get("filled"),
            joined.get("missing"),
            args.exec_lag,
            "on" if args.use_lev else "1x",
        )
        if not args.once:
            out_dir = args.search_dir or (args.db.parent / "search")
            return search_loop(
                hourly,
                out_dir=out_dir,
                equity=args.equity,
                gross_pct=args.gross_pct,
                max_pair_share=args.max_pair_share,
                seed=args.seed,
            )
        variants = [dict(v) for v in DEFAULT_VARIANTS]
        if args.strategy:
            if args.strategy not in NAME_TO_ID:
                raise SystemExit(f"Unknown strategy {args.strategy!r}")
            variants = [v for v in variants if v["name"] == args.strategy]
        if args.enter_top >= 0:
            for v in variants:
                v["enter_top"] = args.enter_top
        if args.slots > 0:
            for v in variants:
                v["slots"] = args.slots
        out_dir = args.search_dir or (args.db.parent / "search")
        store = ResultStore(out_dir, data_span(hourly))
        log.info("Writing CSV results to %s", store.directory)
        for step in steps:
            for hold_h in holds:
                for variant in variants:
                    spec = {
                        "engine": "named",
                        "family": variant["name"],
                        "name": variant["name"],
                        "step_h": int(step),
                        "min_hold_h": int(hold_h),
                        "enter_top": int(variant.get("enter_top") or 0),
                        "slots": max(1, int(variant.get("slots") or 5)),
                        "exec_lag": int(args.exec_lag),
                        "use_lev": 1 if args.use_lev else 0,
                        "k": int(variant.get("k") or 1),
                        "lookback": int(variant.get("lookback") or 1),
                        "a": float(variant.get("a") or 0),
                        "b": float(variant.get("b") or 0),
                        "mode": 0,
                        "zscore": 0,
                        "enter_th": 0.0,
                        "exit_th": 0.0,
                        "min_agree": 0.0,
                        "min_wallets": 0.0,
                        "max_abs_funding": 0.0,
                        "require_improve": 0,
                        "weights": [0.0] * N_FEAT,
                    }
                    row = run_one(
                        hourly,
                        spec,
                        equity=args.equity,
                        gross_pct=args.gross_pct,
                        max_pair_share=args.max_pair_share,
                    )
                    if row is not None:
                        store.append(row)
        store.write_csvs()
        ranked = list(store.by_sharpe)
        show = ranked[: max(1, args.top)]
        for r in show:
            log.info(
                "sharpe=%s ret=%s%% trips=%s family=%s tf=%sh hold=%sh",
                r.get("sharpe"),
                r.get("return_pct"),
                r.get("round_trips"),
                r.get("family"),
                r.get("step_h"),
                r.get("min_hold_h"),
            )
        log.info(
            "Wrote %s strategies. Open in Excel and sort as you like:\n  Sharpe  %s\n  Profit  %s\n  Trades  %s\nAll rows %s",
            store.n,
            store.csv_sharpe,
            store.csv_return,
            store.csv_trades,
            store.csv_all,
        )
        log.info("Copy one row into rank_live.csv, then: python run_rank.py")
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(ranked, indent=2), encoding="utf-8")
            log.info("Also wrote JSON %s", args.json_out)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
