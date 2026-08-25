"""Crowd strategy backtest from research gather data (Numba PnL, param search).

Loads full local gather under data-local/research/:
  books.jsonl + crowd_ticks.jsonl  — wallet books / compact votes
  marks.jsonl                      — mark, funding, oi, basis, day_vol
  candles/<coin>/{1m,15m,1h}.jsonl — OHLCV for dump/trend/vol/rsi gates

Live and backtest share pmf.strategy_exec.pick_trade_votes + pmf.price_engine
(identical refine + market filters + price gates). Fee model: 0.05%/side.

Search is two-stage by default: a wide, shallow sweep of every knob across all
strategies, then a dense search around the winner of the best --top-k strategies.

  python profit-meta-follower/run_backtest.py
  python profit-meta-follower/run_backtest.py --days 7 --max-combos 500 --all-strategies
  python profit-meta-follower/run_backtest.py --meta-timing --days 7 --max-combos 300
  python profit-meta-follower/run_backtest.py --single-stage      # old behaviour
  python profit-meta-follower/apply_cloud_tune.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252 and crash on arrows/× in the progress output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import config as cfg
from pmf.backtest import run_backtest_suite


def _data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir:
        p = Path(args.data_dir)
        return p if p.is_absolute() else ROOT / p
    import os

    prof = os.environ.get("PMF_PROFILE", getattr(cfg, "PMF_PROFILE", "local"))
    return ROOT / f"data-{prof}"


MTF_MIN_BARS = 60  # src.mtf.prepare_interval_biases drops intervals below this


def _print_results(results, ds) -> None:
    if ds is not None:
        eng = getattr(ds, "price_engine", None)
        n_candle = 0
        n_aux = 0
        mtf_ready = 0
        if eng is not None and ds.n_ticks:
            t_end = float(ds.ts[-1])
            for c in ds.index_coin:
                if any(eng.candle_span_s(c, iv) > 0 for iv in ("1m", "15m", "1h")):
                    n_candle += 1
                if eng.day_vol_at(c, t_end) > 0 or abs(eng.funding_at(c, t_end)) > 0:
                    n_aux += 1
                htf = sum(
                    1
                    for iv in ("15m", "1h")
                    if len(eng.candles_as_hl_dicts(c, iv, t_end, max_bars=200)) >= MTF_MIN_BARS
                )
                if htf >= 1:  # 1m always qualifies; consensus needs ≥2 timeframes
                    mtf_ready += 1
        print(
            f"Dataset: ticks={ds.n_ticks} coins={ds.n_coins} "
            f"span={ds.span_days:.2f}d days={','.join(ds.day_labels)} source={ds.source}"
        )
        print(
            f"Crowd: cloud_listed={ds.live_basket_target} cloud_basket={len(ds.cloud_basket_addrs)} "
            f"live_holders={len(ds.live_holder_addrs)}/{ds.live_basket_target} "
            f"labeled_holders={len(ds.holder_addrs)} research_pool={ds.pool_size}"
        )
        print(
            f"PriceEngine: candle_coins={n_candle}/{ds.n_coins} "
            f"mark_aux_coins={n_aux}/{ds.n_coins} (funding/oi/basis/day_vol) "
            f"mtf_ready={mtf_ready}/{ds.n_coins}"
        )
        if mtf_ready < max(1, ds.n_coins // 4):
            print(
                f"  NOTE: multi-timeframe consensus needs ≥{MTF_MIN_BARS} bars on 15m or 1h "
                f"(≈15h / 2.5d of gather). mtf_meta_holders will show 0 trips until then; "
                f"swing_meta_holders and the crowd strategies still trade."
            )
    if not results:
        print("\nNo results — need books.jsonl under data-local/research/.")
        return
    print(f"\nRanked by profit (end open legs force-closed at last price) | {len(results)} configs\n")
    for i, r in enumerate(results[:15], 1):
        p = ", ".join(f"{k}={v}" for k, v in sorted(r.params.items()))
        print(
            f"  #{i:02d} [{r.strategy}] ret={r.return_pct:+.2f}% score={r.score:+.2f} "
            f"dd={r.max_dd_pct:.1f}% trips={r.round_trips} end_open={r.open_legs} "
            f"tpd={r.trades_per_day:.2f} win={r.win_rate_pct:.0f}%"
        )
        print(f"       {p}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Numba crowd backtest on research gather data")
    ap.add_argument("--days", type=int, default=7, help="Use last N calendar days (default 7)")
    ap.add_argument("--data-dir", type=str, default="", help="Data folder (default data-{PMF_PROFILE})")
    ap.add_argument("--max-combos", type=int, default=96, help="Random/grid combos per strategy (default 96; 500×16 is multi-hour)")
    ap.add_argument(
        "--all-strategies",
        action="store_true",
        help=(
            "Search all strategies including extra styles. Default already includes "
            "holders/all, dump/btcdump, and the 4 meta-timing strategies "
            "(mtf/swing × holders/all). "
            "Data load is the same either way; this only multiplies search combos."
        ),
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Also run apply_cloud_tune (write cloud_tuned.json + config_profiles)",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Stage-2 finalists: how many strategies get the deep search (default 2)",
    )
    ap.add_argument(
        "--coarse-combos",
        type=int,
        default=0,
        help="Stage-1 combos per strategy (default 25%% of --max-combos, min 8)",
    )
    ap.add_argument(
        "--single-stage",
        action="store_true",
        help="Disable two-stage search: run the full combo budget on every strategy",
    )
    ap.add_argument(
        "--meta-timing",
        action="store_true",
        help=(
            "Deep-search only the 4 meta-timing strategies: "
            "mtf_meta_holders, mtf_meta_all, swing_meta_holders, swing_meta_all "
            "(multi-candle / indicator entry-exit; holder filter on and off). "
            "Ignores all other strategies. Off by default. Overrides --all-strategies."
        ),
    )
    args = ap.parse_args()

    data_dir = _data_dir(args)
    research = data_dir / "research"
    if not research.exists():
        print(f"No research folder at {research}")
        print("Run local gather until books.jsonl appears.")
        sys.exit(1)

    from pmf.bt_replay import STRATEGY_REPLAYERS
    from pmf.bt_tune import LIVE_STRATEGIES, META_TIMING_STRATEGIES, sim_size_from_cfg

    margin_frac, lev = sim_size_from_cfg(cfg)
    if args.meta_timing:
        strats = [s for s in META_TIMING_STRATEGIES if s in STRATEGY_REPLAYERS]
        # All meta-timing strategies get the full --max-combos budget.
        two_stage = False
        top_k = len(strats) or 1
    else:
        strats = list(STRATEGY_REPLAYERS.keys()) if args.all_strategies else list(LIVE_STRATEGIES)
        two_stage = not args.single_stage
        top_k = max(1, args.top_k)
    coarse = max(0, args.coarse_combos) or None
    print(
        f"Backtest data={data_dir} days={args.days} fee=0.05%/side combos={args.max_combos} "
        f"size={margin_frac:.1%}×{lev:g}x strategy={'+'.join(strats)}"
    )
    if args.meta_timing:
        print(
            f"Search plan: meta-timing ONLY — deep search "
            f"{max(8, args.max_combos)} combos × {len(strats)} strategies "
            f"(mtf/swing × holders/all); other strategies ignored"
        )
    elif two_stage and len(strats) > top_k:
        stage1 = coarse or max(8, round(max(8, args.max_combos) * 0.25))
        print(
            f"Search plan: stage 1 = wide screen, {stage1} combos × {len(strats)} strategies "
            f"→ stage 2 = deep dive on top {top_k} (up to {max(8, args.max_combos)} combos each)"
        )
    print("Loading research data then searching (progress on the next line)...")

    ds, results = run_backtest_suite(
        cfg,
        data_dir,
        max_days=max(1, args.days),
        max_combos=max(8, args.max_combos),
        apply_path=None,
        strategies=strats,
        two_stage=two_stage,
        top_k=top_k,
        coarse_combos=coarse,
    )

    if ds is None:
        books = list(research.glob("*/books.jsonl"))
        ticks = list(research.glob("*/crowd_ticks.jsonl"))
        print(f"\nNo usable timeline yet. books files={len(books)} tick files={len(ticks)}")
        sys.exit(1)

    _print_results(results, ds)

    if not results:
        sys.exit(1)

    from pmf.bt_tune import save_tuned

    latest = data_dir / "backtest_latest.json"
    save_tuned(latest, results[0], dataset=ds)
    print(
        f"\nSaved top profit result "
        f"(ret={results[0].return_pct:+.2f}% trips={results[0].round_trips} end_open={results[0].open_legs}) "
        f"→ {latest}"
    )
    print("Apply to cloud:  python profit-meta-follower/apply_cloud_tune.py")
    if args.apply:
        from apply_cloud_tune import apply_from_payload
        from pmf.bt_tune import load_tuned

        payload = load_tuned(latest)
        if payload:
            apply_from_payload(payload)


if __name__ == "__main__":
    main()
