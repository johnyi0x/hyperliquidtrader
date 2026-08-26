"""Apply the latest backtest winner to cloud settings (commit + deploy).

Reads data-local/backtest_latest.json (written by run_backtest.py), then:
  1. Writes profit-meta-follower/cloud_tuned.json  (tracked in git → Railway picks it up)
  2. Updates _TRADE knobs in config_profiles.py (signal + indicator + strategy name)

Usage (after a successful backtest):

  python profit-meta-follower/run_backtest.py --days 7 --max-combos 500
  python profit-meta-follower/apply_cloud_tune.py
  python profit-meta-follower/apply_cloud_tune.py --reverse   # invert winner (fade ↔ follow)

Winner strategy sets:
  BACKTEST_LIVE_STRATEGY = strategy name (live runner executes identical pick_trade_votes)
  BASKET_FILTER_MODE = holder | off from *_holders vs *_all
  TUNABLE_KEYS + INDICATOR_KEYS + MTF_KEYS + SWING_KEYS
  LIVE_CANDLE_SEED = True only when strategy needs 1m/15m/1h candles

--reverse flips the winner's direction:
  TRADE_MODE follow↔reverse (crowd refine / dump / vol / …)
  SWING_META_MODE follow↔reverse (swing_meta_*)
  MTF_META_MODE follow↔reverse (mtf_meta_*)
  MTF_PRESET long↔short pairs when a known opposite exists
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from pmf.bt_tune import INDICATOR_KEYS, MTF_KEYS, SWING_KEYS, TUNABLE_KEYS, load_tuned
from pmf.strategy_spec import parse_strategy

CLOUD_TUNED_PATH = ROOT / "cloud_tuned.json"
PROFILES_PATH = ROOT / "config_profiles.py"
DEFAULT_SOURCE = ROOT / "data-local" / "backtest_latest.json"

APPLY_KEYS = (
    set(TUNABLE_KEYS)
    | set(INDICATOR_KEYS)
    | set(MTF_KEYS)
    | set(SWING_KEYS)
    | {
        "BASKET_FILTER_MODE",
        "BACKTEST_LIVE_STRATEGY",
        "TRADE_MODE",
        "STICKY_BOOK_SLOTS",
        "LIVE_CANDLE_SEED",
        "LIVE_CANDLES_PER_TICK",
        "LIVE_CANDLE_COOLDOWN_S",
        "LIVE_CANDLE_BARS_1M",
        "LIVE_CANDLE_BARS_15M",
        "LIVE_CANDLE_BARS_1H",
        "MTF_EXEC_IV",
        "MTF_WEIGHT_POWER",
    }
)

# Known MTF preset long↔short mirrors (only pairs that exist in MTF_PRESETS).
_MTF_PRESET_FLIP: dict[str, str] = {
    "rsi_long_30": "rsi_short_70",
    "rsi_long_35": "rsi_short_70",
    "rsi_short_70": "rsi_long_30",
    "ema_x_long": "ema_x_short",
    "ema_x_short": "ema_x_long",
    "dump_bounce": "pump_fade",
    "pump_fade": "dump_bounce",
}


def filter_mode_for_strategy(strategy: str) -> str:
    return parse_strategy(strategy).filter_mode


def _flip_follow_reverse(value: Any, *, default: str = "follow") -> str:
    cur = str(value if value is not None else default).strip().lower()
    if cur in ("reverse", "invert", "fade"):
        return "follow"
    return "reverse"


def reverse_winner_params(params: dict[str, Any], *, strategy: str) -> dict[str, Any]:
    """Exact direction invert of a backtest winner for live apply."""
    out = dict(params)
    spec = parse_strategy(strategy)
    flips: list[str] = []

    # Crowd refine / price-gate styles use TRADE_MODE for side invert.
    if spec.style in ("refine", "direct", "flow", "logged") or spec.gate:
        before = out.get("TRADE_MODE", "follow")
        out["TRADE_MODE"] = _flip_follow_reverse(before)
        flips.append(f"TRADE_MODE {before}->{out['TRADE_MODE']}")

    if spec.style == "swing_meta" or "SWING_META_MODE" in out:
        before = out.get("SWING_META_MODE", "follow")
        out["SWING_META_MODE"] = _flip_follow_reverse(before)
        flips.append(f"SWING_META_MODE {before}->{out['SWING_META_MODE']}")

    if spec.style == "mtf_meta" or "MTF_META_MODE" in out:
        before = out.get("MTF_META_MODE", "follow")
        out["MTF_META_MODE"] = _flip_follow_reverse(before)
        flips.append(f"MTF_META_MODE {before}->{out['MTF_META_MODE']}")
        preset = str(out.get("MTF_PRESET") or "").strip()
        if preset in _MTF_PRESET_FLIP:
            flipped = _MTF_PRESET_FLIP[preset]
            flips.append(f"MTF_PRESET {preset}->{flipped}")
            out["MTF_PRESET"] = flipped

    out["_REVERSE_APPLIED"] = True
    out["_REVERSE_FLIPS"] = flips
    return out


def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        if v == int(v) and abs(v) >= 1.0:
            return f"{float(v):.1f}"
        return repr(float(v))
    if isinstance(v, int):
        return str(v)
    return repr(v)


def _patch_trade_block(text: str, params: dict[str, Any]) -> str:
    """Replace or insert apply keys inside the _TRADE dict."""
    lines = text.splitlines(keepends=True)
    in_trade = False
    depth = 0
    out: list[str] = []
    trade_keys = APPLY_KEYS & set(params.keys())
    seen: set[str] = set()
    indent = "    "

    for line in lines:
        if not in_trade and re.match(r"^_TRADE:\s*dict\s*=\s*\{", line):
            in_trade = True
            depth = line.count("{") - line.count("}")
            out.append(line)
            continue
        if in_trade:
            depth += line.count("{") - line.count("}")
            m = re.match(r'^(\s*)"([A-Z_0-9]+)"\s*:\s*.+,?\s*$', line)
            if m:
                indent = m.group(1)
                key = m.group(2)
                if key in trade_keys:
                    seen.add(key)
                    val = _fmt_value(params[key])
                    out.append(f'{indent}"{key}": {val},\n')
                else:
                    out.append(line)
            elif depth <= 0:
                # Closing brace of _TRADE — insert any missing apply keys first.
                missing = sorted(trade_keys - seen)
                for key in missing:
                    out.append(f'{indent}"{key}": {_fmt_value(params[key])},\n')
                    seen.add(key)
                out.append(line)
                in_trade = False
            else:
                out.append(line)
            continue
        out.append(line)
    return "".join(out)


def _ensure_strategy_comment(
    text: str,
    strategy: str,
    metrics: dict[str, Any],
    *,
    reversed_apply: bool = False,
) -> str:
    tag = " REVERSED" if reversed_apply else ""
    banner = (
        f"# Cloud _TRADE last tuned: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} "
        f"strategy={strategy}{tag} score={metrics.get('score', '?')} ret={metrics.get('return_pct', '?')}%"
    )
    if "Cloud _TRADE last tuned:" in text:
        text = re.sub(r"# Cloud _TRADE last tuned:.*\n", banner + "\n", text, count=1)
    else:
        text = text.replace(
            "# Live trading — cloud only. Leave alone while Railway is live.\n",
            "# Live trading — cloud only. Leave alone while Railway is live.\n" + banner + "\n",
            1,
        )
    return text


def apply_from_payload(
    payload: dict[str, Any],
    *,
    write_profiles: bool = True,
    reverse: bool = False,
) -> None:
    params = payload.get("params")
    if not isinstance(params, dict) or not params:
        raise SystemExit("backtest file has no params — run backtest first")
    strategy = str(payload.get("strategy") or "cloud_holders")
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    spec = parse_strategy(strategy)
    params = dict(params)
    if reverse:
        params = reverse_winner_params(params, strategy=strategy)
        flips = params.pop("_REVERSE_FLIPS", [])
        params.pop("_REVERSE_APPLIED", None)
        print("Reverse apply — flipped:")
        for line in flips:
            print(f"  {line}")
        if not flips:
            print("  (no mode keys found; set TRADE_MODE=reverse as fallback)")
            params["TRADE_MODE"] = "reverse"
    params["BASKET_FILTER_MODE"] = spec.filter_mode
    params["BACKTEST_LIVE_STRATEGY"] = spec.name
    # Candle seed only for price-gated strategies (1m+15m+1h, rate-limited).
    params["LIVE_CANDLE_SEED"] = bool(spec.needs_candles)
    if spec.needs_candles:
        params.setdefault("LIVE_CANDLES_PER_TICK", 1)
        params.setdefault("LIVE_CANDLE_COOLDOWN_S", 8.0)
        params.setdefault("LIVE_CANDLE_BARS_1M", 120)
        params.setdefault("LIVE_CANDLE_BARS_15M", 64)
        params.setdefault("LIVE_CANDLE_BARS_1H", 48)
    if spec.style == "mtf_meta":
        # Original MTF needs ~80 exec bars + 60 HTF bars.
        params["LIVE_CANDLES_PER_TICK"] = max(int(params.get("LIVE_CANDLES_PER_TICK") or 1), 2)
        params["LIVE_CANDLE_BARS_1M"] = max(int(params.get("LIVE_CANDLE_BARS_1M") or 0), 160)
        params["LIVE_CANDLE_BARS_15M"] = max(int(params.get("LIVE_CANDLE_BARS_15M") or 0), 80)
        params["LIVE_CANDLE_BARS_1H"] = max(int(params.get("LIVE_CANDLE_BARS_1H") or 0), 80)
        params.setdefault("MTF_EXEC_IV", "1m")
        params.setdefault("MTF_WEIGHT_POWER", 0.5)
        params.setdefault("STICKY_BOOK_SLOTS", True)
    if spec.style == "swing_meta":
        # Swing timing reads RSI/EMA/returns off 1m (+15m/1h context) and owns exits.
        params["LIVE_CANDLES_PER_TICK"] = max(int(params.get("LIVE_CANDLES_PER_TICK") or 1), 2)
        params["LIVE_CANDLE_BARS_1M"] = max(int(params.get("LIVE_CANDLE_BARS_1M") or 0), 160)
        params["LIVE_CANDLE_BARS_15M"] = max(int(params.get("LIVE_CANDLE_BARS_15M") or 0), 64)
        params["LIVE_CANDLE_BARS_1H"] = max(int(params.get("LIVE_CANDLE_BARS_1H") or 0), 48)
        params.setdefault("STICKY_BOOK_SLOTS", True)
        for key, default in (
            ("SWING_META_MODE", "follow"),
            ("SWING_ENTRY", "rsi_dip"),
            ("SWING_TF", "15m"),
            ("SWING_RSI_BUY", 35.0),
            ("SWING_RSI_SELL", 65.0),
            ("SWING_BAND_PCT", 0.008),
            ("SWING_BREAK_PCT", 0.010),
            ("SWING_LOOKBACK_S", 1800.0),
            ("SWING_TP_PCT", 1.2),
            ("SWING_SL_PCT", 1.8),
            ("SWING_MAX_HOLD_S", 14400.0),
            ("SWING_EXIT_RSI", 0.0),
            ("SWING_REENTRY_S", 900.0),
        ):
            params.setdefault(key, default)

    out = dict(payload)
    out["params"] = params
    out["strategy"] = spec.name
    out["applied_at"] = time.time()
    out["applied_to"] = "cloud"
    out["reversed"] = bool(reverse)
    CLOUD_TUNED_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {CLOUD_TUNED_PATH}")

    if write_profiles and PROFILES_PATH.exists():
        text = PROFILES_PATH.read_text(encoding="utf-8")
        text = _patch_trade_block(text, params)
        text = _ensure_strategy_comment(text, strategy, metrics, reversed_apply=reverse)
        PROFILES_PATH.write_text(text, encoding="utf-8")
        print(f"Updated {PROFILES_PATH} (_TRADE params)")
        updated = ", ".join(f"{k}={params[k]}" for k in sorted(params) if k in APPLY_KEYS)
        print(f"  {updated}")
    print(
        f"  strategy={spec.name} style={spec.style} gate={spec.gate or '-'} "
        f"BASKET_FILTER_MODE={params['BASKET_FILTER_MODE']} candles={'on' if spec.needs_candles else 'off'}"
        f"{' REVERSED' if reverse else ''}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply latest backtest winner to cloud config")
    ap.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE),
        help="backtest_latest.json from run_backtest.py (default: data-local/backtest_latest.json)",
    )
    ap.add_argument("--no-profiles", action="store_true", help="Only write cloud_tuned.json")
    ap.add_argument(
        "--reverse",
        action="store_true",
        help=(
            "Apply the exact reverse of the backtest winner: flip TRADE_MODE / "
            "SWING_META_MODE / MTF_META_MODE (follow↔reverse), and flip known "
            "MTF_PRESET long↔short pairs. Same strategy name + other knobs."
        ),
    )
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_absolute():
        src = ROOT / src
    payload = load_tuned(src)
    if payload is None:
        print(f"No backtest results at {src}")
        print("Run:  python profit-meta-follower/run_backtest.py --days 7 --max-combos 120")
        sys.exit(1)

    apply_from_payload(payload, write_profiles=not args.no_profiles, reverse=args.reverse)
    print("\nNext: git add/commit/push, then redeploy Railway (PMF_PROFILE=cloud).")


if __name__ == "__main__":
    main()
