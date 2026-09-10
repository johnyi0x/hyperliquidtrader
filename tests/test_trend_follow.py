"""Trend-follow: ride list-side pumps, flatten dumps, skip chases."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.trend_follow import (
    TrendParams,
    TrendStore,
    TrendTrade,
    entry_reason,
    exit_reason,
    last_state,
    live_stop_px,
    simulate,
)


STEP_15M = 900_000
STEP_1H = 3_600_000
T0 = 1_700_000_000_000


def _bar(t: int, px: float, rng: float = 0.004) -> dict:
    return {
        "t": int(t),
        "o": float(px),
        "h": float(px) * (1.0 + rng),
        "l": float(px) * (1.0 - rng),
        "c": float(px),
    }


def _series(n: int, start: float, drift: float, step_ms: int, t0: int) -> list[dict]:
    out = []
    px = float(start)
    for i in range(n):
        px *= 1.0 + drift
        out.append(_bar(t0 + i * step_ms, px))
    return out


def _htf_from_exec(exec_c: list[dict]) -> list[dict]:
    """One 1h bar per four 15m bars, slightly below close so longs pass HTF."""
    out = []
    for i, c in enumerate(exec_c):
        if i % 4 != 3:
            continue
        px = float(c["c"]) * 0.995
        out.append(_bar(int(c["t"]), px, rng=0.003))
    return out


class TrendFollowTests(unittest.TestCase):
    def test_chase_skips_stretched_long(self) -> None:
        why = entry_reason(
            side=1,
            close=110.0,
            ema_v=100.0,
            atr_v=0.5,
            htf_ema=99.0,
            chase_pct=8.0,
            entry_buf_atr=0.35,
            min_atr_pct=0.20,
        )
        self.assertTrue(why.startswith("chase"), why)

    def test_htf_blocks_countertrend(self) -> None:
        why = entry_reason(
            side=1,
            close=101.0,
            ema_v=100.0,
            atr_v=0.5,
            htf_ema=102.0,
            chase_pct=8.0,
            entry_buf_atr=0.35,
            min_atr_pct=0.20,
        )
        self.assertEqual(why, "htf_against")

    def test_entry_ok_on_pullback_long(self) -> None:
        why = entry_reason(
            side=1,
            close=101.0,
            ema_v=100.0,
            atr_v=0.5,
            htf_ema=99.0,
            chase_pct=8.0,
            entry_buf_atr=0.35,
            min_atr_pct=0.20,
        )
        self.assertEqual(why, "")

    def test_exit_on_ema_break(self) -> None:
        why = exit_reason(
            side=1,
            close=98.0,
            ema_v=100.0,
            atr_v=0.4,
            extreme=110.0,
            k=3.0,
        )
        self.assertIn(why, ("ema", "trail"))

    def test_live_stop_is_tighter_of_ema_and_chandelier(self) -> None:
        sl = live_stop_px(1, ema_v=100.0, atr_v=1.0, extreme=110.0, k=3.0)
        # chandelier = 110 - 3 = 107, tighter vs EMA 100 is the higher stop 107
        self.assertAlmostEqual(sl, 107.0)
        sl2 = live_stop_px(1, ema_v=108.0, atr_v=1.0, extreme=110.0, k=3.0)
        self.assertAlmostEqual(sl2, 108.0)

    def test_pump_then_dump_exits_instead_of_holding(self) -> None:
        grind = _series(90, 100.0, 0.0015, STEP_15M, T0)
        pump = _series(
            40,
            float(grind[-1]["c"]),
            0.006,
            STEP_15M,
            int(grind[-1]["t"]) + STEP_15M,
        )
        dump = _series(
            25,
            float(pump[-1]["c"]),
            -0.035,
            STEP_15M,
            int(pump[-1]["t"]) + STEP_15M,
        )
        exec_c = grind + pump + dump
        htf = _htf_from_exec(exec_c)
        params = TrendParams(
            interval="15m",
            htf_interval="1h",
            ema_period=50,
            atr_period=14,
            atr_k=3.0,
            chase_pct=10.0,
            entry_buf_atr=0.35,
            cooldown_bars=2,
            min_atr_pct=0.10,
            taker_fee_pct=0.045,
        )
        stats = simulate(exec_c, htf, side=1, params=params)
        self.assertGreaterEqual(stats["trades"], 1)
        hold_px = float(exec_c[-1]["c"]) / float(exec_c[60]["c"]) - 1.0
        # Buy-hold through the dump is ugly; trail/EMA exit should not keep that hole.
        self.assertGreater(stats["return_pct"] / 100.0, hold_px - 0.02)
        st = last_state(exec_c, htf, params)
        self.assertIsNotNone(st)
        why = exit_reason(
            side=1,
            close=float(st["close"]),
            ema_v=float(st["ema"]),
            atr_v=float(st["atr"]),
            extreme=max(float(c["h"]) for c in pump),
            k=params.atr_k,
        )
        self.assertTrue(why, "dump close should trip EMA or trail")

    def test_store_roundtrip_exit_bar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trend.json"
            store = TrendStore(path)
            store.open_trade(
                TrendTrade(
                    coin="ZEC",
                    side="long",
                    entry_px=50.0,
                    extreme=51.0,
                    sl_px=48.0,
                    opened_bar_t=T0,
                    opened_at=1.0,
                    params={"ema_period": 50},
                )
            )
            store.close_trade("ZEC", T0 + 5 * STEP_15M)
            store2 = TrendStore(path)
            self.assertEqual(store2.last_exit_bar.get("ZEC"), T0 + 5 * STEP_15M)
            self.assertNotIn("ZEC", store2.trades)


if __name__ == "__main__":
    unittest.main()
