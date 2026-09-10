"""Unit tests for majority pair-mode wiring (no Hyperliquid calls)."""

from __future__ import annotations

import unittest

from pathlib import Path
from types import SimpleNamespace

from src.majority_universe import (
    _index_markets,
    _match_market,
    bucket_for_side,
    majority_resolve_kwargs,
)
from src.pair_universe import (
    VolumePair,
    is_auto_pair_mode,
    is_majority_mode,
    is_mover_mode,
    is_side_locked_mode,
    valid_pair_modes,
)


class MajorityModeTests(unittest.TestCase):
    def test_mode_flags(self) -> None:
        self.assertTrue(is_majority_mode("majority"))
        self.assertTrue(is_majority_mode("wallet_majority"))
        self.assertTrue(is_majority_mode("meta_follower"))
        self.assertFalse(is_majority_mode("top_movers"))
        self.assertTrue(is_mover_mode("top_movers"))
        self.assertFalse(is_mover_mode("majority"))
        self.assertTrue(is_auto_pair_mode("majority"))
        self.assertTrue(is_side_locked_mode("majority"))
        self.assertTrue(is_side_locked_mode("top_movers"))
        self.assertFalse(is_side_locked_mode("top_volume"))
        self.assertIn("majority", valid_pair_modes())
        self.assertIn("top_movers", valid_pair_modes())

    def test_bucket_maps_wallet_side(self) -> None:
        self.assertEqual(bucket_for_side("long"), "gainer")
        self.assertEqual(bucket_for_side("LONG"), "gainer")
        self.assertEqual(bucket_for_side("short"), "loser")
        self.assertEqual(bucket_for_side("short "), "loser")

    def test_majority_kwargs_include_xyz_even_if_native(self) -> None:
        cfg = SimpleNamespace(
            XYZ_PAIR_MODE="native",
            INCLUDE_XYZ_PAIRS=False,
            MIN_DAY_NOTIONAL_USD=1_000_000,
            MIN_MAX_LEVERAGE=3,
        )
        kw = majority_resolve_kwargs(cfg, data_dir=Path("data"))
        self.assertEqual(kw["majority_xyz_mode"], "include")
        self.assertEqual(kw["majority_min_day_notional"], 0.0)

    def test_match_hip3_stock_names(self) -> None:
        p = VolumePair(
            api_coin="xyz:NVDA",
            symbol="NVDA",
            perp_dex="xyz",
            max_leverage=3,
            day_ntl_vlm=10.0,
            sz_decimals=4,
            only_isolated=False,
        )
        idx = _index_markets([p])
        self.assertIsNotNone(_match_market("xyz:NVDA", idx))
        self.assertIsNotNone(_match_market("xyz:nvda", idx))
        self.assertIsNotNone(_match_market("NVDA", idx))


if __name__ == "__main__":
    unittest.main()
