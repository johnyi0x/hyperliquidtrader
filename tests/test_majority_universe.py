"""Unit tests for majority pair-mode wiring (no Hyperliquid calls)."""

from __future__ import annotations

import unittest

from src.majority_universe import bucket_for_side
from src.pair_universe import (
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


if __name__ == "__main__":
    unittest.main()
