import unittest


class ReverseLiveTests(unittest.TestCase):
    def test_reverse_swaps_the_order_side(self) -> None:
        from bagrank.live import reverse_hold_sides

        holds = [
            {"coin": "BTC", "side": "long", "px": 1.0},
            {"coin": "ETH", "side": "short", "px": 2.0},
        ]
        out = reverse_hold_sides(holds)
        self.assertEqual([row["side"] for row in out], ["short", "long"])
        self.assertEqual([row["coin"] for row in out], ["BTC", "ETH"])
        self.assertEqual(holds[0]["side"], "long")

    def test_live_row_loads_reverse_on(self) -> None:
        from bagrank.specio import default_live_csv, load_csv_row, spec_from_row

        spec = spec_from_row(load_csv_row(default_live_csv()))
        self.assertEqual(spec["family"], "oi_thrust")
        self.assertEqual(spec["reverse"], 1)
        self.assertEqual(spec_from_row({"family": "oi_thrust", "engine": "score"})["reverse"], 0)
