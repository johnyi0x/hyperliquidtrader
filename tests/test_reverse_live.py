import unittest
from types import SimpleNamespace


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

    def test_live_row_loads_reverse_off(self) -> None:
        from bagrank.specio import default_live_csv, load_csv_row, spec_from_row

        spec = spec_from_row(load_csv_row(default_live_csv()))
        self.assertEqual(spec["family"], "composite")
        self.assertEqual(spec["reverse"], 0)
        self.assertEqual(spec_from_row({"family": "composite", "engine": "score"})["reverse"], 0)

    def test_strategy_change_marks_rotate_flat(self) -> None:
        from bagrank.live import mark_strategy_rotation, strategy_fingerprint

        old = {
            "family": "zscore_composite",
            "name": "zscore_composite",
            "engine": "score",
            "mode": 1,
            "step_h": 4,
            "min_hold_h": 2,
            "run_id": "run_old",
            "weights": [0.1, 0.2],
        }
        new = dict(old)
        new["family"] = "anti_rank"
        new["name"] = "anti_rank"
        new["mode"] = 0
        new["run_id"] = "run_new"
        state = {"opened": {}, "strategy_fp": strategy_fingerprint(old)}
        self.assertTrue(mark_strategy_rotation(state, new))
        self.assertTrue(state["rotate_flat"])
        self.assertEqual(state["strategy_fp"], strategy_fingerprint(new))
        self.assertFalse(mark_strategy_rotation({"opened": {}, "strategy_fp": strategy_fingerprint(new)}, new))

    def test_rotate_flat_closes_old_before_new_open(self) -> None:
        from bagrank.live import PaperBook, _apply_paper

        book = PaperBook(1000.0)
        book.positions["SOL"] = {"side": "long", "size": 1.0, "entry": 100.0}
        state = {
            "opened": {"SOL": 1.0},
            "rotate_flat": True,
            "strategy_fp": "anti_rank",
        }
        desired = [
            {"coin": "ETH", "side": "short", "size": 0.5, "px": 2000.0, "notional": 1000.0}
        ]
        _apply_paper(book, desired, state, min_hold_h=24, marks={"SOL": 100.0, "ETH": 2000.0}, slots=1)
        self.assertEqual(book.positions, {})
        self.assertFalse(state.get("rotate_flat"))
        _apply_paper(book, desired, state, min_hold_h=0, marks={"SOL": 100.0, "ETH": 2000.0}, slots=1)
        self.assertEqual(list(book.positions), ["ETH"])
        self.assertEqual(book.positions["ETH"]["side"], "short")

    def test_open_stops_when_other_position_explains_margin(self) -> None:
        from bagrank import live as live_mod

        class _Pos:
            def __init__(self, side: str):
                self.side = side

        class _Client:
            sz_decimals = 4
            info = None

            def get_account_value(self, force=False):
                return 175.0

            def set_leverage(self, lev):
                return None

            def cancel_entry_orders_for_coin(self):
                return None

            def fetch_open_positions(self, force=False):
                return True, [("SOL", _Pos("long"))]

            def l2_book(self):
                return {"levels": [[{"px": "2000"}], [{"px": "2001"}]]}

        class _Ex:
            def execute_mid_open(self, is_buy, sz):
                raise RuntimeError("Insufficient margin to place order. asset=1")

        def _fake_resolve(info, coin):
            return SimpleNamespace(coin=coin, symbol=coin)

        def _fake_executor(client):
            return _Ex()

        client = _Client()
        client.apply_market = lambda market: None
        saved_executor = live_mod._order_executor
        saved_mids = live_mod._mids
        live_mod._order_executor = _fake_executor
        live_mod._mids = lambda client: {"ETH": 2000.0}
        from src import market_resolver

        saved = market_resolver.resolve_market
        market_resolver.resolve_market = _fake_resolve
        try:
            ok = live_mod._live_maker_open(client, "ETH", "short", 0.5, 8)
        finally:
            market_resolver.resolve_market = saved
            live_mod._order_executor = saved_executor
            live_mod._mids = saved_mids
        self.assertFalse(ok)
