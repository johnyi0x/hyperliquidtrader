import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock


class LeverageModeTests(unittest.TestCase):
    def test_isolated_only_coin_uses_isolated_leverage(self) -> None:
        from src.exchange_client import HyperliquidClient

        client = SimpleNamespace(only_isolated=True)
        self.assertFalse(HyperliquidClient.resolve_leverage_is_cross(client))
        self.assertFalse(HyperliquidClient.resolve_leverage_is_cross(client, True))
        self.assertFalse(HyperliquidClient.resolve_leverage_is_cross(client, False))

    def test_normal_coin_defaults_to_cross(self) -> None:
        from src.exchange_client import HyperliquidClient

        client = SimpleNamespace(only_isolated=False)
        self.assertTrue(HyperliquidClient.resolve_leverage_is_cross(client))
        self.assertTrue(HyperliquidClient.resolve_leverage_is_cross(client, True))
        self.assertFalse(HyperliquidClient.resolve_leverage_is_cross(client, False))

    def test_set_leverage_sends_isolated_for_only_isolated_market(self) -> None:
        from src.exchange_client import HyperliquidClient

        client = MagicMock(spec=HyperliquidClient)
        client.coin = "xyz:EWY"
        client.max_leverage = 20
        client.only_isolated = True
        client.perp_dex = "xyz"
        client.exchange = MagicMock()
        client.exchange.info.name_to_coin = {"xyz:EWY": "xyz:EWY"}
        client.exchange.update_leverage.return_value = {"status": "ok"}
        client.logger = MagicMock()
        client.invalidate_user_state = MagicMock()
        client._ensure_sdk_asset_maps = MagicMock()
        client.resolve_leverage_is_cross = (
            lambda is_cross=None: HyperliquidClient.resolve_leverage_is_cross(client, is_cross)
        )

        HyperliquidClient.set_leverage(client, 6)
        client.exchange.update_leverage.assert_called_once_with(6, "xyz:EWY", is_cross=False)

    def test_set_leverage_raises_when_exchange_rejects(self) -> None:
        from src.exchange_client import HyperliquidClient

        client = MagicMock(spec=HyperliquidClient)
        client.coin = "ETH"
        client.max_leverage = 50
        client.only_isolated = False
        client.perp_dex = ""
        client.exchange = MagicMock()
        client.exchange.info.name_to_coin = {"ETH": "ETH"}
        client.exchange.update_leverage.return_value = {"status": "err", "response": "nope"}
        client.logger = MagicMock()
        client.invalidate_user_state = MagicMock()
        client._ensure_sdk_asset_maps = MagicMock()
        client.resolve_leverage_is_cross = (
            lambda is_cross=None: HyperliquidClient.resolve_leverage_is_cross(client, is_cross)
        )

        with self.assertRaises(RuntimeError):
            HyperliquidClient.set_leverage(client, 8)

    def test_live_open_aborts_when_leverage_update_fails(self) -> None:
        from bagrank import live as live_mod
        from src import market_resolver

        class _Client:
            sz_decimals = 4
            info = None
            only_isolated = True

            def apply_market(self, market):
                return None

            def get_account_value(self, force=False):
                return 150.0

            def set_leverage(self, lev, is_cross=None):
                raise RuntimeError("Leverage update failed")

            def cancel_entry_orders_for_coin(self):
                return None

            def l2_book(self):
                return {"levels": [[{"px": "150"}], [{"px": "151"}]]}

        saved_resolve = market_resolver.resolve_market
        saved_mids = live_mod._mids
        market_resolver.resolve_market = lambda info, coin: SimpleNamespace(
            coin=coin, symbol=coin, api_coin=coin, only_isolated=True
        )
        live_mod._mids = lambda client: {"xyz:EWY": 150.0}
        opened = {"called": False}

        def _boom(*args, **kwargs):
            opened["called"] = True
            raise AssertionError("must not open after leverage failure")

        saved_ex = live_mod._order_executor
        live_mod._order_executor = _boom
        try:
            ok = live_mod._live_maker_open(_Client(), "xyz:EWY", "short", 5.0, 6)
        finally:
            market_resolver.resolve_market = saved_resolve
            live_mod._mids = saved_mids
            live_mod._order_executor = saved_ex
        self.assertFalse(ok)
        self.assertFalse(opened["called"])


if __name__ == "__main__":
    unittest.main()
