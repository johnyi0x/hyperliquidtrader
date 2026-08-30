"""Multi-pair watchlist for RSI bot entry scanning."""

from __future__ import annotations

from dataclasses import dataclass

from .exchange_client import HyperliquidClient, Position
from .market_resolver import MarketSpec, resolve_market


@dataclass(frozen=True)
class PairSetup:
    coin_input: str
    market: MarketSpec
    leverage: int
    use_cross_margin: bool
    tp_pct: float
    sl_pct: float

    @property
    def api_coin(self) -> str:
        return self.market.api_coin

    @property
    def symbol(self) -> str:
        return self.market.symbol

    def position_coin_names(self) -> frozenset[str]:
        names = {self.market.api_coin, self.market.symbol}
        if self.market.perp_dex:
            names.add(f"{self.market.perp_dex}:{self.market.symbol}")
        return frozenset(names)


def parse_coin_list(coin: str, coins: tuple[str, ...] | list[str]) -> list[str]:
    """COINS when set, else single COIN."""
    if coins:
        out = [c.strip() for c in coins if str(c).strip()]
        if not out:
            raise ValueError("COINS is set but empty")
        return out
    raw = coin.strip()
    if not raw:
        raise ValueError("Set COIN or COINS")
    return [raw]


def entry_matches_position_coin(entry: PairSetup, position_coin: str) -> bool:
    return position_coin in entry.position_coin_names()


def find_position_in_watchlist(
    client: HyperliquidClient,
    watch: list[PairSetup],
    *,
    force: bool = False,
) -> tuple[PairSetup, Position] | None:
    """First open position whose coin is in the watchlist (watch order)."""
    if force:
        client.invalidate_user_state()
    matched: dict[str, PairSetup] = {}
    for entry in watch:
        for name in entry.position_coin_names():
            matched[name] = entry

    for state in client._iter_clearinghouse_states():
        for ap in state.get("assetPositions", []):
            pos = ap.get("position", ap)
            coin = str(pos.get("coin", ""))
            entry = matched.get(coin)
            if entry is None:
                continue
            szi = float(pos.get("szi", 0))
            if abs(szi) < 1e-12:
                continue
            entry_px = pos.get("entryPx")
            position = Position(
                side="long" if szi > 0 else "short",
                size=abs(szi),
                entry_price=float(entry_px) if entry_px is not None else None,
            )
            return entry, position
    return None


def activate_pair(client: HyperliquidClient, entry: PairSetup) -> None:
    # Skip meta round-trip; watchlist already has MarketSpec.
    client.apply_market(entry.market)


def activate_pair_for_trade(client: HyperliquidClient, entry: PairSetup) -> None:
    """Select pair and ensure exchange leverage matches this market's cap."""
    activate_pair(client, entry)
    client.set_leverage(entry.leverage, is_cross=entry.use_cross_margin)


def resolve_watchlist(
    info: object,
    coin_inputs: list[str],
    default_perp_dex: str | None,
) -> list[MarketSpec]:
    markets: list[MarketSpec] = []
    for raw in coin_inputs:
        dex = default_perp_dex
        if ":" in raw.strip():
            dex = None
        markets.append(resolve_market(info, raw, dex))
    return markets


def cleanup_watchlist_orders(
    client: HyperliquidClient,
    watch: list[PairSetup],
) -> None:
    seen: set[str] = set()
    for entry in watch:
        if entry.api_coin in seen:
            continue
        seen.add(entry.api_coin)
        client.cancel_all_orders_for_coin_named(entry.api_coin)
    client.sweep_orphan_orders()
