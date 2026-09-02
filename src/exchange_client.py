"""Hyperliquid connection, positions, candles, leverage."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import eth_account
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

from .candles import INTERVAL_MS
from .candle_book import CandleBook
from .hl_rate_limit import (
    RequestGuard,
    ThrottledInfo,
    bind_throttled_exchange_post,
    default_shared_budget,
)
from .market_resolver import (
    MarketSpec,
    find_perp_asset_location,
    parse_coin_input,
    perp_dexs_for_sdk,
    resolve_market,
    sdk_perp_dexs_for_dexes,
)
from .pricing import ceil_size, floor_size, round_price, round_size, size_step
from .tp_sl import tp_sl_from_entry
from .ema import EmaSnapshot, build_snapshot
from .rsi import RsiValues, compute_rsi, parse_closes, wilder_rsi_series


# Keys that appear on a clearinghouseState blob and are not builder-dex names.
_CLEARINGHOUSE_META_KEYS = {
    "assetPositions",
    "marginSummary",
    "crossMarginSummary",
    "withdrawable",
    "time",
    "agentAddress",
    "cumLedger",
    "perpDexStates",
}


def canonical_position_coin(coin: str, dex: str | None) -> str:
    """HIP-3 books often return a bare ticker; keep the xyz:COIN form used live."""
    raw = str(coin or "").strip()
    if not raw:
        return ""
    if ":" in raw:
        return raw
    dex_s = str(dex or "").strip()
    return f"{dex_s}:{raw}" if dex_s else raw


def iter_clearinghouse_states(
    raw: Any, *, default_dex: str = ""
) -> list[tuple[str, dict[str, Any]]]:
    """
    Pull every (dex, state) out of a clearinghouseState payload.

    ALL_DEXES is a nested dict or a list of [dex, state] pairs. A top-level
    assetPositions key must not hide sibling HIP-3 books.
    """
    tagged: list[tuple[str, dict[str, Any]]] = []
    seen: set[int] = set()

    def add(dex: str, state: dict[str, Any]) -> None:
        if "assetPositions" not in state:
            return
        sid = id(state)
        if sid in seen:
            return
        seen.add(sid)
        tagged.append((str(dex or ""), state))

    def is_pair(obj: Any) -> bool:
        return (
            isinstance(obj, (list, tuple))
            and len(obj) == 2
            and isinstance(obj[0], str)
            and isinstance(obj[1], dict)
            and "assetPositions" in obj[1]
        )

    def walk(obj: Any, dex_hint: str) -> None:
        if obj is None:
            return
        if is_pair(obj):
            add(obj[0], obj[1])
            return
        if isinstance(obj, dict):
            if "assetPositions" in obj:
                add(dex_hint, obj)
            for key, val in obj.items():
                if key == "assetPositions":
                    continue
                next_dex = dex_hint
                if (
                    isinstance(key, str)
                    and key not in _CLEARINGHOUSE_META_KEYS
                    and isinstance(val, dict)
                    and "assetPositions" in val
                ):
                    next_dex = key
                walk(val, next_dex)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item, dex_hint)

    walk(raw, default_dex or "")
    return tagged


@dataclass
class Position:
    side: str  # "long" | "short"
    size: float
    entry_price: float | None = None


@dataclass(frozen=True)
class OrderSizeEstimate:
    ok: bool
    size: float
    notional_usd: float
    available_margin: float
    mark_price: float
    leverage: int
    sz_decimals: int
    reason: str = ""


class HyperliquidClient:
    def __init__(
        self,
        wallet_address: str,
        private_key: str,
        coin: str,
        logger: logging.Logger,
        use_testnet: bool = False,
        *,
        perp_dex: str | None = None,
        market: MarketSpec | None = None,
        sdk_perp_dexs: list[str] | None = None,
        api_timeout_s: float = 30.0,
        request_guard: RequestGuard | None = None,
        ip_weight_reserve: int | None = None,
    ) -> None:
        self.address = wallet_address
        self.logger = logger
        base_url = constants.TESTNET_API_URL if use_testnet else constants.MAINNET_API_URL

        account: LocalAccount = eth_account.Account.from_key(private_key)
        if account.address.lower() != wallet_address.lower():
            self.logger.warning(
                "Private key address %s differs from HYPE_WALLET_ADDRESS %s; "
                "using wallet address for queries and key for signing.",
                account.address,
                wallet_address,
            )

        self._hl_guard = request_guard or RequestGuard(
            min_interval_s=0.12,
            max_429_retries=8,
            logger=logger,
            shared_budget=default_shared_budget(reserve=ip_weight_reserve),
            ip_reserve=ip_weight_reserve,
        )
        bootstrap_info = ThrottledInfo(
            base_url, skip_ws=True, guard=self._hl_guard, timeout=api_timeout_s
        )
        if market is None:
            market = resolve_market(bootstrap_info, coin, perp_dex)
        self.market = market
        self.coin = market.api_coin
        self.perp_dex = market.perp_dex
        self.only_isolated = market.only_isolated

        sdk_perp_dexs = sdk_perp_dexs if sdk_perp_dexs is not None else perp_dexs_for_sdk(market.perp_dex)
        self._sdk_perp_dexs: list[str] = list(sdk_perp_dexs) if sdk_perp_dexs else [""]
        self.info = ThrottledInfo(
            base_url,
            skip_ws=True,
            guard=self._hl_guard,
            perp_dexs=sdk_perp_dexs,
            timeout=api_timeout_s,
        )
        self.exchange = Exchange(
            account,
            base_url,
            account_address=wallet_address,
            perp_dexs=sdk_perp_dexs,
        )
        bind_throttled_exchange_post(self.exchange, self._hl_guard)

        self._user_state_cache: dict[str, Any] | None = None
        self._user_state_monotonic = 0.0
        self._user_state_ttl_s = 2.0

        self._l2_cache: dict[str, Any] | None = None
        self._l2_monotonic = 0.0
        self._l2_ttl_s = 0.35

        self.sz_decimals = market.sz_decimals
        self.max_leverage = market.max_leverage
        # Headroom for fees/funding/open-order bookkeeping to avoid margin rejects.
        self._order_margin_buffer = 0.97
        self._user_abstraction: str | None = None
        self.candle_book = CandleBook(self.info, logger)
        self._ensure_sdk_asset_maps()

    def _clearinghouse_dex(self) -> str:
        return self.perp_dex or ""

    def _sdk_info_targets(self) -> tuple[Any, ...]:
        """Client Info and Exchange's internal Info must share HIP-3 aliases."""
        return (self.info, self.exchange.info)

    def _lookup_asset_id(self, info: Any, symbol: str) -> int | None:
        """Resolve HIP-3 asset id from whatever name the SDK already registered."""
        dex = self.perp_dex
        if dex:
            candidates = [self.coin, f"{dex}:{symbol}"]
            if self.coin == f"{dex}:{symbol}":
                candidates.append(symbol)
        else:
            candidates = [self.coin, symbol]
        for name in candidates:
            if name in info.coin_to_asset:
                return int(info.coin_to_asset[name])
            if name in info.name_to_coin:
                mapped = info.name_to_coin[name]
                if mapped in info.coin_to_asset:
                    return int(info.coin_to_asset[mapped])
        for key, asset_id in info.coin_to_asset.items():
            key_str = str(key)
            if key_str in candidates:
                continue
            if dex:
                if key_str == self.coin or key_str == f"{dex}:{symbol}":
                    return int(asset_id)
            elif key_str == symbol or key_str == self.coin:
                return int(asset_id)
        return None

    def _register_sdk_aliases(
        self,
        info: Any,
        *,
        asset_id: int,
        sz_dec: int,
    ) -> None:
        aliases = {
            self.coin,
            self.market.symbol,
            f"{self.perp_dex}:{self.market.symbol}" if self.perp_dex else "",
        }
        for alias in aliases:
            if not alias:
                continue
            # Identity map so name_to_asset("xyz:SPCX") -> coin_to_asset["xyz:SPCX"].
            info.coin_to_asset[alias] = asset_id
            info.name_to_coin[alias] = alias
        info.asset_to_sz_decimals[asset_id] = sz_dec

    def _ensure_sdk_asset_maps(self) -> None:
        """HIP-3: ensure xyz:SPCX-style names work on both Info instances."""
        if not self.perp_dex:
            return
        if self.coin in self.exchange.info.name_to_coin:
            return

        exchange_info = self.exchange.info
        symbol = self.market.symbol

        # SDK meta(dex) often registers bare symbol (e.g. "SPCX") but not "xyz:SPCX".
        asset_id = self._lookup_asset_id(exchange_info, symbol)
        if asset_id is not None:
            sz_dec = int(exchange_info.asset_to_sz_decimals.get(asset_id, self.sz_decimals))
            for info in self._sdk_info_targets():
                self._register_sdk_aliases(info, asset_id=asset_id, sz_dec=sz_dec)
            self.logger.info(
                "Registered HIP-3 aliases for %s (asset_id=%s)",
                self.coin,
                asset_id,
            )
            return

        located = find_perp_asset_location(
            self.info,
            api_coin=self.coin,
            symbol=symbol,
            perp_dex=self.perp_dex,
        )
        if located is None:
            self.logger.error(
                "Could not find %s in allPerpMetas — leverage/orders will fail",
                self.coin,
            )
            return

        sz_dec = int(located.asset.get("szDecimals", self.sz_decimals))
        for info in self._sdk_info_targets():
            self._register_sdk_aliases(info, asset_id=located.asset_id, sz_dec=sz_dec)
        self.logger.info(
            "Registered HIP-3 asset maps for %s (asset_id=%s meta_idx=%s)",
            self.coin,
            located.asset_id,
            located.meta_idx,
        )

    def _all_mids_for_market(self) -> dict[str, Any]:
        return self.info.all_mids(dex=self._clearinghouse_dex())

    def _mid_price_from_book(self, mids: dict[str, Any]) -> float:
        for key in (self.coin, self.market.symbol, f"{self.perp_dex}:{self.market.symbol}" if self.perp_dex else ""):
            if key and key in mids:
                return float(mids[key])
        raise KeyError(f"No mid price for {self.coin} in allMids(dex={self._clearinghouse_dex()!r})")

    def invalidate_user_state(self) -> None:
        self._user_state_cache = None
        self._l2_cache = None

    def get_user_state(self, *, force: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        if (
            not force
            and self._user_state_cache is not None
            and now - self._user_state_monotonic < self._user_state_ttl_s
        ):
            return self._user_state_cache
        state = self.info.user_state(self.address, dex=self._clearinghouse_dex())
        self._user_state_cache = state
        self._user_state_monotonic = now
        return state

    def configure_coin(
        self,
        coin: str,
        *,
        perp_dex: str | None = None,
        sz_decimals: int | None = None,
        max_leverage: int | None = None,
    ) -> None:
        """Switch active perp; refresh sizing metadata when coin changes."""
        _symbol, effective_dex = parse_coin_input(coin, perp_dex)
        market = resolve_market(self.info, coin, effective_dex)
        if market.api_coin != self.coin:
            self.coin = market.api_coin
            self.market = market
            self.perp_dex = market.perp_dex
            self.only_isolated = market.only_isolated
            self.invalidate_user_state()
        if sz_decimals is not None:
            self.sz_decimals = int(sz_decimals)
        else:
            self.sz_decimals = market.sz_decimals
        if max_leverage is not None:
            self.max_leverage = max(1, int(max_leverage))
        else:
            self.max_leverage = market.max_leverage
        self._ensure_sdk_asset_maps()

    def apply_market(self, market: MarketSpec) -> None:
        """Switch active perp using a known MarketSpec (no meta round-trip)."""
        if (
            self.coin == market.api_coin
            and self.market.sz_decimals == market.sz_decimals
            and (self.perp_dex or "") == (market.perp_dex or "")
        ):
            return
        self.market = market
        self.coin = market.api_coin
        self.perp_dex = market.perp_dex
        self.only_isolated = market.only_isolated
        self.sz_decimals = market.sz_decimals
        self.max_leverage = market.max_leverage
        self.invalidate_user_state()
        self._ensure_sdk_asset_maps()

    def set_leverage(self, leverage: int, is_cross: bool = True) -> None:
        self._ensure_sdk_asset_maps()
        if self.coin not in self.exchange.info.name_to_coin:
            raise KeyError(
                f"{self.coin} is not registered in the Hyperliquid SDK for dex "
                f"{self.perp_dex!r}. Check COIN/PERP_DEX config."
            )
        requested = max(1, int(leverage))
        effective = min(requested, self.max_leverage)
        if effective != requested:
            self.logger.warning(
                "Leverage %sx exceeds %s max %sx — clamping to %sx",
                requested,
                self.coin,
                self.max_leverage,
                effective,
            )
        result = self.exchange.update_leverage(effective, self.coin, is_cross=is_cross)
        self.invalidate_user_state()
        self.logger.info("Leverage update (%sx): %s", effective, result)

    def get_user_abstraction(self, *, force: bool = False) -> str:
        if not force and self._user_abstraction is not None:
            return self._user_abstraction
        try:
            raw = self.info.post(
                "/info",
                {"type": "userAbstraction", "user": self.address},
            )
            mode = str(raw).strip().strip('"')
        except Exception:
            mode = "default"
        self._user_abstraction = mode
        return mode

    def uses_unified_collateral(self) -> bool:
        mode = self.get_user_abstraction()
        return mode in ("unifiedAccount", "portfolioMargin")

    def get_spot_usdc_balance(self, *, force: bool = False) -> tuple[float, float]:
        """Return (available, total) USDC from spot clearinghouse."""
        try:
            state = self.info.spot_user_state(self.address)
        except Exception:
            return 0.0, 0.0
        for bal in state.get("balances", []):
            if str(bal.get("coin", "")).upper() != "USDC":
                continue
            total = float(bal.get("total", 0) or 0)
            hold = float(bal.get("hold", 0) or 0)
            return max(0.0, total - hold), total
        return 0.0, 0.0

    def get_isolated_margin_locked(self, *, force: bool = False) -> float:
        if force:
            self.invalidate_user_state()
        locked = 0.0
        for state in self._iter_clearinghouse_states():
            for ap in state.get("assetPositions", []):
                pos = ap.get("position", ap)
                lev = pos.get("leverage") or {}
                if lev.get("type") != "isolated":
                    continue
                locked += float(pos.get("marginUsed", 0) or 0)
        return locked

    def get_margin_summary(self, *, force: bool = False) -> dict[str, Any]:
        """
        Perp margin bucket for sizing. Cross-margin positions use crossMarginSummary
        (matches Hyperliquid clearinghouseState); falls back to marginSummary.
        """
        state = self.get_user_state(force=force)
        cross = state.get("crossMarginSummary")
        if isinstance(cross, dict) and cross:
            return cross
        return state["marginSummary"]

    def get_account_value(self, *, force: bool = False) -> float:
        """Total account equity in USD (unified spot USDC or perp summary)."""
        if self.uses_unified_collateral():
            _available, total = self.get_spot_usdc_balance(force=force)
            return total
        return float(self.get_margin_summary(force=force)["accountValue"])

    def get_mark_price(self) -> float:
        """Spot/mid price for the active perp (main or HIP-3 dex)."""
        return self._mid_price_from_book(self._all_mids_for_market())

    def get_available_margin(self, *, force: bool = False) -> float:
        """
        Collateral free for new cross margin.

        Standard perp account: crossMarginSummary accountValue - totalMarginUsed.
        Unified / portfolio margin: spot USDC available minus isolated locks.
        """
        if self.uses_unified_collateral():
            available, _total = self.get_spot_usdc_balance(force=force)
            isolated = self.get_isolated_margin_locked(force=force)
            return max(0.0, available - isolated)

        summary = self.get_margin_summary(force=force)
        account_value = float(summary["accountValue"])
        margin_used = float(summary.get("totalMarginUsed") or 0)
        return max(0.0, account_value - margin_used)

    def _matches_active_coin(self, coin: str) -> bool:
        raw = str(coin or "")
        if not raw:
            return False
        if raw == self.coin or raw == self.market.symbol:
            return True
        if self.perp_dex and raw == f"{self.perp_dex}:{self.market.symbol}":
            return True
        return canonical_position_coin(raw, self.perp_dex) == self.coin

    def _position_dexes_to_query(self) -> list[str]:
        """Native book plus every HIP-3 dex this client was opened with."""
        dexes: list[str] = []

        def add(name: str | None) -> None:
            key = name or ""
            if key not in dexes:
                dexes.append(key)

        for name in self._sdk_perp_dexs:
            add(name)
        add(self.perp_dex)
        add("")
        return dexes

    def _iter_tagged_clearinghouse_states(self) -> list[tuple[str, dict[str, Any]]]:
        """
        HIP-3 positions live on the builder dex. ALL_DEXES plus a per-dex
        fill-in so a native-active client still sees xyz:NVDA.
        """
        tagged: list[tuple[str, dict[str, Any]]] = []
        seen_dexes: set[str] = set()
        try:
            raw = self.info.post(
                "/info",
                {
                    "type": "clearinghouseState",
                    "user": self.address,
                    "dex": "ALL_DEXES",
                },
            )
            for dex, state in iter_clearinghouse_states(raw):
                tagged.append((dex, state))
                seen_dexes.add(dex)
        except Exception as exc:
            self.logger.debug("ALL_DEXES clearinghouseState failed: %s", exc)

        for dex in self._position_dexes_to_query():
            if dex in seen_dexes:
                continue
            try:
                raw = self.info.user_state(self.address, dex=dex)
            except Exception as exc:
                self.logger.debug("clearinghouseState dex=%r failed: %s", dex, exc)
                continue
            extra = iter_clearinghouse_states(raw, default_dex=dex)
            if extra:
                tagged.extend(extra)
                seen_dexes.add(dex)
            elif isinstance(raw, dict) and "assetPositions" in raw:
                tagged.append((dex, raw))
                seen_dexes.add(dex)
        return tagged

    def _iter_clearinghouse_states(self) -> list[dict[str, Any]]:
        return [state for _dex, state in self._iter_tagged_clearinghouse_states()]

    def _position_from_state(self, state: dict[str, Any]) -> Position | None:
        for ap in state.get("assetPositions", []):
            pos = ap.get("position", ap)
            if not self._matches_active_coin(str(pos.get("coin", ""))):
                continue
            szi = float(pos.get("szi", 0))
            if abs(szi) < 1e-12:
                continue
            entry_px = pos.get("entryPx")
            return Position(
                side="long" if szi > 0 else "short",
                size=abs(szi),
                entry_price=float(entry_px) if entry_px is not None else None,
            )
        return None

    def get_position(self, *, force: bool = False) -> Position | None:
        if force:
            self.invalidate_user_state()
        for state in self._iter_clearinghouse_states():
            found = self._position_from_state(state)
            if found is not None:
                return found
        return None

    def get_ema_snapshot(self, interval: str, period: int, min_bars: int) -> tuple[EmaSnapshot | None, int | None]:
        candles = self.get_closed_candles(interval, min_bars=min_bars)
        snap = build_snapshot(candles, period)
        if snap is None:
            return None, None
        return snap, snap.candle_t

    @staticmethod
    def _interval_ms(interval: str) -> int:
        return INTERVAL_MS[interval]

    def _fetch_candles(self, interval: str, bars: int) -> list[dict]:
        return self.candle_book._fetch_raw(self.coin, interval, bars)

    def get_closed_candles(self, interval: str, min_bars: int = 40) -> list[dict]:
        """Only fully closed candles; refetch a TF only when that TF has a new closed bar."""
        return self.candle_book.get(self.coin, interval, min_bars)

    def get_closed_candles_for(
        self, coin: str, interval: str, min_bars: int = 40
    ) -> list[dict]:
        return self.candle_book.get(coin, interval, min_bars)

    def get_rsi(
        self,
        interval: str,
        period: int = 14,
        smooth_period: int = 14,
        *,
        min_bars: int | None = None,
        lookback: int = 0,
    ) -> tuple[RsiValues | None, int | None, list[float] | None]:
        """
        RSI on the latest closed candle (TradingView rsi + sma smoothing).

        When lookback > 0, also returns the prior `lookback` raw Wilder RSI values
        (excluding the current bar) for rolling-extreme entry logic.
        """
        need = min_bars if min_bars is not None else period + smooth_period + 15
        candles = self.get_closed_candles(interval, min_bars=need)
        if len(candles) < period + 2:
            return None, None, None
        closes = parse_closes(candles)
        values = compute_rsi(closes, period, smooth_period)
        if values is None:
            return None, None, None
        prior: list[float] | None = None
        if lookback > 0:
            series = wilder_rsi_series(closes, period)
            if len(series) < lookback + 1:
                return None, None, None
            prior = series[-(lookback + 1) : -1]
        return values, int(candles[-1]["t"]), prior

    def calc_order_size(
        self,
        balance_pct: float,
        leverage: int,
        *,
        min_notional_usd: float = 10.0,
    ) -> float:
        sz, notional = self._calc_order_size_raw(balance_pct, leverage)
        if sz <= 0:
            raise ValueError("Computed order size is zero; check balance_pct and leverage.")
        if notional < min_notional_usd:
            raise ValueError(
                f"Order notional ${notional:.2f} below Hyperliquid minimum ${min_notional_usd}."
            )
        return sz

    def estimate_order_size(
        self,
        balance_pct: float,
        leverage: int,
        *,
        sz_decimals: int | None = None,
        mark_price: float | None = None,
        min_notional_usd: float = 10.0,
        force_margin: bool = True,
        margin_from: str = "available",
    ) -> OrderSizeEstimate:
        """Size/notional for the active or explicit pair parameters.

        margin_from:
          "available" — % of free collateral (legacy).
          "equity" — % of account equity, then capped by free collateral.
        """
        dec = self.sz_decimals if sz_decimals is None else int(sz_decimals)
        try:
            available = self.get_available_margin(force=force_margin)
            if available <= 0:
                return OrderSizeEstimate(
                    ok=False,
                    size=0.0,
                    notional_usd=0.0,
                    available_margin=available,
                    mark_price=0.0,
                    leverage=int(leverage),
                    sz_decimals=dec,
                    reason="no available margin (unified spot USDC or perp free collateral is 0)",
                )
            mid = mark_price if mark_price is not None else self.get_mark_price()
            if mid <= 0:
                return OrderSizeEstimate(
                    ok=False,
                    size=0.0,
                    notional_usd=0.0,
                    available_margin=available,
                    mark_price=mid,
                    leverage=int(leverage),
                    sz_decimals=dec,
                    reason=f"invalid mark price for {self.coin}",
                )
        except KeyError as exc:
            return OrderSizeEstimate(
                ok=False,
                size=0.0,
                notional_usd=0.0,
                available_margin=0.0,
                mark_price=0.0,
                leverage=int(leverage),
                sz_decimals=dec,
                reason=f"price lookup failed for {self.coin}: {exc}",
            )

        balance_pct = max(0.0, min(float(balance_pct), 95.0))
        if str(margin_from).strip().lower() == "equity":
            try:
                equity = self.get_account_value(force=force_margin)
            except Exception:
                equity = 0.0
            if equity <= 0:
                return OrderSizeEstimate(
                    ok=False,
                    size=0.0,
                    notional_usd=0.0,
                    available_margin=available,
                    mark_price=mid,
                    leverage=int(leverage),
                    sz_decimals=dec,
                    reason="account equity unavailable for equity-% sizing",
                )
            base = equity
            base_label = f"equity ${equity:.2f}"
        else:
            base = available
            base_label = f"free ${available:.2f}"
        margin = base * (balance_pct / 100.0) * self._order_margin_buffer
        if margin > available + 1e-9:
            return OrderSizeEstimate(
                ok=False,
                size=0.0,
                notional_usd=0.0,
                available_margin=available,
                mark_price=mid,
                leverage=int(leverage),
                sz_decimals=dec,
                reason=(
                    f"needs ${margin:.2f} margin ({balance_pct:.1f}% of {base_label}) "
                    f"but only ${available:.2f} free"
                ),
            )
        notional_target = margin * max(1, int(leverage))
        raw_sz = notional_target / mid
        step = size_step(dec)
        sz = floor_size(raw_sz, dec)
        if sz <= 0 and raw_sz > 0:
            sz = step
        notional = sz * mid
        if notional < min_notional_usd and available > 0:
            need_sz = min_notional_usd / mid
            sz = ceil_size(need_sz, dec)
            notional = sz * mid
            need_margin = notional / max(1, int(leverage))
            if need_margin > margin * 1.05:
                return OrderSizeEstimate(
                    ok=False,
                    size=sz,
                    notional_usd=notional,
                    available_margin=available,
                    mark_price=mid,
                    leverage=int(leverage),
                    sz_decimals=dec,
                    reason=(
                        f"notional ${notional:.2f} needs ${need_margin:.2f} margin "
                        f"but only ${margin:.2f} allocated ({balance_pct:.0f}% of ${available:.2f})"
                    ),
                )

        if sz <= 0:
            return OrderSizeEstimate(
                ok=False,
                size=0.0,
                notional_usd=0.0,
                available_margin=available,
                mark_price=mid,
                leverage=int(leverage),
                sz_decimals=dec,
                reason=f"computed size rounded to 0 (raw={raw_sz:.8f}, step={step})",
            )
        if notional < min_notional_usd:
            return OrderSizeEstimate(
                ok=False,
                size=sz,
                notional_usd=notional,
                available_margin=available,
                mark_price=mid,
                leverage=int(leverage),
                sz_decimals=dec,
                reason=f"notional ${notional:.2f} below ${min_notional_usd:.2f} minimum",
            )
        return OrderSizeEstimate(
            ok=True,
            size=sz,
            notional_usd=notional,
            available_margin=available,
            mark_price=mid,
            leverage=int(leverage),
            sz_decimals=dec,
        )

    def try_calc_order_size(
        self,
        balance_pct: float,
        leverage: int,
        *,
        sz_decimals: int | None = None,
        mark_price: float | None = None,
        min_notional_usd: float = 10.0,
    ) -> float | None:
        est = self.estimate_order_size(
            balance_pct,
            leverage,
            sz_decimals=sz_decimals,
            mark_price=mark_price,
            min_notional_usd=min_notional_usd,
        )
        return est.size if est.ok else None

    def _calc_order_size_raw(
        self, balance_pct: float, leverage: int, *, force_margin: bool = True
    ) -> tuple[float, float]:
        est = self.estimate_order_size(
            balance_pct,
            leverage,
            force_margin=force_margin,
        )
        if not est.ok:
            raise ValueError(est.reason or "order size unavailable")
        return est.size, est.notional_usd

    def _open_orders_on_dex(self, dex: str) -> list[dict]:
        dex_s = dex or ""
        try:
            return self.info.frontend_open_orders(self.address, dex=dex_s)
        except Exception:
            try:
                return self.info.open_orders(self.address, dex=dex_s)
            except Exception:
                return []

    def _all_frontend_orders(self) -> list[dict]:
        out: list[dict] = []
        seen: set[Any] = set()
        for dex in self._position_dexes_to_query():
            for order in self._open_orders_on_dex(dex):
                oid = order.get("oid")
                key = oid if oid is not None else id(order)
                if key in seen:
                    continue
                seen.add(key)
                out.append(order)
        return out

    def _frontend_orders_for_coin(self) -> list[dict]:
        orders = self._open_orders_on_dex(self._clearinghouse_dex())
        return [
            o
            for o in orders
            if self._matches_active_coin(str(o.get("coin") or ""))
        ]

    def _position_coins(self, *, force: bool = False) -> set[str]:
        ok, positions = self.fetch_open_positions(force=force)
        if not ok:
            return set()
        return {coin for coin, _pos in positions}

    def has_any_open_position(self, *, force: bool = False) -> bool:
        """True if any position exists. On API failure, True (fail closed)."""
        ok, positions = self.fetch_open_positions(force=force)
        if not ok:
            return True
        return bool(positions)

    def fetch_open_positions(
        self, *, force: bool = False
    ) -> tuple[bool, list[tuple[str, "Position"]]]:
        """
        Return (ok, positions) for every open perp on the account.

        ok=False means the API failed (e.g. 502). Callers must NOT treat that as
        flat — otherwise a second entry can open while a position still exists.
        """
        if force:
            self.invalidate_user_state()

        tagged = self._iter_tagged_clearinghouse_states()
        if not tagged:
            self.logger.warning(
                "Position query failed — treating as NOT flat: empty clearinghouse response"
            )
            return False, []

        positions: list[tuple[str, Position]] = []
        seen: set[str] = set()
        for dex, state in tagged:
            for ap in state.get("assetPositions", []):
                pos = ap.get("position", ap)
                coin = canonical_position_coin(str(pos.get("coin", "")), dex)
                if not coin or coin in seen:
                    continue
                szi = float(pos.get("szi", 0))
                if abs(szi) < 1e-12:
                    continue
                seen.add(coin)
                entry_px = pos.get("entryPx")
                positions.append(
                    (
                        coin,
                        Position(
                            side="long" if szi > 0 else "short",
                            size=abs(szi),
                            entry_price=(
                                float(entry_px) if entry_px is not None else None
                            ),
                        ),
                    )
                )
        return True, positions

    def cancel_all_orders_for_coin_named(self, coin: str) -> None:
        _symbol, dex = parse_coin_input(coin)
        saved_coin = self.coin
        saved_dex = self.perp_dex
        try:
            self.coin = coin
            if dex:
                self.perp_dex = dex
            self.cancel_all_orders_for_coin()
        finally:
            self.coin = saved_coin
            self.perp_dex = saved_dex

    def sweep_orphan_orders(self) -> int:
        """Cancel open orders on coins that have no position (stale TP/SL)."""
        ok, open_positions = self.fetch_open_positions(force=True)
        if not ok:
            self.logger.warning(
                "Skipping orphan order sweep — position query failed (not cancelling)"
            )
            return 0
        positions = {coin for coin, _pos in open_positions}
        cancelled = 0
        for order in self._all_frontend_orders():
            coin = order.get("coin")
            if not coin or coin in positions:
                continue
            oid = order.get("oid")
            if oid is None:
                continue
            try:
                self.exchange.cancel(coin, oid)
                cancelled += 1
                self.logger.info(
                    "Cancelled orphan order oid=%s on %s (flat on that coin)",
                    oid,
                    coin,
                )
            except Exception as exc:
                self.logger.warning("Orphan cancel failed %s oid=%s: %s", coin, oid, exc)
        if cancelled:
            self.invalidate_user_state()
        return cancelled

    def cancel_open_orders_for_coin(self) -> None:
        self.cancel_all_orders_for_coin()

    def cancel_all_orders_for_coin(self) -> None:
        self.invalidate_user_state()
        for order in self._frontend_orders_for_coin():
            oid = order.get("oid")
            if oid is None:
                continue
            try:
                self.exchange.cancel(self.coin, oid)
                self.invalidate_user_state()
                self.logger.info("Cancelled open order oid=%s", oid)
            except Exception as exc:
                self.logger.warning("Cancel failed oid=%s: %s", oid, exc)

    def cancel_entry_orders_for_coin(self) -> None:
        """Cancel non-trigger orders only (leave TP/SL triggers in place)."""
        self.invalidate_user_state()
        for order in self._frontend_orders_for_coin():
            if order.get("isTrigger"):
                continue
            oid = order.get("oid")
            if oid is None:
                continue
            try:
                self.exchange.cancel(self.coin, oid)
                self.invalidate_user_state()
                self.logger.info("Cancelled entry/limit oid=%s", oid)
            except Exception as exc:
                self.logger.warning("Cancel failed oid=%s: %s", oid, exc)

    def cancel_tp_triggers_for_coin(self) -> None:
        """Cancel take-profit triggers only (leave stop-loss in place)."""
        self.invalidate_user_state()
        for order in self._reduce_only_triggers():
            label = self._order_type_label(order)
            tpsl = str(order.get("tpsl") or "").lower()
            is_sl = tpsl == "sl" or ("stop" in label and "take profit" not in label)
            is_tp = tpsl == "tp" or "take profit" in label
            if is_sl or not is_tp:
                continue
            oid = order.get("oid")
            if oid is None:
                continue
            try:
                self.exchange.cancel(self.coin, oid)
                self.invalidate_user_state()
                self.logger.info("Cancelled TP trigger oid=%s", oid)
            except Exception as exc:
                self.logger.warning("Cancel TP failed oid=%s: %s", oid, exc)

    def tp_sl_prices_for_entry(
        self,
        side: str,
        entry: float,
        take_profit_pct: float,
        stop_loss_pct: float,
    ) -> tuple[float, float]:
        return tp_sl_from_entry(
            side, entry, take_profit_pct, stop_loss_pct, self.sz_decimals
        )

    @staticmethod
    def _order_type_label(order: dict) -> str:
        return str(order.get("orderType") or order.get("triggerCondition") or "").lower()

    def _reduce_only_triggers(self) -> list[dict]:
        out: list[dict] = []
        for order in self._frontend_orders_for_coin():
            if order.get("isTrigger") and order.get("reduceOnly"):
                out.append(order)
        return out

    def has_exchange_tpsl(self) -> bool:
        """Require both a TP and an SL trigger (reduce-only) on this coin."""
        has_tp = False
        has_sl = False
        for order in self._reduce_only_triggers():
            label = self._order_type_label(order)
            if "take profit" in label:
                has_tp = True
            if "stop" in label:
                has_sl = True
        return has_tp and has_sl

    def has_open_entry_orders(self) -> bool:
        for order in self._frontend_orders_for_coin():
            if order.get("isTrigger"):
                continue
            if order.get("reduceOnly"):
                continue
            return True
        return False

    @staticmethod
    def _bulk_status_error(status: dict) -> str | None:
        if "error" in status:
            return str(status["error"])
        return None

    def _place_position_tpsl_raw(
        self,
        side: str,
        sz: float,
        tp_trigger_px: float,
        sl_trigger_px: float,
    ) -> tuple[bool, str | None]:
        """Place positionTpsl; return (api_ok, error_message)."""
        close_is_buy = side == "short"
        sz = round_size(sz, self.sz_decimals)
        tp_trigger_px = round_price(tp_trigger_px, self.sz_decimals)
        sl_trigger_px = round_price(sl_trigger_px, self.sz_decimals)
        # SL limit_px slightly through trigger so market SL can fill in fast moves (HL docs).
        if side == "long":
            sl_limit_px = round_price(sl_trigger_px * 0.999, self.sz_decimals)
        else:
            sl_limit_px = round_price(sl_trigger_px * 1.001, self.sz_decimals)

        orders = [
            {
                "coin": self.coin,
                "is_buy": close_is_buy,
                "sz": sz,
                "limit_px": tp_trigger_px,
                "order_type": {
                    "trigger": {
                        "isMarket": True,
                        "triggerPx": tp_trigger_px,
                        "tpsl": "tp",
                    }
                },
                "reduce_only": True,
            },
            {
                "coin": self.coin,
                "is_buy": close_is_buy,
                "sz": sz,
                "limit_px": sl_limit_px,
                "order_type": {
                    "trigger": {
                        "isMarket": True,
                        "triggerPx": sl_trigger_px,
                        "tpsl": "sl",
                    }
                },
                "reduce_only": True,
            },
        ]
        try:
            result = self.exchange.bulk_orders(orders, grouping="positionTpsl")
        except Exception as exc:
            return False, str(exc)
        self.invalidate_user_state()
        if result.get("status") != "ok":
            return False, str(result)

        statuses = result.get("response", {}).get("data", {}).get("statuses", [])
        errors: list[str] = []
        for i, leg in enumerate(("TP", "SL")):
            if i >= len(statuses):
                errors.append(f"{leg}: missing status")
                continue
            err = self._bulk_status_error(statuses[i])
            if err:
                errors.append(f"{leg}: {err}")
        if errors:
            return False, "; ".join(errors)
        return True, None

    def attach_position_tpsl(
        self,
        position: Position,
        take_profit_pct: float,
        stop_loss_pct: float,
        *,
        max_attempts: int = 3,
    ) -> bool:
        """
        Attach TP+SL for full position size from exchange entry price.
        Clears orphan triggers first, then verifies both legs appear on book.
        """
        import time as time_mod

        self.invalidate_user_state()
        live = self.get_position(force=True)
        if live is None or live.size < 1e-12:
            self.logger.warning(
                "attach_position_tpsl skipped — no open position on %s",
                self.coin,
            )
            self.cancel_all_orders_for_coin()
            return False
        position = live
        entry = position.entry_price or self.get_mark_price()
        tp_px, sl_px = self.tp_sl_prices_for_entry(
            position.side,
            entry,
            take_profit_pct,
            stop_loss_pct,
        )

        for attempt in range(1, max_attempts + 1):
            live = self.get_position(force=True)
            if live is None or live.size < 1e-12:
                self.logger.warning(
                    "attach_position_tpsl aborted — position closed on %s",
                    self.coin,
                )
                self.cancel_all_orders_for_coin()
                return False
            position = live
            entry = position.entry_price or self.get_mark_price()
            tp_px, sl_px = self.tp_sl_prices_for_entry(
                position.side,
                entry,
                take_profit_pct,
                stop_loss_pct,
            )

            if self.has_exchange_tpsl():
                return True

            # Remove broken/partial trigger sets before re-placing
            for order in self._reduce_only_triggers():
                oid = order.get("oid")
                if oid is not None:
                    try:
                        self.exchange.cancel(self.coin, oid)
                    except Exception:
                        pass
            self.invalidate_user_state()
            self.cancel_entry_orders_for_coin()

            self.logger.info(
                "positionTpsl attempt %s/%s | %s sz=%s entry=%.2f tp=%.2f sl=%.2f",
                attempt,
                max_attempts,
                position.side,
                position.size,
                entry,
                tp_px,
                sl_px,
            )
            ok, err = self._place_position_tpsl_raw(
                position.side,
                position.size,
                tp_px,
                sl_px,
            )
            if not ok:
                self.logger.warning("positionTpsl place failed: %s", err)
                time_mod.sleep(1.0)
                continue

            time_mod.sleep(0.8)
            if self.has_exchange_tpsl():
                self.logger.info("Exchange TP and SL confirmed on book")
                return True
            self.logger.warning("TP/SL placed but not both visible yet — retrying")

        return False

    def place_limit(
        self,
        is_buy: bool,
        sz: float,
        limit_px: float,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Post-only limit (ALO) — fills as maker when resting order is hit."""
        sz = round_size(sz, self.sz_decimals)
        limit_px = round_price(limit_px, self.sz_decimals)
        result = self.exchange.order(
            self.coin,
            is_buy,
            sz,
            limit_px,
            {"limit": {"tif": "Alo"}},
            reduce_only=reduce_only,
        )
        self.invalidate_user_state()
        return result

    def place_market_open(
        self,
        is_buy: bool,
        sz: float,
        *,
        slippage: float = 0.05,
    ) -> dict[str, Any]:
        sz = round_size(sz, self.sz_decimals)
        result = self.exchange.market_open(
            self.coin,
            is_buy,
            sz,
            slippage=slippage,
        )
        self.invalidate_user_state()
        return result

    def place_market_close(self, sz: float | None = None) -> dict[str, Any]:
        sz_arg = round_size(sz, self.sz_decimals) if sz is not None else None
        result = self.exchange.market_close(self.coin, sz=sz_arg)
        self.invalidate_user_state()
        return result

    @staticmethod
    def _is_alo_rejection(message: str) -> bool:
        lower = message.lower()
        return any(
            phrase in lower
            for phrase in (
                "add liquidity only",
                "post only",
                "post-only",
                "alo",
                "immediately matched",
                "would have immediately",
            )
        )

    def parse_fill_from_result(
        self, result: dict
    ) -> tuple[float, int | None, bool]:
        """
        Return (filled_sz, resting_oid, alo_rejected).
        alo_rejected=True means HL canceled post-only (would have been taker); retry price.
        """
        if result.get("status") != "ok":
            return 0.0, None, False
        statuses = result["response"]["data"]["statuses"]
        if not statuses:
            return 0.0, None, False
        status = statuses[0]
        if "filled" in status:
            return float(status["filled"]["totalSz"]), None, False
        if "resting" in status:
            return 0.0, status["resting"]["oid"], False
        if "error" in status:
            msg = status["error"]
            if self._is_alo_rejection(msg):
                return 0.0, None, True
            raise RuntimeError(msg)
        return 0.0, None, False

    def query_filled_size(self, oid: int, original_sz: float) -> tuple[float, bool]:
        """Return (filled_sz, is_done). is_done=True if order gone or fully filled."""
        try:
            status_resp = self.info.query_order_by_oid(self.address, oid)
        except Exception:
            return 0.0, True

        if not status_resp or "order" not in status_resp:
            return 0.0, True

        order_info = status_resp["order"]
        if order_info is None:
            return original_sz, True

        order = order_info.get("order", order_info)
        orig = float(order.get("origSz", original_sz))
        remaining = float(order.get("sz", 0))
        filled = max(0.0, orig - remaining)
        is_open = order_info.get("status") == "open"
        return filled, not is_open

    def l2_book(self) -> dict:
        now = time.monotonic()
        if self._l2_cache is not None and now - self._l2_monotonic < self._l2_ttl_s:
            return self._l2_cache
        book = self.info.l2_snapshot(self.coin)
        self._l2_cache = book
        self._l2_monotonic = now
        return book
