"""Resolve Hyperliquid main perps and HIP-3 builder-dex markets (e.g. xyz:SPCX)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PerpAssetLocation:
    meta_idx: int
    universe_idx: int
    asset: dict[str, Any]
    asset_id: int


@dataclass(frozen=True)
class MarketSpec:
    """Trading pair as Hyperliquid API expects it."""

    api_coin: str  # e.g. "BTC" or "xyz:SPCX"
    symbol: str  # display symbol without dex prefix, e.g. "SPCX"
    perp_dex: str | None  # e.g. "xyz" for HIP-3 builder dexes
    sz_decimals: int
    max_leverage: int
    only_isolated: bool


def _strip_quote_suffix(raw: str) -> str:
    for suffix in ("-USDC", "/USDC", "-USD", "/USD"):
        if raw.upper().endswith(suffix.upper()):
            return raw[: -len(suffix)]
    return raw


def parse_coin_input(coin: str, perp_dex: str | None = None) -> tuple[str, str | None]:
    """
    Parse bot COIN config into (base_symbol, perp_dex).

    Accepts:
      - "BTC"
      - "SPCX-USDC" + PERP_DEX="xyz"
      - "xyz:SPCX"
    """
    raw = coin.strip()
    dex = (perp_dex or "").strip() or None

    if ":" in raw:
        left, right = raw.split(":", 1)
        symbol = _strip_quote_suffix(right.strip())
        return symbol, left.strip() or dex

    symbol = _strip_quote_suffix(raw)
    return symbol, dex


def api_coin_name(symbol: str, perp_dex: str | None) -> str:
    if perp_dex:
        return f"{perp_dex}:{symbol}"
    return symbol


def perp_dexs_for_sdk(perp_dex: str | None) -> list[str] | None:
    """SDK perp_dexs list: main book + optional HIP-3 dex."""
    if not perp_dex:
        return None
    return ["", perp_dex]


def sdk_perp_dexs_for_dexes(dexes: set[str | None]) -> list[str] | None:
    """Union of HIP-3 dexes needed for a multi-pair watchlist."""
    named = sorted({d for d in dexes if d})
    if not named:
        return None
    return [""] + named


def _max_leverage_from_asset(asset: dict[str, Any]) -> int:
    for key in ("maxLeverage", "max_leverage", "maxLev", "leverage", "maxLeveragex"):
        raw = asset.get(key)
        if raw is None:
            continue
        try:
            return max(1, int(float(raw)))
        except (TypeError, ValueError):
            continue
    return 40


def _match_asset_name(asset_name: str, api_coin: str, symbol: str, perp_dex: str | None) -> bool:
    if asset_name == api_coin:
        return True
    if perp_dex and asset_name == symbol:
        return True
    if perp_dex and asset_name == f"{perp_dex}:{symbol}":
        return True
    return False


def _asset_from_universe(
    universe: list[dict[str, Any]],
    *,
    api_coin: str,
    symbol: str,
    perp_dex: str | None,
) -> dict[str, Any] | None:
    for asset in universe:
        name = str(asset.get("name", ""))
        if _match_asset_name(name, api_coin, symbol, perp_dex):
            return asset
    return None


def fetch_all_perp_metas(info: Any) -> list[dict[str, Any]]:
    raw = info.post("/info", {"type": "allPerpMetas"})
    if not isinstance(raw, list):
        return []
    metas: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict) and "universe" in item:
            metas.append(item)
    return metas


def _resolve_from_all_perp_metas(
    info: Any,
    *,
    api_coin: str,
    symbol: str,
    perp_dex: str | None,
) -> dict[str, Any] | None:
    metas = fetch_all_perp_metas(info)
    if not metas:
        return None

    if perp_dex:
        dex_names = list_perp_dex_names(info)
        try:
            dex_idx = dex_names.index(perp_dex)
        except ValueError:
            dex_idx = -1
            for i, dex in enumerate(dex_names):
                if str(dex).lower() == perp_dex.lower():
                    dex_idx = i
                    break
        # allPerpMetas[0] is main dex; HIP-3 dexes follow perpDexs order
        meta_idx = dex_idx + 1 if dex_idx >= 0 else -1
        if 0 <= meta_idx < len(metas):
            found = _asset_from_universe(
                metas[meta_idx].get("universe", []),
                api_coin=api_coin,
                symbol=symbol,
                perp_dex=perp_dex,
            )
            if found is not None:
                return found

    for meta in metas:
        found = _asset_from_universe(
            meta.get("universe", []),
            api_coin=api_coin,
            symbol=symbol,
            perp_dex=perp_dex,
        )
        if found is not None:
            return found
    return None


def list_perp_dex_names(info: Any) -> list[str]:
    try:
        raw = info.perp_dexs()
    except Exception:
        raw = info.post("/info", {"type": "perpDexs"})
    names: list[str] = []
    if not isinstance(raw, list):
        return names
    for item in raw:
        if isinstance(item, dict):
            name = item.get("name")
            if name is not None:
                names.append(str(name))
        elif item is not None:
            names.append(str(item))
    return names


def asset_id_for_meta_slot(meta_idx: int, universe_idx: int) -> int:
    """Match Hyperliquid SDK offsets: main dex 0..N, HIP-3 from 110000."""
    if meta_idx <= 0:
        return universe_idx
    return 110000 + (meta_idx - 1) * 10000 + universe_idx


def find_perp_asset_location(
    info: Any,
    *,
    api_coin: str,
    symbol: str,
    perp_dex: str | None,
) -> PerpAssetLocation | None:
    """Locate a perp in allPerpMetas and compute its SDK asset id."""
    metas = fetch_all_perp_metas(info)
    for meta_idx, meta in enumerate(metas):
        universe = meta.get("universe", [])
        for universe_idx, asset in enumerate(universe):
            name = str(asset.get("name", ""))
            if not _match_asset_name(name, api_coin, symbol, perp_dex):
                continue
            asset_id = asset_id_for_meta_slot(meta_idx, universe_idx)
            return PerpAssetLocation(
                meta_idx=meta_idx,
                universe_idx=universe_idx,
                asset=asset,
                asset_id=asset_id,
            )
    return None


def _suggest_similar_markets(
    info: Any,
    *,
    symbol: str,
    perp_dex: str | None,
    limit: int = 5,
) -> list[str]:
    """Best-effort fuzzy suggestions when a ticker is missing."""
    needle = symbol.upper()
    names: list[str] = []
    try:
        if perp_dex:
            meta = info.meta(perp_dex)
            names.extend(str(a.get("name", "")) for a in (meta.get("universe") or []))
        else:
            for meta in fetch_all_perp_metas(info):
                names.extend(str(a.get("name", "")) for a in (meta.get("universe") or []))
    except Exception:
        return []

    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        bare = name.split(":", 1)[-1].upper()
        if needle in bare or bare in needle:
            score = abs(len(bare) - len(needle))
            scored.append((score, name))
            continue
        # shared prefix length (e.g. SKHYNIX -> SKHY / SKHX)
        shared = 0
        for a, b in zip(needle, bare):
            if a != b:
                break
            shared += 1
        if shared >= 3:
            scored.append((10 - shared + abs(len(bare) - len(needle)), name))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [n for _, n in scored[:limit]]


def resolve_market(info: Any, coin: str, perp_dex: str | None = None) -> MarketSpec:
    """Look up sizing/leverage metadata for a main or HIP-3 perp."""
    symbol, dex = parse_coin_input(coin, perp_dex)
    api_coin = api_coin_name(symbol, dex)
    asset: dict[str, Any] | None = None

    if dex:
        asset = _resolve_from_all_perp_metas(
            info,
            api_coin=api_coin,
            symbol=symbol,
            perp_dex=dex,
        )
        if asset is None:
            try:
                meta = info.meta(dex)
                asset = _asset_from_universe(
                    meta.get("universe", []),
                    api_coin=api_coin,
                    symbol=symbol,
                    perp_dex=dex,
                )
            except Exception:
                asset = None
    else:
        meta = info.meta()
        asset = _asset_from_universe(
            meta.get("universe", []),
            api_coin=api_coin,
            symbol=symbol,
            perp_dex=None,
        )

    if asset is None:
        hint = _suggest_similar_markets(info, symbol=symbol, perp_dex=dex)
        raise ValueError(
            f"Unknown perp market {api_coin!r}. "
            f"For HIP-3 pairs set PERP_DEX (e.g. 'xyz') or use 'xyz:SYMBOL'."
            + (f" Did you mean: {', '.join(hint)}?" if hint else "")
        )

    resolved_name = str(asset.get("name", api_coin))
    if ":" in resolved_name:
        api_coin = resolved_name
        if not dex and ":" in resolved_name:
            dex_part, sym_part = resolved_name.split(":", 1)
            dex = dex_part
            symbol = _strip_quote_suffix(sym_part)

    only_isolated = bool(asset.get("onlyIsolated") or asset.get("marginMode") == "strictIsolated")
    return MarketSpec(
        api_coin=api_coin,
        symbol=symbol,
        perp_dex=dex,
        sz_decimals=int(asset.get("szDecimals", 4)),
        max_leverage=_max_leverage_from_asset(asset),
        only_isolated=only_isolated,
    )
