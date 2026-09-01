"""
Discover Hyperliquid perp pairs by 24h notional volume or 24h % movers.

Used by PAIR_SELECTION_MODE = "top_volume" / "top_movers".
Manual PAIRS mode does not use this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from .market_resolver import (
    _max_leverage_from_asset,
    api_coin_name,
    list_perp_dex_names,
    parse_coin_input,
)

VOLUME_MODES = frozenset({"top_volume", "volume", "auto_volume"})
MOVER_MODES = frozenset(
    {
        "top_movers",
        "movers",
        "gainers_losers",
        "top_gainer_loser",
        "top_gainer_losers",
        "gainer_loser",
        "gainer_losers",
    }
)


def is_volume_mode(mode: str | None) -> bool:
    return str(mode or "").strip().lower() in VOLUME_MODES


def is_mover_mode(mode: str | None) -> bool:
    return str(mode or "").strip().lower() in MOVER_MODES


def is_auto_pair_mode(mode: str | None) -> bool:
    return is_volume_mode(mode) or is_mover_mode(mode)


def valid_pair_modes() -> tuple[str, ...]:
    return ("manual", *sorted(VOLUME_MODES | MOVER_MODES))


@dataclass(frozen=True)
class VolumePair:
    api_coin: str
    symbol: str
    perp_dex: str | None
    max_leverage: int
    day_ntl_vlm: float
    sz_decimals: int
    only_isolated: bool
    day_chg_pct: float = 0.0
    mark_px: float = 0.0
    prev_day_px: float = 0.0


@dataclass
class PairUniverse:
    """[(api_coin_or_input, leverage), ...] plus optional 24h gainer/loser tags."""

    pairs: list[tuple[str, int]]
    buckets: dict[str, str] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)


# Stamp on saved params. Tune WITH the 24h move so MTF consensus can fire.
# Live reverse is a separate order-time flip (REVERSE_STRATEGY only).
MOVER_TUNE_LOCK = "with_trend"


def mover_tune_side(bucket: str) -> int:
    """
    Backtest/tune side for a 24h mover (never reversed).

    Gainers → LONG, losers → SHORT. That matches HTF consensus on a pump/dump
    so MTF can actually trigger. Live only flips if REVERSE_STRATEGY is on.
    """
    return 1 if str(bucket or "").strip().lower() == "gainer" else -1


def fade_tune_side(bucket: str, reverse: bool = False) -> int:
    """Alias of mover_tune_side. `reverse` is ignored (live flip is separate)."""
    return mover_tune_side(bucket)


def allowed_sides_for_movers(
    buckets: dict[str, str] | None,
    reverse: bool = False,
) -> dict[str, int]:
    out: dict[str, int] = {}
    for coin, bucket in (buckets or {}).items():
        b = str(bucket or "").strip().lower()
        if b not in ("gainer", "loser"):
            continue
        out[str(coin)] = mover_tune_side(b)
    return out


def mover_half_counts(n: int) -> tuple[int, int]:
    """(gainers, losers). Odd leftover goes to gainers."""
    total = max(1, int(n))
    return (total + 1) // 2, total // 2


def _as_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _fetch_meta_and_ctxs(info: Any, dex: str = "") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    POST metaAndAssetCtxs. Empty dex = native HL perps.
    Returns (meta, assetCtxs) aligned by universe index.
    """
    body: dict[str, Any] = {"type": "metaAndAssetCtxs"}
    if dex:
        body["dex"] = dex
    raw = info.post("/info", body)
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return {}, []
    meta = raw[0] if isinstance(raw[0], dict) else {}
    ctxs = raw[1] if isinstance(raw[1], list) else []
    return meta, ctxs


def _pairs_from_meta_ctxs(
    meta: dict[str, Any],
    ctxs: list[dict[str, Any]],
    *,
    perp_dex: str | None,
) -> list[VolumePair]:
    universe = meta.get("universe") or []
    out: list[VolumePair] = []
    for i, asset in enumerate(universe):
        if not isinstance(asset, dict):
            continue
        if asset.get("isDelisted") or asset.get("is_delisted"):
            continue
        name = str(asset.get("name", "") or "").strip()
        if not name:
            continue
        # HIP-3 names may already be "xyz:TICKER" or bare ticker depending on dex.
        if ":" in name:
            symbol, dex = parse_coin_input(name)
            api = name
            use_dex = dex
        else:
            symbol = name
            use_dex = perp_dex
            api = api_coin_name(symbol, use_dex)
        ctx = ctxs[i] if i < len(ctxs) and isinstance(ctxs[i], dict) else {}
        vol = _as_float(ctx.get("dayNtlVlm"), 0.0)
        if vol <= 0:
            continue
        mark = _as_float(ctx.get("markPx") or ctx.get("midPx") or ctx.get("oraclePx"), 0.0)
        prev = _as_float(ctx.get("prevDayPx"), 0.0)
        chg = ((mark - prev) / prev * 100.0) if prev > 0 and mark > 0 else 0.0
        only_iso = bool(
            asset.get("onlyIsolated") or asset.get("marginMode") == "strictIsolated"
        )
        out.append(
            VolumePair(
                api_coin=api,
                symbol=symbol,
                perp_dex=use_dex,
                max_leverage=_max_leverage_from_asset(asset),
                day_ntl_vlm=vol,
                sz_decimals=int(asset.get("szDecimals", 4) or 4),
                only_isolated=only_iso,
                day_chg_pct=chg,
                mark_px=mark,
                prev_day_px=prev,
            )
        )
    return out


def _normalize_xyz_mode(xyz_mode: str, include_xyz: bool) -> str:
    mode = str(xyz_mode or "").strip().lower()
    if mode in ("xyz", "hip3", "hip-3", "only_xyz"):
        return "xyz_only"
    if mode in ("both", "all"):
        return "include"
    if mode in ("main", "hl"):
        return "native"
    if mode in ("native", "include", "xyz_only"):
        return mode
    return "include" if include_xyz else "native"


def _collect_perp_pairs(
    info: Any,
    *,
    xyz_mode: str,
    include_xyz: bool,
    logger: logging.Logger,
    scan_label: str,
) -> tuple[list[VolumePair], str]:
    mode = _normalize_xyz_mode(xyz_mode, include_xyz)
    collected: list[VolumePair] = []

    if mode in ("native", "include"):
        meta, ctxs = _fetch_meta_and_ctxs(info, "")
        native = _pairs_from_meta_ctxs(meta, ctxs, perp_dex=None)
        collected.extend(native)
        logger.info("%s scan native: %s markets with dayNtlVlm>0", scan_label, len(native))

    if mode in ("include", "xyz_only"):
        try:
            dex_names = list_perp_dex_names(info)
        except Exception as exc:
            logger.warning("Could not list perp dexes for HIP-3 %s scan: %s", scan_label, exc)
            dex_names = []
        hip_total = 0
        for dex in dex_names:
            dex_s = str(dex or "").strip()
            if not dex_s:
                continue
            try:
                m, c = _fetch_meta_and_ctxs(info, dex_s)
                hip = _pairs_from_meta_ctxs(m, c, perp_dex=dex_s)
                collected.extend(hip)
                hip_total += len(hip)
                logger.info("%s scan %s: %s markets with dayNtlVlm>0", scan_label, dex_s, len(hip))
            except Exception as exc:
                logger.warning("%s scan failed for dex %s: %s", scan_label, dex_s, exc)
        if mode == "xyz_only" and hip_total == 0:
            logger.warning("XYZ_PAIR_MODE=xyz_only found no HIP-3 markets")

    logger.info("%s scan scope=%s collected=%s", scan_label, mode, len(collected))
    return collected, mode


def _dedupe_by_api_coin(collected: Iterable[VolumePair]) -> list[VolumePair]:
    best: dict[str, VolumePair] = {}
    for p in collected:
        key = p.api_coin.upper()
        prev = best.get(key)
        if prev is None or p.day_ntl_vlm > prev.day_ntl_vlm:
            best[key] = p
    return list(best.values())


def _filter_by_leverage(
    ranked: list[VolumePair],
    *,
    min_lev: int,
    max_lev_cap: int,
    logger: logging.Logger,
) -> list[VolumePair]:
    if min_lev > 0:
        skipped = [p for p in ranked if int(p.max_leverage) < min_lev]
        ranked = [p for p in ranked if int(p.max_leverage) >= min_lev]
        if skipped:
            sample = ", ".join(f"{p.api_coin}({p.max_leverage}x)" for p in skipped[:8])
            extra = "" if len(skipped) <= 8 else f" +{len(skipped) - 8} more"
            logger.info(
                "Min maxLev ≥%sx: dropped %s low-lev markets (%s%s)",
                min_lev,
                len(skipped),
                sample,
                extra,
            )
    if max_lev_cap > 0:
        skipped_hi = [p for p in ranked if int(p.max_leverage) > max_lev_cap]
        ranked = [p for p in ranked if int(p.max_leverage) <= max_lev_cap]
        if skipped_hi:
            sample = ", ".join(f"{p.api_coin}({p.max_leverage}x)" for p in skipped_hi[:8])
            extra = "" if len(skipped_hi) <= 8 else f" +{len(skipped_hi) - 8} more"
            logger.info(
                "Max maxLev ≤%sx: dropped %s high-lev markets (%s%s)",
                max_lev_cap,
                len(skipped_hi),
                sample,
                extra,
            )
    return ranked


def _lev_label(min_lev: int, max_lev_cap: int) -> str:
    lev_lo = min_lev if min_lev > 0 else 1
    lev_hi = max_lev_cap if max_lev_cap > 0 else None
    return f"{lev_lo}–{lev_hi}x" if lev_hi is not None else f"≥{lev_lo}x"


def _assign_discovered_leverage(
    discovered: list[VolumePair],
    *,
    use_max_leverage: bool,
    leverage_overrides: dict[str, int],
    requested_leverage_for,
) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for p in discovered:
        if use_max_leverage:
            lev = max(1, int(p.max_leverage))
            ov = None
            for key in (p.api_coin, p.symbol, f"{p.perp_dex}:{p.symbol}" if p.perp_dex else ""):
                if key and key in leverage_overrides:
                    ov = int(leverage_overrides[key])
                    break
                for ok, ov_v in leverage_overrides.items():
                    if str(ok).strip().upper() == str(key).upper():
                        ov = int(ov_v)
                        break
                if ov is not None:
                    break
            if ov is not None:
                lev = max(1, min(int(ov), int(p.max_leverage)))
        else:
            lev = max(
                1,
                min(
                    int(requested_leverage_for(p.api_coin, p.symbol)),
                    int(p.max_leverage),
                ),
            )
        out.append((p.api_coin, lev))
    return out


def discover_top_volume_pairs(
    info: Any,
    top_n: int,
    *,
    include_xyz: bool = False,
    xyz_mode: str = "native",
    min_max_leverage: int = 0,
    max_max_leverage: int = 0,
    logger: logging.Logger | None = None,
) -> list[VolumePair]:
    """
    Rank perps by 24h notional volume and return the top `top_n`.

    xyz_mode:
      - "native"   → main HL book only
      - "include"  → native + HIP-3
      - "xyz_only" → HIP-3 builder dexes only
    include_xyz is legacy: True maps to "include" when xyz_mode is blank/native
    and caller still passes the old flag alone.
    min_max_leverage → keep only markets whose exchange maxLev ≥ this
    max_max_leverage → keep only markets whose exchange maxLev ≤ this (0 = off)
    Then take the top `top_n` by volume among those.
    """
    log = logger or logging.getLogger("hl-multi")
    n = max(1, int(top_n))
    min_lev = max(0, int(min_max_leverage or 0))
    max_lev_cap = max(0, int(max_max_leverage or 0))
    if max_lev_cap > 0 and min_lev > 0 and max_lev_cap < min_lev:
        log.warning(
            "MAX_MAX_LEVERAGE (%s) < MIN_MAX_LEVERAGE (%s) — no markets can qualify",
            max_lev_cap,
            min_lev,
        )

    collected, mode = _collect_perp_pairs(
        info,
        xyz_mode=xyz_mode,
        include_xyz=include_xyz,
        logger=log,
        scan_label="Volume",
    )
    ranked = sorted(_dedupe_by_api_coin(collected), key=lambda p: p.day_ntl_vlm, reverse=True)
    ranked = _filter_by_leverage(
        ranked, min_lev=min_lev, max_lev_cap=max_lev_cap, logger=log
    )
    top = ranked[:n]
    lev_label = _lev_label(min_lev, max_lev_cap)
    if top:
        if len(top) < n:
            log.warning(
                "Only %s/%s qualifying markets (scope=%s maxLev %s) — using all of them",
                len(top),
                n,
                mode,
                lev_label,
            )
        log.info(
            "Top volume %s/%s (scope=%s maxLev %s) | #1 %s $%.0f %sx | #%s %s $%.0f %sx",
            len(top),
            len(ranked),
            mode,
            lev_label,
            top[0].api_coin,
            top[0].day_ntl_vlm,
            top[0].max_leverage,
            len(top),
            top[-1].api_coin,
            top[-1].day_ntl_vlm,
            top[-1].max_leverage,
        )
    else:
        log.warning(
            "Top volume scan returned no markets (scope=%s maxLev %s)",
            mode,
            lev_label,
        )
    return top


def discover_top_mover_pairs(
    info: Any,
    top_n: int,
    *,
    include_xyz: bool = False,
    xyz_mode: str = "native",
    min_max_leverage: int = 0,
    max_max_leverage: int = 0,
    logger: logging.Logger | None = None,
) -> tuple[list[VolumePair], dict[str, str]]:
    """
    Rank perps by 24h % change vs prevDayPx.

    Returns (pairs, buckets) where buckets maps api_coin → "gainer"|"loser".
    Half the look-set are the strongest gainers, half the weakest losers
    (odd leftover goes to gainers). No coin is in both buckets.
    """
    log = logger or logging.getLogger("hl-multi")
    n = max(1, int(top_n))
    n_gain, n_lose = mover_half_counts(n)
    min_lev = max(0, int(min_max_leverage or 0))
    max_lev_cap = max(0, int(max_max_leverage or 0))
    if max_lev_cap > 0 and min_lev > 0 and max_lev_cap < min_lev:
        log.warning(
            "MAX_MAX_LEVERAGE (%s) < MIN_MAX_LEVERAGE (%s) — no markets can qualify",
            max_lev_cap,
            min_lev,
        )

    collected, mode = _collect_perp_pairs(
        info,
        xyz_mode=xyz_mode,
        include_xyz=include_xyz,
        logger=log,
        scan_label="Mover",
    )
    ranked = _filter_by_leverage(
        _dedupe_by_api_coin(collected),
        min_lev=min_lev,
        max_lev_cap=max_lev_cap,
        logger=log,
    )
    eligible = [p for p in ranked if p.prev_day_px > 0 and p.mark_px > 0]
    skipped_px = len(ranked) - len(eligible)
    if skipped_px:
        log.info("Mover scan: skipped %s markets without prevDayPx/markPx", skipped_px)

    by_chg = sorted(eligible, key=lambda p: p.day_chg_pct, reverse=True)
    gainers = by_chg[:n_gain]
    gainer_keys = {p.api_coin.upper() for p in gainers}
    losers_pool = [p for p in by_chg if p.api_coin.upper() not in gainer_keys]
    losers_pool.sort(key=lambda p: p.day_chg_pct)
    losers = losers_pool[:n_lose]

    # If one side is short (tiny universe), fill leftover slots from the other.
    leftover = n - len(gainers) - len(losers)
    if leftover > 0:
        used = {p.api_coin.upper() for p in gainers} | {p.api_coin.upper() for p in losers}
        rest = [p for p in by_chg if p.api_coin.upper() not in used]
        if len(gainers) < n_gain:
            extra = rest[:leftover]
            gainers.extend(extra)
        else:
            rest_lose = sorted(rest, key=lambda p: p.day_chg_pct)
            losers.extend(rest_lose[:leftover])

    buckets: dict[str, str] = {}
    for p in gainers:
        buckets[p.api_coin] = "gainer"
    for p in losers:
        buckets[p.api_coin] = "loser"

    # Gainers (hottest first) then losers (weakest first) so logs read clearly.
    top = list(gainers) + list(losers)
    lev_label = _lev_label(min_lev, max_lev_cap)
    if top:
        g_txt = ", ".join(f"{p.api_coin}({p.day_chg_pct:+.1f}%)" for p in gainers[:7])
        l_txt = ", ".join(f"{p.api_coin}({p.day_chg_pct:+.1f}%)" for p in losers[:7])
        extra_g = f" +{len(gainers) - 7}" if len(gainers) > 7 else ""
        extra_l = f" +{len(losers) - 7}" if len(losers) > 7 else ""
        log.info(
            "Top movers %s (scope=%s maxLev %s) | %s gainers + %s losers of %s with 24h %%",
            len(top),
            mode,
            lev_label,
            len(gainers),
            len(losers),
            len(eligible),
        )
        if gainers:
            log.info("24h gainers: %s%s", g_txt, extra_g)
        if losers:
            log.info("24h losers: %s%s", l_txt, extra_l)
        if len(top) < n:
            log.warning(
                "Only %s/%s qualifying mover markets (scope=%s maxLev %s)",
                len(top),
                n,
                mode,
                lev_label,
            )
    else:
        log.warning(
            "Top mover scan returned no markets (scope=%s maxLev %s)",
            mode,
            lev_label,
        )
    return top, buckets


def resolve_pair_universe(
    info: Any,
    *,
    mode: str,
    manual_pairs: tuple[str, ...] | list[str],
    top_volume_count: int,
    include_xyz: bool,
    use_max_leverage: bool,
    default_leverage: int,
    leverage_overrides: dict[str, int] | None,
    requested_leverage_for,
    min_max_leverage: int = 0,
    max_max_leverage: int = 0,
    xyz_mode: str | None = None,
    top_mover_count: int | None = None,
    logger: logging.Logger | None = None,
) -> PairUniverse:
    """
    Returns PairUniverse for tune + watch.

    mode:
      - "manual": use manual_pairs + PAIR_LEVERAGE / LEVERAGE
      - "top_volume": discover by dayNtlVlm
      - "top_movers": half 24h gainers + half 24h losers
    """
    log = logger or logging.getLogger("hl-multi")
    mode_s = (mode or "manual").strip().lower()
    overrides = leverage_overrides if isinstance(leverage_overrides, dict) else {}
    scope = str(xyz_mode or "").strip().lower()
    if not scope:
        scope = "include" if include_xyz else "native"

    if is_volume_mode(mode_s):
        discovered = discover_top_volume_pairs(
            info,
            top_volume_count,
            include_xyz=bool(include_xyz),
            xyz_mode=scope,
            min_max_leverage=int(min_max_leverage or 0),
            max_max_leverage=int(max_max_leverage or 0),
            logger=log,
        )
        if not discovered:
            raise RuntimeError(
                "PAIR_SELECTION_MODE=top_volume found no markets — check "
                "XYZ_PAIR_MODE / INCLUDE_XYZ_PAIRS / MIN_MAX_LEVERAGE / MAX_MAX_LEVERAGE"
            )
        pairs = _assign_discovered_leverage(
            discovered,
            use_max_leverage=use_max_leverage,
            leverage_overrides=overrides,
            requested_leverage_for=requested_leverage_for,
        )
        return PairUniverse(pairs=pairs)

    if is_mover_mode(mode_s):
        n = int(top_mover_count) if top_mover_count is not None else int(top_volume_count)
        discovered, buckets = discover_top_mover_pairs(
            info,
            n,
            include_xyz=bool(include_xyz),
            xyz_mode=scope,
            min_max_leverage=int(min_max_leverage or 0),
            max_max_leverage=int(max_max_leverage or 0),
            logger=log,
        )
        if not discovered:
            raise RuntimeError(
                "PAIR_SELECTION_MODE=top_movers found no markets — check "
                "XYZ_PAIR_MODE / INCLUDE_XYZ_PAIRS / MIN_MAX_LEVERAGE / MAX_MAX_LEVERAGE"
            )
        pairs = _assign_discovered_leverage(
            discovered,
            use_max_leverage=use_max_leverage,
            leverage_overrides=overrides,
            requested_leverage_for=requested_leverage_for,
        )
        return PairUniverse(pairs=pairs, buckets=buckets)

    # Manual selection (existing behavior)
    pair_list = [str(x).strip() for x in manual_pairs if str(x).strip()]
    if not pair_list:
        raise ValueError("PAIRS must list at least one pair in manual mode")
    out: list[tuple[str, int]] = []
    for raw in pair_list:
        sym, dex = parse_coin_input(raw)
        api = f"{dex}:{sym}" if dex else sym
        lev = max(1, int(requested_leverage_for(raw, api, sym)))
        out.append((raw, lev))
    log.info(
        "Manual pair mode: %s pairs | default lev=%sx | overrides=%s",
        len(out),
        default_leverage,
        len(overrides),
    )
    return PairUniverse(pairs=out)
