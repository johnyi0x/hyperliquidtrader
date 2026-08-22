"""
Discover Hyperliquid perp pairs by 24h notional volume (dayNtlVlm).

Used by PAIR_SELECTION_MODE = "top_volume". Manual PAIRS mode does not use this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .market_resolver import (
    _max_leverage_from_asset,
    api_coin_name,
    list_perp_dex_names,
    parse_coin_input,
)


@dataclass(frozen=True)
class VolumePair:
    api_coin: str
    symbol: str
    perp_dex: str | None
    max_leverage: int
    day_ntl_vlm: float
    sz_decimals: int
    only_isolated: bool


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
            )
        )
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
    mode = str(xyz_mode or "").strip().lower()
    if mode in ("xyz", "hip3", "hip-3", "only_xyz"):
        mode = "xyz_only"
    elif mode in ("both", "all"):
        mode = "include"
    elif mode in ("main", "hl"):
        mode = "native"
    if mode not in ("native", "include", "xyz_only"):
        mode = "include" if include_xyz else "native"

    collected: list[VolumePair] = []

    if mode in ("native", "include"):
        meta, ctxs = _fetch_meta_and_ctxs(info, "")
        native = _pairs_from_meta_ctxs(meta, ctxs, perp_dex=None)
        collected.extend(native)
        log.info("Volume scan native: %s markets with dayNtlVlm>0", len(native))

    if mode in ("include", "xyz_only"):
        try:
            dex_names = list_perp_dex_names(info)
        except Exception as exc:
            log.warning("Could not list perp dexes for HIP-3 volume scan: %s", exc)
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
                log.info("Volume scan %s: %s markets with dayNtlVlm>0", dex_s, len(hip))
            except Exception as exc:
                log.warning("Volume scan failed for dex %s: %s", dex_s, exc)
        if mode == "xyz_only" and hip_total == 0:
            log.warning("XYZ_PAIR_MODE=xyz_only found no HIP-3 markets")

    log.info("Volume scan scope=%s collected=%s", mode, len(collected))

    # De-dupe by api_coin (keep highest volume if duplicates).
    best: dict[str, VolumePair] = {}
    for p in collected:
        key = p.api_coin.upper()
        prev = best.get(key)
        if prev is None or p.day_ntl_vlm > prev.day_ntl_vlm:
            best[key] = p

    ranked = sorted(best.values(), key=lambda p: p.day_ntl_vlm, reverse=True)
    if min_lev > 0:
        skipped = [p for p in ranked if int(p.max_leverage) < min_lev]
        ranked = [p for p in ranked if int(p.max_leverage) >= min_lev]
        if skipped:
            sample = ", ".join(
                f"{p.api_coin}({p.max_leverage}x)" for p in skipped[:8]
            )
            extra = "" if len(skipped) <= 8 else f" +{len(skipped) - 8} more"
            log.info(
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
            sample = ", ".join(
                f"{p.api_coin}({p.max_leverage}x)" for p in skipped_hi[:8]
            )
            extra = "" if len(skipped_hi) <= 8 else f" +{len(skipped_hi) - 8} more"
            log.info(
                "Max maxLev ≤%sx: dropped %s high-lev markets (%s%s)",
                max_lev_cap,
                len(skipped_hi),
                sample,
                extra,
            )
    top = ranked[:n]
    lev_lo = min_lev if min_lev > 0 else 1
    lev_hi = max_lev_cap if max_lev_cap > 0 else None
    lev_label = f"{lev_lo}–{lev_hi}x" if lev_hi is not None else f"≥{lev_lo}x"
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
    logger: logging.Logger | None = None,
) -> list[tuple[str, int]]:
    """
    Returns [(api_coin_or_input, leverage), ...] for tune + watch.

    mode:
      - "manual": use manual_pairs + PAIR_LEVERAGE / LEVERAGE
      - "top_volume": discover by dayNtlVlm; leverage = exchange max (unless override)
    """
    log = logger or logging.getLogger("hl-multi")
    mode_s = (mode or "manual").strip().lower()
    overrides = leverage_overrides if isinstance(leverage_overrides, dict) else {}
    scope = str(xyz_mode or "").strip().lower()
    if not scope:
        scope = "include" if include_xyz else "native"

    if mode_s in ("top_volume", "volume", "auto_volume"):
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
        out: list[tuple[str, int]] = []
        for p in discovered:
            # Prefer exchange max; allow rare manual overrides by api/symbol.
            if use_max_leverage:
                lev = max(1, int(p.max_leverage))
                # Optional overrides still win if present.
                ov = None
                for key in (p.api_coin, p.symbol, f"{p.perp_dex}:{p.symbol}" if p.perp_dex else ""):
                    if key and key in overrides:
                        ov = int(overrides[key])
                        break
                    for ok, ov_v in overrides.items():
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

    # Manual selection (existing behavior)
    pairs = [str(x).strip() for x in manual_pairs if str(x).strip()]
    if not pairs:
        raise ValueError("PAIRS must list at least one pair in manual mode")
    out = []
    for raw in pairs:
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
    return out
