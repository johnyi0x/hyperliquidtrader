"""Pair+side universe from top-wallet majority holds (meta-follower style).

Imports profit-meta-follower's leaderboard / snapshot / tally helpers. Does not
edit that tree. Used when PAIR_SELECTION_MODE is majority / wallet_majority.

Long → gainer bucket, short → loser bucket so existing MTF side-lock works.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .pair_universe import (
    VolumePair,
    _assign_discovered_leverage,
    _collect_perp_pairs,
    _dedupe_by_api_coin,
    _normalize_xyz_mode,
)

_PMF_ROOT = Path(__file__).resolve().parents[1] / "profit-meta-follower"


def bucket_for_side(side: str) -> str:
    return "gainer" if str(side or "").strip().lower() == "long" else "loser"


def _ensure_pmf_path() -> Path:
    root = _PMF_ROOT
    if not root.is_dir():
        raise RuntimeError(
            f"profit-meta-follower not found at {root} — needed for majority pair mode"
        )
    p = str(root)
    if p not in sys.path:
        sys.path.insert(0, p)
    return root


def _window_roi(row: Any, name: str) -> float:
    alias = {"month": "perpMonth", "week": "perpWeek", "day": "perpDay"}.get(name, name)
    windows = getattr(row, "windows", None) or {}
    w = windows.get(name) or windows.get(alias)
    if w is None:
        return float("-inf")
    try:
        return float(w.roi)
    except (TypeError, ValueError, AttributeError):
        return float("-inf")


def _pmf_tally_cfg(
    *,
    xyz_mode: str,
    max_pairs: int,
    min_hold_pct: float,
    min_agreement: float,
    min_notional: float,
    stale_s: float,
) -> SimpleNamespace:
    scope = str(xyz_mode or "native").strip().lower()
    if scope not in ("native", "include", "xyz_only"):
        scope = "native"
    return SimpleNamespace(
        DEX_SCOPE=scope,
        ALLOW_COINS=(),
        DENY_COINS=(),
        STALE_SNAPSHOT_S=float(stale_s),
        MAJORITY_MIN_NOTIONAL_USD=float(min_notional),
        MAJORITY_MIN_HOLD_PCT=float(min_hold_pct),
        MAJORITY_EXIT_HOLD_PCT=min(float(min_hold_pct), 0.03),
        MAJORITY_MIN_SIDE_AGREEMENT=float(min_agreement),
        MAJORITY_SINGLE_PAIR=False,
        MAJORITY_STICKY=False,
        MAX_COINS_IN_BOOK=max(1, int(max_pairs)),
        OUR_GROSS_MARGIN_PCT=90.0,
        MAJORITY_MAX_PAIR_SHARE=0.70,
        OUR_MIN_LEVERAGE=1,
        OUR_MAX_LEVERAGE=50,
        MAJORITY_LEVERAGE_DIV=1.0,
    )


def _index_markets(pairs: list[VolumePair]) -> dict[str, VolumePair]:
    idx: dict[str, VolumePair] = {}
    for p in pairs:
        idx[p.api_coin.upper()] = p
        idx[p.symbol.upper()] = p
        if p.perp_dex:
            idx[f"{p.perp_dex}:{p.symbol}".upper()] = p
    return idx


def _match_market(coin: str, idx: dict[str, VolumePair]) -> VolumePair | None:
    raw = str(coin or "").strip()
    if not raw:
        return None
    hit = idx.get(raw.upper())
    if hit is not None:
        return hit
    if ":" in raw:
        return idx.get(raw.split(":", 1)[-1].upper())
    return None


def _pair_payload(p: VolumePair, side: str) -> dict[str, Any]:
    return {
        "api_coin": p.api_coin,
        "symbol": p.symbol,
        "perp_dex": p.perp_dex,
        "max_leverage": int(p.max_leverage),
        "day_ntl_vlm": float(p.day_ntl_vlm),
        "sz_decimals": int(p.sz_decimals),
        "only_isolated": bool(p.only_isolated),
        "day_chg_pct": float(p.day_chg_pct),
        "mark_px": float(p.mark_px),
        "prev_day_px": float(p.prev_day_px),
        "side": str(side),
    }


def _pair_from_payload(row: dict[str, Any]) -> VolumePair:
    dex = row.get("perp_dex")
    return VolumePair(
        api_coin=str(row.get("api_coin") or ""),
        symbol=str(row.get("symbol") or ""),
        perp_dex=str(dex) if dex else None,
        max_leverage=int(row.get("max_leverage") or 1),
        day_ntl_vlm=float(row.get("day_ntl_vlm") or 0),
        sz_decimals=int(row.get("sz_decimals") or 4),
        only_isolated=bool(row.get("only_isolated")),
        day_chg_pct=float(row.get("day_chg_pct") or 0),
        mark_px=float(row.get("mark_px") or 0),
        prev_day_px=float(row.get("prev_day_px") or 0),
    )


def _cache_path(data_dir: Path) -> Path:
    return Path(data_dir) / "majority_universe.json"


def _load_cache(path: Path, max_age_h: float) -> dict[str, Any] | None:
    if not path.exists() or max_age_h <= 0:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    age_h = (time.time() - float(raw.get("fetched_at") or 0)) / 3600.0
    if age_h > max_age_h:
        return None
    pairs = raw.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        return None
    raw["_age_h"] = age_h
    return raw


def _save_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _log_board(log: logging.Logger, annotated: list[Any], *, n: int = 16) -> None:
    from pmf.majority import compact_hold_board

    log.info("MAJORITY board | %s", compact_hold_board(annotated, n=n))


def discover_majority_pairs(
    info: Any,
    *,
    data_dir: Path,
    max_pairs: int,
    basket_size: int,
    snap_sleep_s: float,
    refresh_hours: float,
    rank_window: str,
    xyz_mode: str,
    include_xyz: bool,
    use_max_leverage: bool,
    leverage_overrides: dict[str, int] | None,
    requested_leverage_for,
    min_max_leverage: int = 0,
    max_max_leverage: int = 0,
    min_day_notional: float = 0.0,
    min_hold_pct: float = 0.05,
    min_agreement: float = 0.55,
    min_wallet_notional: float = 50.0,
    leaderboard_cache_hours: float = 6.0,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[list[VolumePair], dict[str, str]]:
    """Top-N majority (coin, side) mapped onto live HL markets.

    Returns (VolumePair list in rank order, buckets gainer/loser).
    """
    log = logger or logging.getLogger("hl-multi")
    max_n = max(1, int(max_pairs or 1))
    cache_h = max(0.05, float(refresh_hours or 2.5))
    cache_file = _cache_path(Path(data_dir))
    want_xyz = _normalize_xyz_mode(xyz_mode, include_xyz)
    if not force:
        cached = _load_cache(cache_file, cache_h)
        if cached is not None and str(cached.get("xyz_mode") or "") != want_xyz:
            log.info(
                "MAJORITY cache ignore — scope %s vs want %s (HIP-3 eligible)",
                cached.get("xyz_mode") or "?",
                want_xyz,
            )
            cached = None
        if cached is not None:
            discovered: list[VolumePair] = []
            buckets: dict[str, str] = {}
            for row in cached.get("pairs") or []:
                if not isinstance(row, dict):
                    continue
                p = _pair_from_payload(row)
                if not p.api_coin:
                    continue
                side = str(row.get("side") or "long")
                discovered.append(p)
                buckets[p.api_coin] = bucket_for_side(side)
            if discovered:
                log.info(
                    "MAJORITY cache hit (%.1fh old, max_pairs=%s) | %s",
                    float(cached.get("_age_h") or 0),
                    max_n,
                    ", ".join(
                        f"{p.api_coin} {str(r.get('side') or '?')}"
                        for p, r in zip(discovered, cached.get("pairs") or [])
                    ),
                )
                for line in cached.get("log_lines") or []:
                    log.info("%s", line)
                return discovered[:max_n], buckets

    _ensure_pmf_path()
    from pmf.leaderboard import load_leaderboard
    from pmf.majority import compact_hold_board, pick_majority_targets, tally_holds
    from pmf.snapshots import SnapshotClient

    collected, mode = _collect_perp_pairs(
        info,
        xyz_mode=xyz_mode,
        include_xyz=include_xyz,
        logger=log,
        scan_label="MAJORITY markets",
    )
    listed = _dedupe_by_api_coin(collected)
    idx = _index_markets(listed)
    min_lev = max(0, int(min_max_leverage or 0))
    max_lev_cap = max(0, int(max_max_leverage or 0))
    min_vol = float(min_day_notional or 0.0)

    lb_path = Path(data_dir) / "majority_leaderboard.json"
    rows = load_leaderboard(lb_path, float(leaderboard_cache_hours), log)
    win = str(rank_window or "week").strip().lower() or "week"
    ranked = sorted(rows, key=lambda r: _window_roi(r, win), reverse=True)
    n_basket = max(8, int(basket_size or 120))
    addrs = [r.address for r in ranked[:n_basket] if getattr(r, "address", None)]
    if len(addrs) < 8:
        raise RuntimeError(
            f"MAJORITY leaderboard too small ({len(addrs)} wallets after {win} ROI rank)"
        )
    log.info(
        "MAJORITY snap %s wallets (top %s by %s ROI, sleep=%.2fs, scope=%s)",
        len(addrs),
        n_basket,
        win,
        float(snap_sleep_s or 0),
        mode,
    )

    snapper = SnapshotClient(info, log, [""])
    now = time.time()
    snaps = []
    sleep_s = max(0.0, float(snap_sleep_s or 0.0))
    errors = 0
    for i, addr in enumerate(addrs, start=1):
        snaps.append(snapper.snapshot(addr, now))
        if snaps[-1].error:
            errors += 1
        if i == 1 or i == len(addrs) or i % 25 == 0:
            log.info(
                "MAJORITY snap progress %s/%s errors=%s",
                i,
                len(addrs),
                errors,
            )
        if sleep_s > 0 and i < len(addrs):
            time.sleep(sleep_s)

    tally_cfg = _pmf_tally_cfg(
        xyz_mode=mode,
        max_pairs=max(max_n, 16),
        min_hold_pct=float(min_hold_pct),
        min_agreement=float(min_agreement),
        min_notional=float(min_wallet_notional),
        stale_s=1800.0,
    )
    hold_rows, stats = tally_holds(snaps, tally_cfg, now=time.time())
    log.info(
        "MAJORITY tally ok=%s/%s empty=%s errors=%s stale=%s coins=%s",
        stats.get("ok"),
        stats.get("snapped"),
        stats.get("empty"),
        stats.get("errors"),
        stats.get("stale"),
        stats.get("coins"),
    )
    n_ok = int(stats.get("ok") or 0)
    if n_ok < 20:
        raise RuntimeError(
            f"MAJORITY only {n_ok} usable wallet snapshots — not enough votes (API errors={errors})"
        )

    tally_cfg.MAX_COINS_IN_BOOK = max(max_n, 16)
    _targets, annotated, _meta = pick_majority_targets(
        hold_rows,
        tally_cfg,
        managed=set(),
        markets={},
    )
    _log_board(log, annotated, n=16)

    extra_skip_logs: list[str] = []
    selected: list[tuple[VolumePair, Any]] = []
    overflow: list[Any] = []
    for row in annotated:
        why = str(getattr(row, "skip", "") or "")
        if why:
            extra_skip_logs.append(
                "MAJORITY skip %s %s — %s hold=%.1f%% agr=%.0f%% L=%s S=%s"
                % (
                    row.coin,
                    row.side,
                    why,
                    float(row.hold_pct) * 100.0,
                    float(row.agreement) * 100.0,
                    row.long_n,
                    row.short_n,
                )
            )
            continue
        market = _match_market(row.coin, idx)
        if market is None:
            extra_skip_logs.append(
                "MAJORITY skip %s %s — not_listed hold=%.1f%% agr=%.0f%% L=%s S=%s"
                % (
                    row.coin,
                    row.side,
                    float(row.hold_pct) * 100.0,
                    float(row.agreement) * 100.0,
                    row.long_n,
                    row.short_n,
                )
            )
            continue
        # HIP-3 (xyz: stocks, etc.) stay eligible even when movers/volume
        # scans use XYZ_PAIR_MODE=native plus min lev / min volume cuts.
        if not market.perp_dex:
            if min_lev > 0 and int(market.max_leverage) < min_lev:
                extra_skip_logs.append(
                    "MAJORITY skip %s %s — min_lev %sx<%sx hold=%.1f%%"
                    % (
                        market.api_coin,
                        row.side,
                        market.max_leverage,
                        min_lev,
                        float(row.hold_pct) * 100.0,
                    )
                )
                continue
            if max_lev_cap > 0 and int(market.max_leverage) > max_lev_cap:
                extra_skip_logs.append(
                    "MAJORITY skip %s %s — max_lev %sx>%sx hold=%.1f%%"
                    % (
                        market.api_coin,
                        row.side,
                        market.max_leverage,
                        max_lev_cap,
                        float(row.hold_pct) * 100.0,
                    )
                )
                continue
            if min_vol > 0 and float(market.day_ntl_vlm) < min_vol:
                extra_skip_logs.append(
                    "MAJORITY skip %s %s — min_vol $%.0f<$%.0f hold=%.1f%%"
                    % (
                        market.api_coin,
                        row.side,
                        market.day_ntl_vlm,
                        min_vol,
                        float(row.hold_pct) * 100.0,
                    )
                )
                continue
        if len(selected) >= max_n:
            overflow.append(row)
            continue
        selected.append((market, row))

    for line in extra_skip_logs[:24]:
        log.info("%s", line)
    if len(extra_skip_logs) > 24:
        log.info("MAJORITY skip … +%s more", len(extra_skip_logs) - 24)
    for row in overflow[:16]:
        log.info(
            "MAJORITY not selected %s %s — rank cap (max_pairs=%s) hold=%.1f%% agr=%.0f%% L=%s S=%s",
            row.coin,
            row.side,
            max_n,
            float(row.hold_pct) * 100.0,
            float(row.agreement) * 100.0,
            row.long_n,
            row.short_n,
        )

    if not selected:
        raise RuntimeError(
            "PAIR_SELECTION_MODE=majority found no tradeable pairs — "
            "check XYZ_PAIR_MODE / MIN_MAX_LEVERAGE / MIN_DAY_NOTIONAL_USD / "
            "MAJORITY_MIN_HOLD_PCT"
        )

    discovered = [p for p, _ in selected]
    buckets = {p.api_coin: bucket_for_side(r.side) for p, r in selected}
    pick_txt = ", ".join(
        "%s %s hold=%.1f%% agr=%.0f%% wallets=%s lev_med=%sx"
        % (
            p.api_coin,
            r.side,
            float(r.hold_pct) * 100.0,
            float(r.agreement) * 100.0,
            r.wallets,
            r.median_leverage,
        )
        for p, r in selected
    )
    why_txt = (
        "top wallets by %s ROI currently hold these names on the majority side "
        "(min hold %.0f%%, min agreement %.0f%%, no sticky — refresh drops off-list names)"
        % (win, float(min_hold_pct) * 100.0, float(min_agreement) * 100.0)
    )
    log.info(
        "MAJORITY pick %s/%s eligible_listed=%s skipped_or_unlisted=%s overflow=%s | %s",
        len(selected),
        max_n,
        len(selected) + len(overflow),
        len(extra_skip_logs),
        len(overflow),
        pick_txt,
    )
    log.info("MAJORITY why | %s", why_txt)

    log_lines = [
        "MAJORITY board | %s" % compact_hold_board(annotated, n=16),
        "MAJORITY pick %s/%s | %s" % (len(selected), max_n, pick_txt),
        "MAJORITY why | %s" % why_txt,
        *extra_skip_logs[:24],
        *[
            "MAJORITY not selected %s %s — rank cap (max_pairs=%s) hold=%.1f%%"
            % (r.coin, r.side, max_n, float(r.hold_pct) * 100.0)
            for r in overflow[:16]
        ],
    ]
    _save_cache(
        cache_file,
        {
            "fetched_at": time.time(),
            "max_pairs": max_n,
            "xyz_mode": want_xyz,
            "pairs": [_pair_payload(p, r.side) for p, r in selected],
            "stats": stats,
            "board": [asdict(r) if hasattr(r, "__dataclass_fields__") else {} for r in annotated[:20]],
            "log_lines": log_lines,
        },
    )
    return discovered, buckets


def assign_majority_leverage(
    discovered: list[VolumePair],
    *,
    use_max_leverage: bool,
    leverage_overrides: dict[str, int] | None,
    requested_leverage_for,
) -> list[tuple[str, int]]:
    return _assign_discovered_leverage(
        discovered,
        use_max_leverage=use_max_leverage,
        leverage_overrides=leverage_overrides if isinstance(leverage_overrides, dict) else {},
        requested_leverage_for=requested_leverage_for,
    )


def majority_resolve_kwargs(cfg: Any, *, data_dir: Path, force: bool = False) -> dict[str, Any]:
    """Extra resolve_pair_universe kwargs for majority mode."""
    return {
        "data_dir": Path(data_dir),
        "majority_force": bool(force),
        "majority_max_pairs": int(getattr(cfg, "MAJORITY_MAX_PAIRS", 2) or 2),
        "majority_basket_size": int(getattr(cfg, "MAJORITY_BASKET_SIZE", 120) or 120),
        "majority_snap_sleep_s": float(getattr(cfg, "MAJORITY_SNAP_SLEEP_S", 0.22) or 0.22),
        "majority_refresh_hours": float(getattr(cfg, "MAJORITY_LIST_REFRESH_HOURS", 2.5) or 2.5),
        "majority_rank_window": str(getattr(cfg, "MAJORITY_RANK_WINDOW", "week") or "week"),
        "majority_min_hold_pct": float(getattr(cfg, "MAJORITY_MIN_HOLD_PCT", 0.05) or 0.05),
        "majority_min_agreement": float(
            getattr(cfg, "MAJORITY_MIN_SIDE_AGREEMENT", 0.55) or 0.55
        ),
        "majority_min_wallet_notional": float(
            getattr(cfg, "MAJORITY_MIN_WALLET_NOTIONAL_USD", 50.0) or 50.0
        ),
        "majority_leaderboard_cache_hours": float(
            getattr(cfg, "MAJORITY_LEADERBOARD_CACHE_HOURS", 6.0) or 6.0
        ),
        # Wallet list may rank HIP-3 (xyz:) names. Do not inherit XYZ_PAIR_MODE=native.
        "majority_xyz_mode": "include",
        "majority_min_day_notional": 0.0,
    }
