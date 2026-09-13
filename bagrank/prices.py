"""Join collector coin_prices onto the rank panel.

Fill order matches live: snapshot mark at that hour, then 1h open, then 1h close.
Hyperliquid candle fetch is only an optional gap-fill for hours the collector missed.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .store import candle_span, load_coin_prices, upsert_candles
from .timeutil import HOUR_S, to_unix

log = logging.getLogger("bagrank.prices")


def _fpos(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        px = float(value)
    except (TypeError, ValueError):
        return 0.0
    return px if px > 0.0 else 0.0


def row_fill_px(row: dict[str, Any]) -> float:
    """Collector mark at cycle time; fall back to that hour's candle."""
    for key in ("mark_px", "ohlc_open", "ohlc_close"):
        px = _fpos(row.get(key))
        if px > 0.0:
            return px
    return 0.0


def _fany(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fill_panel_from_collector(conn: Any, panel: Any, *, venue: str) -> dict[str, int]:
    rows = load_coin_prices(conn, venue=venue)
    t_ix = {int(ts): i for i, ts in enumerate(panel.cycle_unix)}
    c_ix = {c: j for j, c in enumerate(panel.coins)}
    filled = 0
    for row in rows:
        ts = to_unix(row["cycle_ts"])
        coin = str(row.get("coin") or "")
        if ts not in t_ix or coin not in c_ix:
            continue
        i = t_ix[ts]
        j = c_ix[coin]
        px = row_fill_px(row)
        if px > 0.0:
            panel.marks[i, j] = px
            filled += 1
        panel.funding[i, j] = _fany(row.get("funding"))
        panel.oi[i, j] = _fany(row.get("open_interest"))
        panel.premium[i, j] = _fany(row.get("premium"))
        panel.volume[i, j] = _fany(row.get("day_ntl_vlm") or row.get("ohlc_volume"))
        panel.prev_day[i, j] = _fany(row.get("prev_day_px"))
    needed = int((panel.rank > 0).sum()) if panel.n_times and panel.n_coins else 0
    missing = 0
    if needed:
        missing = int(((panel.rank > 0) & (panel.marks <= 0.0)).sum())
    log.debug(
        "Collector prices | joined=%s on-board-missing=%s",
        filled,
        missing,
    )
    return {"filled": filled, "missing": missing, "needed": needed}


def _info_client():
    from hyperliquid.info import Info
    from hyperliquid.utils import constants

    return Info(constants.MAINNET_API_URL, skip_ws=True)


def _parse_candles(coin: str, interval: str, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in raw or []:
        try:
            open_ts = int(c["t"]) // 1000
            out.append(
                {
                    "coin": coin,
                    "interval": interval,
                    "open_ts": open_ts,
                    "open": float(c["o"]),
                    "high": float(c["h"]),
                    "low": float(c["l"]),
                    "close": float(c["c"]),
                    "volume": float(c.get("v") or 0),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def fetch_coin_candles(
    info: Any,
    coin: str,
    *,
    start_unix: int,
    end_unix: int,
    interval: str = "1h",
) -> list[dict[str, Any]]:
    start_ms = int(start_unix) * 1000
    end_ms = int(end_unix) * 1000
    raw = info.candles_snapshot(coin, interval, start_ms, end_ms)
    if not isinstance(raw, list):
        return []
    return _parse_candles(coin, interval, raw)


def sync_hl_gaps(
    conn: Any,
    panel: Any,
    *,
    sleep_s: float = 0.12,
    info: Any | None = None,
) -> dict[str, int]:
    """Optional: fetch HL 1h candles only for coins still missing a fill."""
    missing_coins: list[str] = []
    for j, coin in enumerate(panel.coins):
        if not bool(((panel.rank[:, j] > 0) & (panel.marks[:, j] <= 0.0)).any()):
            continue
        missing_coins.append(coin)
    if not missing_coins:
        return {"fetched": 0, "cached": 0, "failed": 0, "filled": 0}
    client = info or _info_client()
    lo = int(panel.cycle_unix[0]) - HOUR_S
    hi = int(panel.cycle_unix[-1]) + HOUR_S
    fetched = 0
    failed = 0
    for coin in missing_coins:
        span = candle_span(conn, coin, "1h")
        need_lo = lo
        if span is not None:
            have_lo, have_hi = span
            if have_lo <= lo + HOUR_S and have_hi >= hi - HOUR_S:
                continue
            if have_hi >= lo:
                need_lo = have_hi - HOUR_S
        try:
            rows = fetch_coin_candles(
                client, coin, start_unix=need_lo, end_unix=hi, interval="1h"
            )
        except Exception as exc:
            failed += 1
            log.warning("HL gap-fill failed %s: %s", coin, exc)
            continue
        if rows:
            upsert_candles(conn, rows)
            fetched += 1
        if sleep_s > 0:
            time.sleep(sleep_s)
    conn.commit()
    extra = fill_panel_from_hl_cache(conn, panel)
    log.info("HL gap-fill | coins=%s fetched=%s failed=%s extra_cells=%s", len(missing_coins), fetched, failed, extra)
    return {"fetched": fetched, "cached": 0, "failed": failed, "filled": extra}


def fill_panel_from_hl_cache(conn: Any, panel: Any, *, bar_seconds: int = HOUR_S) -> int:
    n = 0
    for j, coin in enumerate(panel.coins):
        rows = conn.execute(
            """
            SELECT open_ts, close FROM price_candles
            WHERE coin = ? AND interval = ?
            ORDER BY open_ts
            """,
            (coin, "1h"),
        ).fetchall()
        if not rows:
            continue
        by_open = {int(r["open_ts"]): float(r["close"]) for r in rows}
        for i, ts in enumerate(panel.cycle_unix):
            if panel.marks[i, j] > 0.0:
                continue
            px = by_open.get(int(ts), 0.0) or by_open.get(int(ts) - int(bar_seconds), 0.0)
            if px > 0.0:
                panel.marks[i, j] = px
                n += 1
    return n


def _fctx(ctx: dict[str, Any], *keys: str) -> float:
    for key in keys:
        v = _fany(ctx.get(key))
        if v != 0.0:
            return v
    return 0.0


def overlay_hl_market(panel: Any, info: Any) -> int:
    """Overwrite the latest bar's mark/funding/OI/premium/volume from Hyperliquid."""
    if panel is None or panel.n_times < 1 or panel.n_coins < 1:
        return 0
    try:
        raw = info.meta_and_asset_ctxs()
    except Exception:
        try:
            raw = info.post("/info", {"type": "metaAndAssetCtxs"})
        except Exception as exc:
            log.warning("HL metaAndAssetCtxs failed: %s", exc)
            return 0
    meta: dict[str, Any] = {}
    ctxs: list[Any] = []
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        meta = raw[0] if isinstance(raw[0], dict) else {}
        ctxs = raw[1] if isinstance(raw[1], list) else []
    universe = meta.get("universe") or []
    by_coin: dict[str, dict[str, Any]] = {}
    for i, asset in enumerate(universe):
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "").strip()
        if not name:
            continue
        ctx = ctxs[i] if i < len(ctxs) and isinstance(ctxs[i], dict) else {}
        by_coin[name] = ctx
        if ":" in name:
            by_coin[name.split(":")[-1]] = ctx
    try:
        mids_raw = info.all_mids()
    except Exception:
        mids_raw = {}
    mids: dict[str, float] = {}
    if isinstance(mids_raw, dict):
        for coin, px in mids_raw.items():
            try:
                v = float(px)
            except (TypeError, ValueError):
                continue
            if v > 0:
                mids[str(coin)] = v
    t = panel.n_times - 1
    filled = 0
    for j, coin in enumerate(panel.coins):
        ctx = by_coin.get(coin) or by_coin.get(str(coin).split(":")[-1]) or {}
        px = mids.get(coin) or mids.get(str(coin).split(":")[-1]) or 0.0
        if px <= 0:
            px = _fctx(ctx, "markPx", "midPx", "oraclePx")
        if px > 0:
            panel.marks[t, j] = px
            filled += 1
        fund = _fctx(ctx, "funding")
        if fund != 0.0 or "funding" in ctx:
            panel.funding[t, j] = float(ctx.get("funding") or 0)
        oi = _fctx(ctx, "openInterest")
        if oi > 0:
            panel.oi[t, j] = oi
        prem = _fctx(ctx, "premium")
        if prem != 0.0 or "premium" in ctx:
            panel.premium[t, j] = float(ctx.get("premium") or 0)
        vol = _fctx(ctx, "dayNtlVlm")
        if vol > 0:
            panel.volume[t, j] = vol
        prev = _fctx(ctx, "prevDayPx")
        if prev > 0:
            panel.prev_day[t, j] = prev
    log.debug("HL live overlay | last-bar marks=%s coins=%s", filled, panel.n_coins)
    return filled
