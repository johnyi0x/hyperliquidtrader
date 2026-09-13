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
    log.info(
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
