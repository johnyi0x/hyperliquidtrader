"""Candle fetch with dynamic Hyperliquid max-limit probing."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
}

# Sensible default; probed and cached at runtime.
DEFAULT_MAX_CANDLES = 5000
_PROBE_CACHE_NAME = "hl_candle_limit.json"


def interval_ms(interval: str) -> int:
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval {interval!r}; known={list(INTERVAL_MS)}")
    return INTERVAL_MS[interval]


def _cache_path(data_dir: Path | None) -> Path | None:
    if data_dir is None:
        return None
    return Path(data_dir) / _PROBE_CACHE_NAME


def load_cached_limit(data_dir: Path | None) -> int | None:
    path = _cache_path(data_dir)
    if path is None or not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        n = int(raw.get("max_candles", 0))
        # Re-probe at least weekly.
        age = time.time() - float(raw.get("ts", 0))
        if n >= 100 and age < 7 * 86400:
            return n
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def save_cached_limit(data_dir: Path | None, max_candles: int) -> None:
    path = _cache_path(data_dir)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"max_candles": int(max_candles), "ts": time.time()},
            indent=2,
        ),
        encoding="utf-8",
    )


def _fetch_raw(
    info: Any,
    coin: str,
    interval: str,
    n: int,
    *,
    end_ms: int | None = None,
) -> list[dict]:
    end = int(end_ms if end_ms is not None else time.time() * 1000)
    start = end - n * interval_ms(interval)
    return info.candles_snapshot(coin, interval, start, end)


def probe_max_candles(
    info: Any,
    coin: str,
    interval: str = "1m",
    *,
    data_dir: Path | None = None,
    logger: logging.Logger | None = None,
    candidates: tuple[int, ...] = (5000, 4500, 4000, 3000, 2000, 1000),
) -> int:
    """
    Discover the largest candleSnapshot window HL currently accepts.
    Tries cached value first, then descending candidates, then binary-ish shrink.
    """
    log = logger or logging.getLogger("hl-multi")
    cached = load_cached_limit(data_dir)
    if cached is not None:
        try:
            rows = _fetch_raw(info, coin, interval, cached)
            if isinstance(rows, list) and len(rows) > 0:
                log.info("Candle limit cache hit: max=%s (got %s)", cached, len(rows))
                return cached
        except Exception as exc:
            log.warning("Cached candle limit %s failed (%s) — re-probing", cached, exc)

    best = 0
    for n in candidates:
        try:
            rows = _fetch_raw(info, coin, interval, n)
            if isinstance(rows, list) and len(rows) > 0:
                best = max(best, n)
                log.info("Candle probe ok n=%s got=%s", n, len(rows))
                break
        except Exception as exc:
            log.info("Candle probe n=%s rejected: %s", n, exc)
            continue

    if best <= 0:
        # Last resort: small known-good size.
        best = 500
        log.warning("Candle probe failed — falling back to %s", best)
    else:
        # Try a bit above the first success (in case limit grew).
        for bump in (best + 500, best + 1000, 6000, 8000):
            if bump <= best:
                continue
            try:
                rows = _fetch_raw(info, coin, interval, bump)
                if isinstance(rows, list) and len(rows) > 0:
                    best = bump
                    log.info("Candle probe raised max to %s (got %s)", bump, len(rows))
                else:
                    break
            except Exception:
                break

    save_cached_limit(data_dir, best)
    log.info("Hyperliquid candleSnapshot max ≈ %s bars", best)
    return best


def fetch_closed_candles(
    info: Any,
    coin: str,
    interval: str,
    requested: int,
    *,
    max_candles: int | None = None,
    data_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> list[dict]:
    """
    Fetch up to `requested` closed candles, capped by probed HL max.
    Drops the still-forming last bar when its end time is in the future.
    """
    log = logger or logging.getLogger("hl-multi")
    limit = int(max_candles) if max_candles else probe_max_candles(
        info, coin, interval, data_dir=data_dir, logger=log
    )
    n = max(50, min(int(requested), int(limit)))
    raw = _fetch_raw(info, coin, interval, n)
    if not isinstance(raw, list):
        return []
    now = time.time() * 1000
    step = interval_ms(interval)
    closed: list[dict] = []
    for c in raw:
        try:
            t = int(c["t"])
        except (KeyError, TypeError, ValueError):
            continue
        # Closed if bar end is in the past.
        if t + step <= now + 1:
            closed.append(c)
    if len(closed) < len(raw) and closed:
        log.debug(
            "Dropped %s open bar(s) for %s %s",
            len(raw) - len(closed),
            coin,
            interval,
        )
    return closed
