"""Download and parse the public Hyperliquid stats leaderboard."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests

from .store import atomic_write_json, read_json
from .types import LeaderboardRow, WindowPerf, fnum

STATS_URLS = (
    "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard",
    "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard.json",
)


def _windows_from_row(row: dict[str, Any]) -> dict[str, WindowPerf]:
    raw = row.get("windowPerformances") or row.get("window_performances") or []
    out: dict[str, WindowPerf] = {}
    if isinstance(raw, dict):
        items = raw.items()
    else:
        items = raw
    for item in items:
        name = ""
        block: Any = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            name, block = str(item[0]), item[1]
        elif isinstance(item, dict):
            name = str(item.get("window") or item.get("name") or "")
            block = item
        if not name or not isinstance(block, dict):
            continue
        out[name] = WindowPerf(
            pnl=fnum(block.get("pnl")),
            roi=fnum(block.get("roi")),
            volume=fnum(block.get("vlm", block.get("volume"))),
        )
    return out


def parse_leaderboard(payload: Any) -> list[LeaderboardRow]:
    if isinstance(payload, dict):
        rows = (
            payload.get("leaderboardRows")
            or payload.get("leaderboard_rows")
            or payload.get("leaderboard")
            or []
        )
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    out: list[LeaderboardRow] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        addr = str(row.get("ethAddress") or row.get("eth_address") or "").strip().lower()
        if not addr.startswith("0x") or len(addr) != 42:
            continue
        out.append(
            LeaderboardRow(
                address=addr,
                account_value=fnum(row.get("accountValue") or row.get("account_value")),
                display_name=(str(row["displayName"]) if row.get("displayName") else None),
                windows=_windows_from_row(row),
            )
        )
    return out


def load_leaderboard(
    cache_path: Path,
    cache_hours: float,
    logger: logging.Logger,
    timeout_s: float = 90.0,
) -> list[LeaderboardRow]:
    cached = read_json(cache_path, None)
    if isinstance(cached, dict):
        age_h = (time.time() - float(cached.get("fetched_at") or 0)) / 3600.0
        if age_h <= cache_hours and cached.get("payload") is not None:
            rows = parse_leaderboard(cached["payload"])
            if rows:
                logger.info("Leaderboard cache hit (%s rows, %.1fh old)", len(rows), age_h)
                return rows

    last_err: Exception | None = None
    payload: Any = None
    for url in STATS_URLS:
        try:
            logger.info("Downloading leaderboard %s", url)
            resp = requests.get(url, timeout=timeout_s)
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception as exc:
            last_err = exc
            logger.warning("Leaderboard fetch failed (%s): %s", url, exc)

    if payload is None:
        if isinstance(cached, dict) and cached.get("payload") is not None:
            logger.warning("Using stale leaderboard cache after fetch failure")
            return parse_leaderboard(cached["payload"])
        raise RuntimeError(f"Could not download HL leaderboard: {last_err}")

    rows = parse_leaderboard(payload)
    if not rows:
        raise RuntimeError("Leaderboard payload parsed to 0 wallets")
    atomic_write_json(
        cache_path,
        {"fetched_at": time.time(), "count": len(rows), "payload": payload},
    )
    logger.info("Leaderboard stored (%s wallets)", len(rows))
    return rows
