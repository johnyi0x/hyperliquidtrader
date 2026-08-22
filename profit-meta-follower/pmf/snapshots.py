"""Hyperliquid info helpers for wallet snapshots (native + HIP-3)."""

from __future__ import annotations

import logging
from typing import Any

from .types import WalletPos, WalletSnapshot, coin_key, fnum, split_coin


def _iter_states(raw: Any) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    if raw is None:
        return states
    if isinstance(raw, dict):
        if "assetPositions" in raw:
            states.append(raw)
        else:
            for v in raw.values():
                if isinstance(v, dict) and "assetPositions" in v:
                    states.append(v)
                elif isinstance(v, list) and len(v) == 2 and isinstance(v[1], dict):
                    if "assetPositions" in v[1]:
                        states.append(v[1])
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and "assetPositions" in item:
                states.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], dict):
                if "assetPositions" in item[1]:
                    states.append(item[1])
    return states


def parse_positions(states: list[dict[str, Any]], account_value: float) -> list[WalletPos]:
    raw: list[tuple[str, str, float, float, float | None, int, bool]] = []
    seen: set[str] = set()
    for state in states:
        for ap in state.get("assetPositions") or []:
            pos = ap.get("position", ap) if isinstance(ap, dict) else {}
            szi = fnum(pos.get("szi"))
            if abs(szi) < 1e-12:
                continue
            coin = coin_key(str(pos.get("coin") or ""))
            if not coin or coin in seen:
                continue
            seen.add(coin)
            lev_raw = pos.get("leverage") or {}
            if isinstance(lev_raw, dict):
                lev = max(1, int(fnum(lev_raw.get("value"), 1)))
                isolated = str(lev_raw.get("type") or "").lower() == "isolated"
            else:
                lev = max(1, int(fnum(lev_raw, 1)))
                isolated = False
            notional = abs(fnum(pos.get("positionValue")))
            if notional <= 0:
                entry = fnum(pos.get("entryPx"))
                notional = abs(szi) * entry if entry > 0 else 0.0
            raw.append(
                (
                    coin,
                    "long" if szi > 0 else "short",
                    abs(szi),
                    notional,
                    fnum(pos.get("entryPx")) or None,
                    lev,
                    isolated,
                )
            )
    equity = max(account_value, sum(p[3] for p in raw), 1e-9)
    out: list[WalletPos] = []
    for coin, side, size, notional, entry_px, lev, isolated in raw:
        signed = 1.0 if side == "long" else -1.0
        out.append(
            WalletPos(
                coin=coin,
                side=side,
                size=size,
                notional=notional,
                entry_px=entry_px,
                leverage=lev,
                isolated=isolated,
                conviction=(notional / equity) * signed,
            )
        )
    return out


def fingerprint(positions: list[WalletPos]) -> str:
    parts = []
    for p in sorted(positions, key=lambda x: x.coin):
        parts.append(f"{p.coin}:{p.side}:{round(p.conviction, 3)}")
    return "|".join(parts)


def account_value_from_states(states: list[dict[str, Any]]) -> float:
    best = 0.0
    for state in states:
        for key in ("marginSummary", "crossMarginSummary"):
            block = state.get(key) or {}
            if isinstance(block, dict):
                best = max(best, fnum(block.get("accountValue")))
    return best


class SnapshotClient:
    def __init__(self, info: Any, logger: logging.Logger, dex_names: list[str]) -> None:
        self.info = info
        self.log = logger
        self.dex_names = dex_names  # includes "" for native

    def fetch_user_states(self, address: str) -> list[dict[str, Any]]:
        try:
            raw = self.info.post(
                "/info",
                {"type": "clearinghouseState", "user": address, "dex": "ALL_DEXES"},
            )
            states = _iter_states(raw)
            if states:
                return states
        except Exception as exc:
            self.log.debug("ALL_DEXES snapshot failed for %s: %s", address[:10], exc)
        states: list[dict[str, Any]] = []
        for dex in self.dex_names:
            try:
                raw = self.info.user_state(address, dex=dex)
                states.extend(_iter_states(raw) or ([raw] if isinstance(raw, dict) else []))
            except Exception as exc:
                self.log.debug("snapshot dex=%r %s: %s", dex, address[:10], exc)
        return states

    def snapshot(self, address: str, now: float) -> WalletSnapshot:
        try:
            states = self.fetch_user_states(address)
            equity = account_value_from_states(states)
            positions = parse_positions(states, equity)
            return WalletSnapshot(
                address=address.lower(),
                account_value=equity,
                positions=positions,
                fetched_at=now,
                fingerprint=fingerprint(positions),
                dexes_ok=True,
            )
        except Exception as exc:
            return WalletSnapshot(
                address=address.lower(),
                account_value=0.0,
                positions=[],
                fetched_at=now,
                fingerprint="",
                dexes_ok=False,
                error=str(exc)[:200],
            )


def list_dex_query_names(info: Any, scope: str) -> list[str]:
    from src.market_resolver import list_perp_dex_names

    named: list[str] = []
    try:
        named = list_perp_dex_names(info)
    except Exception:
        named = []
    scope_s = (scope or "native").strip().lower()
    if scope_s == "native":
        return [""]
    if scope_s == "xyz_only":
        return [d for d in named if d]
    return [""] + [d for d in named if d]
