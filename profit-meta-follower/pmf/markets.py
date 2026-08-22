"""Cached mark/funding/volume for native + HIP-3 (weight-20 calls, not on every tick)."""

from __future__ import annotations

import logging
import time
from typing import Any

from .types import MarketCtx, coin_key, fnum, split_coin


class MarketCache:
    def __init__(self, info: Any, dex_names: list[str], logger: logging.Logger, ttl_s: float) -> None:
        self.info = info
        self.dex_names = dex_names
        self.log = logger
        self.ttl_s = ttl_s
        self.ctxs: dict[str, MarketCtx] = {}
        self._fetched_at = 0.0

    def refresh_if_needed(self, now: float, coins: list[str] | None = None) -> None:
        if now - self._fetched_at < self.ttl_s and self.ctxs:
            return
        dexes = list(self.dex_names)
        if coins:
            need = {""}
            for c in coins:
                _sym, dex = split_coin(c)
                need.add(dex or "")
            dexes = [d for d in self.dex_names if d in need] or dexes[:1]
        out: dict[str, MarketCtx] = {}
        for dex in dexes:
            try:
                body: dict[str, Any] = {"type": "metaAndAssetCtxs"}
                if dex:
                    body["dex"] = dex
                raw = self.info.post("/info", body)
            except Exception as exc:
                self.log.debug("metaAndAssetCtxs dex=%r: %s", dex, exc)
                continue
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                continue
            meta = raw[0] if isinstance(raw[0], dict) else {}
            ctxs = raw[1] if isinstance(raw[1], list) else []
            universe = meta.get("universe") or []
            for i, asset in enumerate(universe):
                if not isinstance(asset, dict):
                    continue
                name = str(asset.get("name") or "").strip()
                if not name:
                    continue
                api = coin_key(name, dex or None)
                ctx = ctxs[i] if i < len(ctxs) and isinstance(ctxs[i], dict) else {}
                mark = fnum(ctx.get("markPx") or ctx.get("midPx"))
                oracle = fnum(ctx.get("oraclePx"))
                basis = 0.0
                if oracle > 0 and mark > 0:
                    basis = (mark - oracle) / oracle
                out[api] = MarketCtx(
                    coin=api,
                    day_volume=fnum(ctx.get("dayNtlVlm")),
                    funding=fnum(ctx.get("funding")),
                    open_interest=fnum(ctx.get("openInterest")),
                    basis=basis,
                )
        if out:
            self.ctxs = out
            self._fetched_at = now
            self.log.info("Market cache %s coins (dexes=%s)", len(out), len(dexes))

    def get(self, coin: str) -> MarketCtx | None:
        if coin in self.ctxs:
            return self.ctxs[coin]
        sym, dex = split_coin(coin)
        if dex:
            return self.ctxs.get(f"{dex}:{sym}")
        return self.ctxs.get(sym)
