"""Qualify leaderboard wallets: drop lottery tickets, MMs, deposit-inflated PnL."""

from __future__ import annotations

import logging
import math
import statistics
import time
from typing import Any

from .types import LeaderboardRow, QualifiedWallet, fnum


def holder_filter_on(cfg: Any) -> bool:
    return str(getattr(cfg, "BASKET_FILTER_MODE", "off") or "off").strip().lower() in (
        "holder",
        "holders",
        "filter",
    )


def _window(row: LeaderboardRow, name: str):
    return row.windows.get(name) or row.windows.get(
        {"month": "perpMonth", "week": "perpWeek", "day": "perpDay"}.get(name, name)
    )


def is_holder_tape(fills: list[Any], now_ms: int, cfg: Any) -> tuple[bool, str]:
    """True if recent fills look like sitting / slow trading, not a scalper."""
    lookback_d = float(getattr(cfg, "HOLD_LOOKBACK_DAYS", 7.0) or 7.0)
    start_ms = now_ms - int(lookback_d * 86400_000)
    times: list[int] = []
    for item in fills or []:
        if not isinstance(item, dict):
            continue
        ts = int(item.get("time") or 0)
        if ts <= 0 or ts < start_ms:
            continue
        times.append(ts)
    n = len(times)
    if n == 0:
        return True, "no_fills"
    max_fills = int(getattr(cfg, "HOLD_MAX_FILLS", 18) or 18)
    max_per_day = float(getattr(cfg, "HOLD_MAX_FILLS_PER_DAY", 5.0) or 5.0)
    min_gap = float(getattr(cfg, "HOLD_MIN_MEDIAN_GAP_S", 3600.0) or 3600.0)
    per_day = n / max(lookback_d, 0.1)
    times.sort()
    gaps = [(times[i] - times[i - 1]) / 1000.0 for i in range(1, n)]
    median_gap = float(statistics.median(gaps)) if gaps else lookback_d * 86400.0
    if n <= max(3, max_fills // 3):
        return True, f"fills={n}"
    if median_gap >= min_gap and per_day <= max_per_day:
        return True, f"fills={n} gap={median_gap:.0f}s"
    if n > max_fills or per_day > max_per_day or median_gap < min_gap:
        return False, f"scalp fills={n} {per_day:.1f}/d gap={median_gap:.0f}s"
    return True, f"fills={n} gap={median_gap:.0f}s"


def shortlist(rows: list[LeaderboardRow], cfg: Any) -> list[QualifiedWallet]:
    rank_n = str(cfg.RANK_WINDOW)
    confirm_n = str(getattr(cfg, "CONFIRM_WINDOW", "") or "").strip()
    holder = holder_filter_on(cfg)
    min_eq = float(cfg.MIN_ACCOUNT_VALUE or 0) if holder else 0.0
    min_pnl = float(cfg.MIN_WINDOW_PNL or 0) if holder else 0.0
    max_roi = float(cfg.MAX_WINDOW_ROI or 0) if holder else 0.0
    min_roi = float(cfg.MIN_WINDOW_ROI or 0) if holder else 0.0
    min_confirm = float(cfg.MIN_CONFIRM_PNL or 0) if holder else 0.0
    min_vol = float(cfg.MIN_WINDOW_VOLUME or 0) if holder else 0.0
    max_ve = float(cfg.MAX_VOLUME_TO_EQUITY or 0) if holder else 0.0
    tilt_pnl = bool(getattr(cfg, "RANK_TILT_PNL", False))
    pool_n = int(cfg.CANDIDATE_POOL)
    if holder:
        scan = int(getattr(cfg, "HOLDER_SCAN_POOL", 0) or 0)
        if scan > 0:
            pool_n = max(pool_n, scan)
    if bool(getattr(cfg, "RESEARCH_DATA_ENABLED", False)):
        pool_n = max(pool_n, int(getattr(cfg, "RESEARCH_POOL_SIZE", 0) or 0))
    scored: list[QualifiedWallet] = []
    for row in rows:
        w = _window(row, rank_n)
        if w is None:
            continue
        if min_eq > 0 and row.account_value < min_eq:
            continue
        if min_pnl != 0 and w.pnl < min_pnl:
            continue
        confirm_pnl = 0.0
        if confirm_n:
            c = _window(row, confirm_n)
            confirm_pnl = c.pnl if c is not None else 0.0
            if min_confirm != 0 and confirm_pnl < min_confirm:
                continue
        if min_vol > 0 and w.volume < min_vol:
            continue
        if max_ve > 0 and w.volume / max(row.account_value, 1.0) > max_ve:
            continue
        if max_roi > 0 and w.roi > max_roi:
            continue
        if min_roi != 0 and w.roi < min_roi:
            continue
        score = w.roi
        if tilt_pnl:
            score = w.roi * math.log10(max(w.pnl, 250.0) + 50.0)
        scored.append(
            QualifiedWallet(
                address=row.address,
                account_value=row.account_value,
                rank_pnl=w.pnl,
                rank_roi=w.roi,
                rank_volume=w.volume,
                confirm_pnl=confirm_pnl,
                score=score,
            )
        )
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:pool_n]


def copy_rank_window(cfg: Any) -> str:
    """Leaderboard window for copy / copy_reverse only (day=24h, week=7d)."""
    return str(getattr(cfg, "COPY_RANK_WINDOW", "") or getattr(cfg, "RANK_WINDOW", "week") or "week")


def shortlist_copy_roi(
    rows: list[LeaderboardRow],
    cfg: Any,
    *,
    scan_n: int,
    min_equity: float = 0.0,
) -> list[QualifiedWallet]:
    """Top wallets by COPY_RANK_WINDOW ROI (HL leaderboard column)."""
    rank_n = copy_rank_window(cfg)
    out: list[QualifiedWallet] = []
    for row in rows:
        w = _window(row, rank_n)
        if w is None:
            continue
        if min_equity > 0 and row.account_value < min_equity:
            continue
        if w.roi <= 0:
            continue
        out.append(
            QualifiedWallet(
                address=row.address,
                account_value=row.account_value,
                rank_pnl=w.pnl,
                rank_roi=w.roi,
                rank_volume=w.volume,
                confirm_pnl=0.0,
                score=w.roi,
            )
        )
    out.sort(key=lambda x: x.rank_roi, reverse=True)
    return out[: max(1, int(scan_n))]


def shortlist_copy_profit(
    rows: list[LeaderboardRow],
    cfg: Any,
    *,
    scan_n: int,
    min_equity: float = 0.0,
) -> list[QualifiedWallet]:
    """Deprecated alias — copy mode ranks by ROI like the HL UI."""
    return shortlist_copy_roi(rows, cfg, scan_n=scan_n, min_equity=min_equity)


def _ledger_net_deposit(entries: list[Any], start_ms: int) -> float:
    net = 0.0
    for item in entries or []:
        if not isinstance(item, dict):
            continue
        ts = int(item.get("time") or 0)
        if ts and ts < start_ms:
            continue
        delta = item.get("delta") or {}
        if not isinstance(delta, dict):
            continue
        kind = str(delta.get("type") or "")
        usdc = fnum(delta.get("usdc") or delta.get("amount") or delta.get("usdcValue"))
        if kind in ("deposit", "vaultWithdraw", "vaultDistribution", "rewardsClaim"):
            net += abs(usdc) if kind == "deposit" else usdc
        elif kind in ("withdraw", "vaultDeposit"):
            net -= abs(usdc)
        elif kind == "internalTransfer":
            # inbound vs outbound is ambiguous; ignore unless destination is self
            pass
        elif kind == "subAccountTransfer":
            net += usdc
    return net


def _fill_closed_pnl(fills: list[Any]) -> tuple[float, float, int]:
    pnl = 0.0
    fees = 0.0
    n = 0
    for fill in fills or []:
        if not isinstance(fill, dict):
            continue
        n += 1
        pnl += fnum(fill.get("closedPnl"))
        fees += fnum(fill.get("fee"))
    return pnl, fees, n


class Qualifier:
    def __init__(self, info: Any, logger: logging.Logger, cfg: Any) -> None:
        self.info = info
        self.log = logger
        self.cfg = cfg

    def _is_vault(self, address: str) -> bool:
        if not bool(self.cfg.SKIP_VAULTS):
            return False
        try:
            raw = self.info.post("/info", {"type": "userRole", "user": address})
        except Exception as exc:
            self.log.debug("userRole %s: %s", address[:10], exc)
            return False
        if isinstance(raw, dict):
            role = str(raw.get("role") or "").lower()
        else:
            role = str(raw).lower()
        return role == "vault"

    def _deposit_ratio(self, address: str, account_value: float) -> float:
        start_ms = int((time.time() - 40 * 86400) * 1000)
        try:
            raw = self.info.post(
                "/info",
                {
                    "type": "userNonFundingLedgerUpdates",
                    "user": address,
                    "startTime": start_ms,
                },
            )
        except Exception as exc:
            self.log.debug("ledger %s: %s", address[:10], exc)
            return 0.0
        net = _ledger_net_deposit(raw if isinstance(raw, list) else [], start_ms)
        return net / max(account_value, 1.0)

    def _fill_mismatch(self, address: str, rank_pnl: float) -> bool:
        if not bool(self.cfg.AUDIT_FILLS) or rank_pnl <= 0:
            return False
        try:
            fills = self.info.post("/info", {"type": "userFills", "user": address})
        except Exception as exc:
            self.log.debug("fills %s: %s", address[:10], exc)
            return False
        pnl, fees, n = _fill_closed_pnl(fills if isinstance(fills, list) else [])
        realized = pnl - abs(fees)
        if n < 8:
            return False
        # Leaderboard window PnL is much larger than recent fill sample — not a mismatch
        # by itself. Only reject when recent realized is clearly negative.
        if realized < 0 and abs(realized) > abs(rank_pnl) * float(self.cfg.FILL_PNL_MISMATCH_RATIO):
            return True
        return False

    def _recent_fills(self, address: str, start_ms: int) -> list[Any] | None:
        raw: Any = None
        try:
            raw = self.info.post(
                "/info",
                {"type": "userFillsByTime", "user": address, "startTime": start_ms},
            )
        except Exception as exc:
            self.log.debug("userFillsByTime %s: %s", address[:10], exc)
            try:
                raw = self.info.post("/info", {"type": "userFills", "user": address})
            except Exception as exc2:
                self.log.debug("userFills %s: %s", address[:10], exc2)
                return None
        if not isinstance(raw, list):
            return []
        out: list[Any] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            ts = int(item.get("time") or 0)
            if ts and ts < start_ms:
                continue
            out.append(item)
        return out

    def _pick_holders(self, pool: list[QualifiedWallet]) -> list[QualifiedWallet]:
        want = int(self.cfg.BASKET_SIZE)
        lookback_d = float(getattr(self.cfg, "HOLD_LOOKBACK_DAYS", 7.0) or 7.0)
        start_ms = int((time.time() - lookback_d * 86400) * 1000)
        now_ms = int(time.time() * 1000)
        keep: list[QualifiedWallet] = []
        scanned = 0
        skipped = 0
        self.log.info(
            "Holder scan: checking recent fills on up to %s ROI-ranked wallets (want %s)",
            len(pool),
            want,
        )
        for w in pool:
            if len(keep) >= want:
                break
            scanned += 1
            fills = self._recent_fills(w.address, start_ms)
            if fills is None:
                skipped += 1
                self.log.info("Skip %s — fills request failed", w.address[:10])
                continue
            ok, why = is_holder_tape(fills, now_ms, self.cfg)
            if not ok:
                skipped += 1
                self.log.info("Skip scalper %s roi=%.1f%% %s", w.address[:10], w.rank_roi * 100.0, why)
                continue
            w.reasons = [why]
            keep.append(w)
            self.log.info(
                "Keep holder %s/%s %s roi=%.1f%% pnl=$%.0f %s",
                len(keep),
                want,
                w.address[:10],
                w.rank_roi * 100.0,
                w.rank_pnl,
                why,
            )
        self.log.info("Holder basket %s wallets (scanned %s, skipped %s)", len(keep), scanned, skipped)
        return keep

    def label_research_pool(self, pool: list[QualifiedWallet]) -> list[dict[str, Any]]:
        """Label top RESEARCH_POOL_SIZE wallets holder yes/no (no trade basket)."""
        research_n = int(getattr(self.cfg, "RESEARCH_POOL_SIZE", 0) or 0) or len(pool)
        lookback_d = float(getattr(self.cfg, "HOLD_LOOKBACK_DAYS", 7.0) or 7.0)
        start_ms = int((time.time() - lookback_d * 86400) * 1000)
        now_ms = int(time.time() * 1000)
        research: list[dict[str, Any]] = []
        self.log.info(
            "Research label: fill-tape on top %s ROI wallets (holders + scalpers kept for offline)",
            min(research_n, len(pool)),
        )
        for i, w in enumerate(pool):
            if i >= research_n:
                break
            fills = self._recent_fills(w.address, start_ms)
            if fills is None:
                research.append(
                    {
                        "address": w.address,
                        "account_value": round(w.account_value, 2),
                        "rank_pnl": round(w.rank_pnl, 2),
                        "rank_roi": round(w.rank_roi, 6),
                        "rank_volume": round(w.rank_volume, 2),
                        "score": round(w.score, 6),
                        "holder": None,
                        "why": "fills_failed",
                    }
                )
                continue
            ok, why = is_holder_tape(fills, now_ms, self.cfg)
            research.append(
                {
                    "address": w.address,
                    "account_value": round(w.account_value, 2),
                    "rank_pnl": round(w.rank_pnl, 2),
                    "rank_roi": round(w.rank_roi, 6),
                    "rank_volume": round(w.rank_volume, 2),
                    "score": round(w.score, 6),
                    "holder": bool(ok),
                    "why": why,
                }
            )
            tag = "holder" if ok else "scalper"
            self.log.info(
                "Research %s %s/%s %s roi=%.1f%% %s",
                tag,
                len(research),
                research_n,
                w.address[:10],
                w.rank_roi * 100.0,
                why,
            )
        holders = sum(1 for r in research if r.get("holder") is True)
        self.log.info("Research pool labeled %s (holders=%s)", len(research), holders)
        return research

    def pick_holders_and_research(
        self, pool: list[QualifiedWallet]
    ) -> tuple[list[QualifiedWallet], list[dict[str, Any]]]:
        """
        One fill-pass: label top RESEARCH_POOL_SIZE for offline filter on/off,
        and keep BASKET_SIZE holders for live trading.
        """
        want = int(self.cfg.BASKET_SIZE)
        research_n = max(want, int(getattr(self.cfg, "RESEARCH_POOL_SIZE", 0) or 0))
        lookback_d = float(getattr(self.cfg, "HOLD_LOOKBACK_DAYS", 7.0) or 7.0)
        start_ms = int((time.time() - lookback_d * 86400) * 1000)
        now_ms = int(time.time() * 1000)
        keep: list[QualifiedWallet] = []
        research: list[dict[str, Any]] = []
        self.log.info(
            "Research+holder scan: label top %s ROI wallets, keep %s holders for trade",
            min(research_n, len(pool)),
            want,
        )
        for i, w in enumerate(pool):
            need_label = i < research_n
            need_more_holders = len(keep) < want
            if not need_label and not need_more_holders:
                break
            fills = self._recent_fills(w.address, start_ms)
            if fills is None:
                if need_label:
                    research.append(
                        {
                            "address": w.address,
                            "account_value": w.account_value,
                            "rank_pnl": w.rank_pnl,
                            "rank_roi": w.rank_roi,
                            "rank_volume": w.rank_volume,
                            "score": w.score,
                            "holder": None,
                            "why": "fills_failed",
                        }
                    )
                continue
            ok, why = is_holder_tape(fills, now_ms, self.cfg)
            if need_label:
                research.append(
                    {
                        "address": w.address,
                        "account_value": round(w.account_value, 2),
                        "rank_pnl": round(w.rank_pnl, 2),
                        "rank_roi": round(w.rank_roi, 6),
                        "rank_volume": round(w.rank_volume, 2),
                        "score": round(w.score, 6),
                        "holder": bool(ok),
                        "why": why,
                    }
                )
            if ok and need_more_holders:
                w.reasons = [why]
                keep.append(w)
                self.log.info(
                    "Keep holder %s/%s %s roi=%.1f%% %s",
                    len(keep),
                    want,
                    w.address[:10],
                    w.rank_roi * 100.0,
                    why,
                )
            elif need_label and not ok:
                self.log.info("Research scalper %s roi=%.1f%% %s", w.address[:10], w.rank_roi * 100.0, why)
        self.log.info(
            "Research pool labeled %s | trade holders %s",
            len(research),
            len(keep),
        )
        return keep, research

    def deep_audit(self, pool: list[QualifiedWallet]) -> list[QualifiedWallet]:
        if holder_filter_on(self.cfg):
            return self._pick_holders(pool)
        n = int(self.cfg.BASKET_SIZE)
        if not bool(getattr(self.cfg, "DEEP_AUDIT", False)):
            picked = pool[:n]
            self.log.info(
                "Fast basket: top %s by %s ROI (no extra /info audit)",
                len(picked),
                getattr(self.cfg, "RANK_WINDOW", "week"),
            )
            for i, w in enumerate(picked[:5], 1):
                self.log.info(
                    "  #%s %s roi=%.1f%% pnl=$%.0f eq=$%.0f",
                    i,
                    w.address[:10],
                    w.rank_roi * 100.0,
                    w.rank_pnl,
                    w.account_value,
                )
            return picked
        keep: list[QualifiedWallet] = []
        for i, w in enumerate(pool):
            self.log.info(
                "Audit %s/%s %s equity=$%.0f pnl=$%.0f roi=%.1f%%",
                i + 1,
                len(pool),
                w.address[:10],
                w.account_value,
                w.rank_pnl,
                w.rank_roi * 100.0,
            )
            reasons: list[str] = []
            dep = self._deposit_ratio(w.address, w.account_value)
            if dep > float(self.cfg.MAX_DEPOSIT_TO_EQUITY):
                self.log.info("Skip deposit-heavy %s ratio=%.2f", w.address[:10], dep)
                continue
            if self._fill_mismatch(w.address, w.rank_pnl):
                self.log.info("Skip fill/PnL mismatch %s", w.address[:10])
                continue
            if self._is_vault(w.address):
                self.log.info("Skip vault %s", w.address[:10])
                continue
            reasons.append(f"dep_ratio={dep:.2f}")
            w.reasons = reasons
            keep.append(w)
            if len(keep) >= int(self.cfg.BASKET_SIZE):
                break
        return keep
