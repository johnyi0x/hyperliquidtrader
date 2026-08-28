"""Main loop: refresh basket, snapshot wallets, rebalance our book."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.exchange_client import HyperliquidClient
from src.hl_rate_limit import RequestGuard, ThrottledInfo, default_shared_budget
from src.market_resolver import list_perp_dex_names, sdk_perp_dexs_for_dexes
from hyperliquid.utils import constants

from types import SimpleNamespace

from .consensus import BookEngine, build_votes, crashed_wallets, min_live_voters, min_wallets_on_coin, votes_to_targets
from .copy_exec import copy_targets_from_leaders, min_fresh_copy_leaders
from .copy_score import (
    leaders_from_state,
    leaders_to_basket,
    leaders_to_state,
    pick_copy_leaders,
)
from .leaderboard import load_leaderboard
from .markets import MarketCache
from .price_engine import PriceEngine
from .qualify import Qualifier, copy_rank_window, holder_filter_on, shortlist, shortlist_copy_roi
from .rebalancer import PaperBook, Rebalancer
from .research import ResearchWriter, compact_research_votes
from .snapshots import SnapshotClient, list_dex_query_names
from .store import StateStore, atomic_write_json, read_json
from .strategy_exec import pick_trade_votes
from .mtf_exec import MtfTrader
from .swing_exec import SwingTrader
from .strategy_spec import strategy_from_cfg
from .telemetry import TelemetryWriter, compact_votes
from .tracker import BasketTracker
from .types import QualifiedWallet, WalletSnapshot


def _basket_from_state(raw: list[Any]) -> list[QualifiedWallet]:
    out: list[QualifiedWallet] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        addr = str(item.get("address") or "").lower()
        if not addr.startswith("0x"):
            continue
        out.append(
            QualifiedWallet(
                address=addr,
                account_value=float(item.get("account_value") or 0),
                rank_pnl=float(item.get("rank_pnl") or 0),
                rank_roi=float(item.get("rank_roi") or 0),
                rank_volume=float(item.get("rank_volume") or 0),
                confirm_pnl=float(item.get("confirm_pnl") or 0),
                score=float(item.get("score") or 0),
                reasons=list(item.get("reasons") or []),
            )
        )
    return out


def _basket_to_state(wallets: list[QualifiedWallet]) -> list[dict]:
    return [
        {
            "address": w.address,
            "account_value": w.account_value,
            "rank_pnl": w.rank_pnl,
            "rank_roi": w.rank_roi,
            "rank_volume": w.rank_volume,
            "confirm_pnl": w.confirm_pnl,
            "score": w.score,
            "reasons": w.reasons,
        }
        for w in wallets
    ]


def _copy_sig(cfg: Any) -> str:
    return "|".join(
        [
            "copy-v13",
            str(getattr(cfg, "COPY_TOP_N", "")),
            str(getattr(cfg, "COPY_RANK_WINDOW", "")),
            str(getattr(cfg, "COPY_BOARD_SCAN", "")),
            str(getattr(cfg, "COPY_MIN_PNL_VOLUME_RATIO", "")),
            str(getattr(cfg, "COPY_MIN_MEDIAN_GAP_S", "")),
            str(getattr(cfg, "RUN_MODE", "copy")),
        ]
    )


def _basket_sig(cfg: Any) -> str:
    return "|".join(
        [
            str(getattr(cfg, "INSTANCE_NAME", "") or getattr(cfg, "PMF_PROFILE", "")),
            str(getattr(cfg, "RANK_WINDOW", "")),
            str(getattr(cfg, "MIN_ACCOUNT_VALUE", "")),
            str(getattr(cfg, "MAX_WINDOW_ROI", "")),
            str(getattr(cfg, "MIN_WINDOW_PNL", "")),
            str(getattr(cfg, "BASKET_SIZE", "")),
            str(getattr(cfg, "BASKET_FILTER_MODE", "off")),
            "v7",
        ]
    )


def _research_sig(cfg: Any) -> str:
    return "|".join(
        [
            str(getattr(cfg, "INSTANCE_NAME", "") or getattr(cfg, "PMF_PROFILE", "")),
            str(getattr(cfg, "RANK_WINDOW", "")),
            str(getattr(cfg, "RESEARCH_POOL_SIZE", "")),
            "research-v2",
        ]
    )


class ProfitMetaRunner:
    def __init__(
        self,
        cfg: Any,
        *,
        client: HyperliquidClient,
        data_dir: Path,
        logger: logging.Logger,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.data_dir = data_dir
        self.log = logger
        self.store = StateStore(data_dir)
        dex_names = list_dex_query_names(client.info, cfg.DEX_SCOPE)
        self.snapper = SnapshotClient(client.info, logger, dex_names)
        self.tracker = BasketTracker(self.snapper, cfg, logger)
        self.book = BookEngine()
        self.book.load(self.store.data.get("book_engine"))
        self.markets = MarketCache(
            client.info,
            dex_names,
            logger,
            float(getattr(cfg, "MARKET_CACHE_S", 60.0) or 60.0),
        )
        self.price_book = PriceEngine(logger=logger)
        self._strategy = strategy_from_cfg(cfg)
        self.mtf_trader = MtfTrader.from_dump(self.store.data.get("mtf_trader"))
        self.swing_trader = SwingTrader.from_dump(self.store.data.get("swing_trader"))
        paper = None
        if bool(cfg.PAPER_TRADING):
            paper = PaperBook(float(cfg.PAPER_START_BALANCE), 0.045, logger)
            saved = read_json(data_dir / "paper_book.json", None)
            if isinstance(saved, dict):
                paper.cash = float(saved.get("cash") or paper.cash)
                from .types import OurPos

                for raw in saved.get("positions") or []:
                    if not isinstance(raw, dict):
                        continue
                    paper.positions[str(raw["coin"])] = OurPos(
                        coin=str(raw["coin"]),
                        side=str(raw["side"]),
                        size=float(raw["size"]),
                        notional=float(raw.get("notional") or 0),
                        entry_px=raw.get("entry_px"),
                        leverage=int(raw.get("leverage") or 1),
                    )
        self.paper = paper
        self.rebalancer = Rebalancer(client, self.snapper, cfg, logger, paper=paper)
        inst = str(getattr(cfg, "INSTANCE_NAME", "") or getattr(cfg, "PMF_PROFILE", "local"))
        self.telemetry = TelemetryWriter(
            data_dir,
            enabled=bool(getattr(cfg, "TELEMETRY_ENABLED", True)),
            instance=inst,
        )
        research_on = bool(getattr(cfg, "RESEARCH_DATA_ENABLED", False))
        self.research_only = bool(getattr(cfg, "RESEARCH_ONLY", False))
        self.research = ResearchWriter(
            data_dir, enabled=research_on, instance=inst, logger=logger
        )
        self.research.load_resume(self.store.data.get("research_resume"))
        self.research_book = BookEngine()
        self.research_book.load(self.store.data.get("research_book_engine"))
        self.research_tracker: BasketTracker | None = None
        if research_on:
            research_cfg = type("ResearchCfg", (), {})()
            research_cfg.WALLETS_PER_TICK = int(getattr(cfg, "RESEARCH_WALLETS_PER_TICK", 3) or 3)
            research_cfg.SNAPSHOT_INTERVAL_S = float(
                getattr(cfg, "RESEARCH_SNAPSHOT_INTERVAL_S", 60.0) or 60.0
            )
            research_cfg.STALE_SNAPSHOT_S = float(getattr(cfg, "STALE_SNAPSHOT_S", 480.0) or 480.0)
            research_cfg.MAX_BOOK_CHANGES_PER_HOUR = 0
            research_cfg.BASKET_FILTER_MODE = "off"
            self.research_tracker = BasketTracker(self.snapper, research_cfg, logger)
        self._research_addrs: list[str] = [
            str(a).lower() for a in (self.store.data.get("research_addrs") or []) if str(a).startswith("0x")
        ]
        if self.research_tracker is not None and self._research_addrs and self.research_only:
            self.research_tracker.set_basket(
                [QualifiedWallet(a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for a in self._research_addrs]
            )
        self._last_equity: float | None = None

    def _is_copy_mode(self) -> bool:
        mode = str(getattr(self.cfg, "RUN_MODE", "crowd") or "crowd").strip().lower()
        return mode in ("copy", "copy_reverse")

    def _copy_reverse(self) -> bool:
        return str(getattr(self.cfg, "RUN_MODE", "") or "").strip().lower() == "copy_reverse"

    def _copy_age_h(self) -> float:
        if str(self.store.data.get("copy_rank") or "") != _copy_sig(self.cfg):
            return 1e9
        built = float(self.store.data.get("copy_built_at") or 0)
        if built <= 0:
            return 1e9
        return (time.time() - built) / 3600.0

    def _copy_leaders(self) -> list:
        return leaders_from_state(self.store.data.get("copy_leaders") or [])

    def _copy_watchlist_full(self, leaders: list | None = None) -> bool:
        want = max(1, int(getattr(self.cfg, "COPY_TOP_N", 5) or 5))
        cur = leaders if leaders is not None else self._copy_leaders()
        if not bool(getattr(self.cfg, "COPY_REQUIRE_FULL_WATCHLIST", True)):
            return len(cur) >= 1
        ready_flag = self.store.data.get("copy_ready")
        if ready_flag is False:
            return False
        return len(cur) >= want

    def _copy_reject_addrs(self) -> set[str]:
        """Addresses already fetched/rejected — never re-query in later waves."""
        ttl_h = float(getattr(self.cfg, "COPY_REJECT_TTL_H", 24.0) or 24.0)
        now = time.time()
        out: set[str] = set()

        raw = self.store.data.get("copy_rejects") or {}
        alive: dict[str, Any] = {}
        if isinstance(raw, dict):
            for addr, meta in raw.items():
                a = str(addr).lower()
                if not a.startswith("0x"):
                    continue
                ts = 0.0
                reason = ""
                if isinstance(meta, dict):
                    ts = float(meta.get("ts") or 0)
                    reason = str(meta.get("why") or "")
                elif isinstance(meta, (int, float)):
                    ts = float(meta)
                if ttl_h > 0 and ts > 0 and (now - ts) > ttl_h * 3600.0:
                    continue
                alive[a] = {"ts": ts or now, "why": reason}
                out.add(a)
        self.store.data["copy_rejects"] = alive

        scanned_raw = self.store.data.get("copy_scanned") or {}
        scanned_alive: dict[str, Any] = {}
        if isinstance(scanned_raw, dict):
            for addr, meta in scanned_raw.items():
                a = str(addr).lower()
                if not a.startswith("0x"):
                    continue
                ts = float(meta.get("ts") or 0) if isinstance(meta, dict) else float(meta or 0)
                if ttl_h > 0 and ts > 0 and (now - ts) > ttl_h * 3600.0:
                    continue
                scanned_alive[a] = {"ts": ts or now}
                out.add(a)
        elif isinstance(scanned_raw, list):
            for addr in scanned_raw:
                a = str(addr).lower()
                if a.startswith("0x"):
                    scanned_alive[a] = {"ts": now}
                    out.add(a)
        self.store.data["copy_scanned"] = scanned_alive
        return out

    def _merge_copy_rejects(self, rejects: dict[str, str]) -> None:
        if not rejects:
            return
        cur = self.store.data.get("copy_rejects") or {}
        if not isinstance(cur, dict):
            cur = {}
        now = time.time()
        for addr, why in rejects.items():
            a = str(addr).lower()
            cur[a] = {"ts": now, "why": str(why or "")[:80]}
        if len(cur) > 5000:
            items = sorted(
                cur.items(),
                key=lambda kv: float((kv[1] or {}).get("ts") or 0) if isinstance(kv[1], dict) else 0,
            )
            cur = dict(items[-4000:])
        self.store.data["copy_rejects"] = cur

    def _merge_copy_scanned(self, scanned: list[str]) -> None:
        if not scanned:
            return
        cur = self.store.data.get("copy_scanned") or {}
        if not isinstance(cur, dict):
            cur = {}
        now = time.time()
        for addr in scanned:
            a = str(addr).lower()
            if a.startswith("0x"):
                cur[a] = {"ts": now}
        if len(cur) > 8000:
            items = sorted(
                cur.items(),
                key=lambda kv: float((kv[1] or {}).get("ts") or 0) if isinstance(kv[1], dict) else 0,
            )
            cur = dict(items[-6000:])
        self.store.data["copy_scanned"] = cur

    def refresh_copy_leaders_if_needed(self) -> None:
        existing = self._copy_leaders()
        want = max(1, int(getattr(self.cfg, "COPY_TOP_N", 5) or 5))
        require_full = bool(getattr(self.cfg, "COPY_REQUIRE_FULL_WATCHLIST", True))
        refresh_h = float(getattr(self.cfg, "COPY_REFRESH_HOURS", 12.0) or 12.0)
        retry_min = float(getattr(self.cfg, "COPY_INCOMPLETE_RETRY_MIN", 10.0) or 10.0)
        full = self._copy_watchlist_full(existing)
        sig = _copy_sig(self.cfg)
        old_sig = str(self.store.data.get("copy_rank") or "")
        # Only reset progress when config actually changed from a prior known sig.
        sig_changed = bool(old_sig) and old_sig != sig

        if existing and full and self._copy_age_h() < refresh_h and not sig_changed:
            self.log.info(
                "Using saved copy leaders (%s/%s wallets, %.1fh old)",
                len(existing),
                want,
                self._copy_age_h(),
            )
            self.tracker.set_basket(leaders_to_basket(existing))
            return

        failed_at = float(self.store.data.get("copy_scan_failed_at") or 0)
        backoff_s = retry_min * 60.0 if not full else 1800.0
        if (
            failed_at > 0
            and (time.time() - failed_at) < backoff_s
            and not full
            and not sig_changed
        ):
            last_log = float(self.store.data.get("copy_backoff_log_at") or 0)
            if time.time() - last_log >= 120.0:
                self.log.warning(
                    "Copy scan backoff — last wave %.0f min ago (have %s/%s next_idx=%s seen=%s)",
                    (time.time() - failed_at) / 60.0,
                    len(existing),
                    want,
                    int(self.store.data.get("copy_scan_offset") or 0),
                    len(self.store.data.get("copy_scanned") or {}),
                )
                self.store.data["copy_backoff_log_at"] = time.time()
                self.store.save()
            if existing:
                self.tracker.set_basket(leaders_to_basket(existing))
            return

        if sig_changed:
            self.log.info("Copy config changed (%s → %s) — reset scan progress", old_sig[:24], sig[:24])
            self.store.data["copy_rejects"] = {}
            self.store.data["copy_scanned"] = {}
            self.store.data["copy_scan_offset"] = 0
            existing = []

        offset = int(self.store.data.get("copy_scan_offset") or 0)
        self.log.info(
            "Refreshing copy-trade leaders (need full=%s want=%s have=%s next_idx=%s)",
            require_full,
            want,
            len(existing),
            offset,
        )
        rows = load_leaderboard(
            self.data_dir / "leaderboard_cache.json",
            float(self.cfg.LEADERBOARD_CACHE_HOURS),
            self.log,
        )
        scan_n = int(getattr(self.cfg, "COPY_CANDIDATE_SCAN", 2000) or 2000)
        min_eq = float(getattr(self.cfg, "COPY_MIN_EQUITY", 1000.0) or 0.0)
        pool = shortlist_copy_roi(
            rows, self.cfg, scan_n=scan_n, min_equity=min_eq
        )
        self.log.info(
            "Copy shortlist %s/%s by %s ROI (min_eq=$%.0f) — HL leaderboard order",
            len(pool),
            len(rows),
            copy_rank_window(self.cfg),
            min_eq,
        )
        if not pool:
            self.log.error("No wallets on copy shortlist")
            self.store.data["copy_scan_failed_at"] = time.time()
            self.store.data["copy_ready"] = False
            self.store.data["copy_rank"] = sig
            self.store.save()
            return

        # Keep partial watchlist; only hunt for missing slots.
        keep = existing if (existing and not full) else []
        if existing and full and self._copy_age_h() >= refresh_h:
            keep = []
            self.store.data["copy_scan_offset"] = 0
            offset = 0

        skip = self._copy_reject_addrs()
        # If cursor past board end, restart from head but still skip already-seen addrs
        # (picks up new high-ROI wallets after leaderboard churn).
        if offset >= len(pool):
            self.log.info(
                "Copy cursor past board end (%s>=%s) — wrap to 0, still skipping %s seen",
                offset,
                len(pool),
                len(skip),
            )
            offset = 0

        qualifier = Qualifier(self.client.info, self.log, self.cfg)
        result = pick_copy_leaders(
            pool,
            qualifier,
            self.cfg,
            logger=self.log,
            keep=keep,
            skip_addrs=skip,
            start_offset=offset,
        )
        self._merge_copy_rejects(result.rejects)
        self._merge_copy_scanned(result.scanned)
        self.store.data["copy_scan_offset"] = int(result.next_offset)
        self.store.data["copy_rank"] = sig  # always — avoids false "config changed" resets

        leaders = result.leaders
        if not leaders:
            if existing and full:
                self.log.warning(
                    "Copy scan found 0 — keeping previous full list (%s)",
                    len(existing),
                )
                self.tracker.set_basket(leaders_to_basket(existing))
                return
            self.log.error(
                "Copy wave found 0 leaders — next wave at idx %s (seen=%s); retry in %.0fm",
                self.store.data["copy_scan_offset"],
                len(self.store.data.get("copy_scanned") or {}),
                retry_min,
            )
            self.store.data["copy_scan_failed_at"] = time.time()
            self.store.data["copy_ready"] = False
            self.store.save()
            return

        ready = (not require_full) or (len(leaders) >= want)
        self.store.data["copy_leaders"] = leaders_to_state(leaders)
        self.store.data["copy_ready"] = ready
        if ready:
            self.store.data["copy_built_at"] = time.time()
            self.store.data["copy_scan_failed_at"] = 0
            # Keep seen/rejects so a later full reselect still skips known HFT.
            self.store.data["copy_scan_offset"] = 0
            self.log.info("Copy watchlist READY %s/%s — trading enabled", len(leaders), want)
        else:
            self.store.data["copy_built_at"] = 0
            self.store.data["copy_scan_failed_at"] = time.time()
            self.log.warning(
                "Copy watchlist INCOMPLETE %s/%s — next wave idx=%s seen=%s; retry in %.0fm",
                len(leaders),
                want,
                self.store.data["copy_scan_offset"],
                len(self.store.data.get("copy_scanned") or {}),
                retry_min,
            )
        self.store.save()
        self.tracker.set_basket(leaders_to_basket(leaders))
        self.telemetry.record_event(
            ts=time.time(),
            kind="copy_leaders_refresh",
            payload={
                "count": len(leaders),
                "want": want,
                "ready": ready,
                "offset": int(self.store.data.get("copy_scan_offset") or 0),
                "seen": len(self.store.data.get("copy_scanned") or {}),
                "leaders": [ld.address[:10] for ld in leaders],
            },
        )

    def cycle_copy(self) -> None:
        """Copy top-N leader books — separate from crowd consensus."""
        now = time.time()
        n = self.tracker.poll_some(now)
        snaps = self.tracker.live_snapshots()
        leaders = self._copy_leaders()
        want = max(1, int(getattr(self.cfg, "COPY_TOP_N", 5) or 5))
        if not leaders:
            last_wait = float(self.store.data.get("copy_empty_log_at") or 0)
            if time.time() - last_wait >= 120.0:
                self.log.warning("Copy mode — no leaders loaded yet")
                self.store.data["copy_empty_log_at"] = time.time()
                self.store.save()
            return
        if not self._copy_watchlist_full(leaders):
            last_wait = float(self.store.data.get("copy_wait_log_at") or 0)
            if time.time() - last_wait >= 120.0:
                self.log.info(
                    "Copy waiting for full watchlist %s/%s — no trades yet",
                    len(leaders),
                    want,
                )
                self.store.data["copy_wait_log_at"] = time.time()
                self.store.save()
            self.store.heartbeat(
                {"mode": "copy", "ready": False, "leaders": len(leaders), "want": want}
            )
            self._persist_tracker()
            return

        try:
            self.markets.refresh_if_needed(now, [p.coin for s in snaps for p in s.positions])
        except Exception as exc:
            self.log.debug("Market cache: %s", exc)

        by_addr = {s.address.lower(): s for s in snaps}
        fresh = self._fresh_count(by_addr, [ld.address for ld in leaders], now=now)
        need = min_fresh_copy_leaders(self.cfg, len(leaders))
        managed = set(str(x) for x in (self.store.data.get("managed_coins") or []))
        targets = copy_targets_from_leaders(
            leaders,
            snaps,
            self.cfg,
            now=now,
            reverse=self._copy_reverse(),
        )
        trade_keys = [f"{t.side}:{t.coin}" for t in targets]

        try:
            if self.paper is not None:
                marks = {c: 1.0 for c in {p.coin for s in snaps for p in s.positions}}
                self._last_equity = self.paper.equity(marks)
            else:
                _, self._last_equity = self.rebalancer.current_book()
        except Exception:
            pass

        self.telemetry.record_tick(
            ts=now,
            voters=fresh,
            listed=len(leaders),
            raw=[],
            trade=trade_keys,
            managed=sorted(managed),
            equity=self._last_equity,
        )
        self.log.info(
            "Copy tick mode=%s snaps=%s fresh=%s/%s leaders=%s targets=%s",
            "reverse" if self._copy_reverse() else "follow",
            n,
            fresh,
            len(leaders),
            ", ".join(f"{ld.address[:8]}({ld.recent.win_rate:.0%})" for ld in leaders),
            ", ".join(trade_keys) or "-",
        )

        if fresh < need:
            self.log.info(
                "Copy warm-up — need %s/%s fresh leader snapshots",
                need,
                len(leaders),
            )
            self.store.heartbeat({"mode": "copy", "fresh": fresh, "warm": True})
            self._persist_tracker()
            return

        # Copy mode uses its own short cooldown (crowd cloud_tuned cooldown is unrelated).
        saved_cd = getattr(self.cfg, "REBALANCE_COOLDOWN_S", 180.0)
        copy_cd = float(getattr(self.cfg, "COPY_REBALANCE_COOLDOWN_S", 45.0) or 45.0)
        try:
            setattr(self.cfg, "REBALANCE_COOLDOWN_S", copy_cd)
            last_reb = float(self.store.data.get("last_rebalance_at") or 0)
            result = self.rebalancer.run(targets, managed, last_reb, now)
        finally:
            setattr(self.cfg, "REBALANCE_COOLDOWN_S", saved_cd)
        if not isinstance(result, tuple) or len(result) != 2:
            self.log.error("rebalancer.run returned %r — skipping this tick", result)
            new_managed, attempted = managed, False
        else:
            new_managed, attempted = result
        if attempted or new_managed != managed:
            self.store.data["managed_coins"] = sorted(new_managed)
            self.store.data["last_targets"] = [
                {
                    "coin": t.coin,
                    "side": t.side,
                    "leverage": t.leverage,
                    "margin_pct": t.margin_pct,
                    "conviction": t.conviction,
                }
                for t in targets
            ]
            if attempted:
                self.store.data["last_rebalance_at"] = now
                self.telemetry.record_event(
                    ts=now,
                    kind="rebalance",
                    payload={
                        "mode": "copy",
                        "targets": trade_keys,
                        "managed": sorted(new_managed),
                        "equity": self._last_equity,
                    },
                )
            self.store.save()
            self._save_paper()
        self._persist_tracker()
        self.store.heartbeat(
            {
                "mode": "copy",
                "fresh": fresh,
                "targets": len(targets),
                "managed": sorted(new_managed),
            }
        )

    def _save_paper(self) -> None:
        if self.paper is None:
            return
        atomic_write_json(
            self.data_dir / "paper_book.json",
            {
                "cash": self.paper.cash,
                "positions": [
                    {
                        "coin": p.coin,
                        "side": p.side,
                        "size": p.size,
                        "notional": p.notional,
                        "entry_px": p.entry_px,
                        "leverage": p.leverage,
                    }
                    for p in self.paper.positions.values()
                ],
            },
        )

    def _basket_age_h(self) -> float:
        if str(self.store.data.get("basket_rank") or "") != _basket_sig(self.cfg):
            return 1e9
        built = float(self.store.data.get("basket_built_at") or 0)
        if built <= 0:
            return 1e9
        return (time.time() - built) / 3600.0

    def _research_pool_age_h(self) -> float:
        if str(self.store.data.get("research_rank") or "") != _research_sig(self.cfg):
            return 1e9
        built = float(self.store.data.get("research_built_at") or 0)
        if built <= 0:
            return 1e9
        return (time.time() - built) / 3600.0

    def _holder_addrs(self) -> set[str]:
        out: set[str] = set()
        for row in self.store.data.get("research_pool") or []:
            if not isinstance(row, dict):
                continue
            if row.get("holder") is True:
                addr = str(row.get("address") or "").lower()
                if addr.startswith("0x"):
                    out.add(addr)
        return out

    def _fresh_count(
        self,
        snaps_by_addr: dict[str, WalletSnapshot],
        addrs: list[str],
        *,
        now: float,
    ) -> int:
        stale = float(getattr(self.cfg, "STALE_SNAPSHOT_S", 480.0) or 480.0)
        n = 0
        for a in addrs:
            s = snaps_by_addr.get(a)
            if s is None or s.error:
                continue
            if now - s.fetched_at > stale:
                continue
            # Successful clearinghouse read counts (empty book is still valid data).
            n += 1
        return n

    def _coverage(
        self,
        snaps_by_addr: dict[str, WalletSnapshot],
        *,
        now: float,
        holders: set[str],
    ) -> dict[str, Any]:
        listed = len(self._research_addrs)
        fresh = self._fresh_count(snaps_by_addr, self._research_addrs, now=now)
        holder_list = [a for a in self._research_addrs if a in holders]
        holders_listed = len(holder_list)
        holders_fresh = self._fresh_count(snaps_by_addr, holder_list, now=now)
        frac = (fresh / listed) if listed else 0.0
        return {
            "fresh": fresh,
            "listed": listed,
            "holders_fresh": holders_fresh,
            "holders_listed": holders_listed,
            "frac": round(frac, 4),
        }

    def _build_votes_raw(
        self,
        snaps_by_addr: dict[str, WalletSnapshot],
        addrs: list[str],
        *,
        now: float,
    ) -> list[Any]:
        """Live-parity raw CoinVotes from a wallet subset (same gates as cloud)."""
        snaps = [snaps_by_addr[a] for a in addrs if a in snaps_by_addr]
        if len(snaps) < 3:
            return []
        vote_cfg = SimpleNamespace(
            DEX_SCOPE=getattr(self.cfg, "DEX_SCOPE", "include"),
            ALLOW_COINS=getattr(self.cfg, "ALLOW_COINS", ()),
            DENY_COINS=getattr(self.cfg, "DENY_COINS", ()),
            STALE_SNAPSHOT_S=float(getattr(self.cfg, "STALE_SNAPSHOT_S", 480) or 480),
            MIN_WALLET_CONVICTION=float(getattr(self.cfg, "MIN_WALLET_CONVICTION", 0.015) or 0.015),
            # Same warm-up fraction as live — empty ticks when coverage is thin.
            MIN_LIVE_VOTERS_PCT=float(getattr(self.cfg, "MIN_LIVE_VOTERS_PCT", 0.45) or 0.45),
            MIN_LIVE_VOTERS=int(getattr(self.cfg, "MIN_LIVE_VOTERS", 0) or 0),
            MIN_WALLETS_ON_COIN_PCT=float(getattr(self.cfg, "MIN_WALLETS_ON_COIN_PCT", 0.08) or 0.08),
            MIN_WALLETS_ON_COIN=0,
            MIN_SIDE_AGREEMENT=float(getattr(self.cfg, "MIN_SIDE_AGREEMENT", 0.10) or 0.10),
            EXIT_SIDE_AGREEMENT=float(getattr(self.cfg, "EXIT_SIDE_AGREEMENT", 0.05) or 0.05),
            MIN_AVG_CONVICTION=float(getattr(self.cfg, "MIN_AVG_CONVICTION", 0.022) or 0.022),
            MAX_COINS_IN_BOOK=int(getattr(self.cfg, "MAX_COINS_IN_BOOK", 3) or 3),
            MAX_LIVE_EQUITY_DROP=0.0,
            PREFERRED_COINS=(),
            BASKET_SIZE=len(addrs),
        )
        return build_votes(snaps, set(), vote_cfg, now, listed=len(addrs))

    def _crowd_votes_for_research(
        self,
        snaps_by_addr: dict[str, WalletSnapshot],
        addrs: list[str],
        *,
        now: float,
    ) -> list[dict[str, Any]]:
        return compact_research_votes(self._build_votes_raw(snaps_by_addr, addrs, now=now))

    def _trade_votes_for_research(
        self,
        snaps_by_addr: dict[str, WalletSnapshot],
        holder_addrs: list[str],
        *,
        now: float,
    ) -> list[dict[str, Any]]:
        """Cloud-parity refined book (flow / raw_flow / sticky) — no orders."""
        raw = self._build_votes_raw(snaps_by_addr, holder_addrs, now=now)
        managed = {str(x) for x in (self.store.data.get("research_managed_coins") or [])}
        refined = self.research_book.refine(
            raw,
            markets=self.markets.ctxs,
            managed=managed,
            cfg=self.cfg,
            now=now,
            log=None,
        )
        self.store.data["research_managed_coins"] = sorted({v.coin for v in refined})
        self.store.data["research_book_engine"] = self.research_book.dump()
        return compact_research_votes(refined)

    def _live_trade_votes(self, raw: list, managed: set[str], now: float) -> list:
        """Identical strategy path as backtest (pick_trade_votes)."""
        spec = strategy_from_cfg(self.cfg)
        self._strategy = spec
        return pick_trade_votes(
            raw,
            book=self.book,
            markets=self.markets.ctxs,
            managed=managed,
            cfg=self.cfg,
            now=now,
            spec=spec,
            price=self.price_book,
            log=self.log,
            mtf_trader=self.mtf_trader,
            swing_trader=self.swing_trader,
        )

    def _maybe_seed_live_candles(self, now: float, coins: list[str], *, raw_coins: list[str] | None = None) -> None:
        spec = self._strategy
        if not spec.needs_candles:
            return
        if not bool(getattr(self.cfg, "LIVE_CANDLE_SEED", True)):
            return
        want = []
        seen: set[str] = set()
        # BTC anchor + crowd raw names first (swing/MTF need 1h on alts even with no position).
        for c in ["BTC", *(raw_coins or []), *coins]:
            if not c or c in seen:
                continue
            seen.add(c)
            want.append(c)
        self.price_book.queue_candle_jobs(want, spec.candle_intervals or ("1m", "15m", "1h"))
        try:
            self.price_book.maybe_fetch_candles(
                self.client.info,
                now=now,
                data_dir=self.data_dir,
                per_tick=int(getattr(self.cfg, "LIVE_CANDLES_PER_TICK", 1) or 1),
                cooldown_s=float(getattr(self.cfg, "LIVE_CANDLE_COOLDOWN_S", 8.0) or 8.0),
                bars_1m=int(getattr(self.cfg, "LIVE_CANDLE_BARS_1M", 120) or 120),
                bars_15m=int(getattr(self.cfg, "LIVE_CANDLE_BARS_15M", 64) or 64),
                bars_1h=int(getattr(self.cfg, "LIVE_CANDLE_BARS_1H", 48) or 48),
            )
        except Exception as exc:
            self.log.debug("Live candle seed: %s", exc)

    def _persist_research_resume(self) -> None:
        self.store.data["research_addrs"] = list(self._research_addrs)
        self.store.data["research_resume"] = self.research.dump_resume()
        self.store.data["research_book_engine"] = self.research_book.dump()
        self.store.save()

    def _set_research_tracker(self, addrs: list[str]) -> None:
        self._research_addrs = [a.lower() for a in addrs if str(a).startswith("0x")]
        if self.research_tracker is None:
            return
        self.research_tracker.set_basket(
            [QualifiedWallet(a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for a in self._research_addrs]
        )

    def refresh_research_pool_if_needed(self) -> None:
        """Build/resume labeled top-ROI pool for gather-only mode."""
        existing = [str(a).lower() for a in (self.store.data.get("research_addrs") or [])]
        if existing and self._research_pool_age_h() < float(self.cfg.BASKET_REFRESH_HOURS):
            self.log.info(
                "Resuming research pool (%s wallets, %.1fh old)",
                len(existing),
                self._research_pool_age_h(),
            )
            self._set_research_tracker(existing)
            self.research.load_resume(self.store.data.get("research_resume"))
            return

        self.log.info("Refreshing research pool from leaderboard (label holders + scalpers)")
        rows = load_leaderboard(
            self.data_dir / "leaderboard_cache.json",
            float(self.cfg.LEADERBOARD_CACHE_HOURS),
            self.log,
        )
        pool = shortlist(rows, self.cfg)
        self.log.info("Research shortlist %s/%s leaderboard rows", len(pool), len(rows))
        if not pool:
            raise RuntimeError("No wallets on leaderboard shortlist — loosen config")
        qualifier = Qualifier(self.client.info, self.log, self.cfg)
        self.store.data["refresh"] = {"started_at": time.time(), "pool": _basket_to_state(pool)}
        self.store.save()
        research_rows = qualifier.label_research_pool(pool)
        if len(research_rows) < 5:
            raise RuntimeError(f"Research pool too small ({len(research_rows)})")
        addrs = [str(r["address"]).lower() for r in research_rows if r.get("address")]
        self.store.data["research_pool"] = research_rows
        self.store.data["research_addrs"] = addrs
        self.store.data["research_built_at"] = time.time()
        self.store.data["research_rank"] = _research_sig(self.cfg)
        self.store.data["refresh"] = {}
        self.store.save()
        self._set_research_tracker(addrs)
        self.research.save_pool(ts=time.time(), wallets=research_rows)
        self.telemetry.record_event(
            ts=time.time(),
            kind="research_pool_refresh",
            payload={
                "count": len(research_rows),
                "holders": sum(1 for r in research_rows if r.get("holder") is True),
            },
        )
        self.log.info(
            "Research pool ready: %s wallets | holders=%s | #1 %s roi=%.1f%%",
            len(research_rows),
            sum(1 for r in research_rows if r.get("holder") is True),
            addrs[0][:10],
            float(research_rows[0].get("rank_roi") or 0) * 100.0,
        )

    def refresh_basket_if_needed(self) -> None:
        if self.research_only:
            self.refresh_research_pool_if_needed()
            return
        if self._is_copy_mode():
            self.refresh_copy_leaders_if_needed()
            return
        existing = _basket_from_state(self.store.data.get("basket") or [])
        if existing and self._basket_age_h() < float(self.cfg.BASKET_REFRESH_HOURS):
            self.log.info(
                "Using saved basket (%s wallets, %.1fh old)",
                len(existing),
                self._basket_age_h(),
            )
            self.tracker.set_basket(existing)
            self._research_addrs = [str(a).lower() for a in (self.store.data.get("research_addrs") or [])]
            if self.research_tracker is not None and self._research_addrs:
                trade_set = {w.address.lower() for w in existing}
                research_only = [a for a in self._research_addrs if a not in trade_set]
                self.research_tracker.set_basket(
                    [
                        QualifiedWallet(a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                        for a in research_only
                    ]
                )
            return

        self.log.info("Refreshing wallet basket from leaderboard")
        rows = load_leaderboard(
            self.data_dir / "leaderboard_cache.json",
            float(self.cfg.LEADERBOARD_CACHE_HOURS),
            self.log,
        )
        pool = shortlist(rows, self.cfg)
        self.log.info("Shortlist %s/%s leaderboard rows", len(pool), len(rows))
        if not pool:
            raise RuntimeError("No wallets passed the leaderboard filters — loosen config")
        qualifier = Qualifier(self.client.info, self.log, self.cfg)
        # Resume-friendly: persist pool so a disconnect mid-audit can skip re-download.
        self.store.data["refresh"] = {
            "started_at": time.time(),
            "pool": _basket_to_state(pool),
        }
        self.store.save()
        research_rows: list[dict[str, Any]] = []
        research_on = bool(getattr(self.cfg, "RESEARCH_DATA_ENABLED", False))
        if research_on and holder_filter_on(self.cfg):
            audited, research_rows = qualifier.pick_holders_and_research(pool)
        else:
            audited = qualifier.deep_audit(pool)
            if research_on:
                # Filter off: research pool = same top-N as trade list (labels = all tradeable).
                research_n = int(getattr(self.cfg, "RESEARCH_POOL_SIZE", 0) or 0) or len(audited)
                research_rows = [
                    {
                        "address": w.address,
                        "account_value": round(w.account_value, 2),
                        "rank_pnl": round(w.rank_pnl, 2),
                        "rank_roi": round(w.rank_roi, 6),
                        "rank_volume": round(w.rank_volume, 2),
                        "score": round(w.score, 6),
                        "holder": None,
                        "why": "filter_off",
                    }
                    for w in pool[:research_n]
                ]
        if len(audited) < max(5, min_live_voters(self.cfg, len(audited))):
            raise RuntimeError(
                f"Only {len(audited)} wallets survived deep audit — loosen filters"
            )
        self.store.data["basket"] = _basket_to_state(audited)
        self.store.data["basket_built_at"] = time.time()
        self.store.data["basket_rank"] = _basket_sig(self.cfg)
        self.store.data["refresh"] = {}
        self.store.data["hyper_wallets"] = []
        self._research_addrs = [str(r["address"]).lower() for r in research_rows if r.get("address")]
        self.store.data["research_addrs"] = self._research_addrs
        self.store.save()
        self.tracker.set_basket(audited)
        if self.research_tracker is not None and self._research_addrs:
            trade_set = {w.address.lower() for w in audited}
            research_only_addrs = [a for a in self._research_addrs if a not in trade_set]
            self.research_tracker.set_basket(
                [
                    QualifiedWallet(a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    for a in research_only_addrs
                ]
            )
            self.log.info(
                "Research track %s wallets (%s also in trade basket — reuse those snaps)",
                len(self._research_addrs),
                len(self._research_addrs) - len(research_only_addrs),
            )
        if research_rows:
            self.research.save_pool(ts=time.time(), wallets=research_rows)
        self.telemetry.save_basket_snapshot(ts=time.time(), wallets=_basket_to_state(audited))
        self.telemetry.record_event(
            ts=time.time(),
            kind="basket_refresh",
            payload={
                "count": len(audited),
                "filter": str(getattr(self.cfg, "BASKET_FILTER_MODE", "off")),
                "research": len(research_rows),
            },
        )
        self.log.info(
            "Basket ready: %s wallets ranked by %s ROI | #1 %s roi=%.1f%% pnl=$%.0f",
            len(audited),
            getattr(self.cfg, "RANK_WINDOW", "week"),
            audited[0].address[:10],
            audited[0].rank_roi * 100.0,
            audited[0].rank_pnl,
        )

    def _persist_tracker(self) -> None:
        self.store.data["book_engine"] = self.book.dump()
        self.store.data["mtf_trader"] = self.mtf_trader.dump()
        self.store.data["swing_trader"] = self.swing_trader.dump()
        self.store.save()

    def _run_research_after_trade(self, now: float, trade_snaps: list) -> None:
        """Side-channel gather after rebalance (trade profiles only)."""
        if self.research_only:
            return
        if not self.research.enabled or self.research_tracker is None:
            return
        if not self._research_addrs:
            return
        try:
            self.research_tracker.poll_some(now)
            research_snaps = self.research_tracker.live_snapshots()
            coin_hint = [p.coin for s in trade_snaps for p in s.positions]
            coin_hint.extend(p.coin for s in research_snaps for p in s.positions)
            try:
                self.markets.refresh_if_needed(now, coin_hint)
            except Exception as exc:
                self.log.debug("Research market cache: %s", exc)
            by_addr = {s.address: s for s in trade_snaps}
            for s in research_snaps:
                prev = by_addr.get(s.address)
                if prev is None or s.fetched_at >= prev.fetched_at:
                    by_addr[s.address] = s
            wrote_books, wrote_marks = self.research.maybe_record_crowd(
                ts=now,
                books_interval_s=float(getattr(self.cfg, "RESEARCH_RECORD_INTERVAL_S", 60.0) or 60.0),
                marks_interval_s=float(getattr(self.cfg, "RESEARCH_MARKS_INTERVAL_S", 30.0) or 30.0),
                snaps_by_addr=by_addr,
                research_addrs=self._research_addrs,
                markets=self.markets.ctxs,
            )
            if wrote_books or wrote_marks:
                self._persist_research_resume()
                self.log.info(
                    "Research sample books=%s marks=%s wallets=%s/%s coins=%s",
                    wrote_books,
                    wrote_marks,
                    sum(1 for a in self._research_addrs if a in by_addr),
                    len(self._research_addrs),
                    len(self.research.seen_coins),
                )
        except Exception as exc:
            self.log.warning("Research gather failed (trade unaffected): %s", exc)

    def cycle_research_only(self) -> None:
        """Gather-only: books first (live-parity), then marks, then candles. No orders."""
        now = time.time()
        if self.research_tracker is None or not self._research_addrs:
            self.log.warning("Research pool empty — waiting for refresh")
            time.sleep(2.0)
            return
        # 1) Crowd snapshots first — same style as live wallet polling.
        n = self.research_tracker.poll_some(now)
        snaps = self.research_tracker.live_snapshots()
        by_addr = {s.address: s for s in snaps}
        coin_hint = [p.coin for s in snaps for p in s.positions]
        try:
            self.markets.refresh_if_needed(now, coin_hint or None)
        except Exception as exc:
            self.log.debug("Research market cache: %s", exc)

        holders = self._holder_addrs()
        cov = self._coverage(by_addr, now=now, holders=holders)
        holder_list = [a for a in self._research_addrs if a in holders]
        crowd_all = self._crowd_votes_for_research(by_addr, self._research_addrs, now=now)
        crowd_holders = self._crowd_votes_for_research(by_addr, holder_list, now=now)
        crowd_trade = self._trade_votes_for_research(by_addr, holder_list, now=now)

        # 2–3) Persist books + live-like ticks + marks (before any candle API).
        wrote_books, wrote_marks = self.research.maybe_record_crowd(
            ts=now,
            books_interval_s=float(getattr(self.cfg, "RESEARCH_RECORD_INTERVAL_S", 60.0) or 60.0),
            marks_interval_s=float(getattr(self.cfg, "RESEARCH_MARKS_INTERVAL_S", 30.0) or 30.0),
            snaps_by_addr=by_addr,
            research_addrs=self._research_addrs,
            markets=self.markets.ctxs,
            coverage=cov,
            min_coverage=float(getattr(self.cfg, "RESEARCH_MIN_COVERAGE", 0) or 0),
            min_fresh_wallets=int(getattr(self.cfg, "RESEARCH_MIN_FRESH_WALLETS", 20) or 20),
            crowd_all=crowd_all,
            crowd_holders=crowd_holders,
            crowd_trade=crowd_trade,
        )

        # 4) Candles last — never delays book sampling above.
        candle_calls = 0
        try:
            candle_calls = self.research.maybe_fetch_candles(
                ts=now,
                info=self.client.info,
                data_dir=self.data_dir,
                intervals=tuple(getattr(self.cfg, "RESEARCH_CANDLE_INTERVALS", ("1m", "15m", "1h"))),
                bars=int(getattr(self.cfg, "RESEARCH_CANDLE_BARS", 300) or 300),
                per_tick=int(getattr(self.cfg, "RESEARCH_CANDLES_PER_TICK", 1) or 1),
                cooldown_s=float(getattr(self.cfg, "RESEARCH_CANDLE_COOLDOWN_S", 2.0) or 2.0),
            )
        except Exception as exc:
            self.log.debug("Research candles: %s", exc)

        self.log.info(
            "Research tick snap=%s cov=%s/%s (%.0f%%) holders=%s/%s books=%s marks=%s candles=%s coins=%s trade=%s",
            n,
            cov["fresh"],
            cov["listed"],
            float(cov["frac"]) * 100.0,
            cov["holders_fresh"],
            cov["holders_listed"],
            wrote_books,
            wrote_marks,
            candle_calls,
            len(self.research.seen_coins),
            len(crowd_trade),
        )
        if wrote_books or wrote_marks or candle_calls:
            self._persist_research_resume()
        else:
            # Keep flow EMA / managed state even when books skipped (warm-up).
            self.store.data["research_book_engine"] = self.research_book.dump()
            self.store.save()
        self.store.heartbeat(
            {
                "mode": "research",
                "fresh": cov["fresh"],
                "listed": cov["listed"],
                "cov": cov["frac"],
                "coins": len(self.research.seen_coins),
                "holders": cov["holders_listed"],
            }
        )

    def cycle(self) -> None:
        if self.research_only:
            self.cycle_research_only()
            return
        if self._is_copy_mode():
            self.cycle_copy()
            return
        now = time.time()
        equity_baseline = dict(self.tracker.last_equity)
        # --- Live trade path only (identical whether research is on or off) ---
        n = self.tracker.poll_some(now)
        snaps = self.tracker.live_snapshots()
        crashed = crashed_wallets(snaps, equity_baseline, self.cfg)
        if crashed:
            self.log.warning("Muted %s wallet(s) after live equity crash", len(crashed))
        churned = self.tracker.churned_wallets(now)
        if churned:
            self.log.info("Muted %s wallet(s) for book churn (scalper tape)", len(churned))
        skip = set(crashed) | set(churned)
        voters = self.tracker.voter_count(now, skip)
        try:
            self.markets.refresh_if_needed(
                now,
                [p.coin for s in snaps for p in s.positions],
            )
        except Exception as exc:
            self.log.debug("Market cache: %s", exc)
        # Free mark history every tick (no extra API). Candles only if strategy needs them.
        for coin, ctx in (self.markets.ctxs or {}).items():
            self.price_book.ingest_mark(
                coin,
                now,
                float(getattr(ctx, "mark", 0) or 0),
                funding=float(getattr(ctx, "funding", 0) or 0),
                basis=float(getattr(ctx, "basis", 0) or 0),
                oi=float(getattr(ctx, "open_interest", 0) or 0),
                day_vol=float(getattr(ctx, "day_volume", 0) or 0),
            )
        managed = set(str(x) for x in (self.store.data.get("managed_coins") or []))
        seed_coins = sorted(
            {
                *[p.coin for s in snaps for p in s.positions],
                *managed,
            }
        )
        raw = build_votes(
            snaps,
            churned,
            self.cfg,
            now,
            baseline_equity=equity_baseline,
            listed=len(self.tracker.addrs),
        )
        self._maybe_seed_live_candles(now, seed_coins, raw_coins=[v.coin for v in raw])
        votes = self._live_trade_votes(raw, managed, now)
        targets = votes_to_targets(votes, self.cfg)
        trade_keys = [f"{v.side}:{v.coin}" for v in votes]
        try:
            if self.paper is not None:
                marks = {c: 1.0 for c in {p.coin for s in snaps for p in s.positions}}
                self._last_equity = self.paper.equity(marks)
            else:
                _, self._last_equity = self.rebalancer.current_book()
        except Exception:
            pass
        self.telemetry.record_tick(
            ts=now,
            voters=voters,
            listed=len(self.tracker.addrs),
            raw=compact_votes(raw),
            trade=trade_keys,
            managed=sorted(managed),
            equity=self._last_equity,
        )
        self.log.info(
            "Tick snapshots=%s voters=%s/%s crash=%s raw=%s trade=%s",
            n,
            voters,
            len(self.tracker.addrs),
            len(crashed),
            ", ".join(
                f"{v.side} {v.coin} agr={v.agreement:.0%} conv={v.avg_conviction:+.3f} flow={v.flow:+.3f}"
                for v in raw[:6]
            )
            or "-",
            ", ".join(f"{v.side}:{v.coin}" for v in votes) or "-",
        )
        listed = len(self.tracker.addrs)
        if voters < min_live_voters(self.cfg, listed):
            self.log.info(
                "Not enough fresh voters yet — no trade (reconnect warm-up, need %s/%s)",
                min_live_voters(self.cfg, listed),
                listed or int(getattr(self.cfg, "BASKET_SIZE", 0) or 0),
            )
            self.store.heartbeat({"voters": voters, "warm": True})
            self._persist_tracker()
            self._run_research_after_trade(now, snaps)
            return
        if not raw:
            self.log.info(
                "Basket is split — no crowded side yet (need ≥%s wallets, ≥%.0f%% of list)",
                min_wallets_on_coin(self.cfg),
                float(self.cfg.MIN_SIDE_AGREEMENT) * 100.0,
            )

        last_reb = float(self.store.data.get("last_rebalance_at") or 0)
        result = self.rebalancer.run(targets, managed, last_reb, now)
        if not isinstance(result, tuple) or len(result) != 2:
            self.log.error("rebalancer.run returned %r — skipping this tick", result)
            new_managed, attempted = managed, False
        else:
            new_managed, attempted = result
        if attempted or new_managed != managed:
            self.store.data["managed_coins"] = sorted(new_managed)
            self.store.data["last_targets"] = [
                {
                    "coin": t.coin,
                    "side": t.side,
                    "leverage": t.leverage,
                    "margin_pct": t.margin_pct,
                    "conviction": t.conviction,
                }
                for t in targets
            ]
            if attempted:
                self.store.data["last_rebalance_at"] = now
                self.telemetry.record_event(
                    ts=now,
                    kind="rebalance",
                    payload={
                        "targets": trade_keys,
                        "managed": sorted(new_managed),
                        "equity": self._last_equity,
                    },
                )
            self.store.save()
            self._save_paper()
        self._persist_tracker()
        self.store.heartbeat(
            {
                "voters": voters,
                "crash": len(crashed),
                "targets": len(targets),
                "managed": sorted(new_managed),
            }
        )
        # --- Research AFTER trade (local only; cloud research=False is a no-op) ---
        self._run_research_after_trade(now, snaps)

    def run_forever(self) -> None:
        self.refresh_basket_if_needed()
        if self.research_only:
            self.log.info(
                "RESEARCH mode (no live orders) | profile=%s pool=%s snap=%ss/%s books=%ss marks=%ss min_wallets=%s candles=%s cloud_exit=%s",
                getattr(self.cfg, "PMF_PROFILE", "research"),
                len(self._research_addrs),
                getattr(self.cfg, "RESEARCH_SNAPSHOT_INTERVAL_S", 10),
                getattr(self.cfg, "RESEARCH_WALLETS_PER_TICK", 20),
                getattr(self.cfg, "RESEARCH_RECORD_INTERVAL_S", 60),
                getattr(self.cfg, "RESEARCH_MARKS_INTERVAL_S", 30),
                getattr(self.cfg, "RESEARCH_MIN_FRESH_WALLETS", 20),
                ",".join(getattr(self.cfg, "RESEARCH_CANDLE_INTERVALS", ("1m", "15m", "1h"))),
                getattr(self.cfg, "EXIT_RAW_FLOW", None),
            )
        else:
            if self._is_copy_mode():
                self.log.info(
                    "Copy-trade running | mode=%s profile=%s paper=%s leaders=%s rank=%s scan=%s "
                    "gap=%.0f-%.0fs min_wr=%.0f%% refresh=%sh scope=%s",
                    "reverse" if self._copy_reverse() else "follow",
                    getattr(self.cfg, "PMF_PROFILE", "local"),
                    bool(self.cfg.PAPER_TRADING),
                    int(getattr(self.cfg, "COPY_TOP_N", 2) or 2),
                    copy_rank_window(self.cfg),
                    int(getattr(self.cfg, "COPY_CANDIDATE_SCAN", 200) or 200),
                    float(getattr(self.cfg, "COPY_MIN_MEDIAN_GAP_S", 90) or 0),
                    float(getattr(self.cfg, "COPY_MAX_MEDIAN_GAP_S", 0) or 0),
                    float(getattr(self.cfg, "COPY_MIN_WIN_RATE", 0.40) or 0) * 100,
                    float(getattr(self.cfg, "COPY_REFRESH_HOURS", 4) or 4),
                    self.cfg.DEX_SCOPE,
                )
            else:
                spec = strategy_from_cfg(self.cfg)
                tuned = str(getattr(self.cfg, "TUNED_STRATEGY", "") or spec.name)
                tuned_at = float(getattr(self.cfg, "TUNED_AT", 0) or 0)
                self.log.info(
                    "Profit-meta follower running | profile=%s paper=%s trade=%s size=%s filter=%s "
                    "strategy=%s gate=%s candles=%s research=%s scope=%s basket=%s interval=%ss "
                    "need_voters=%s enter=%.0f%% exit=%.0f%% raw_exit=%s agr_gb=%.0f%% tuned=%s",
                    getattr(self.cfg, "PMF_PROFILE", "local"),
                    bool(self.cfg.PAPER_TRADING),
                    str(getattr(self.cfg, "TRADE_MODE", "follow") or "follow"),
                    str(getattr(self.cfg, "SIZE_MODE", "fixed") or "fixed"),
                    str(getattr(self.cfg, "BASKET_FILTER_MODE", "off") or "off"),
                    spec.name,
                    spec.gate or "-",
                    "on-demand" if spec.needs_candles else "off",
                    bool(getattr(self.cfg, "RESEARCH_DATA_ENABLED", False)),
                    self.cfg.DEX_SCOPE,
                    len(self.tracker.addrs),
                    self.cfg.SNAPSHOT_INTERVAL_S,
                    min_live_voters(self.cfg),
                    float(self.cfg.MIN_SIDE_AGREEMENT) * 100.0,
                    float(getattr(self.cfg, "EXIT_SIDE_AGREEMENT", 0) or 0) * 100.0,
                    getattr(self.cfg, "EXIT_RAW_FLOW", 0),
                    float(getattr(self.cfg, "EXIT_AGREEMENT_GIVEBACK", 0) or 0) * 100.0,
                    tuned or "none",
                )
                if tuned_at > 0:
                    self.log.info("Tuned params loaded (saved_at=%.0f strategy=%s)", tuned_at, tuned or "cloud_refine")
        fails = 0
        while True:
            try:
                refresh_due = (
                    self._research_pool_age_h()
                    if self.research_only
                    else (
                        self._copy_age_h()
                        if self._is_copy_mode()
                        else self._basket_age_h()
                    )
                ) >= float(
                    getattr(
                        self.cfg,
                        "COPY_REFRESH_HOURS" if self._is_copy_mode() else "BASKET_REFRESH_HOURS",
                        12.0,
                    )
                )
                if refresh_due:
                    try:
                        self.refresh_basket_if_needed()
                    except Exception as exc:
                        self.log.error("Pool/basket refresh failed (keeping old): %s", exc)
                self.cycle()
                fails = 0
            except Exception as exc:
                fails += 1
                delay = min(20.0, 4.0 * (2 ** min(fails, 3)))
                self.log.exception("Cycle error (%s) — retry in %.0fs: %s", fails, delay, exc)
                time.sleep(delay)
                continue
            time.sleep(float(self.cfg.LOOP_SLEEP_S))


def build_live_client(
    cfg: Any,
    wallet: str,
    key: str,
    logger: logging.Logger,
) -> HyperliquidClient:
    use_testnet = bool(cfg.USE_TESTNET) and not bool(cfg.PAPER_TRADING)
    base_url = constants.TESTNET_API_URL if use_testnet else constants.MAINNET_API_URL
    guard = RequestGuard(
        min_interval_s=0.12,
        max_429_retries=8,
        logger=logger,
        shared_budget=default_shared_budget(),
    )
    info = ThrottledInfo(base_url, skip_ws=True, guard=guard)
    try:
        named = set(list_perp_dex_names(info))
    except Exception:
        named = set()
    sdk = sdk_perp_dexs_for_dexes(named)  # type: ignore[arg-type]
    return HyperliquidClient(
        wallet_address=wallet,
        private_key=key,
        coin="BTC",
        logger=logger,
        use_testnet=use_testnet,
        sdk_perp_dexs=sdk,
    )
