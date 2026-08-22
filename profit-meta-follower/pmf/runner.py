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

from .consensus import BookEngine, build_votes, crashed_wallets, min_live_voters, min_wallets_on_coin, votes_to_targets
from .leaderboard import load_leaderboard
from .markets import MarketCache
from .qualify import Qualifier, shortlist
from .rebalancer import PaperBook, Rebalancer
from .snapshots import SnapshotClient, list_dex_query_names
from .store import StateStore, atomic_write_json, read_json
from .telemetry import TelemetryWriter, compact_votes
from .tracker import BasketTracker
from .types import QualifiedWallet


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
        self._last_equity: float | None = None

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

    def refresh_basket_if_needed(self) -> None:
        existing = _basket_from_state(self.store.data.get("basket") or [])
        if existing and self._basket_age_h() < float(self.cfg.BASKET_REFRESH_HOURS):
            self.log.info(
                "Using saved basket (%s wallets, %.1fh old)",
                len(existing),
                self._basket_age_h(),
            )
            self.tracker.set_basket(existing)
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
        audited = qualifier.deep_audit(pool)
        if len(audited) < max(5, min_live_voters(self.cfg, len(audited))):
            raise RuntimeError(
                f"Only {len(audited)} wallets survived deep audit — loosen filters"
            )
        self.store.data["basket"] = _basket_to_state(audited)
        self.store.data["basket_built_at"] = time.time()
        self.store.data["basket_rank"] = _basket_sig(self.cfg)
        self.store.data["refresh"] = {}
        self.store.data["hyper_wallets"] = []
        self.store.save()
        self.tracker.set_basket(audited)
        self.telemetry.save_basket_snapshot(ts=time.time(), wallets=_basket_to_state(audited))
        self.telemetry.record_event(
            ts=time.time(),
            kind="basket_refresh",
            payload={"count": len(audited), "filter": str(getattr(self.cfg, "BASKET_FILTER_MODE", "off"))},
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
        self.store.save()

    def cycle(self) -> None:
        now = time.time()
        equity_baseline = dict(self.tracker.last_equity)
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
        raw = build_votes(
            snaps,
            churned,
            self.cfg,
            now,
            baseline_equity=equity_baseline,
            listed=len(self.tracker.addrs),
        )
        managed = set(str(x) for x in (self.store.data.get("managed_coins") or []))
        votes = self.book.refine(
            raw,
            markets=self.markets.ctxs,
            managed=managed,
            cfg=self.cfg,
            now=now,
            log=self.log,
        )
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

    def run_forever(self) -> None:
        self.refresh_basket_if_needed()
        self.log.info(
            "Profit-meta follower running | profile=%s paper=%s trade=%s size=%s filter=%s scope=%s basket=%s interval=%ss need_voters=%s enter=%.0f%% exit=%.0f%% raw_exit=%s agr_gb=%.0f%%",
            getattr(self.cfg, "PMF_PROFILE", "local"),
            bool(self.cfg.PAPER_TRADING),
            str(getattr(self.cfg, "TRADE_MODE", "follow") or "follow"),
            str(getattr(self.cfg, "SIZE_MODE", "fixed") or "fixed"),
            str(getattr(self.cfg, "BASKET_FILTER_MODE", "off") or "off"),
            self.cfg.DEX_SCOPE,
            len(self.tracker.addrs),
            self.cfg.SNAPSHOT_INTERVAL_S,
            min_live_voters(self.cfg),
            float(self.cfg.MIN_SIDE_AGREEMENT) * 100.0,
            float(getattr(self.cfg, "EXIT_SIDE_AGREEMENT", 0) or 0) * 100.0,
            getattr(self.cfg, "EXIT_RAW_FLOW", 0),
            float(getattr(self.cfg, "EXIT_AGREEMENT_GIVEBACK", 0) or 0) * 100.0,
        )
        fails = 0
        while True:
            try:
                if self._basket_age_h() >= float(self.cfg.BASKET_REFRESH_HOURS):
                    try:
                        self.refresh_basket_if_needed()
                    except Exception as exc:
                        self.log.error("Basket refresh failed (keeping old basket): %s", exc)
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
