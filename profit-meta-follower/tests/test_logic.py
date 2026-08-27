from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_REPO))

from pmf.consensus import BookEngine, build_votes, crashed_wallets, votes_to_targets
from pmf.leaderboard import parse_leaderboard
from pmf.qualify import is_holder_tape, shortlist
from pmf.rebalancer import Rebalancer, plan_actions
from pmf.snapshots import fingerprint, parse_positions
from pmf.types import (
    CoinVote,
    LeaderboardRow,
    MarketCtx,
    OurPos,
    QualifiedWallet,
    TargetPos,
    WalletPos,
    WalletSnapshot,
    WindowPerf,
)


class _Cfg:
    DEX_SCOPE = "include"
    ALLOW_COINS = ()
    DENY_COINS = ()
    STALE_SNAPSHOT_S = 180.0
    MIN_WALLET_CONVICTION = 0.02
    MIN_LIVE_VOTERS = 3
    MIN_WALLETS_ON_COIN = 2
    MIN_SIDE_AGREEMENT = 0.22
    MIN_AVG_CONVICTION = 0.035
    MAX_COINS_IN_BOOK = 4
    OUR_GROSS_MARGIN_PCT = 40.0
    MAX_MARGIN_PER_COIN_PCT = 50.0
    OUR_MIN_LEVERAGE = 2
    OUR_MAX_LEVERAGE = 10
    SINGLE_NAME_SIZE_MULT = 0.55
    MAX_LIVE_EQUITY_DROP = 0.40
    MIN_ENTRY_FLOW = 0.0
    EXIT_FLOW = -0.008
    EXIT_AVG_CONVICTION = 0.020
    OPEN_CONFIRM_S = 45.0
    FLOW_EMA_ALPHA = 0.30
    CONV_GIVEBACK = 0.30
    MIN_COIN_DAY_VOLUME = 1000.0
    MAX_HOSTILE_FUNDING = 0.0004
    MAX_BASIS_ABS = 0.05
    MAX_SPREAD_PCT = 0.01
    MANAGED_ONLY = True
    FLATTEN_WHEN_DROPPED = True
    REBALANCE_DRIFT_PCT = 25.0
    MAX_ACTIONS_PER_CYCLE = 3
    SIZE_MODE = "fixed"
    LEVERAGE_MODE = "auto"
    COPY_MARGIN_CAP_PCT = 100.0
    TRADE_MODE = "follow"
    RANK_WINDOW = "month"
    CONFIRM_WINDOW = "week"
    MIN_ACCOUNT_VALUE = 1000.0
    MIN_WINDOW_PNL = 100.0
    MIN_CONFIRM_PNL = 0.0
    MIN_WINDOW_VOLUME = 1000.0
    MAX_VOLUME_TO_EQUITY = 250.0
    MAX_WINDOW_ROI = 1.5
    MIN_WINDOW_ROI = 0.03
    CANDIDATE_POOL = 10
    BASKET_FILTER_MODE = "holder"
    MAX_VOLUME_TO_PROFIT = 150.0


def _snap(addr: str, equity: float, convs: dict[str, float], ts: float = 1_000.0, leverage: int = 5) -> WalletSnapshot:
    positions = []
    for coin, conv in convs.items():
        positions.append(
            WalletPos(
                coin=coin,
                side="long" if conv > 0 else "short",
                size=1.0,
                notional=abs(conv) * equity,
                entry_px=100.0,
                leverage=leverage,
                isolated=False,
                conviction=conv,
            )
        )
    return WalletSnapshot(
        address=addr,
        account_value=equity,
        positions=positions,
        fetched_at=ts,
        fingerprint=fingerprint(positions),
    )


class ParseTests(unittest.TestCase):
    def test_leaderboard_camel_and_snake(self) -> None:
        payload = {
            "leaderboardRows": [
                {
                    "ethAddress": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "accountValue": "12000",
                    "windowPerformances": [
                        ["month", {"pnl": "5000", "roi": "0.2", "vlm": "200000"}],
                        ["week", {"pnl": "400", "roi": "0.04", "vlm": "20000"}],
                    ],
                }
            ]
        }
        rows = parse_leaderboard(payload)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].windows["month"].pnl, 5000.0)

    def test_positions_all_dexes_shape(self) -> None:
        states = [
            {
                "marginSummary": {"accountValue": "10000"},
                "assetPositions": [
                    {
                        "position": {
                            "coin": "xyz:SNDK",
                            "szi": "0.5",
                            "positionValue": "2000",
                            "entryPx": "4000",
                            "leverage": {"type": "cross", "value": 10},
                        }
                    }
                ],
            }
        ]
        pos = parse_positions(states, 10000)
        self.assertEqual(pos[0].coin, "xyz:SNDK")
        self.assertAlmostEqual(pos[0].conviction, 0.2)


class QualifyTests(unittest.TestCase):
    def test_drops_lottery_and_mm(self) -> None:
        good = LeaderboardRow(
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            20000,
            None,
            {
                "month": WindowPerf(5000, 0.2, 200000),
                "week": WindowPerf(800, 0.04, 40000),
            },
        )
        lottery = LeaderboardRow(
            "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            3000,
            None,
            {
                "month": WindowPerf(8000, 4.0, 20000),
                "week": WindowPerf(8000, 4.0, 20000),
            },
        )
        spike = LeaderboardRow(
            "0xdddddddddddddddddddddddddddddddddddddddd",
            6000,
            None,
            {
                "month": WindowPerf(5766, 7.53, 80000),
                "week": WindowPerf(5766, 7.53, 80000),
            },
        )
        mm = LeaderboardRow(
            "0xcccccccccccccccccccccccccccccccccccccccc",
            20000,
            None,
            {
                "month": WindowPerf(3000, 0.1, 20_000_000),
                "week": WindowPerf(400, 0.02, 4_000_000),
            },
        )
        out = shortlist([good, lottery, spike, mm], _Cfg())
        self.assertEqual([w.address for w in out], [good.address])

    def test_off_mode_keeps_high_volume(self) -> None:
        cfg = _Cfg()
        cfg.BASKET_FILTER_MODE = "off"
        mm = LeaderboardRow(
            "0xcccccccccccccccccccccccccccccccccccccccc",
            20000,
            None,
            {
                "month": WindowPerf(3000, 0.1, 20_000_000),
                "week": WindowPerf(400, 0.02, 4_000_000),
            },
        )
        out = shortlist([mm], cfg)
        self.assertEqual(len(out), 1)

    def test_holder_tape_no_fills_is_holder(self) -> None:
        cfg = _Cfg()
        ok, why = is_holder_tape([], 1_700_000_000_000, cfg)
        self.assertTrue(ok)
        self.assertEqual(why, "no_fills")

    def test_holder_tape_dense_fills_is_scalper(self) -> None:
        cfg = _Cfg()
        now = 1_700_000_000_000
        fills = [{"time": now - i * 20_000} for i in range(40)]
        ok, _why = is_holder_tape(fills, now, cfg)
        self.assertFalse(ok)

    def test_holder_tape_few_slow_fills_is_holder(self) -> None:
        cfg = _Cfg()
        now = 1_700_000_000_000
        fills = [{"time": now - i * 3 * 3600_000} for i in range(5)]
        ok, _why = is_holder_tape(fills, now, cfg)
        self.assertTrue(ok)


class ConsensusTests(unittest.TestCase):
    def test_average_conviction_not_whale_dollars(self) -> None:
        now = 1_000.0
        # Tiny skilled wallet heavy long BTC; whale with dust BTC.
        snaps = [
            _snap("0x1", 10_000, {"BTC": 0.40}, now),
            _snap("0x2", 10_000, {"BTC": 0.30}, now),
            _snap("0x3", 10_000_000, {"BTC": 0.005}, now),  # below dust threshold
            _snap("0x4", 10_000, {"ETH": -0.25}, now),
        ]
        votes = build_votes(snaps, set(), _Cfg(), now)
        btc = next(v for v in votes if v.coin == "BTC")
        self.assertEqual(btc.side, "long")
        self.assertGreater(btc.avg_conviction, 0.1)
        self.assertEqual(btc.wallets_long, 2)

    def test_stale_and_hyper_excluded(self) -> None:
        now = 10_000.0
        snaps = [
            _snap("0x1", 10_000, {"BTC": 0.4}, now),
            _snap("0x2", 10_000, {"BTC": 0.4}, now),
            _snap("0x3", 10_000, {"BTC": 0.4}, now - 500),  # stale
            _snap("0x4", 10_000, {"BTC": 0.4}, now),
        ]
        cfg = _Cfg()
        cfg.MIN_LIVE_VOTERS = 2
        votes = build_votes(snaps, {"0x4"}, cfg, now)
        self.assertTrue(votes)
        self.assertEqual(votes[0].voters, 2)

    def test_crowd_pct_of_wallet_list(self) -> None:
        now = 1_000.0
        snaps = [
            _snap("0x1", 10_000, {"BTC": 0.4}, now),
            _snap("0x2", 10_000, {"BTC": 0.4}, now),
            _snap("0x3", 10_000, {"BTC": 0.4}, now),
            _snap("0x4", 10_000, {"ETH": -0.4}, now),
        ]
        cfg = _Cfg()
        cfg.BASKET_SIZE = 20
        cfg.MIN_LIVE_VOTERS_PCT = 0.15
        cfg.MIN_LIVE_VOTERS = 99
        cfg.MIN_WALLETS_ON_COIN_PCT = 0.10
        cfg.MIN_SIDE_AGREEMENT = 0.10
        cfg.MIN_AVG_CONVICTION = 0.02
        votes = build_votes(snaps, set(), cfg, now)
        btc = next(v for v in votes if v.coin == "BTC")
        self.assertAlmostEqual(btc.agreement, 3 / 20, places=4)
        cfg.MIN_SIDE_AGREEMENT = 0.50
        cfg.EXIT_SIDE_AGREEMENT = 0.50
        self.assertFalse(build_votes(snaps, set(), cfg, now))

    def test_scalpers_still_vote(self) -> None:
        now = 1_000.0
        snaps = [
            _snap("0x1", 10_000, {"BTC": 0.4}, now),
            _snap("0x2", 10_000, {"BTC": 0.4}, now),
            _snap("0x3", 10_000, {"BTC": 0.3}, now),
        ]
        votes = build_votes(snaps, set(), _Cfg(), now)
        self.assertTrue(votes)
        self.assertEqual(votes[0].wallets_long, 3)

    def test_targets_split_margin(self) -> None:
        votes = [
            CoinVote("BTC", "long", 8, 1, 10, 0.8, 0.2, 8, 1.6),
            CoinVote("ETH", "short", 1, 6, 10, 0.6, -0.1, 5, 0.6),
        ]
        t = votes_to_targets(votes, _Cfg())
        self.assertEqual(len(t), 2)
        self.assertLessEqual(sum(x.margin_pct for x in t), 40.01)
        self.assertGreater(t[0].margin_pct, t[1].margin_pct)

    def test_single_name_shrinks(self) -> None:
        votes = [CoinVote("BTC", "long", 8, 1, 10, 0.8, 0.2, 8, 1.6)]
        t = votes_to_targets(votes, _Cfg())
        self.assertAlmostEqual(t[0].margin_pct, 40.0 * 0.55, places=4)
        self.assertEqual(t[0].leverage, 8)

    def test_wallet_avg_copies_margin_and_mean_leverage(self) -> None:
        cfg = _Cfg()
        cfg.SIZE_MODE = "wallet_avg"
        cfg.LEVERAGE_MODE = "auto"
        vote = CoinVote(
            "BTC",
            "long",
            2,
            0,
            2,
            1.0,
            4.0,
            10,
            1.0,
            mean_leverage=10.0,
            avg_margin_pct=40.0,
        )
        t = votes_to_targets([vote], cfg)
        self.assertEqual(len(t), 1)
        self.assertAlmostEqual(t[0].margin_pct, 40.0)
        self.assertEqual(t[0].leverage, 10)

    def test_wallet_avg_build_votes_margin(self) -> None:
        now = 1_000.0
        # 30% margin @10x → conviction 3.0; 50% margin @10x → conviction 5.0
        a = _snap("0x1", 10_000, {"BTC": 3.0}, now, leverage=10)
        b = _snap("0x2", 10_000, {"BTC": 5.0}, now, leverage=10)
        cfg = _Cfg()
        cfg.MIN_LIVE_VOTERS = 2
        cfg.MIN_WALLETS_ON_COIN = 2
        cfg.MIN_SIDE_AGREEMENT = 0.5
        cfg.MIN_AVG_CONVICTION = 0.02
        votes = build_votes([a, b], set(), cfg, now)
        self.assertTrue(votes)
        self.assertAlmostEqual(votes[0].avg_margin_pct, 40.0, places=4)
        self.assertAlmostEqual(votes[0].mean_leverage, 10.0, places=4)

    def test_reverse_mode_opens_opposite_side(self) -> None:
        cfg = _Cfg()
        cfg.TRADE_MODE = "reverse"
        eng = BookEngine()
        vote = CoinVote("BTC", "long", 8, 1, 10, 0.8, 0.2, 5, 1.0)
        mk = {"BTC": MarketCtx("BTC", 5_000_000, 0.00001, 1.0, 0.0)}
        first = eng.refine([vote], markets=mk, managed=set(), cfg=cfg, now=1000.0)
        self.assertEqual(first, [])
        later = eng.refine(
            [CoinVote("BTC", "long", 8, 1, 10, 0.8, 0.2, 5, 1.0)],
            markets=mk,
            managed=set(),
            cfg=cfg,
            now=1100.0,
        )
        self.assertEqual(len(later), 1)
        self.assertEqual(later[0].side, "short")
        t = votes_to_targets(later, cfg)
        self.assertEqual(t[0].side, "short")
        self.assertAlmostEqual(t[0].margin_pct, 40.0 * 0.55, places=4)

    def test_reverse_mode_funding_uses_our_side(self) -> None:
        cfg = _Cfg()
        cfg.TRADE_MODE = "reverse"
        eng = BookEngine()
        eng.ok_since = {"BTC|long": 0.0}
        mk = {"BTC": MarketCtx("BTC", 5_000_000, 0.002, 1.0, 0.0)}
        out = eng.refine(
            [CoinVote("BTC", "long", 8, 1, 10, 0.8, 0.2, 5, 1.0)],
            markets=mk,
            managed=set(),
            cfg=cfg,
            now=5000.0,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].side, "short")

    def test_equity_crash_mutes(self) -> None:
        now = 1_000.0
        snaps = [
            _snap("0x1", 10_000, {"BTC": 0.4}, now),
            _snap("0x2", 10_000, {"BTC": 0.4}, now),
            _snap("0x3", 2_000, {"BTC": 0.9}, now),  # crashed vs 10k baseline
        ]
        crashed = crashed_wallets(snaps, {"0x1": 10_000, "0x2": 10_000, "0x3": 10_000}, _Cfg())
        self.assertIn("0x3", crashed)


class GateTests(unittest.TestCase):
    def _btc_vote(self, conv: float = 0.2) -> CoinVote:
        return CoinVote("BTC", "long", 8, 1, 10, 0.8, conv, 5, 1.0)

    def test_waits_for_persist_then_allows(self) -> None:
        cfg = _Cfg()
        eng = BookEngine()
        mk = {"BTC": MarketCtx("BTC", 5_000_000, 0.00001, 1.0, 0.0)}
        first = eng.refine([self._btc_vote()], markets=mk, managed=set(), cfg=cfg, now=1000.0)
        self.assertEqual(first, [])
        later = eng.refine([self._btc_vote()], markets=mk, managed=set(), cfg=cfg, now=1100.0)
        self.assertEqual(len(later), 1)

    def test_blocks_dumping_new_entry(self) -> None:
        cfg = _Cfg()
        eng = BookEngine()
        eng.ema = {"BTC": 0.25}
        mk = {"BTC": MarketCtx("BTC", 5_000_000, 0.00001, 1.0, 0.0)}
        out = eng.refine([self._btc_vote(0.10)], markets=mk, managed=set(), cfg=cfg, now=2000.0)
        self.assertEqual(out, [])

    def test_exits_hold_on_dump(self) -> None:
        cfg = _Cfg()
        eng = BookEngine()
        eng.ema = {"BTC": 0.20}
        eng.peak = {"BTC": 0.20}
        eng.ok_since = {"BTC|long": 1000.0}
        mk = {"BTC": MarketCtx("BTC", 5_000_000, 0.00001, 1.0, 0.0)}
        out = eng.refine([self._btc_vote(0.04)], markets=mk, managed={"BTC"}, cfg=cfg, now=2000.0)
        self.assertEqual(out, [])
        self.assertNotIn("BTC|long", eng.ok_since)

    def test_exit_resets_persist_so_reopen_must_wait(self) -> None:
        """A fade must not reopen on the next tick just because persist already elapsed."""
        cfg = _Cfg()
        cfg.OPEN_CONFIRM_S = 45.0
        eng = BookEngine()
        eng.ema = {"BTC": 0.20}
        eng.peak = {"BTC": 0.20}
        eng.ok_since = {"BTC|long": 1000.0}
        mk = {"BTC": MarketCtx("BTC", 5_000_000, 0.00001, 1.0, 0.0)}
        dumped = eng.refine([self._btc_vote(0.04)], markets=mk, managed={"BTC"}, cfg=cfg, now=2000.0)
        self.assertEqual(dumped, [])
        rebound = eng.refine(
            [self._btc_vote(0.20)],
            markets=mk,
            managed=set(),
            cfg=cfg,
            now=2010.0,
        )
        self.assertEqual(rebound, [])
        self.assertIn("BTC|long", eng.ok_since)
        later = eng.refine(
            [self._btc_vote(0.20)],
            markets=mk,
            managed=set(),
            cfg=cfg,
            now=2060.0,
        )
        self.assertEqual(len(later), 1)

    def test_tiny_negative_flow_keeps_persist(self) -> None:
        cfg = _Cfg()
        eng = BookEngine()
        mk = {"BTC": MarketCtx("BTC", 5_000_000, 0.00001, 1.0, 0.0)}
        eng.refine([self._btc_vote(0.20)], markets=mk, managed=set(), cfg=cfg, now=1000.0)
        self.assertIn("BTC|long", eng.ok_since)
        later = eng.refine([self._btc_vote(0.199)], markets=mk, managed=set(), cfg=cfg, now=1020.0)
        self.assertEqual(later, [])
        self.assertIn("BTC|long", eng.ok_since)

    def test_raw_flow_exit_when_holding(self) -> None:
        cfg = _Cfg()
        cfg.EXIT_RAW_FLOW = -0.015
        eng = BookEngine()
        eng.prev_raw = {"BTC": 0.30}
        mk = {"BTC": MarketCtx("BTC", 5_000_000, 0.00001, 1.0, 0.0)}
        out = eng.refine([self._btc_vote(0.10)], markets=mk, managed={"BTC"}, cfg=cfg, now=2000.0)
        self.assertEqual(out, [])

    def test_agreement_giveback_exit(self) -> None:
        cfg = _Cfg()
        cfg.EXIT_AGREEMENT_GIVEBACK = 0.25
        eng = BookEngine()
        eng.peak_agreement = {"BTC": 0.40}
        mk = {"BTC": MarketCtx("BTC", 5_000_000, 0.00001, 1.0, 0.0)}
        v = CoinVote("BTC", "long", 4, 1, 10, 0.28, 0.20, 5, 1.0)
        out = eng.refine([v], markets=mk, managed={"BTC"}, cfg=cfg, now=2000.0)
        self.assertEqual(out, [])

    def test_sticky_slots_keep_holds_over_new_name(self) -> None:
        cfg = _Cfg()
        cfg.MAX_COINS_IN_BOOK = 3
        cfg.STICKY_BOOK_SLOTS = True
        cfg.OPEN_CONFIRM_S = 0.0
        eng = BookEngine()
        coins = ("BTC", "ETH", "ZEC", "HYPE")
        mk = {c: MarketCtx(c, 5_000_000, 0.00001, 1.0, 0.0) for c in coins}

        def vote(coin: str, score: float) -> CoinVote:
            return CoinVote(coin, "long", 8, 1, 10, 0.8, 0.20, 5, score)

        raw = [vote("HYPE", 9.0), vote("BTC", 3.0), vote("ETH", 2.0), vote("ZEC", 1.0)]
        held = eng.refine(raw, markets=mk, managed={"BTC", "ETH", "ZEC"}, cfg=cfg, now=5000.0)
        self.assertEqual({v.coin for v in held}, {"BTC", "ETH", "ZEC"})
        fill = eng.refine(raw, markets=mk, managed={"BTC", "ETH"}, cfg=cfg, now=5000.0)
        self.assertEqual({v.coin for v in fill}, {"BTC", "ETH", "HYPE"})

    def test_close_bypasses_rebalance_cooldown(self) -> None:
        import logging
        import time as _time

        cfg = _Cfg()
        cfg.REBALANCE_COOLDOWN_S = 9999.0
        cfg.FLATTEN_WHEN_DROPPED = True
        reb = Rebalancer(None, None, cfg, logging.getLogger("t"))  # type: ignore[arg-type]
        reb.current_book = lambda: ([OurPos("BTC", "long", 1.0, 1000.0, 100.0, 5)], 1000.0)  # type: ignore[method-assign]
        reb._close = lambda coin, size: True  # type: ignore[method-assign]
        now = _time.time()
        managed, attempted = reb.run([], {"BTC"}, now - 10, now)
        self.assertNotIn("BTC", managed)
        self.assertTrue(attempted)

    def test_research_compact_book(self) -> None:
        from pmf.research import ResearchWriter, compact_book, compact_candle, compact_mkt, compact_research_votes
        from pmf.types import CoinVote, MarketCtx

        snap = _snap("0x1", 10_000, {"BTC": 0.4, "ETH": -0.2}, 1000.0, leverage=8)
        rows = compact_book(snap)
        self.assertEqual(len(rows), 2)
        by_coin = {r[0]: r for r in rows}
        self.assertEqual(by_coin["BTC"][1], 1)
        self.assertEqual(by_coin["ETH"][1], -1)
        self.assertAlmostEqual(by_coin["BTC"][2], 0.4)
        self.assertEqual(by_coin["BTC"][3], 8)
        m = compact_mkt(MarketCtx("BTC", 1e6, 0.0001, 2e9, 0.001, mark=70000.0))
        self.assertEqual(len(m), 5)
        self.assertAlmostEqual(m[0], 70000.0)
        bar = compact_candle({"t": 1000, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10})
        self.assertEqual(bar[0], 1000)
        self.assertAlmostEqual(bar[4], 1.5)
        vote = CoinVote(
            coin="BTC",
            side="long",
            wallets_long=10,
            wallets_short=2,
            voters=12,
            agreement=0.2,
            avg_conviction=0.05,
            median_leverage=5,
            score=1.0,
        )
        vote.raw_conviction = 0.05
        vote.ema = 0.04
        vote.flow = 0.001
        vote.raw_flow = -0.03
        vote.persist_s = 120.0
        packed = compact_research_votes([vote])
        self.assertEqual(packed[0]["c"], "BTC")
        self.assertAlmostEqual(packed[0]["rflow"], -0.03)
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            w = ResearchWriter(Path(td), enabled=True, instance="research")
            w.seen_coins.add("BTC")
            w._last_books_at = 12.0
            dump = w.dump_resume()
            w2 = ResearchWriter(Path(td), enabled=True, instance="research")
            w2.load_resume(dump)
            self.assertIn("BTC", w2.seen_coins)
            self.assertAlmostEqual(w2._last_books_at, 12.0)
            # Coverage gate skips books when frac too low.
            wrote, _ = w.maybe_record_crowd(
                ts=100.0,
                books_interval_s=1.0,
                marks_interval_s=999.0,
                snaps_by_addr={"0x1": snap},
                research_addrs=["0x1"],
                markets={},
                coverage={"fresh": 0, "listed": 10, "frac": 0.0},
                min_fresh_wallets=50,
                crowd_all=[],
                crowd_holders=[],
                crowd_trade=[],
            )
            self.assertFalse(wrote)
            wrote2, _ = w.maybe_record_crowd(
                ts=200.0,
                books_interval_s=1.0,
                marks_interval_s=999.0,
                snaps_by_addr={"0x1": snap},
                research_addrs=["0x1"],
                markets={},
                coverage={"fresh": 8, "listed": 10, "frac": 0.8},
                min_fresh_wallets=1,
                crowd_all=[{"c": "BTC"}],
                crowd_holders=[],
                crowd_trade=[{"c": "BTC", "rflow": -0.02}],
            )
            self.assertTrue(wrote2)
            day = Path(td) / "research"
            books = list(day.glob("*/books.jsonl"))
            self.assertEqual(len(books), 1)
            line = books[0].read_text(encoding="utf-8").strip().splitlines()[-1]
            import json

            payload = json.loads(line)
            self.assertEqual(payload["schema"], 2)
            self.assertEqual(len(payload["w"][0]), 4)  # addr, eq, pos, fetched_at
            ticks = list(day.glob("*/crowd_ticks.jsonl"))
            self.assertEqual(len(ticks), 1)
            tline = json.loads(ticks[0].read_text(encoding="utf-8").strip().splitlines()[-1])
            self.assertIn("trade", tline)

    def test_hostile_funding_blocks_long(self) -> None:
        cfg = _Cfg()
        eng = BookEngine()
        eng.ok_since = {"BTC|long": 0.0}
        mk = {"BTC": MarketCtx("BTC", 5_000_000, 0.002, 1.0, 0.0)}
        out = eng.refine([self._btc_vote()], markets=mk, managed=set(), cfg=cfg, now=5000.0)
        self.assertEqual(out, [])


class BacktestTests(unittest.TestCase):
    def test_numba_sim_fees_and_pnl(self) -> None:
        import numpy as np
        from pmf.bt_numba import PMF_FEE_PCT, simulate_portfolio

        marks = np.array([[100.0], [100.0], [120.0]], dtype=np.float64)
        target_coin = np.array([[0, -1, -1], [0, -1, -1], [-1, -1, -1]], dtype=np.int32)
        target_side = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int8)
        out = simulate_portfolio(
            marks,
            target_coin,
            target_side,
            cooldown_s=0.0,
            tick_interval_s=60.0,
            fee_rate=PMF_FEE_PCT,
            margin_frac=0.10,
            leverage=1.0,
            initial_equity=1000.0,
            day_ids=np.array([0, 0, 0], dtype=np.int32),
        )
        self.assertEqual(out["round_trips"], 1)
        self.assertGreater(out["return_pct"], 1.0)
        self.assertGreater(out["total_fees"], 0.0)

    def test_open_hold_marked_to_last_price(self) -> None:
        import numpy as np
        from pmf.bt_numba import simulate_portfolio

        # Enter and stay long — force-closed at last mark as one completed trip.
        marks = np.array([[100.0], [100.0], [120.0]], dtype=np.float64)
        target_coin = np.array([[0, -1, -1], [0, -1, -1], [0, -1, -1]], dtype=np.int32)
        target_side = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.int8)
        out = simulate_portfolio(
            marks,
            target_coin,
            target_side,
            cooldown_s=0.0,
            tick_interval_s=60.0,
            fee_rate=0.0005,
            margin_frac=0.10,
            leverage=1.0,
            initial_equity=1000.0,
            day_ids=np.array([0, 0, 0], dtype=np.int32),
        )
        self.assertEqual(out["round_trips"], 1)
        self.assertEqual(out["open_legs"], 1)  # were open at end before force-close
        self.assertGreater(out["return_pct"], 1.0)

    def test_hold_can_rank_over_churn(self) -> None:
        from pmf.bt_tune import rank_results, TuneResult

        # Hold: one force-close at end, higher PnL than churny 4-trip result.
        hold = TuneResult("hold", {}, 10.0, 2.0, 1, 100.0, 9.5, 0.0, 0.0, 0.0, open_legs=1)
        churn = TuneResult("churn", {}, 4.0, 3.0, 4, 75.0, 3.0, 12.0, 1.0, 1.0, open_legs=0)
        ranked = rank_results([churn, hold], span_days=0.3)
        self.assertEqual(ranked[0].strategy, "hold")

    def test_rank_by_profit_when_enough_trades(self) -> None:
        from pmf.bt_tune import rank_results, TuneResult

        low = TuneResult("low", {}, 0.5, 2.0, 6, 50.0, 0.4, 2.0, 1.0, 0.4)
        high = TuneResult("high", {}, 4.0, 3.0, 6, 50.0, 3.5, 2.0, 1.0, 0.4)
        ranked = rank_results([low, high], span_days=1.0)
        self.assertEqual(ranked[0].strategy, "high")

    def test_build_dataset_from_synthetic_books(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from pmf.research_load import build_dataset

        snap = _snap("0xabc", 10000.0, {"BTC": 0.5}, 1000.0)
        row = {
            "ts": 1000.0,
            "w": [[snap.address, snap.account_value, [["BTC", 1, 0.5, 5]], snap.fetched_at]],
            "mkt": {"BTC": [100.0, 0.0, 0.0, 0.0, 0.0]},
            "cov": {"fresh": 10, "listed": 10, "frac": 1.0},
        }
        with tempfile.TemporaryDirectory() as td:
            day = Path(td) / "research" / "2026-08-23"
            day.mkdir(parents=True)
            (day / "books.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (day / "marks.jsonl").write_text(
                json.dumps({"ts": 1000.0, "mkt": {"BTC": [100.0, 0, 0, 0, 0]}}) + "\n",
                encoding="utf-8",
            )
            ds = build_dataset(Path(td), max_days=7)
            self.assertIsNotNone(ds)
            assert ds is not None
            self.assertEqual(ds.n_ticks, 1)
            self.assertIn("BTC", ds.coin_index)
            self.assertEqual(ds.live_basket_target, 50)

    def test_crowd_all_uses_full_pool(self) -> None:
        from types import SimpleNamespace

        from pmf.bt_replay import _crowd, _filter_snaps

        basket = frozenset(f"0x{i:040x}" for i in range(50))
        ds = SimpleNamespace(
            live_basket_target=50,
            pool_size=50,
            live_holder_addrs=set(),
            holder_addrs=set(),
            cloud_basket_addrs=basket,
        )
        holders, listed = _crowd(ds, "all")
        self.assertEqual(listed, 50)
        self.assertEqual(holders, basket)

        snaps = [_snap(f"0x{i:040x}", 1000.0, {}, 1000.0) for i in range(200)]
        self.assertEqual(len(_filter_snaps(snaps, basket, "all", listed=50)), 50)

    def test_live_holders_are_first_n_in_roi_order(self) -> None:
        from pmf.research_load import holders_from_ranked_pool

        rows = [{"address": f"0x{i:040x}", "holder": i % 3 == 0} for i in range(20)]
        all_h, live = holders_from_ranked_pool(rows, basket_size=3)
        self.assertEqual(len(all_h), 7)
        self.assertEqual(len(live), 3)
        self.assertIn(f"0x{0:040x}", live)
        self.assertIn(f"0x{3:040x}", live)
        self.assertIn(f"0x{6:040x}", live)
        self.assertNotIn(f"0x{9:040x}", live)

    def test_apply_maps_strategy_to_holder_filter(self) -> None:
        from apply_cloud_tune import filter_mode_for_strategy
        from pmf.strategy_spec import parse_strategy

        self.assertEqual(filter_mode_for_strategy("cloud_holders"), "holder")
        self.assertEqual(filter_mode_for_strategy("direct_holders"), "holder")
        self.assertEqual(filter_mode_for_strategy("flow_holders"), "holder")
        self.assertEqual(filter_mode_for_strategy("mtf_meta_holders"), "holder")
        self.assertEqual(filter_mode_for_strategy("cloud_all"), "off")
        self.assertEqual(filter_mode_for_strategy("direct_all"), "off")
        self.assertEqual(filter_mode_for_strategy("crowd_dump_holders"), "holder")
        self.assertEqual(filter_mode_for_strategy("crowd_dump_all"), "off")
        self.assertEqual(filter_mode_for_strategy("crowd_btcdump_holders"), "holder")

        dump = parse_strategy("crowd_dump_holders")
        self.assertEqual(dump.gate, "dump")
        self.assertTrue(dump.needs_candles)
        self.assertEqual(dump.candle_intervals, ("1m", "15m", "1h"))
        self.assertEqual(dump.style, "refine")
        plain = parse_strategy("cloud_holders")
        self.assertIsNone(plain.gate)
        self.assertFalse(plain.needs_candles)
        self.assertEqual(plain.candle_intervals, ())
        direct = parse_strategy("direct_all")
        self.assertEqual(direct.style, "direct")
        self.assertEqual(direct.filter_mode, "off")
        rsi = parse_strategy("crowd_rsi_holders")
        self.assertEqual(rsi.gate, "rsi")
        self.assertTrue(rsi.needs_candles)
        mtf = parse_strategy("mtf_meta_holders")
        self.assertEqual(mtf.style, "mtf_meta")
        self.assertTrue(mtf.needs_candles)
        self.assertEqual(mtf.filter_mode, "holder")
        self.assertIsNone(mtf.gate)
        self.assertEqual(mtf.candle_intervals, ("1m", "15m", "1h"))

    def test_price_gate_blocks_long_on_dump(self) -> None:
        from types import SimpleNamespace

        from pmf.price_gates import price_gate_ok

        cfg = SimpleNamespace(DUMP_RET_PCT=-0.02, DUMP_LOOKBACK_S=1800.0)
        ok = price_gate_ok(
            gate="dump",
            coin="ETH",
            side="long",
            managed=set(),
            cfg=cfg,
            ret=lambda c, look: -0.05,
            ema_bias=lambda c: 0.0,
            atr_pct=lambda c: 0.01,
        )
        self.assertFalse(ok)
        ok2 = price_gate_ok(
            gate="dump",
            coin="ETH",
            side="long",
            managed=set(),
            cfg=cfg,
            ret=lambda c, look: -0.01,
            ema_bias=lambda c: 0.0,
            atr_pct=lambda c: 0.01,
        )
        self.assertTrue(ok2)

    def test_live_price_book_ret(self) -> None:
        from pmf.price_engine import PriceEngine

        eng = PriceEngine()
        eng.ingest_mark("ETH", 1000.0, 100.0)
        eng.ingest_mark("ETH", 2800.0, 95.0)
        self.assertLess(eng.ret("ETH", 1800.0, 2800.0), -0.04)
        self.assertFalse(eng.has_coin("BTC"))

    def test_price_engine_candle_range_dump(self) -> None:
        from pmf.price_engine import PriceEngine

        eng = PriceEngine()
        # 1m bars: dumped from high 110 to close 100
        base_ms = 1_700_000_000_000
        for i in range(20):
            o = 100.0 + i * 0.5
            h = o + 2
            eng.ingest_bar(
                "ETH",
                "1m",
                [base_ms + i * 60_000, o, h, o - 1, o, 10.0],
            )
        # last bar dumps
        eng.ingest_bar("ETH", "1m", [base_ms + 20 * 60_000, 108.0, 110.0, 99.0, 100.0, 50.0])
        asof = base_ms / 1000.0 + 21 * 60
        rd = eng.range_dump("ETH", 1800.0, asof, tf="1m")
        self.assertLess(rd, -0.05)

    def test_pick_trade_votes_parity_direct(self) -> None:
        from types import SimpleNamespace

        from pmf.consensus import BookEngine
        from pmf.strategy_exec import pick_trade_votes
        from pmf.strategy_spec import parse_strategy
        from pmf.types import CoinVote

        raw = [
            CoinVote("BTC", "long", 10, 2, 20, 0.5, 0.1, 10, 1.0),
            CoinVote("ETH", "short", 2, 8, 20, 0.4, 0.1, 10, 0.9),
        ]
        cfg = SimpleNamespace(MAX_COINS_IN_BOOK=2, MIN_ENTRY_FLOW=0.0, EXIT_RAW_FLOW=-0.02)
        out = pick_trade_votes(
            raw,
            book=BookEngine(),
            markets={},
            managed=set(),
            cfg=cfg,
            now=1.0,
            spec=parse_strategy("direct_holders"),
            price=None,
        )
        self.assertEqual([v.coin for v in out], ["BTC", "ETH"])

    def test_price_features_detect_dump(self) -> None:
        from pmf.price_engine import PriceEngine

        eng = PriceEngine()
        # Coin dumps ~4% from t=0 to t=1800.
        for t, px in ((0.0, 100.0), (600.0, 99.0), (1200.0, 98.0), (1800.0, 96.0)):
            eng.ingest_mark("AAA", t, px)
        ret = eng.ret("AAA", 1800.0, 1800.0)
        self.assertLess(ret, -0.03)

    def test_sim_size_matches_live_fixed_margin(self) -> None:
        from types import SimpleNamespace

        from pmf.bt_tune import sim_size_from_cfg

        cfg = SimpleNamespace(
            OUR_GROSS_MARGIN_PCT=90.0,
            MAX_MARGIN_PER_COIN_PCT=33.33,
            OUR_MIN_LEVERAGE=2,
            OUR_MAX_LEVERAGE=20,
        )
        margin, lev = sim_size_from_cfg(cfg)
        self.assertAlmostEqual(margin, 0.30, places=3)
        self.assertEqual(lev, 10.0)

    def test_mark_aux_not_clobbered_by_densify(self) -> None:
        from pmf.price_engine import PriceEngine

        eng = PriceEngine()
        eng.ingest_mark("ETH", 1000.0, 100.0, funding=0.0002, basis=0.001, oi=1e6, day_vol=5e6)
        eng.ingest_mark("ETH", 1000.0, 101.0, aux=False)  # densify same ts
        self.assertAlmostEqual(eng.funding_at("ETH", 1000.0), 0.0002)
        self.assertAlmostEqual(eng.basis_at("ETH", 1000.0), 0.001)
        self.assertAlmostEqual(eng.day_vol_at("ETH", 1000.0), 5e6)
        self.assertAlmostEqual(eng.price_at("ETH", 1000.0), 101.0)

    def test_backtest_markets_use_funding_from_engine(self) -> None:
        from pmf.bt_replay import _markets_for_row
        from pmf.price_engine import PriceEngine
        from pmf.research_load import BookRow

        eng = PriceEngine()
        eng.ingest_mark("BTC", 50.0, 70000.0, funding=0.00015, basis=0.002, oi=2e9, day_vol=1e8)
        book = BookRow(ts=50.0, wallets=[], marks={"BTC": 70000.0}, mkt={})
        mkts = _markets_for_row(book, {"BTC": 0}, price=eng, asof=50.0)
        self.assertAlmostEqual(mkts["BTC"].funding, 0.00015)
        self.assertAlmostEqual(mkts["BTC"].day_volume, 1e8)
        self.assertGreater(mkts["BTC"].mark, 0)

    def test_mtf_meta_follow_hold_and_max_hold(self) -> None:
        from types import SimpleNamespace

        from pmf.mtf_exec import MtfTrader
        from pmf.price_engine import PriceEngine
        from pmf.types import CoinVote

        class Stub(MtfTrader):
            def mtf_entry_side(self, coin, asof, price, setup):
                return 1

            def mtf_should_exit(self, coin, asof, price, setup, pos):
                return False

        v = CoinVote("ETH", "long", 10, 1, 20, 0.5, 0.1, 10, 1.0)
        cfg = SimpleNamespace(
            MTF_PRESET="rsi_long_30",
            MTF_META_MODE="follow",
            MTF_EXIT="none",
            MTF_MAX_HOLD_S=100.0,
            MTF_EMA=50,
            MTF_MIN_AGREE=2,
            MTF_MIN_SCORE=0.3,
        )
        eng = PriceEngine()
        eng.ingest_mark("ETH", 50.0, 100.0)
        t = Stub()
        out = t.apply([v], managed=set(), cfg=cfg, now=50.0, price=eng, max_n=3)
        self.assertEqual([x.coin for x in out], ["ETH"])
        self.assertEqual(out[0].side, "long")
        t.mtf_entry_side = lambda *a, **k: 0  # type: ignore
        out2 = t.apply([v], managed={"ETH"}, cfg=cfg, now=80.0, price=eng, max_n=3)
        self.assertEqual([x.coin for x in out2], ["ETH"])
        out3 = t.apply([v], managed={"ETH"}, cfg=cfg, now=200.0, price=eng, max_n=3)
        self.assertEqual(out3, [])

    def test_mtf_meta_reverse_fades_crowd(self) -> None:
        from types import SimpleNamespace

        from pmf.mtf_exec import MtfTrader
        from pmf.price_engine import PriceEngine
        from pmf.types import CoinVote

        class Stub(MtfTrader):
            def mtf_entry_side(self, coin, asof, price, setup):
                return 1

            def mtf_should_exit(self, coin, asof, price, setup, pos):
                return False

        cfg = SimpleNamespace(
            MTF_PRESET="rsi_long_30",
            MTF_META_MODE="reverse",
            MTF_EXIT="none",
            MTF_MAX_HOLD_S=86400.0,
            MTF_EMA=50,
            MTF_MIN_AGREE=2,
            MTF_MIN_SCORE=0.3,
        )
        eng = PriceEngine()
        eng.ingest_mark("ETH", 50.0, 100.0)
        t = Stub()
        long_crowd = CoinVote("ETH", "long", 10, 1, 20, 0.5, 0.1, 10, 1.0)
        self.assertEqual(
            t.apply([long_crowd], managed=set(), cfg=cfg, now=50.0, price=eng, max_n=3),
            [],
        )
        short_crowd = CoinVote("ETH", "short", 1, 10, 20, 0.5, 0.1, 10, 1.0)
        out = t.apply([short_crowd], managed=set(), cfg=cfg, now=50.0, price=eng, max_n=3)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].side, "long")

    def test_logged_trade_replaced_by_mtf_meta(self) -> None:
        from pmf.bt_replay import STRATEGY_REPLAYERS
        from pmf.bt_tune import LIVE_STRATEGIES

        self.assertIn("mtf_meta_holders", STRATEGY_REPLAYERS)
        self.assertIn("mtf_meta_holders", LIVE_STRATEGIES)
        self.assertNotIn("logged_trade", STRATEGY_REPLAYERS)

    def test_meta_timing_strategies_are_mtf_and_swing_only(self) -> None:
        from pmf.bt_replay import STRATEGY_REPLAYERS
        from pmf.bt_tune import META_TIMING_STRATEGIES

        self.assertEqual(
            list(META_TIMING_STRATEGIES),
            [
                "mtf_meta_holders",
                "mtf_meta_all",
                "swing_meta_holders",
                "swing_meta_all",
            ],
        )
        for name in META_TIMING_STRATEGIES:
            self.assertIn(name, STRATEGY_REPLAYERS)
        # Must not silently include crowd gates / refine / flow.
        for name in META_TIMING_STRATEGIES:
            self.assertTrue(name.startswith("mtf_") or name.startswith("swing_"))
        from pmf.strategy_spec import parse_strategy

        self.assertEqual(parse_strategy("mtf_meta_all").filter_mode, "off")
        self.assertEqual(parse_strategy("swing_meta_all").wallet_mode, "all")
        self.assertEqual(parse_strategy("mtf_meta_holders").filter_mode, "holder")

    def test_direct_all_replaced_by_swing_meta(self) -> None:
        from pmf.bt_replay import STRATEGY_REPLAYERS
        from pmf.bt_tune import LIVE_STRATEGIES
        from pmf.strategy_spec import parse_strategy

        self.assertIn("swing_meta_holders", STRATEGY_REPLAYERS)
        self.assertIn("swing_meta_holders", LIVE_STRATEGIES)
        self.assertNotIn("direct_all", STRATEGY_REPLAYERS)
        # Live must never keep cloud_holders out of the roster.
        self.assertIn("cloud_holders", STRATEGY_REPLAYERS)
        spec = parse_strategy("swing_meta_holders")
        self.assertEqual(spec.style, "swing_meta")
        self.assertIsNone(spec.gate)
        self.assertTrue(spec.needs_candles)
        self.assertEqual(spec.filter_mode, "holder")
        self.assertEqual(spec.candle_intervals, ("1m", "15m", "1h"))

    def _swing_cfg(self, **over):
        from types import SimpleNamespace

        base = dict(
            SWING_META_MODE="follow",
            SWING_ENTRY="rsi_dip",
            SWING_TF="15m",
            SWING_RSI_BUY=35.0,
            SWING_RSI_SELL=65.0,
            SWING_BAND_PCT=0.008,
            SWING_BREAK_PCT=0.01,
            SWING_LOOKBACK_S=1800.0,
            SWING_TP_PCT=1.0,
            SWING_SL_PCT=2.0,
            SWING_MAX_HOLD_S=14400.0,
            SWING_EXIT_RSI=0.0,
        )
        base.update(over)
        return SimpleNamespace(**base)

    def test_swing_meta_entry_exit_round_trip(self) -> None:
        from pmf.price_engine import PriceEngine
        from pmf.swing_exec import SwingTrader
        from pmf.types import CoinVote

        eng = PriceEngine()
        eng.ingest_mark("ETH", 100.0, 100.0)
        v = CoinVote("ETH", "long", 10, 1, 20, 0.5, 0.1, 10, 1.0)

        class Dipped(SwingTrader):
            def entry_ok(self, coin, want, asof, price, sc):
                return want > 0

        t = Dipped()
        cfg = self._swing_cfg()
        out = t.apply([v], managed=set(), cfg=cfg, now=100.0, price=eng, max_n=3)
        self.assertEqual([o.side for o in out], ["long"])
        self.assertIn("ETH", t.positions)

        # +1.5% > take-profit 1.0% → closed, and the cooldown blocks same-tick re-entry.
        eng.ingest_mark("ETH", 160.0, 101.5)
        out2 = t.apply([v], managed={"ETH"}, cfg=cfg, now=160.0, price=eng, max_n=3)
        self.assertEqual(out2, [])
        self.assertNotIn("ETH", t.positions)

        # Still cooling down 10 min later (SWING_REENTRY_S default 900s), open after.
        eng.ingest_mark("ETH", 760.0, 101.5)
        self.assertEqual(t.apply([v], managed=set(), cfg=cfg, now=760.0, price=eng, max_n=3), [])
        eng.ingest_mark("ETH", 1100.0, 101.5)
        self.assertEqual(
            len(t.apply([v], managed=set(), cfg=cfg, now=1100.0, price=eng, max_n=3)), 1
        )

    def test_swing_meta_stop_loss_and_max_hold(self) -> None:
        from pmf.price_engine import PriceEngine
        from pmf.swing_exec import SwingTrader
        from pmf.types import CoinVote

        eng = PriceEngine()
        eng.ingest_mark("ETH", 100.0, 100.0)
        v = CoinVote("ETH", "long", 10, 1, 20, 0.5, 0.1, 10, 1.0)

        class Always(SwingTrader):
            def entry_ok(self, coin, want, asof, price, sc):
                return True

        t = Always()
        cfg = self._swing_cfg(SWING_TP_PCT=5.0, SWING_SL_PCT=2.0, SWING_MAX_HOLD_S=300.0)
        t.apply([v], managed=set(), cfg=cfg, now=100.0, price=eng, max_n=3)
        eng.ingest_mark("ETH", 150.0, 97.0)  # −3% → stop out
        self.assertEqual(t.apply([v], managed={"ETH"}, cfg=cfg, now=150.0, price=eng, max_n=3), [])

        t2 = Always()
        eng2 = PriceEngine()
        eng2.ingest_mark("ETH", 100.0, 100.0)
        t2.apply([v], managed=set(), cfg=cfg, now=100.0, price=eng2, max_n=3)
        eng2.ingest_mark("ETH", 401.0, 100.2)  # flat but past 300s max hold
        self.assertEqual(t2.apply([v], managed={"ETH"}, cfg=cfg, now=401.0, price=eng2, max_n=3), [])

    def test_swing_meta_reverse_fades_crowd(self) -> None:
        from pmf.price_engine import PriceEngine
        from pmf.swing_exec import SwingTrader
        from pmf.types import CoinVote

        eng = PriceEngine()
        eng.ingest_mark("ETH", 100.0, 100.0)
        v = CoinVote("ETH", "long", 10, 1, 20, 0.5, 0.1, 10, 1.0)

        class Always(SwingTrader):
            def entry_ok(self, coin, want, asof, price, sc):
                return True

        t = Always()
        out = t.apply(
            [v],
            managed=set(),
            cfg=self._swing_cfg(SWING_META_MODE="reverse"),
            now=100.0,
            price=eng,
            max_n=3,
        )
        self.assertEqual([o.side for o in out], ["short"])

    def test_swing_meta_needs_price_engine(self) -> None:
        from pmf.swing_exec import SwingTrader
        from pmf.types import CoinVote

        v = CoinVote("ETH", "long", 10, 1, 20, 0.5, 0.1, 10, 1.0)
        t = SwingTrader()
        self.assertEqual(
            t.apply([v], managed=set(), cfg=self._swing_cfg(), now=1.0, price=None, max_n=3), []
        )

    def test_swing_state_round_trips_through_store_dump(self) -> None:
        from pmf.swing_exec import SwingTrader, _SwingPos

        t = SwingTrader()
        t.positions["ETH"] = _SwingPos(side="short", entry_px=123.5, entry_ts=42.0)
        back = SwingTrader.from_dump(t.dump())
        self.assertEqual(back.positions["ETH"].side, "short")
        self.assertAlmostEqual(back.positions["ETH"].entry_px, 123.5)
        self.assertAlmostEqual(back.positions["ETH"].entry_ts, 42.0)

    def test_apply_sets_swing_defaults_and_candle_seed(self) -> None:
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock

        import apply_cloud_tune as ac

        payload = {
            "strategy": "swing_meta_holders",
            "params": {"MIN_SIDE_AGREEMENT": 0.1, "SWING_ENTRY": "breakout"},
            "metrics": {"score": 1.0, "return_pct": 2.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "cloud_tuned.json"
            with mock.patch.object(ac, "CLOUD_TUNED_PATH", dest):
                ac.apply_from_payload(payload, write_profiles=False)
            out = json.loads(dest.read_text(encoding="utf-8"))["params"]
        self.assertEqual(out["BACKTEST_LIVE_STRATEGY"], "swing_meta_holders")
        self.assertEqual(out["BASKET_FILTER_MODE"], "holder")
        self.assertTrue(out["LIVE_CANDLE_SEED"])
        self.assertEqual(out["SWING_ENTRY"], "breakout")  # winner preserved
        self.assertEqual(out["SWING_TF"], "15m")  # default filled
        self.assertGreaterEqual(out["LIVE_CANDLE_BARS_1M"], 160)
        for key in ("SWING_TP_PCT", "SWING_SL_PCT", "SWING_MAX_HOLD_S", "SWING_META_MODE"):
            self.assertIn(key, out)

    def test_apply_reverse_flips_modes(self) -> None:
        from apply_cloud_tune import reverse_winner_params

        swing = reverse_winner_params(
            {"SWING_META_MODE": "reverse", "SWING_ENTRY": "ema_pullback"},
            strategy="swing_meta_all",
        )
        self.assertEqual(swing["SWING_META_MODE"], "follow")
        self.assertEqual(swing["SWING_ENTRY"], "ema_pullback")

        crowd = reverse_winner_params({"MIN_SIDE_AGREEMENT": 0.1}, strategy="crowd_vol_all")
        self.assertEqual(crowd["TRADE_MODE"], "reverse")

        mtf = reverse_winner_params(
            {"MTF_META_MODE": "follow", "MTF_PRESET": "ema_x_long"},
            strategy="mtf_meta_holders",
        )
        self.assertEqual(mtf["MTF_META_MODE"], "reverse")
        self.assertEqual(mtf["MTF_PRESET"], "ema_x_short")

    def test_coarse_grid_is_wide_and_deep_grid_focuses(self) -> None:
        from pmf.bt_tune import (
            DEFAULT_GRID,
            _grid_for_strategy,
            coarse_grid,
            deep_grid_around,
            top_strategies,
            TuneResult,
        )

        wide = coarse_grid(_grid_for_strategy("crowd_dump_holders", DEFAULT_GRID))
        span_wide = max(wide["DUMP_RET_PCT"]) - min(wide["DUMP_RET_PCT"])
        span_deep = max(DEFAULT_GRID["FLOW_EMA_ALPHA"]) - min(DEFAULT_GRID["FLOW_EMA_ALPHA"])
        self.assertGreater(span_wide, 0.02)
        self.assertGreater(max(wide["FLOW_EMA_ALPHA"]) - min(wide["FLOW_EMA_ALPHA"]), span_deep)

        deep = deep_grid_around(
            _grid_for_strategy("cloud_holders", DEFAULT_GRID), {"FLOW_EMA_ALPHA": 0.28}
        )
        vals = deep["FLOW_EMA_ALPHA"]
        self.assertIn(0.28, vals)
        self.assertGreater(len(vals), len(DEFAULT_GRID["FLOW_EMA_ALPHA"]))
        self.assertLessEqual(max(vals) - min(vals), span_wide + 0.4)

    def test_top_strategies_picks_best_per_strategy(self) -> None:
        from pmf.bt_tune import TuneResult, top_strategies

        def mk(name: str, ret: float, trips: int = 2) -> TuneResult:
            return TuneResult(
                strategy=name,
                params={"X": ret},
                return_pct=ret,
                max_dd_pct=1.0,
                round_trips=trips,
                win_rate_pct=50.0,
                score=ret,
                trades_per_day=1.0,
                active_day_ratio=1.0,
                cluster_share=0.1,
            )

        res = [mk("a", 1.0), mk("a", 5.0), mk("b", 3.0), mk("c", -2.0)]
        top = top_strategies(res, k=2)
        self.assertEqual([name for name, _ in top], ["a", "b"])
        self.assertAlmostEqual(top[0][1].return_pct, 5.0)

        # A strategy that never traded must not steal a deep-dive slot at 0.00%.
        with_inert = [mk("idle", 0.0, trips=0), mk("a", 2.0), mk("b", -1.0)]
        self.assertEqual([n for n, _ in top_strategies(with_inert, k=2)], ["a", "b"])

    def test_ind_panel_matches_price_engine(self) -> None:
        """Panel rsi/ema/atr/ret equal PriceEngine at several synthetic tick timestamps."""
        from types import SimpleNamespace

        import numpy as np

        from pmf.bt_panels import PANEL_LOOKBACKS, PANEL_TFS, PanelPrice, build_ind_panel
        from pmf.price_engine import PriceEngine

        eng = PriceEngine()
        base_ms = 1_700_000_000_000
        # Dense 1m bars + marks so RSI/EMA/ATR/ret have enough history.
        for i in range(80):
            t_ms = base_ms + i * 60_000
            c = 100.0 + (i % 7) * 0.4 - (i % 5) * 0.2
            h = c + 1.5
            l = c - 1.2
            eng.ingest_bar("ETH", "1m", [t_ms, c, h, l, c, 10.0])
            if i % 15 == 0:
                eng.ingest_bar(
                    "ETH",
                    "15m",
                    [t_ms, c, h + 0.5, l - 0.5, c, 50.0],
                )
            if i % 60 == 0:
                eng.ingest_bar("ETH", "1h", [t_ms, c, h + 1.0, l - 1.0, c, 200.0])
            eng.ingest_mark("ETH", t_ms / 1000.0 + 60.0, c)

        # Tick asofs = closed-bar times late in the series.
        tick_ts = np.array(
            [base_ms / 1000.0 + k * 60.0 for k in (40, 50, 60, 70)],
            dtype=np.float64,
        )
        ds = SimpleNamespace(
            ts=tick_ts,
            n_ticks=len(tick_ts),
            n_coins=1,
            index_coin=["ETH"],
            coin_index={"ETH": 0},
            price_engine=eng,
        )
        panel = build_ind_panel(ds, progress=False, eng=eng)
        self.assertIsNotNone(panel)
        assert panel is not None
        wrapped = PanelPrice(eng, panel, ds)

        for i, asof in enumerate(tick_ts):
            wrapped.set_tick(i)
            self.assertAlmostEqual(wrapped.price_at("ETH", float(asof)), eng.price_at("ETH", float(asof)))
            for tf in PANEL_TFS:
                self.assertAlmostEqual(
                    wrapped.rsi("ETH", float(asof), tf=tf),
                    eng.rsi("ETH", float(asof), tf=tf),
                    places=12,
                )
                self.assertAlmostEqual(
                    wrapped.ema_bias("ETH", float(asof), tf=tf),
                    eng.ema_bias("ETH", float(asof), tf=tf),
                    places=12,
                )
            self.assertAlmostEqual(
                wrapped.atr_pct("ETH", float(asof)),
                eng.atr_pct("ETH", float(asof)),
                places=12,
            )
            for lb in PANEL_LOOKBACKS:
                self.assertAlmostEqual(
                    wrapped.ret("ETH", lb, float(asof)),
                    eng.ret("ETH", lb, float(asof)),
                    places=12,
                )
                self.assertAlmostEqual(
                    wrapped.range_dump("ETH", lb, float(asof)),
                    eng.range_dump("ETH", lb, float(asof)),
                    places=12,
                )

        # Off-tick asof must fall back to engine (still identical).
        mid = float(tick_ts[1]) + 0.5
        self.assertAlmostEqual(wrapped.rsi("ETH", mid, tf="15m"), eng.rsi("ETH", mid, tf="15m"), places=12)

    def test_mtf_entry_cache_shared_across_traders(self) -> None:
        from types import SimpleNamespace

        from pmf import mtf_exec
        from pmf.mtf_exec import MtfTrader, resolve_mtf_setup
        from pmf.price_engine import PriceEngine

        eng = PriceEngine()
        base_ms = 1_700_000_000_000
        for i in range(40):
            t_ms = base_ms + i * 60_000
            c = 100.0 + i * 0.1
            eng.ingest_bar("BTC", "1m", [t_ms, c, c + 1, c - 1, c, 5.0])
            eng.ingest_bar("BTC", "15m", [t_ms, c, c + 1, c - 1, c, 5.0])
            eng.ingest_bar("BTC", "1h", [t_ms, c, c + 1, c - 1, c, 5.0])
        asof = base_ms / 1000.0 + 40 * 60
        cfg = SimpleNamespace(
            MTF_PRESET="rsi_long_30",
            MTF_EMA=50,
            MTF_MIN_AGREE=2,
            MTF_MIN_SCORE=0.30,
            MTF_WEIGHT_POWER=0.5,
            MTF_EXIT="none",
            MTF_EXEC_IV="1m",
        )
        setup = resolve_mtf_setup(cfg)
        mtf_exec._ENTRY_CACHE.clear()
        a = MtfTrader()
        b = MtfTrader()
        s1 = a.mtf_entry_side("BTC", asof, eng, setup)
        before = len(mtf_exec._ENTRY_CACHE)
        s2 = b.mtf_entry_side("BTC", asof, eng, setup)
        self.assertEqual(s1, s2)
        self.assertGreaterEqual(before, 1)
        self.assertEqual(len(mtf_exec._ENTRY_CACHE), before)  # second hit, no new key


class PlanTests(unittest.TestCase):
    def test_close_unmanaged_ignored_and_flip_closes_first(self) -> None:
        ours = [
            OurPos("BTC", "long", 1, 1000, 100, 5),
            OurPos("SOL", "long", 10, 500, 50, 5),  # not managed, not a target
        ]
        targets = [TargetPos("BTC", "short", 5, 20.0, -0.1)]
        acts = plan_actions(ours, targets, 5000, _Cfg(), managed={"BTC"})
        kinds = [a.kind for a in acts]
        self.assertIn("close", kinds)
        self.assertIn("open", kinds)
        self.assertTrue(kinds.index("close") < kinds.index("open"))
        self.assertFalse(any(a.coin == "SOL" for a in acts))

    def test_empty_rebalance_returns_two_values(self) -> None:
        import logging

        cfg = _Cfg()
        cfg.REBALANCE_COOLDOWN_S = 0
        reb = Rebalancer(None, None, cfg, logging.getLogger("t"))  # type: ignore[arg-type]
        reb.current_book = lambda: ([], 1000.0)  # type: ignore[method-assign]
        managed, attempted = reb.run([], set(), 0.0, 10_000.0)
        self.assertEqual(managed, set())
        self.assertFalse(attempted)
        managed, attempted = reb.run([], set(), 0.0, 10_000.0)
        equity_zero = Rebalancer(None, None, cfg, logging.getLogger("t"))  # type: ignore[arg-type]
        equity_zero.current_book = lambda: ([], 0.0)  # type: ignore[method-assign]
        managed, attempted = equity_zero.run([], set(), 0.0, 10_000.0)
        self.assertEqual(managed, set())
        self.assertFalse(attempted)


class CopyModeTests(unittest.TestCase):
    def test_copy_filter_rejects_scalper(self) -> None:
        from types import SimpleNamespace

        from pmf.copy_score import FillStats, passes_copy_filters

        cfg = SimpleNamespace(
            COPY_MIN_FILLS=6,
            COPY_MAX_FILLS=120,
            COPY_MIN_MEDIAN_GAP_S=300.0,
            COPY_MAX_MEDIAN_GAP_S=43200.0,
            COPY_MIN_FILLS_PER_DAY=1.5,
            COPY_MAX_FILLS_PER_DAY=18.0,
            COPY_MIN_WIN_RATE=0.52,
            COPY_MIN_HIST_WIN_RATE=0.48,
            COPY_MIN_RECENT_PNL=100.0,
            COPY_MIN_HIST_PNL=200.0,
            COPY_MIN_PROFIT_FACTOR=1.25,
            COPY_MAX_FAST_FLIP_RATIO=0.35,
        )
        recent = FillStats(
            n_fills=20,
            median_gap_s=60.0,
            fills_per_day=10.0,
            win_rate=0.6,
            closed_pnl=100.0,
            wins=6,
            losses=4,
            gross_win=200,
            gross_loss=100,
        )
        hist = FillStats(
            n_fills=40,
            median_gap_s=90.0,
            fills_per_day=8.0,
            win_rate=0.55,
            closed_pnl=200.0,
            wins=20,
            losses=15,
            gross_win=500,
            gross_loss=300,
        )
        ok, why = passes_copy_filters(recent, hist, cfg)
        self.assertFalse(ok)
        self.assertIn("scalpy", why)

    def test_copy_filter_accepts_active_consistent(self) -> None:
        from types import SimpleNamespace

        from pmf.copy_score import FillStats, passes_copy_filters, score_copy_wallet
        from pmf.types import QualifiedWallet

        cfg = SimpleNamespace(
            COPY_MIN_FILLS=6,
            COPY_MAX_FILLS=120,
            COPY_MIN_MEDIAN_GAP_S=300.0,
            COPY_MAX_MEDIAN_GAP_S=43200.0,
            COPY_MIN_FILLS_PER_DAY=1.5,
            COPY_MAX_FILLS_PER_DAY=18.0,
            COPY_MIN_WIN_RATE=0.52,
            COPY_MIN_HIST_WIN_RATE=0.48,
            COPY_MIN_RECENT_PNL=100.0,
            COPY_MIN_HIST_PNL=200.0,
            COPY_MIN_PROFIT_FACTOR=1.25,
            COPY_MAX_FAST_FLIP_RATIO=0.35,
            COPY_IDEAL_GAP_S=2700.0,
        )
        recent = FillStats(
            n_fills=40,
            median_gap_s=2500.0,
            fills_per_day=6.0,
            win_rate=0.60,
            closed_pnl=800.0,
            wins=18,
            losses=12,
            gross_win=2000.0,
            gross_loss=1200.0,
            fast_flips=1,
            round_trips=10,
        )
        hist = FillStats(
            n_fills=100,
            median_gap_s=2600.0,
            fills_per_day=4.0,
            win_rate=0.55,
            closed_pnl=2500.0,
            wins=40,
            losses=30,
            gross_win=5000.0,
            gross_loss=2500.0,
            fast_flips=2,
            round_trips=20,
        )
        ok, why = passes_copy_filters(recent, hist, cfg)
        self.assertTrue(ok, why)
        w = QualifiedWallet("0xabc", 10_000, 1000, 0.2, 0, 0, 0)
        self.assertGreater(score_copy_wallet(w, recent, hist, cfg), 50.0)

    def test_copy_reverse_flips_side(self) -> None:
        from types import SimpleNamespace

        from pmf.copy_exec import copy_targets_from_leaders
        from pmf.copy_score import CopyLeader, FillStats
        from pmf.types import WalletPos, WalletSnapshot

        cfg = SimpleNamespace(
            STALE_SNAPSHOT_S=9999.0,
            OUR_GROSS_MARGIN_PCT=90.0,
            MAX_MARGIN_PER_COIN_PCT=33.33,
            COPY_MARGIN_CAP_PCT=100.0,
            COPY_MAX_POSITIONS=5,
            OUR_MIN_LEVERAGE=2,
            OUR_MAX_LEVERAGE=20,
            DEX_SCOPE="include",
            ALLOW_COINS=(),
            DENY_COINS=(),
        )
        ld = CopyLeader(
            address="0xabc",
            score=10.0,
            rank_roi=0.2,
            rank_pnl=1000.0,
            account_value=10_000.0,
            recent=FillStats(win_rate=0.6),
        )
        snap = WalletSnapshot(
            address="0xabc",
            account_value=10_000.0,
            positions=[
                WalletPos(
                    coin="BTC",
                    side="long",
                    size=1.0,
                    notional=3000.0,
                    entry_px=100.0,
                    leverage=10,
                    isolated=False,
                    conviction=0.3,
                )
            ],
            fetched_at=1000.0,
            fingerprint="x",
        )
        follow = copy_targets_from_leaders([ld], [snap], cfg, now=1001.0, reverse=False)
        reverse = copy_targets_from_leaders([ld], [snap], cfg, now=1001.0, reverse=True)
        self.assertEqual(follow[0].side, "long")
        self.assertEqual(reverse[0].side, "short")
        self.assertEqual(follow[0].coin, reverse[0].coin)

    def test_copy_scan_cfg_shortlist(self) -> None:
        from pmf.qualify import shortlist
        from pmf.types import LeaderboardRow, WindowPerf

        base = _Cfg()
        base.CANDIDATE_POOL = 10
        base.RANK_WINDOW = "week"
        base.BASKET_FILTER_MODE = "off"
        scan_n = 25

        class _ScanCfg:
            def __getattr__(self, name: str):
                if name == "CANDIDATE_POOL":
                    return scan_n
                if name == "RESEARCH_DATA_ENABLED":
                    return False
                return getattr(base, name)

        rows = [
            LeaderboardRow(
                f"0x{i:040x}",
                10_000.0,
                None,
                {"week": WindowPerf(1000, 0.1 + i * 0.01, 50_000)},
            )
            for i in range(30)
        ]
        out = shortlist(rows, _ScanCfg())
        self.assertEqual(len(out), scan_n)

        from types import SimpleNamespace

        from pmf.copy_exec import copy_targets_from_leaders
        from pmf.copy_score import CopyLeader, FillStats
        from pmf.types import WalletPos, WalletSnapshot

        cfg = SimpleNamespace(
            STALE_SNAPSHOT_S=9999.0,
            OUR_GROSS_MARGIN_PCT=90.0,
            MAX_MARGIN_PER_COIN_PCT=33.33,
            COPY_MARGIN_CAP_PCT=100.0,
            COPY_MAX_POSITIONS=3,
            OUR_MIN_LEVERAGE=2,
            OUR_MAX_LEVERAGE=20,
            DEX_SCOPE="include",
            ALLOW_COINS=(),
            DENY_COINS=(),
        )
        ld = CopyLeader(
            address="0xabc",
            score=10.0,
            rank_roi=0.2,
            rank_pnl=1000.0,
            account_value=10_000.0,
            recent=FillStats(win_rate=0.6),
        )
        snap = WalletSnapshot(
            address="0xabc",
            account_value=10_000.0,
            positions=[
                WalletPos(
                    coin="BTC",
                    side="long",
                    size=1.0,
                    notional=3000.0,
                    entry_px=100.0,
                    leverage=10,
                    isolated=False,
                    conviction=0.3,
                )
            ],
            fetched_at=1000.0,
            fingerprint="x",
        )
        targets = copy_targets_from_leaders([ld], [snap], cfg, now=1001.0)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].coin, "BTC")
        self.assertEqual(targets[0].side, "long")


if __name__ == "__main__":
    unittest.main()
