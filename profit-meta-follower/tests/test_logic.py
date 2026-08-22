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

    def test_hostile_funding_blocks_long(self) -> None:
        cfg = _Cfg()
        eng = BookEngine()
        eng.ok_since = {"BTC|long": 0.0}
        mk = {"BTC": MarketCtx("BTC", 5_000_000, 0.002, 1.0, 0.0)}
        out = eng.refine([self._btc_vote()], markets=mk, managed=set(), cfg=cfg, now=5000.0)
        self.assertEqual(out, [])


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


if __name__ == "__main__":
    unittest.main()
