"""
EMA race: watch the 24h gainer/loser list, score who has stretched
farthest from EMA vs the rest of the basket, trade that one pair.

TP/SL are 1:1. Size + stop are set so a fill is ~EMA_RACE_EQUITY_RISK_PCT
of equity after leverage (default 10%). Paper and live both use market orders.

Learning: after every close, Laplace win-rates by (follow|fade, bucket, above|below)
and a per-coin multiplier. Next pick uses those weights so the same losing
context is less likely to win the race.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ema_dev import EmaDevSnap, signed_dev_pct


def ctx_key(policy: str, bucket: str, ema_side: str) -> str:
    b = str(bucket or "na").lower() or "na"
    if b not in ("gainer", "loser"):
        b = "na"
    p = "fade" if str(policy).lower() == "fade" else "follow"
    s = "below" if str(ema_side).lower() == "below" else "above"
    return f"{p}|{b}|{s}"


def policy_side(policy: str, signed_dev_pct: float) -> str:
    """follow = long strength / short weakness. fade = opposite."""
    above = float(signed_dev_pct) >= 0.0
    follow = str(policy).lower() != "fade"
    if follow:
        return "long" if above else "short"
    return "short" if above else "long"


def ema_side_of(signed_dev_pct: float) -> str:
    return "above" if float(signed_dev_pct) >= 0.0 else "below"


def tp_sl_price_pct(leverage: int, risk_pct: float, min_pct: float) -> float:
    """Price % for both TP and SL so leverage * this ≈ account risk %."""
    lev = max(1, int(leverage))
    raw = float(risk_pct) / float(lev)
    return max(float(min_pct), raw)


def margin_pct_for_risk(
    *,
    leverage: int,
    tp_sl_pct: float,
    risk_pct: float,
    cap_pct: float,
) -> float:
    """Equity-% to send to estimate_order_size so SL ≈ risk_pct of equity."""
    lev = max(1, int(leverage))
    sl = max(1e-6, float(tp_sl_pct))
    # notional / equity = risk_pct / sl_pct ; margin/equity = that / lev
    pct = (float(risk_pct) / sl) / lev * 100.0
    return min(float(cap_pct), max(1.0, pct))


@dataclass(frozen=True)
class RaceCandidate:
    coin: str
    close: float
    ema: float
    abs_dev_pct: float
    signed_dev_pct: float
    bar_t: int
    cross_bars: int
    z: float
    rel: float
    rank: int
    policy: str
    side: str
    ctx: str
    score: float
    bucket: str
    weight: float
    coin_mult: float


@dataclass
class RaceTrade:
    coin: str
    side: str
    policy: str
    ctx: str
    bucket: str
    abs_dev_pct: float
    signed_dev_pct: float
    entry_px: float
    entry_ema: float
    tp_sl_pct: float
    leverage: int
    opened_bar_t: int = 0
    opened_at: float = 0.0
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    last_exit_coin: str = ""
    last_exit_bar_t: int = 0
    last_exit_at: float = 0.0
    last_exit_win: bool = True
    entry_ctx: dict = field(default_factory=dict)


@dataclass
class _Bin:
    n: int = 0
    wins: int = 0
    pnl_usd: float = 0.0

    def weight(self) -> float:
        return (self.wins + 1.0) / (self.n + 2.0)

    def as_dict(self) -> dict:
        return {"n": self.n, "wins": self.wins, "pnl_usd": round(self.pnl_usd, 4)}


class RaceLearner:
    """Persisted pick weights. Safe to load on a new box with empty state."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.bins: dict[str, _Bin] = {}
        self.coins: dict[str, dict] = {}
        self.trades: int = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        for key, blob in (raw.get("bins") or {}).items():
            if not isinstance(blob, dict):
                continue
            try:
                self.bins[str(key)] = _Bin(
                    n=int(blob.get("n", 0) or 0),
                    wins=int(blob.get("wins", 0) or 0),
                    pnl_usd=float(blob.get("pnl_usd", 0) or 0),
                )
            except (TypeError, ValueError):
                continue
        for coin, blob in (raw.get("coins") or {}).items():
            if not isinstance(blob, dict):
                continue
            try:
                self.coins[str(coin)] = {
                    "n": int(blob.get("n", 0) or 0),
                    "wins": int(blob.get("wins", 0) or 0),
                    "pnl_usd": float(blob.get("pnl_usd", 0) or 0),
                    "mult": float(blob.get("mult", 1.0) or 1.0),
                }
            except (TypeError, ValueError):
                continue
        try:
            self.trades = int(raw.get("trades", 0) or 0)
        except (TypeError, ValueError):
            self.trades = 0

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trades": self.trades,
            "updated_ts": time.time(),
            "bins": {k: v.as_dict() for k, v in sorted(self.bins.items())},
            "coins": self.coins,
        }
        tmp = self.path.with_suffix(".tmp")
        last_err: OSError | None = None
        for attempt in range(5):
            try:
                tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                os.replace(tmp, self.path)
                return
            except OSError as exc:
                last_err = exc
                time.sleep(0.05 * (attempt + 1))
        if last_err is not None:
            raise last_err

    def weight(self, ctx: str) -> float:
        return self.bins.setdefault(ctx, _Bin()).weight()

    def coin_mult(self, coin: str) -> float:
        row = self.coins.get(coin) or {}
        try:
            m = float(row.get("mult", 1.0) or 1.0)
        except (TypeError, ValueError):
            m = 1.0
        return min(1.6, max(0.35, m))

    def observe(self, *, ctx: str, coin: str, win: bool, pnl_usd: float) -> None:
        b = self.bins.setdefault(ctx, _Bin())
        b.n += 1
        if win:
            b.wins += 1
        b.pnl_usd += float(pnl_usd)
        row = self.coins.setdefault(
            coin, {"n": 0, "wins": 0, "pnl_usd": 0.0, "mult": 1.0}
        )
        row["n"] = int(row.get("n", 0) or 0) + 1
        if win:
            row["wins"] = int(row.get("wins", 0) or 0) + 1
        row["pnl_usd"] = float(row.get("pnl_usd", 0) or 0) + float(pnl_usd)
        m = float(row.get("mult", 1.0) or 1.0)
        row["mult"] = min(1.6, max(0.35, m * (1.12 if win else 0.82)))
        self.trades += 1
        self._save()

    def board(self) -> list[dict]:
        rows = []
        for key, b in sorted(self.bins.items(), key=lambda kv: (-kv[1].n, kv[0])):
            if b.n <= 0:
                continue
            rows.append(
                {
                    "ctx": key,
                    "n": b.n,
                    "wins": b.wins,
                    "wr": round(b.wins / b.n * 100.0, 1),
                    "pnl": round(b.pnl_usd, 2),
                    "w": round(b.weight(), 3),
                }
            )
        return rows

    def best_ctx(self) -> str:
        scored = [(b.n, b.weight(), k) for k, b in self.bins.items() if b.n >= 2]
        if not scored:
            return "follow|na|above"
        scored.sort(key=lambda t: (-t[1], -t[0], t[2]))
        return scored[0][2]


def _mean_std(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n <= 0:
        return 0.0, 0.01
    mean = sum(vals) / n
    if n == 1:
        return mean, 0.01
    var = sum((v - mean) ** 2 for v in vals) / n
    return mean, max(0.01, math.sqrt(var))


def race_candidates(
    snaps: list[EmaDevSnap],
    *,
    buckets: dict[str, str],
    learner: RaceLearner,
    min_dev_pct: float,
    skip_coin: str | None = None,
) -> list[RaceCandidate]:
    """One candidate per (coin, policy). Ranked later by score."""
    usable = [
        s
        for s in snaps
        if s.abs_dev_pct + 1e-12 >= float(min_dev_pct)
        and (not skip_coin or s.coin != skip_coin)
    ]
    if not usable:
        return []
    abs_vals = [float(s.abs_dev_pct) for s in usable]
    mean_abs, std_abs = _mean_std(abs_vals)
    by_abs = sorted(usable, key=lambda s: (-s.abs_dev_pct, s.coin))
    rank_of = {s.coin: i + 1 for i, s in enumerate(by_abs)}
    out: list[RaceCandidate] = []
    for snap in usable:
        z = (float(snap.abs_dev_pct) - mean_abs) / std_abs
        rel = float(snap.abs_dev_pct) / max(mean_abs, 0.2)
        ema_s = ema_side_of(snap.signed_dev_pct)
        bucket = str(buckets.get(snap.coin) or "na")
        rank = int(rank_of.get(snap.coin, 99))
        cm = learner.coin_mult(snap.coin)
        for policy in ("follow", "fade"):
            ctx = ctx_key(policy, bucket, ema_s)
            w = learner.weight(ctx)
            # Farthest vs basket (rel, z) * learned context * coin memory.
            score = rel * (0.55 + 0.9 * z if z > 0 else max(0.15, 0.55 + 0.4 * z))
            score *= w * cm
            if policy == "follow":
                score += 1e-6  # tie-break: follow when learning is still flat
            out.append(
                RaceCandidate(
                    coin=snap.coin,
                    close=float(snap.close),
                    ema=float(snap.ema),
                    abs_dev_pct=float(snap.abs_dev_pct),
                    signed_dev_pct=float(snap.signed_dev_pct),
                    bar_t=int(snap.bar_t),
                    cross_bars=int(snap.cross_bars),
                    z=round(z, 3),
                    rel=round(rel, 3),
                    rank=rank,
                    policy=policy,
                    side=policy_side(policy, snap.signed_dev_pct),
                    ctx=ctx,
                    score=float(score),
                    bucket=bucket if bucket in ("gainer", "loser") else "na",
                    weight=round(w, 3),
                    coin_mult=round(cm, 3),
                )
            )
    out.sort(key=lambda c: (-c.score, c.coin, c.policy))
    return out


def race_entry_reject(
    cand: RaceCandidate,
    *,
    tp_sl_pct: float,
    min_d_to_sl: float = 1.2,
    max_follow_dev_pct: float = 9.0,
) -> str | None:
    """Skip setups the paper+live log stopped out. None = ok."""
    d = float(cand.abs_dev_pct)
    sl = max(0.05, float(tp_sl_pct))
    if cand.policy == "follow" and cand.bucket == "loser" and cand.side == "long":
        return "loser_long"
    if cand.policy == "follow" and cand.bucket == "gainer" and cand.side == "short":
        return "gainer_short"
    if d + 1e-12 < float(min_d_to_sl) * sl:
        return "d<sl"
    if cand.policy == "follow" and d > float(max_follow_dev_pct) + 1e-12:
        return "d_ext"
    return None


def pick_race(cands: list[RaceCandidate]) -> RaceCandidate | None:
    return cands[0] if cands else None


def board_rows(cands: list[RaceCandidate], *, n: int = 8) -> list[dict]:
    """Best policy per coin, farthest first — for logs."""
    best: dict[str, RaceCandidate] = {}
    for c in cands:
        prev = best.get(c.coin)
        if prev is None or c.score > prev.score + 1e-12:
            best[c.coin] = c
    ranked = sorted(best.values(), key=lambda c: (-c.abs_dev_pct, c.coin))
    rows = []
    for c in ranked[: max(1, n)]:
        rows.append(
            {
                "coin": c.coin,
                "d": round(c.signed_dev_pct, 3),
                "abs": round(c.abs_dev_pct, 3),
                "z": c.z,
                "rel": c.rel,
                "rank": c.rank,
                "pol": c.policy,
                "side": c.side,
                "ctx": c.ctx,
                "sc": round(c.score, 3),
                "w": c.weight,
                "cm": c.coin_mult,
                "bkt": c.bucket,
                "xb": c.cross_bars,
            }
        )
    return rows


class RaceStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.trade: RaceTrade | None = None
        self._load()

    def _load(self) -> None:
        self.trade = None
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        try:
            self.trade = RaceTrade(
                coin=str(raw.get("coin", "") or ""),
                side=str(raw.get("side", "") or ""),
                policy=str(raw.get("policy", "") or ""),
                ctx=str(raw.get("ctx", "") or ""),
                bucket=str(raw.get("bucket", "") or ""),
                abs_dev_pct=float(raw.get("abs_dev_pct", 0) or 0),
                signed_dev_pct=float(raw.get("signed_dev_pct", 0) or 0),
                entry_px=float(raw.get("entry_px", 0) or 0),
                entry_ema=float(raw.get("entry_ema", 0) or 0),
                tp_sl_pct=float(raw.get("tp_sl_pct", 0) or 0),
                leverage=int(raw.get("leverage", 0) or 0),
                opened_bar_t=int(raw.get("opened_bar_t", 0) or 0),
                opened_at=float(raw.get("opened_at", 0) or 0),
                mfe_pct=float(raw.get("mfe_pct", 0) or 0),
                mae_pct=float(raw.get("mae_pct", 0) or 0),
                last_exit_coin=str(raw.get("last_exit_coin", "") or ""),
                last_exit_bar_t=int(raw.get("last_exit_bar_t", 0) or 0),
                last_exit_at=float(raw.get("last_exit_at", 0) or 0),
                last_exit_win=bool(raw.get("last_exit_win", True)),
                entry_ctx=dict(raw["entry_ctx"])
                if isinstance(raw.get("entry_ctx"), dict)
                else {},
            )
        except (TypeError, ValueError):
            self.trade = None

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.trade is None:
            if self.path.exists():
                try:
                    self.path.unlink()
                except OSError:
                    pass
            return
        tmp = self.path.with_suffix(".tmp")
        last_err: OSError | None = None
        for attempt in range(5):
            try:
                tmp.write_text(json.dumps(asdict(self.trade), indent=2), encoding="utf-8")
                os.replace(tmp, self.path)
                return
            except OSError as exc:
                last_err = exc
                time.sleep(0.05 * (attempt + 1))
        if last_err is not None:
            raise last_err

    def open_trade(self, trade: RaceTrade) -> RaceTrade:
        prev = self.trade
        if prev is not None:
            trade.last_exit_coin = prev.last_exit_coin
            trade.last_exit_bar_t = prev.last_exit_bar_t
            trade.last_exit_at = prev.last_exit_at
            trade.last_exit_win = bool(prev.last_exit_win)
        self.trade = trade
        self._save()
        return trade

    def close(self, *, coin: str, bar_t: int, win: bool = True) -> None:
        exit_coin = coin
        exit_bar = int(bar_t or 0)
        prev_at = 0.0
        if self.trade is not None:
            exit_coin = self.trade.coin or coin
            if exit_bar <= 0:
                exit_bar = self.trade.opened_bar_t
            prev_at = float(self.trade.last_exit_at or 0)
        self.trade = RaceTrade(
            coin="",
            side="",
            policy="",
            ctx="",
            bucket="",
            abs_dev_pct=0.0,
            signed_dev_pct=0.0,
            entry_px=0.0,
            entry_ema=0.0,
            tp_sl_pct=0.0,
            leverage=0,
            last_exit_coin=exit_coin,
            last_exit_bar_t=exit_bar,
            last_exit_at=time.time() if exit_coin else prev_at,
            last_exit_win=bool(win),
        )
        self._save()

    def active(self) -> RaceTrade | None:
        t = self.trade
        if t is None or not t.coin:
            return None
        return t

    def save(self) -> None:
        self._save()
