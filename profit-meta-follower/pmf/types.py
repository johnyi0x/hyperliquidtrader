from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def coin_key(coin: str, dex: str | None = None) -> str:
    raw = str(coin or "").strip()
    if ":" in raw:
        left, right = raw.split(":", 1)
        dex_s = left.strip()
        sym = right.strip()
        return f"{dex_s}:{sym}" if dex_s else sym
    dex_s = (dex or "").strip()
    return f"{dex_s}:{raw}" if dex_s else raw


def split_coin(api_coin: str) -> tuple[str, str | None]:
    raw = str(api_coin or "").strip()
    if ":" in raw:
        left, right = raw.split(":", 1)
        return right.strip(), left.strip() or None
    return raw, None


def dex_of(api_coin: str) -> str:
    _sym, dex = split_coin(api_coin)
    return dex or ""


@dataclass
class WindowPerf:
    pnl: float = 0.0
    roi: float = 0.0
    volume: float = 0.0


@dataclass
class LeaderboardRow:
    address: str
    account_value: float
    display_name: str | None
    windows: dict[str, WindowPerf] = field(default_factory=dict)


@dataclass
class WalletPos:
    coin: str
    side: str  # long | short
    size: float
    notional: float
    entry_px: float | None
    leverage: int
    isolated: bool
    conviction: float  # signed notional / wallet equity


@dataclass
class WalletSnapshot:
    address: str
    account_value: float
    positions: list[WalletPos]
    fetched_at: float
    fingerprint: str
    dexes_ok: bool = True
    error: str = ""


@dataclass
class QualifiedWallet:
    address: str
    account_value: float
    rank_pnl: float
    rank_roi: float
    rank_volume: float
    confirm_pnl: float
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class CoinVote:
    coin: str
    side: str
    wallets_long: int
    wallets_short: int
    voters: int
    agreement: float
    avg_conviction: float  # signed, equal-wallet mean
    median_leverage: int
    score: float
    flow: float = 0.0  # change in smoothed conviction vs prior cycle
    raw_flow: float = 0.0  # change in raw basket conviction (fast exit lane)
    persist_s: float = 0.0
    ema: float = 0.0
    raw_conviction: float = 0.0  # unsmoothed equal-wallet mean before EMA
    mean_leverage: float = 0.0
    avg_margin_pct: float = 0.0  # equal-wallet mean of (notional / equity / lev) * 100, winning side only


@dataclass
class MarketCtx:
    coin: str
    day_volume: float
    funding: float
    open_interest: float
    basis: float  # (mark - oracle) / oracle



@dataclass
class TargetPos:
    coin: str
    side: str
    leverage: int
    margin_pct: float  # percent of OUR equity to allocate as margin
    conviction: float


@dataclass
class OurPos:
    coin: str
    side: str
    size: float
    notional: float
    entry_px: float | None
    leverage: int


def fnum(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default
