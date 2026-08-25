"""Average-conviction book, then flow / persist / market gates."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from .types import CoinVote, MarketCtx, TargetPos, WalletSnapshot, dex_of


def basket_size(cfg: Any, listed: int | None = None) -> int:
    if listed and listed > 0:
        return int(listed)
    return max(1, int(getattr(cfg, "BASKET_SIZE", 0) or 0) or int(getattr(cfg, "MIN_LIVE_VOTERS", 1) or 1))


def cloud_crowd_listed(cfg: Any, *, tracker_n: int | None = None, dataset_target: int | None = None) -> int:
    """Denominator for agreement / min-voters — must match live len(tracker.addrs)."""
    if tracker_n and tracker_n > 0:
        return int(tracker_n)
    if dataset_target and dataset_target > 0:
        return int(dataset_target)
    return basket_size(cfg)


def min_live_voters(cfg: Any, listed: int | None = None) -> int:
    pct = float(getattr(cfg, "MIN_LIVE_VOTERS_PCT", 0) or 0)
    if pct > 0:
        return max(3, int(math.ceil(basket_size(cfg, listed) * pct)))
    return max(1, int(getattr(cfg, "MIN_LIVE_VOTERS", 1) or 1))


def min_wallets_on_coin(cfg: Any, *, other: bool = False, listed: int | None = None) -> int:
    pct_attr = "MIN_WALLETS_OTHER_PCT" if other else "MIN_WALLETS_ON_COIN_PCT"
    pct = float(getattr(cfg, pct_attr, 0) or 0)
    if other and pct <= 0:
        pct = float(getattr(cfg, "MIN_WALLETS_ON_COIN_PCT", 0) or 0)
    if pct > 0:
        return max(2, int(math.ceil(basket_size(cfg, listed) * pct)))
    if other:
        return max(2, int(getattr(cfg, "MIN_WALLETS_OTHER", 0) or getattr(cfg, "MIN_WALLETS_ON_COIN", 2) or 2))
    return max(2, int(getattr(cfg, "MIN_WALLETS_ON_COIN", 2) or 2))


def agreement_floor(cfg: Any, live_n: int, listed: int | None = None) -> int:
    """Crowd size we compare against: actual wallet list when known."""
    n = int(listed or 0) or int(getattr(cfg, "BASKET_SIZE", 0) or 0)
    return n if n > 0 else max(1, live_n)


def in_scope(coin: str, cfg: Any) -> bool:
    scope = str(cfg.DEX_SCOPE).strip().lower()
    dex = dex_of(coin)
    if scope == "native" and dex:
        return False
    if scope == "xyz_only" and not dex:
        return False
    allow = tuple(str(x).strip() for x in (cfg.ALLOW_COINS or ()) if str(x).strip())
    deny = {str(x).strip().upper() for x in (cfg.DENY_COINS or ()) if str(x).strip()}
    if coin.upper() in deny:
        return False
    if allow and coin not in allow:
        return False
    return True


def crashed_wallets(
    snapshots: list[WalletSnapshot],
    baseline: dict[str, float],
    cfg: Any,
) -> set[str]:
    """Mute only if THIS snapshot source fell vs the last snapshot of the same wallet."""
    drop = float(getattr(cfg, "MAX_LIVE_EQUITY_DROP", 0.40) or 0.40)
    out: set[str] = set()
    if drop <= 0:
        return out
    for s in snapshots:
        base = float(baseline.get(s.address) or 0.0)
        if base <= 0 or s.account_value <= 0:
            continue
        # Ignore tiny books / missing margin summaries — not a real blow-up.
        if base < 500 or s.account_value < 500:
            continue
        if s.account_value < base * (1.0 - drop):
            out.add(s.address)
    return out


def build_votes(
    snapshots: list[WalletSnapshot],
    hyper: set[str],
    cfg: Any,
    now: float,
    *,
    baseline_equity: dict[str, float] | None = None,
    listed: int | None = None,
) -> list[CoinVote]:
    stale = float(cfg.STALE_SNAPSHOT_S)
    min_conv = float(cfg.MIN_WALLET_CONVICTION)
    crashed = crashed_wallets(snapshots, baseline_equity or {}, cfg)
    live: list[WalletSnapshot] = []
    for s in snapshots:
        if s.address in crashed:
            continue
        if hyper and s.address in hyper:
            continue
        if s.error:
            continue
        if now - s.fetched_at > stale:
            continue
        if s.account_value <= 0 and not s.positions:
            continue
        live.append(s)
    if len(live) < min_live_voters(cfg, listed):
        return []

    n = len(live)
    denom = agreement_floor(cfg, n, listed)
    enter_agr = float(cfg.MIN_SIDE_AGREEMENT)
    exit_agr = float(getattr(cfg, "EXIT_SIDE_AGREEMENT", 0) or 0) or enter_agr
    emit_agr = min(enter_agr, exit_agr)
    long_n: dict[str, int] = defaultdict(int)
    short_n: dict[str, int] = defaultdict(int)
    long_margin: dict[str, list[float]] = defaultdict(list)
    short_margin: dict[str, list[float]] = defaultdict(list)
    long_levs: dict[str, list[int]] = defaultdict(list)
    short_levs: dict[str, list[int]] = defaultdict(list)

    for snap in live:
        used: dict[str, float] = {}
        for pos in snap.positions:
            if not in_scope(pos.coin, cfg):
                continue
            if abs(pos.conviction) < min_conv:
                continue
            used[pos.coin] = pos.conviction
            lev = max(1, pos.leverage)
            margin_pct = abs(pos.conviction) / lev * 100.0
            if pos.conviction > 0:
                long_n[pos.coin] += 1
                long_margin[pos.coin].append(margin_pct)
                long_levs[pos.coin].append(lev)
            else:
                short_n[pos.coin] += 1
                short_margin[pos.coin].append(margin_pct)
                short_levs[pos.coin].append(lev)
        snap._used = used  # type: ignore[attr-defined]

    all_coins = set()
    for snap in live:
        all_coins |= set(getattr(snap, "_used", {}))

    votes: list[CoinVote] = []
    preferred = {
        str(x).strip().upper()
        for x in (getattr(cfg, "PREFERRED_COINS", ()) or ())
        if str(x).strip()
    }
    for coin in all_coins:
        series = [getattr(snap, "_used", {}).get(coin, 0.0) for snap in live]
        avg = sum(series) / n
        if abs(avg) < float(cfg.MIN_AVG_CONVICTION):
            continue
        side = "long" if avg > 0 else "short"
        side_n = long_n[coin] if side == "long" else short_n[coin]
        is_pref = (not preferred) or coin.upper() in preferred
        need_wallets = min_wallets_on_coin(cfg, other=bool(preferred) and not is_pref, listed=listed)
        if side_n < need_wallets:
            continue
        agreement = side_n / denom
        if agreement < emit_agr:
            continue
        side_margins = long_margin[coin] if side == "long" else short_margin[coin]
        side_levs = long_levs[coin] if side == "long" else short_levs[coin]
        if not side_levs:
            continue
        med_lev = int(statistics.median(side_levs))
        mean_lev = float(sum(side_levs) / len(side_levs))
        avg_margin = float(sum(side_margins) / len(side_margins)) if side_margins else 0.0
        score = abs(avg) * agreement * side_n
        if preferred and coin.upper() in preferred:
            score *= 1.8
        votes.append(
            CoinVote(
                coin=coin,
                side=side,
                wallets_long=long_n[coin],
                wallets_short=short_n[coin],
                voters=n,
                agreement=agreement,
                avg_conviction=avg,
                median_leverage=med_lev,
                score=score,
                mean_leverage=mean_lev,
                avg_margin_pct=avg_margin,
            )
        )
    votes.sort(key=lambda v: v.score, reverse=True)
    return votes[: int(cfg.MAX_COINS_IN_BOOK) + 4]


def exec_side(signal: str, cfg: Any) -> str:
    """Side we actually trade. reverse = opposite of the basket."""
    if is_reverse_mode(cfg):
        return "short" if str(signal).lower() == "long" else "long"
    return signal


def is_reverse_mode(cfg: Any) -> bool:
    return str(getattr(cfg, "TRADE_MODE", "follow") or "follow").strip().lower() in (
        "reverse",
        "invert",
        "fade",
    )


def _hostile_funding(side: str, funding: float, cap: float) -> bool:
    if cap <= 0:
        return False
    if side == "long" and funding > cap:
        return True
    if side == "short" and funding < -cap:
        return True
    return False


def market_blocks_entry(coin: str, side: str, ctx: MarketCtx | None, cfg: Any) -> str:
    if ctx is None:
        return "no_market_data"
    if ctx.day_volume < float(getattr(cfg, "MIN_COIN_DAY_VOLUME", 0) or 0):
        return "thin_volume"
    if abs(ctx.basis) > float(getattr(cfg, "MAX_BASIS_ABS", 1) or 1):
        return "blown_basis"
    if _hostile_funding(side, ctx.funding, float(getattr(cfg, "MAX_HOSTILE_FUNDING", 0) or 0)):
        return "hostile_funding"
    return ""


class BookEngine:
    """
    Basket = one pile of inventory. Scalper flips cancel unless many go the same way.
    We trade the smoothed pile, and leave when that pile shrinks from its peak.
    """

    def __init__(self) -> None:
        self.ema: dict[str, float] = {}
        self.peak: dict[str, float] = {}
        self.peak_agreement: dict[str, float] = {}
        self.prev_raw: dict[str, float] = {}
        self.ok_since: dict[str, float] = {}

    def load(self, raw: dict[str, Any] | None) -> None:
        data = raw or {}
        self.ema = {str(k): float(v) for k, v in (data.get("ema") or data.get("prev_conv") or {}).items()}
        self.peak = {str(k): float(v) for k, v in (data.get("peak") or {}).items()}
        self.peak_agreement = {str(k): float(v) for k, v in (data.get("peak_agreement") or {}).items()}
        self.prev_raw = {str(k): float(v) for k, v in (data.get("prev_raw") or {}).items()}
        self.ok_since = {str(k): float(v) for k, v in (data.get("ok_since") or {}).items()}

    def dump(self) -> dict[str, Any]:
        return {
            "ema": self.ema,
            "peak": self.peak,
            "peak_agreement": self.peak_agreement,
            "prev_raw": self.prev_raw,
            "ok_since": self.ok_since,
        }

    def refine(
        self,
        raw: list[CoinVote],
        *,
        markets: dict[str, MarketCtx],
        managed: set[str],
        cfg: Any,
        now: float,
        log: Any = None,
    ) -> list[CoinVote]:
        alpha = float(getattr(cfg, "FLOW_EMA_ALPHA", 0.30) or 0.30)
        confirm_s = float(getattr(cfg, "OPEN_CONFIRM_S", 45.0) or 0.0)
        min_flow = float(getattr(cfg, "MIN_ENTRY_FLOW", 0.0) or 0.0)
        exit_flow = float(getattr(cfg, "EXIT_FLOW", -0.008) or -0.008)
        exit_raw_flow = float(getattr(cfg, "EXIT_RAW_FLOW", 0) or 0)
        agr_giveback = float(getattr(cfg, "EXIT_AGREEMENT_GIVEBACK", 0) or 0)
        hold_conv = float(getattr(cfg, "EXIT_AVG_CONVICTION", 0.020) or 0.020)
        giveback = float(getattr(cfg, "CONV_GIVEBACK", 0.30) or 0.30)
        min_ema = float(cfg.MIN_AVG_CONVICTION)
        enter_agr = float(cfg.MIN_SIDE_AGREEMENT)
        exit_agr = float(getattr(cfg, "EXIT_SIDE_AGREEMENT", 0) or 0)
        kept: list[CoinVote] = []
        seen_keys: set[str] = set()
        next_ema: dict[str, float] = {}

        next_raw: dict[str, float] = {}

        for v in raw:
            raw_conv = float(v.avg_conviction)
            v.raw_conviction = raw_conv
            prev_raw = self.prev_raw.get(v.coin, raw_conv)
            v.raw_flow = raw_conv - prev_raw
            next_raw[v.coin] = raw_conv
            prev = self.ema.get(v.coin, raw_conv)
            ema = alpha * raw_conv + (1.0 - alpha) * prev
            next_ema[v.coin] = ema
            v.ema = ema
            v.flow = ema - prev
            v.avg_conviction = ema  # size/weights follow the smoothed basket
            key = f"{v.coin}|{v.side}"
            seen_keys.add(key)
            peak_key = v.coin
            pk = max(abs(ema), float(self.peak.get(peak_key) or 0.0))
            pk_agr = max(v.agreement, float(self.peak_agreement.get(peak_key) or 0.0))
            self.peak_agreement[peak_key] = pk_agr
            if (ema >= 0 and prev >= 0) or (ema <= 0 and prev <= 0):
                self.peak[peak_key] = pk
            else:
                self.peak[peak_key] = abs(ema)
                pk = abs(ema)
            faded = pk > 1e-9 and (pk - abs(ema)) / pk >= giveback
            holding = v.coin in managed
            ctx = markets.get(v.coin)
            # EMA noise like flow=-0.0000 is flat, not a dump. Ride a stable pile.
            flow_flat = 1e-3
            if abs(v.flow) < flow_flat:
                v.flow = 0.0
            dumping = v.flow < (min_flow if min_flow < 0 else -flow_flat)

            if holding:
                agr_faded = (
                    agr_giveback > 0
                    and pk_agr > 1e-9
                    and (pk_agr - v.agreement) / pk_agr >= agr_giveback
                )
                raw_dump = exit_raw_flow < 0 and v.raw_flow <= exit_raw_flow
                if exit_agr > 0 and v.agreement < exit_agr:
                    if log:
                        log.info(
                            "Drop %s — crowd thinned agr=%.0f%% < exit %.0f%%",
                            v.coin,
                            v.agreement * 100.0,
                            exit_agr * 100.0,
                        )
                    self._clear_coin_state(peak_key)
                    continue
                if agr_faded:
                    if log:
                        log.info(
                            "Drop %s — agreement faded agr=%.0f%% peak=%.0f%%",
                            v.coin,
                            v.agreement * 100.0,
                            pk_agr * 100.0,
                        )
                    self._clear_coin_state(peak_key)
                    continue
                if raw_dump:
                    if log:
                        log.info(
                            "Drop %s — raw flow dump raw=%+.3f flow=%+.4f",
                            v.coin,
                            raw_conv,
                            v.raw_flow,
                        )
                    self._clear_coin_state(peak_key)
                    continue
                if abs(ema) < hold_conv or faded or v.flow <= exit_flow:
                    if log:
                        log.info(
                            "Drop %s — flow faded ema=%+.3f peak=%.3f giveback=%s",
                            v.coin,
                            ema,
                            pk,
                            faded,
                        )
                    self._clear_coin_state(peak_key)
                    continue
                reason = market_blocks_entry(v.coin, exec_side(v.side, cfg), ctx, cfg) if ctx is not None else ""
                if reason == "hostile_funding":
                    if log:
                        log.info("Drop %s — funding now hostile", v.coin)
                    self._clear_coin_state(peak_key)
                    continue
                v.persist_s = now - self.ok_since.get(key, now)
                v.side = exec_side(v.side, cfg)
                kept.append(v)
                continue

            trade = exec_side(v.side, cfg)
            reason = market_blocks_entry(v.coin, trade, ctx, cfg)
            if reason:
                if log:
                    log.info("Skip new %s %s — %s", trade, v.coin, reason)
                self.ok_since.pop(key, None)
                continue
            if v.agreement < enter_agr:
                if log:
                    log.info(
                        "Skip new %s %s — agr=%.0f%% < enter %.0f%% of list",
                        v.side,
                        v.coin,
                        v.agreement * 100.0,
                        enter_agr * 100.0,
                    )
                self.ok_since.pop(key, None)
                continue
            if abs(ema) < min_ema:
                if log:
                    log.info("Skip new %s %s — weak ema=%+.3f", v.side, v.coin, ema)
                if abs(ema) < min_ema * 0.6:
                    self.ok_since.pop(key, None)
                continue
            if dumping:
                if log:
                    log.info(
                        "Skip new %s %s — basket shrinking ema=%+.3f flow=%+.4f",
                        v.side,
                        v.coin,
                        ema,
                        v.flow,
                    )
                self.ok_since.pop(key, None)
                continue
            if key not in self.ok_since:
                self.ok_since[key] = now
            v.persist_s = now - self.ok_since[key]
            if v.persist_s < confirm_s:
                if log:
                    log.info("Wait %s %s — persist %.0fs/%ss", v.side, v.coin, v.persist_s, confirm_s)
                continue
            v.side = trade
            kept.append(v)

        for key in list(self.ok_since):
            if key not in seen_keys:
                self.ok_since.pop(key, None)
        for coin in list(self.peak):
            if coin not in next_ema:
                self._clear_coin_state(coin)
        self.ema = next_ema
        self.prev_raw = next_raw
        kept.sort(key=lambda x: x.score, reverse=True)
        max_n = int(cfg.MAX_COINS_IN_BOOK)
        if bool(getattr(cfg, "STICKY_BOOK_SLOTS", False)):
            return _sticky_book(kept, managed, max_n)
        return kept[:max_n]

    def _clear_coin_state(self, coin: str) -> None:
        """Drop hold state. Also reset persist so a fade cannot reopen in seconds."""
        self.peak.pop(coin, None)
        self.peak_agreement.pop(coin, None)
        prefix = f"{coin}|"
        for key in list(self.ok_since):
            if key.startswith(prefix):
                self.ok_since.pop(key, None)


def _sticky_book(kept: list[CoinVote], managed: set[str], max_n: int) -> list[CoinVote]:
    """Keep valid holds in the 3 slots. A 4th name waits until a hold actually exits."""
    if max_n <= 0:
        return []
    held = [v for v in kept if v.coin in managed]
    fresh = [v for v in kept if v.coin not in managed]
    out = held[:max_n]
    if len(out) < max_n:
        out.extend(fresh[: max_n - len(out)])
    return out


def _clamp_leverage(cfg: Any, raw: float) -> int:
    lo = int(cfg.OUR_MIN_LEVERAGE)
    hi = int(cfg.OUR_MAX_LEVERAGE)
    return max(lo, min(hi, int(round(raw))))


def votes_to_targets(votes: list[CoinVote], cfg: Any) -> list[TargetPos]:
    if not votes:
        return []
    size_mode = str(getattr(cfg, "SIZE_MODE", "fixed") or "fixed").strip().lower()
    lev_mode = str(getattr(cfg, "LEVERAGE_MODE", "auto") or "auto").strip().lower()
    if lev_mode == "auto":
        lev_mode = "mean" if size_mode == "wallet_avg" else "median"
    cap_copy = float(getattr(cfg, "COPY_MARGIN_CAP_PCT", 0) or 0)
    weights = [abs(v.avg_conviction) for v in votes]
    total_w = sum(weights) or 1.0
    gross = float(cfg.OUR_GROSS_MARGIN_PCT)
    if len(votes) == 1:
        gross *= float(getattr(cfg, "SINGLE_NAME_SIZE_MULT", 1.0) or 1.0)
        cap = gross
    else:
        cap = gross * (float(cfg.MAX_MARGIN_PER_COIN_PCT) / 100.0)
    targets: list[TargetPos] = []
    for v, w in zip(votes, weights):
        if size_mode == "wallet_avg":
            margin_pct = float(v.avg_margin_pct or 0.0)
            if cap_copy > 0:
                margin_pct = min(margin_pct, cap_copy)
        else:
            margin_pct = min(cap, gross * (w / total_w))
        raw_lev = v.mean_leverage if lev_mode == "mean" and v.mean_leverage > 0 else float(v.median_leverage)
        lev = _clamp_leverage(cfg, raw_lev)
        if margin_pct * lev < 0.5:
            continue
        targets.append(
            TargetPos(
                coin=v.coin,
                side=v.side,
                leverage=lev,
                margin_pct=margin_pct,
                conviction=v.avg_conviction,
            )
        )
    return targets
