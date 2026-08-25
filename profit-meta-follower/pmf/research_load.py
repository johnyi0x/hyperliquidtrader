"""Load research gather files for offline backtests (last 7d or all available)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .snapshots import fingerprint
from .types import WalletPos, WalletSnapshot

log = logging.getLogger("pmf-research-load")

TICKER_FEE_PCT = 0.0005  # 0.05% per buy/sell


def _fmt_dur(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _load_progress(done: int, total: int, t0: float, *, label: str = "") -> None:
    """Single-line load bar on stderr (same style as tune search)."""
    import sys
    import time as _time

    total = max(1, int(total))
    done = min(max(0, int(done)), total)
    elapsed = _time.time() - t0
    frac = done / total
    eta_s = _fmt_dur(elapsed * (total - done) / done) if done > 0 else "?"
    width = 28
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    tag = f" {label}" if label else ""
    sys.stderr.write(
        f"\r[{bar}] {done}/{total} {frac:5.1%}  elapsed {_fmt_dur(elapsed)}  eta {eta_s}{tag}   "
    )
    sys.stderr.flush()
    if done >= total:
        sys.stderr.write("\n")
        sys.stderr.flush()


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("rb") as fh:
        for _ in fh:
            n += 1
    return n


@dataclass
class BookRow:
    ts: float
    wallets: list[WalletSnapshot]
    marks: dict[str, float]
    # Full gather vector per coin: [mark, funding, oi, basis, day_vol]
    mkt: dict[str, list[float]] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)


@dataclass
class TickRow:
    ts: float
    all_votes: list[dict[str, Any]]
    holder_votes: list[dict[str, Any]]
    trade_votes: list[dict[str, Any]]
    coverage: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchDataset:
    """Aligned timeline for crowd + mark backtests."""

    ts: np.ndarray
    day_ids: np.ndarray
    day_labels: list[str]
    books: list[BookRow | None]
    ticks: list[TickRow | None]
    marks: np.ndarray  # (n_ticks, n_coins)
    coin_index: dict[str, int]
    index_coin: list[str]
    holder_addrs: set[str]
    live_holder_addrs: set[str]
    live_listed: int
    live_basket_target: int
    cloud_basket_addrs: frozenset[str]
    pool_size: int
    span_days: float
    source: str
    # Unified marks + 1m/15m/1h candle engine (same object live uses).
    price_engine: Any = None
    # Precomputed MarketCtx per tick (same as live); avoids rebuild every combo.
    markets_by_tick: list | None = None
    # Per-tick indicator panel (rsi/ema/atr/ret/range_dump) for O(1) replay.
    ind_panel: Any = None
    # Deprecated legacy arrays (kept empty; gates use price_engine).
    features: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_ticks(self) -> int:
        return int(len(self.ts))

    @property
    def n_coins(self) -> int:
        return len(self.index_coin)

    @property
    def has_price_features(self) -> bool:
        eng = self.price_engine
        return eng is not None and any(eng.has_coin(c) for c in self.index_coin[:8])


def _parse_day(name: str) -> datetime | None:
    try:
        return datetime.strptime(name, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _research_day_dirs(research_dir: Path, *, max_days: int = 7) -> list[Path]:
    if not research_dir.exists():
        return []
    days: list[tuple[datetime, Path]] = []
    for p in research_dir.iterdir():
        if not p.is_dir():
            continue
        dt = _parse_day(p.name)
        if dt is not None:
            days.append((dt, p))
    days.sort(key=lambda x: x[0])
    if max_days > 0 and len(days) > max_days:
        days = days[-max_days:]
    return [p for _dt, p in days]


def _cloud_basket_size() -> int:
    try:
        from config_profiles import CLOUD

        return max(1, int(CLOUD.get("BASKET_SIZE", 100) or 100))
    except Exception:
        return 100


def _addr(row: dict[str, Any]) -> str:
    addr = str(row.get("address") or "").lower()
    return addr if addr.startswith("0x") else ""


def load_ranked_pool_rows(research_dir: Path, state_path: Path | None = None) -> list[dict[str, Any]]:
    """ROI-ordered labeled pool (same order cloud walks for the holder basket)."""
    if state_path and state_path.exists():
        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
            rows = raw.get("research_pool") or []
            out = [r for r in rows if isinstance(r, dict)]
            if out:
                return out
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    pools = sorted(research_dir.glob("*/pool-*.json"))
    if not pools:
        return []
    try:
        raw = json.loads(pools[-1].read_text(encoding="utf-8"))
        rows = raw.get("wallets") or raw.get("research_pool") or []
        return [r for r in rows if isinstance(r, dict)]
    except (OSError, json.JSONDecodeError, TypeError):
        return []


def holders_from_ranked_pool(
    rows: Iterable[dict[str, Any]],
    *,
    basket_size: int,
) -> tuple[set[str], set[str]]:
    """All labeled holders, plus the first `basket_size` in ROI order (live crowd)."""
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("holder") is not True:
            continue
        addr = _addr(row)
        if not addr or addr in seen:
            continue
        seen.add(addr)
        ordered.append(addr)
    want = max(1, int(basket_size or 0))
    return seen, set(ordered[:want])


def load_cloud_basket_addrs(
    research_dir: Path,
    *,
    basket_size: int,
    state_path: Path | None = None,
) -> frozenset[str]:
    """Cloud trade-basket addresses (top-N ROI wallets live polls).

    Prefer pmf_state.json basket (exact live list). Else ranked pool order.
    """
    want = max(1, int(basket_size or 0))
    candidates: list[Path] = []
    if state_path is not None:
        candidates.append(state_path)
    base = research_dir.parent
    for name in ("pmf_state.json",):
        candidates.append(base / name)
        candidates.append(base.parent / "data-cloud" / name)
        candidates.append(base.parent / "data-local" / name)
    seen_paths: set[Path] = set()
    for sp in candidates:
        if sp in seen_paths or not sp.exists():
            continue
        seen_paths.add(sp)
        try:
            raw = json.loads(sp.read_text(encoding="utf-8"))
            basket = raw.get("basket") or []
            addrs: list[str] = []
            for row in basket:
                if isinstance(row, dict):
                    a = str(row.get("address") or "").lower()
                else:
                    a = str(row).lower()
                if a.startswith("0x"):
                    addrs.append(a)
            if addrs:
                return frozenset(addrs[:want])
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    rows = load_ranked_pool_rows(research_dir, state_path=state_path)
    addrs = []
    for row in rows:
        a = _addr(row)
        if a:
            addrs.append(a)
        if len(addrs) >= want:
            break
    return frozenset(addrs[:want])


def load_holder_labels(research_dir: Path, state_path: Path | None = None) -> set[str]:
    rows = load_ranked_pool_rows(research_dir, state_path=state_path)
    all_h, _live = holders_from_ranked_pool(rows, basket_size=10**9)
    return all_h


def _wallet_from_row(addr: str, equity: float, pos_rows: list[Any], fetched_at: float, ts: float) -> WalletSnapshot:
    positions: list[WalletPos] = []
    for row in pos_rows or []:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        coin = str(row[0])
        sign = int(row[1])
        conv = float(row[2])
        lev = int(row[3]) if len(row) > 3 else 5
        side = "long" if sign >= 0 else "short"
        notional = abs(conv) * max(equity, 1.0)
        positions.append(
            WalletPos(
                coin=coin,
                side=side,
                size=1.0,
                notional=notional,
                entry_px=None,
                leverage=max(1, lev),
                isolated=False,
                conviction=conv if sign >= 0 else -abs(conv),
            )
        )
    fa = float(fetched_at or ts)
    return WalletSnapshot(
        address=str(addr).lower(),
        account_value=float(equity or 0),
        positions=positions,
        fetched_at=fa,
        fingerprint=fingerprint(positions),
    )


def _parse_book_line(row: dict[str, Any]) -> BookRow | None:
    ts = float(row.get("ts") or 0)
    if ts <= 0:
        return None
    wallets: list[WalletSnapshot] = []
    for w in row.get("w") or []:
        if not isinstance(w, (list, tuple)) or len(w) < 3:
            continue
        addr = str(w[0])
        equity = float(w[1] or 0)
        pos = w[2]
        fetched_at = float(w[3]) if len(w) > 3 else ts
        wallets.append(_wallet_from_row(addr, equity, pos, fetched_at, ts))
    mkt_raw = row.get("mkt") or {}
    marks: dict[str, float] = {}
    mkt: dict[str, list[float]] = {}
    if isinstance(mkt_raw, dict):
        for coin, vec in mkt_raw.items():
            if isinstance(vec, (list, tuple)) and vec:
                vals = [float(x or 0) for x in vec]
                mkt[str(coin)] = vals
                px = float(vals[0] or 0)
                if px > 0:
                    marks[str(coin)] = px
    cov = row.get("cov") if isinstance(row.get("cov"), dict) else {}
    if not wallets:
        return None
    return BookRow(ts=ts, wallets=wallets, marks=marks, mkt=mkt, coverage=cov)


def _parse_tick_line(row: dict[str, Any]) -> TickRow | None:
    ts = float(row.get("ts") or 0)
    if ts <= 0:
        return None
    cov = row.get("cov") if isinstance(row.get("cov"), dict) else {}
    return TickRow(
        ts=ts,
        all_votes=list(row.get("all") or []),
        holder_votes=list(row.get("holders") or []),
        trade_votes=list(row.get("trade") or []),
        coverage=cov,
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def load_research_books_and_ticks(
    research_dir: Path,
    *,
    max_days: int = 7,
    progress: bool = True,
) -> tuple[list[BookRow], list[TickRow]]:
    import time as _time

    books: list[BookRow] = []
    ticks: list[TickRow] = []
    day_dirs = _research_day_dirs(research_dir, max_days=max_days)
    total = 0
    for day_dir in day_dirs:
        total += _count_lines(day_dir / "books.jsonl")
        total += _count_lines(day_dir / "crowd_ticks.jsonl")
    total = max(1, total)
    done = 0
    t0 = _time.time()
    if progress:
        _load_progress(0, total, t0, label="books+ticks")
    for day_dir in day_dirs:
        for row in _iter_jsonl(day_dir / "books.jsonl"):
            parsed = _parse_book_line(row)
            if parsed is not None:
                books.append(parsed)
            done += 1
            if progress and (done % 25 == 0):
                _load_progress(min(done, total), total, t0, label="books+ticks")
        for row in _iter_jsonl(day_dir / "crowd_ticks.jsonl"):
            parsed = _parse_tick_line(row)
            if parsed is not None:
                ticks.append(parsed)
            done += 1
            if progress and (done % 50 == 0):
                _load_progress(min(done, total), total, t0, label="books+ticks")
    if progress:
        _load_progress(total, total, t0, label="books+ticks done")
    books.sort(key=lambda b: b.ts)
    ticks.sort(key=lambda t: t.ts)
    return books, ticks


def load_marks_series(
    research_dir: Path,
    *,
    max_days: int = 7,
    progress: bool = True,
) -> list[tuple[float, dict[str, float]]]:
    import time as _time

    out: list[tuple[float, dict[str, float]]] = []
    day_dirs = _research_day_dirs(research_dir, max_days=max_days)
    total = max(1, sum(_count_lines(d / "marks.jsonl") for d in day_dirs))
    done = 0
    t0 = _time.time()
    if progress:
        _load_progress(0, total, t0, label="marks")
    for day_dir in day_dirs:
        for row in _iter_jsonl(day_dir / "marks.jsonl"):
            done += 1
            ts = float(row.get("ts") or 0)
            if ts <= 0:
                if progress and done % 100 == 0:
                    _load_progress(done, total, t0, label="marks")
                continue
            mkt_raw = row.get("mkt") or {}
            marks: dict[str, float] = {}
            if isinstance(mkt_raw, dict):
                for coin, vec in mkt_raw.items():
                    if isinstance(vec, (list, tuple)) and vec:
                        px = float(vec[0] or 0)
                        if px > 0:
                            marks[str(coin)] = px
            if marks:
                out.append((ts, marks))
            if progress and (done % 50 == 0 or done >= total):
                _load_progress(done, total, t0, label="marks")
    if progress:
        _load_progress(total, total, t0, label="marks")
    out.sort(key=lambda x: x[0])
    return out


def _align_marks(
    ts: np.ndarray,
    book_rows: list[BookRow | None],
    mark_rows: list[tuple[float, dict[str, float]]],
    coin_index: dict[str, int],
) -> np.ndarray:
    n = len(ts)
    n_coins = len(coin_index)
    marks = np.full((n, n_coins), np.nan, dtype=np.float64)
    mark_i = 0
    last: dict[str, float] = {}
    for i in range(n):
        t = float(ts[i])
        while mark_i < len(mark_rows) and mark_rows[mark_i][0] <= t + 1e-6:
            last.update(mark_rows[mark_i][1])
            mark_i += 1
        br = book_rows[i]
        if br is not None:
            last.update(br.marks)
        for coin, px in last.items():
            idx = coin_index.get(coin)
            if idx is not None and px > 0:
                marks[i, idx] = px
    # Forward-fill per coin.
    for j in range(n_coins):
        prev = np.nan
        for i in range(n):
            v = marks[i, j]
            if np.isfinite(v) and v > 0:
                prev = v
            elif np.isfinite(prev) and prev > 0:
                marks[i, j] = prev
    return marks


def build_dataset(
    data_dir: Path,
    *,
    max_days: int = 7,
    state_path: Path | None = None,
    live_basket_size: int | None = None,
    progress: bool = False,
) -> ResearchDataset | None:
    import time as _time

    research_dir = data_dir / "research"
    if progress:
        print("Load: books + crowd ticks...", flush=True)
    books_raw, ticks_raw = load_research_books_and_ticks(
        research_dir, max_days=max_days, progress=progress
    )
    if progress:
        print("Load: marks series...", flush=True)
    mark_rows = load_marks_series(research_dir, max_days=max_days, progress=progress)

    if not books_raw and not ticks_raw:
        return None

    # Master timeline from books (canonical); attach nearest tick rows.
    if books_raw:
        ts_list = [b.ts for b in books_raw]
        book_rows: list[BookRow | None] = list(books_raw)
        source = "books"
    else:
        ts_list = [t.ts for t in ticks_raw]
        book_rows = [None] * len(ts_list)
        source = "crowd_ticks"

    ts = np.array(ts_list, dtype=np.float64)
    tick_by_ts = {round(t.ts, 3): t for t in ticks_raw}
    tick_rows: list[TickRow | None] = []
    for t in ts_list:
        tick_rows.append(tick_by_ts.get(round(t, 3)))

    day_labels = sorted({datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d") for t in ts_list})
    day_to_id = {d: i for i, d in enumerate(day_labels)}
    day_ids = np.array(
        [day_to_id[datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")] for t in ts_list],
        dtype=np.int32,
    )

    coin_index: dict[str, int] = {}
    for br in book_rows:
        if br is None:
            continue
        for snap in br.wallets:
            for p in snap.positions:
                coin_index.setdefault(p.coin, len(coin_index))
    for _t, m in mark_rows:
        for c in m:
            coin_index.setdefault(c, len(coin_index))
    for tr in tick_rows:
        if tr is None:
            continue
        for bucket in (tr.all_votes, tr.holder_votes, tr.trade_votes):
            for v in bucket:
                c = str(v.get("c") or "")
                if c:
                    coin_index.setdefault(c, len(coin_index))

    index_coin = [""] * len(coin_index)
    for c, i in coin_index.items():
        index_coin[i] = c

    marks = _align_marks(ts, book_rows, mark_rows, coin_index)
    live_target = max(1, int(live_basket_size or 0) or _cloud_basket_size())
    ranked = load_ranked_pool_rows(research_dir, state_path=state_path)
    holders, live_holders = holders_from_ranked_pool(ranked, basket_size=live_target)
    if not holders:
        holders = load_holder_labels(research_dir, state_path=state_path)
        live_holders = set(holders)
    live_listed = len(live_holders)
    cloud_basket = load_cloud_basket_addrs(
        research_dir,
        basket_size=live_target,
        state_path=state_path,
    )
    pool_n = 0
    for br in book_rows:
        if br is not None:
            pool_n = max(pool_n, len(br.wallets))
    if pool_n <= 0 and ticks_raw:
        for tr in ticks_raw:
            cov = tr.coverage or {}
            pool_n = max(pool_n, int(cov.get("listed") or 0))
    span_days = max(0.25, (float(ts[-1]) - float(ts[0])) / 86400.0) if len(ts) > 1 else 0.25

    from .price_engine import load_research_into_engine

    if progress:
        print("Load: PriceEngine (marks + 1m/15m/1h candles)...", flush=True)
    price_eng = load_research_into_engine(
        research_dir,
        max_days=max_days,
        coins=list(coin_index.keys()),
        progress=progress,
        load_book_mkt=False,  # marks.jsonl already has funding/oi/basis/day_vol
    )
    # Densify mark path at every book tick WITHOUT wiping funding/oi/basis/day_vol.
    n_ticks = len(ts_list)
    densify_t0 = _time.time()
    if progress and n_ticks:
        _load_progress(0, n_ticks, densify_t0, label="densify marks")
    for i, t in enumerate(ts_list):
        for coin, j in coin_index.items():
            px = float(marks[i, j]) if j < marks.shape[1] else 0.0
            if px > 0:
                price_eng.ingest_mark(coin, float(t), px, aux=False)
        if progress and n_ticks and (i % 20 == 0 or i + 1 >= n_ticks):
            _load_progress(i + 1, n_ticks, densify_t0, label="densify marks")
    n_candle_coins = sum(
        1
        for c in coin_index
        if any(price_eng.candle_span_s(c, iv) > 0 for iv in ("1m", "15m", "1h"))
    )
    n_aux = sum(1 for c in coin_index if abs(price_eng.funding_at(c, float(ts_list[-1]))) > 0 or price_eng.day_vol_at(c, float(ts_list[-1])) > 0) if ts_list else 0

    log.info(
        "Research dataset: ticks=%s coins=%s days=%.2f holders=%s live=%s/%s listed=%s "
        "cloud_basket=%s pool=%s source=%s candle_coins=%s mark_aux_coins=%s",
        len(ts),
        len(coin_index),
        span_days,
        len(holders),
        len(live_holders),
        live_target,
        live_listed,
        len(cloud_basket),
        pool_n,
        source,
        n_candle_coins,
        n_aux,
    )
    if progress:
        print(
            f"Load done: ticks={len(ts)} coins={len(coin_index)} "
            f"candles={n_candle_coins} mark_aux={n_aux}",
            flush=True,
        )

    ds = ResearchDataset(
        ts=ts,
        day_ids=day_ids,
        day_labels=day_labels,
        books=book_rows,
        ticks=tick_rows,
        marks=marks,
        coin_index=coin_index,
        index_coin=index_coin,
        holder_addrs=holders,
        live_holder_addrs=live_holders,
        live_listed=live_listed,
        live_basket_target=live_target,
        cloud_basket_addrs=cloud_basket,
        pool_size=max(pool_n, 50),
        span_days=span_days,
        source=source,
        price_engine=price_eng,
        features={},
    )
    # One-time market snapshot per tick — identical to per-combo rebuild, done once.
    from .bt_replay import _markets_for_row

    if progress:
        print("Load: precompute market ctx per tick...", flush=True)
    mkt_t0 = _time.time()
    markets: list = []
    n_m = len(ts_list)
    for i in range(n_m):
        markets.append(
            _markets_for_row(
                book_rows[i],
                coin_index,
                price=price_eng,
                asof=float(ts_list[i]),
            )
        )
        if progress and (i % 40 == 0 or i + 1 >= n_m):
            _load_progress(i + 1, n_m, mkt_t0, label="markets")
    ds.markets_by_tick = markets

    # One-time indicator panel — identical PriceEngine math, O(1) at each tick.
    from .bt_panels import build_ind_panel

    ds.ind_panel = build_ind_panel(ds, progress=progress, eng=price_eng)
    return ds
