"""Staggered wallet snapshots. Scalpers stay in the basket — we average them."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

from .snapshots import SnapshotClient
from .types import QualifiedWallet, WalletSnapshot


class BasketTracker:
    def __init__(self, snapper: SnapshotClient, cfg: Any, logger: logging.Logger) -> None:
        self.snapper = snapper
        self.cfg = cfg
        self.log = logger
        self.snapshots: dict[str, WalletSnapshot] = {}
        self._cursor = 0
        self._next_due: dict[str, float] = {}
        self._addrs: list[str] = []
        self.last_equity: dict[str, float] = {}
        self._fp_changes: dict[str, list[float]] = defaultdict(list)

    def set_basket(
        self,
        wallets: list[QualifiedWallet],
        saved_hyper: list[str] | None = None,
        saved_fps: dict[str, list[str]] | None = None,
    ) -> None:
        addrs = [w.address.lower() for w in wallets]
        self.snapshots = {a: s for a, s in self.snapshots.items() if a in addrs}
        now = time.time()
        for i, addr in enumerate(addrs):
            self._next_due[addr] = now + (i % max(1, int(self.cfg.WALLETS_PER_TICK))) * 0.35
        self._cursor = 0
        self._addrs = addrs

    @property
    def addrs(self) -> list[str]:
        return self._addrs

    @property
    def hyper(self) -> set[str]:
        return set()

    def poll_some(self, now: float) -> int:
        if not self.addrs:
            return 0
        budget = max(1, int(self.cfg.WALLETS_PER_TICK))
        interval = float(self.cfg.SNAPSHOT_INTERVAL_S)
        due: list[tuple[float, str]] = []
        for addr in self.addrs:
            if now < self._next_due.get(addr, 0.0):
                continue
            fetched = 0.0
            snap = self.snapshots.get(addr)
            if snap is not None:
                fetched = float(snap.fetched_at)
            due.append((fetched, addr))
        due.sort()
        n = 0
        for _fetched, addr in due[:budget]:
            prev = self.snapshots.get(addr)
            snap = self.snapper.snapshot(addr, now)
            if (
                prev is not None
                and prev.fingerprint
                and snap.fingerprint
                and prev.fingerprint != snap.fingerprint
            ):
                self._fp_changes[addr].append(now)
            cutoff = now - 3600.0
            if addr in self._fp_changes:
                self._fp_changes[addr] = [t for t in self._fp_changes[addr] if t > cutoff]
            self.snapshots[addr] = snap
            if snap.account_value > 0:
                self.last_equity[addr] = snap.account_value
            self._next_due[addr] = now + interval
            n += 1
        return n

    def churned_wallets(self, now: float) -> set[str]:
        """Wallets whose book flipped too often in the last hour (live scalpers).

        Gated only by MAX_BOOK_CHANGES_PER_HOUR (0 = off). Cloud leaves this at 0
        so scalpers stay in the average; local holder mode uses a small cap.
        """
        cap = int(getattr(self.cfg, "MAX_BOOK_CHANGES_PER_HOUR", 0) or 0)
        if cap <= 0:
            return set()
        cutoff = now - 3600.0
        out: set[str] = set()
        for addr, times in self._fp_changes.items():
            if sum(1 for t in times if t > cutoff) >= cap:
                out.add(addr)
        return out

    def live_snapshots(self) -> list[WalletSnapshot]:
        return list(self.snapshots.values())

    def voter_count(self, now: float, extra_exclude: set[str] | None = None) -> int:
        stale = float(self.cfg.STALE_SNAPSHOT_S)
        skip = set(extra_exclude or ())
        n = 0
        for addr, s in self.snapshots.items():
            if addr in skip or s.error:
                continue
            if now - s.fetched_at <= stale and (s.account_value > 0 or s.positions):
                n += 1
        return n
