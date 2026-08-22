"""Run the profit-meta follower.

  python profit-meta-follower/run.py
  PMF_PROFILE=cloud python profit-meta-follower/run.py

Local and cloud use separate data folders and can use different wallets (.env).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

import config as cfg
from src.logger import setup_logger
from pmf.runner import ProfitMetaRunner, build_live_client


def _data_dir() -> Path:
    override = os.environ.get("PMF_DATA_DIR", "").strip()
    if override:
        p = Path(override)
        return p if p.is_absolute() else ROOT / p
    prof = getattr(cfg, "PMF_PROFILE", os.environ.get("PMF_PROFILE", "local"))
    return ROOT / f"data-{prof}"


def load_secrets() -> tuple[str, str]:
    load_dotenv(REPO / ".env")
    sibling = REPO.parent / "hyperliquid-rsi-bot" / ".env"
    if sibling.exists():
        load_dotenv(sibling, override=False)
    prof = getattr(cfg, "PMF_PROFILE", "local").upper()
    # Per-profile env: HYPE_WALLET_ADDRESS_CLOUD / HYPE_PRIVATE_KEY_CLOUD etc.
    wallet = os.environ.get(f"HYPE_WALLET_ADDRESS_{prof}", "").strip()
    key = os.environ.get(f"HYPE_PRIVATE_KEY_{prof}", "").strip()
    if not wallet:
        wallet = os.environ.get("HYPE_WALLET_ADDRESS", "").strip()
    if not key:
        key = os.environ.get("HYPE_PRIVATE_KEY", "").strip()
    if not wallet or not key:
        raise RuntimeError(
            "Set HYPE_WALLET_ADDRESS and HYPE_PRIVATE_KEY in .env "
            f"(or HYPE_WALLET_ADDRESS_{prof} / HYPE_PRIVATE_KEY_{prof})"
        )
    if not wallet.startswith("0x") or len(wallet) != 42:
        raise RuntimeError("HYPE_WALLET_ADDRESS must be 42-char hex")
    if not key.startswith("0x"):
        key = "0x" + key
    return wallet, key


def main() -> None:
    data_dir = _data_dir()
    log_dir = ROOT / "logs" / getattr(cfg, "PMF_PROFILE", "local")
    logger = setup_logger("profit-meta", log_dir)
    logger.info(
        "Instance profile=%s data=%s filter=%s",
        getattr(cfg, "PMF_PROFILE", "local"),
        data_dir,
        getattr(cfg, "BASKET_FILTER_MODE", "off"),
    )
    if not bool(cfg.PAPER_TRADING):
        logger.warning("LIVE mode — this bot will send real orders on your Hyperliquid account")
    wallet, key = load_secrets()
    client = build_live_client(cfg, wallet, key, logger)
    runner = ProfitMetaRunner(cfg, client=client, data_dir=data_dir, logger=logger)
    try:
        runner.run_forever()
    except KeyboardInterrupt:
        logger.info("Stopped")


if __name__ == "__main__":
    main()
