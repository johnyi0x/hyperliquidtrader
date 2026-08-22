# hl-multi-strategy-bot

Multi-interval Numba backtests + paper/live trading with shared entry/exit/DCA rules.

## First run — do you need `run_backtest.py`?

**No.** `bot_live.py` auto-tunes when:
- there is no saved setup yet, or
- `BACKTEST_REFRESH_HOURS` has elapsed (default 24h) and you are **flat**

`run_backtest.py` is optional (tune only, no trading).

```text
python bot_live.py          # paper or live (see PAPER_TRADING)
python run_backtest.py      # optional one-shot tune
python check_paper.py       # paper balance / last trades
```

## Paper vs live

In `config.py`:

```python
PAPER_TRADING = True   # paper first
# PAPER_TRADING = False  # real orders
```

Same path for both:
- same daily tune (`run_full_tune`)
- same saved winner (`data/params.json`)
- same `OrderExecutor` (paper client simulates fills + fees + TP/SL)
- same closed-bar entry / exit / DCA rules (`src/engine.py` = Numba masks)

## Setup

```bash
cd hl-multi-strategy-bot
pip install -r requirements.txt
```

`.env` (or sibling `hyperliquid-rsi-bot/.env`):

```
HYPE_WALLET_ADDRESS=0x...
HYPE_PRIVATE_KEY=0x...
```

Edit `config.py` (`PAIRS`, leverage, intervals, `TARGET_TRADES_PER_DAY`, exit toggles, …).

## Parity notes (backtest ↔ paper/live)

| Rule | Behavior |
|------|----------|
| Entry | Last **closed** bar of setup interval |
| TP/SL | Spot %; exchange/paper triggers when enabled |
| Exit signal | Same closed bar as backtest |
| DCA | Only with TP/SL on (both tune + live); once per closed bar |
| Max hold | `MAX_POSITION_HOURS` (same hours → bars in tune) |
| Balance % | Capped at 95% like sizing helper |
| Fees | `TAKER_FEE_PCT` both sides |
| `FLIP_EXECUTION` | Live/paper only — leave **False** to match backtest |

Inevitable small differences: market slippage vs bar close, network delays, exchange rounding.

## Recent fixes
- Paper **DCA** (same-side average-in) works — was blocked before
- No exit/DCA on the **entry closed bar** (matches Numba next-bar rule)
- Backtest uses live’s **0.97 margin buffer** for sizing
- Trade-state / params saves **retry** on Windows file locks
- Startup validates intervals and exit-layer config

## Pairs

```python
PAIRS = ("CASHCAT",)           # main
PAIRS = ("xyz:SPCX",)          # HIP-3
PAIRS = ("BTC", "xyz:SPCX")    # multi
```
