# ES Futures Trend Continuation Strategy
### TradeStation API v3 — Python Implementation

---

## Files

| File | Purpose |
|---|---|
| `es_strategy.py` | Live trading engine (real-time streaming) |
| `backtest.py` | Parameter-sweep backtester |

---

## Setup

### 1. Install dependencies
```bash
pip3 install requests pytz pandas
```

### 2. TradeStation API Credentials
Get your credentials from the [TradeStation Developer Portal](https://developer.tradestation.com).

Set via environment variables (recommended):
```bash
export TS_CLIENT_ID="your_client_id"
export TS_CLIENT_SECRET="your_client_secret"
export TS_REFRESH_TOKEN="your_refresh_token"
export TS_ACCOUNT_ID="your_account_id"
```

Or set directly in `StrategyConfig` in `es_strategy.py`.

### 3. Update the symbol
ES futures roll quarterly. Update `symbol = "ESH25"` to the current front-month contract:
- March: `ESH25`, June: `ESM25`, September: `ESU25`, December: `ESZ25`

---

## How to Run

### Backtest first (always)
```bash
python3 backtest.py
```
- Outputs `backtest_results.csv` — all parameter combinations
- Outputs `best_config.json` — top configuration by Expectancy

Set `use_quick_grid=False` in `BacktestRunner` for the full parameter sweep (~thousands of combinations).

### Live / Paper Trading
```bash
python3 es_strategy.py
```
> ⚠️ `paper_trading = True` by default. Only set to `False` after thorough backtesting.

---

## Architecture

```
ESStrategy (orchestrator)
├── TradeStationClient    — API v3 authentication, market data, orders
├── ScenarioEngine        — Classifies price action into 10 trade scenarios (1.1–3.4)
├── FilterEngine          — RSI, VWAP, Volume, Time Window, ADX, daily P&L limits
├── StopCalculator        — Fixed pts/%, ATR, Fibonacci, Level Rebreak
├── TargetCalculator      — Fixed R:R, ATR mult, Fibonacci extension, trailing
└── TradeManager          — Manages trailing stops, breakeven, partial exits
```

---

## Strategy Scenarios

| Scenario | Opening | Condition | Action |
|---|---|---|---|
| 1.1 | Above PDH | No retrace | No Trade |
| **1.2** | Above PDH | Dip → recross PDH up | **LONG** |
| 1.3 | Above PDH | Dip → stays in range | No Trade |
| **1.4** | Above PDH | Dip → breaks PDL | **SHORT** |
| **2.1** | Inside | Breaks PDH | **LONG** |
| 2.2 | Inside | Stays inside | No Trade |
| **2.3** | Inside | Breaks PDL | **SHORT** |
| 3.1 | Below PDL | Stays below | No Trade |
| **3.2** | Below PDL | Bounce → re-break PDL | **SHORT** |
| 3.3 | Below PDL | Moves above → in range | No Trade |
| **3.4** | Below PDL | Moves above → breaks PDH | **LONG** |

---

## Key API Endpoints Used

| Endpoint | Purpose |
|---|---|
| `POST /oauth/token` | OAuth2 refresh token exchange |
| `GET /v3/marketdata/barcharts/{symbol}` | Historical OHLCV bars |
| `GET /v3/marketdata/stream/barcharts/{symbol}` | Real-time streaming bars |
| `GET /v3/marketdata/quotes/{symbol}` | Quote snapshot |
| `GET /v3/brokerage/accounts/{id}/balances` | Account balance |
| `POST /v3/orderexecution/orders` | Place orders (Market, Limit, Stop) |
| `PUT /v3/orderexecution/orders/{id}` | Modify orders (stop adjustment) |
| `DELETE /v3/orderexecution/orders/{id}` | Cancel orders |

Simulation base URL: `https://sim-api.tradestation.com/v3` (paper trading)
Live base URL: `https://api.tradestation.com/v3`

---

## Configurable Parameters

All parameters are in `StrategyConfig`. Key ones:

```python
# Stop Loss
sl_type        = StopLossType.FIXED_PTS     # FIXED_PCT | FIXED_PTS | DYNAMIC_ATR | DYNAMIC_FIBO | LEVEL_REBREAK
sl_pts         = 10.0                        # Points (for FIXED_PTS)
sl_atr_mult    = 1.5                         # ATR multiplier (for DYNAMIC_ATR)

# Entry
entry_type     = EntryType.CONFIRM_LVL       # MARKET_FIRST | CONFIRM_LVL | CONFIRM_LAST
vol_filter     = True                        # Require 1.5x avg volume
rsi_filter     = True                        # RSI > 50 for longs, < 50 for shorts
vwap_filter    = True                        # Price must be above/below VWAP
max_entries_per_day = 2

# Profit Target
pt_type        = ProfitTargetType.FIXED_RR   # FIXED_RR | ATR_MULT | TRAILING_PTS | TRAILING_ATR | FIBO_EXT | PARTIAL
pt_rr_ratio    = 3.0                         # Minimum 1:3 R:R

# Risk Management
daily_loss_limit_pct = 0.02                  # 2% daily loss limit
daily_profit_lock_pct = 0.04                 # Stop at 4% daily gain
account_capital = 50000.0
```

---

## Backtest Parameters (from strategy doc)

The backtest sweeps across all combinations from the master table:

| Parameter | Range |
|---|---|
| SL Type | FIXED_PTS, DYNAMIC_ATR, LEVEL_REBREAK |
| SL Points | 5, 8, 10, 12, 15 |
| ATR Multiplier (SL) | 1.0x, 1.5x, 2.0x, 2.5x |
| Entry Method | MARKET_FIRST, CONFIRM_LVL, CONFIRM_LAST |
| Volume Filter | Off, 1.5x, 2.0x avg |
| RSI Filter | Off, >50/<50, >55/<45, >60/<40 |
| VWAP Filter | Off, On |
| Time Window | All Day, Open Drive, Morning, Power Hour |
| Max Entries/Day | 1, 2, 3 |
| Profit Target | FIXED_RR, TRAILING, ATR_MULT |
| R:R Ratio | 1:2, 1:3, 1:4, 1:5 |
| ATR Multiplier (PT) | 2x, 3x, 4x |
| Trailing Distance | 5, 8, 10, 15 pts |

---

## Important Notes

- **Paper trade first.** Never run `paper_trading = False` without completing a full backtest and walk-forward validation.
- **Update the symbol** each quarterly roll (March/June/September/December).
- **API rate limits** — do not spam bar requests. The streaming endpoint handles real-time data efficiently.
- **Slippage model** — 1 tick per side (0.25 pts) and $4.50 round-trip commission are applied to all backtest P&L calculations.
- Tokens expire after ~20 minutes; the client auto-refreshes using your refresh token.
