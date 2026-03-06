"""
ES Futures Trend Continuation Strategy
TradeStation API v3 Implementation

Strategy: CME E-Mini S&P 500 (ES) intraday trend continuation
based on Previous Day High/Low breakouts with configurable
stop loss, entry, and profit booking methods.

USAGE:
    1. Set your credentials in config.py (or environment variables)
    2. Configure strategy parameters in the StrategyConfig dataclass
    3. Run: python es_strategy.py

REQUIRES:
    pip install requests pytz pandas
"""

import os
import time
import json
import logging
import requests
import pytz
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

# ─────────────────────────────────────────────
# ENVIRONMENT — load .env before anything else
# ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env from the current working directory
except ImportError:
    pass  # dotenv not installed — fall back to real environment variables

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("es_strategy.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class StopLossType(Enum):
    FIXED_PCT   = "FIXED_PCT"       # Fixed % of notional
    FIXED_PTS   = "FIXED_PTS"       # Fixed ES points
    DYNAMIC_ATR = "DYNAMIC_ATR"     # ATR-based stop
    DYNAMIC_FIBO= "DYNAMIC_FIBO"    # Fibonacci retracement stop
    LEVEL_REBREAK="LEVEL_REBREAK"   # Re-break of entry level

class EntryType(Enum):
    MARKET_FIRST  = "MARKET_FIRST"  # Buy/sell on first breakout tick
    CONFIRM_LVL   = "CONFIRM_LVL"   # Wait for next best offer > reference level
    CONFIRM_LAST  = "CONFIRM_LAST"  # Wait for next best offer > last executed price

class ProfitTargetType(Enum):
    FIXED_RR      = "FIXED_RR"      # Fixed Risk:Reward ratio
    FIXED_PTS     = "FIXED_PTS"     # Fixed ES points
    ATR_MULT      = "ATR_MULT"      # ATR multiplier
    TRAILING_PTS  = "TRAILING_PTS"  # Trailing stop (points)
    TRAILING_ATR  = "TRAILING_ATR"  # Trailing stop (ATR)
    FIBO_EXT      = "FIBO_EXT"      # Fibonacci extension
    PARTIAL       = "PARTIAL"       # Partial profit + trailing

class TradeDirection(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    NONE  = "NONE"

class Scenario(Enum):
    S1_1 = "1.1"  # Open above PDH, no retrace → NO TRADE
    S1_2 = "1.2"  # Open above PDH → dip below PDH → recross up → LONG
    S1_3 = "1.3"  # Open above PDH → dip below → stay between → NO TRADE
    S1_4 = "1.4"  # Open above PDH → dip below → break PDL → SHORT
    S2_1 = "2.1"  # Open inside range → break PDH → LONG
    S2_2 = "2.2"  # Open inside range → stay inside → NO TRADE
    S2_3 = "2.3"  # Open inside range → break PDL → SHORT
    S3_1 = "3.1"  # Open below PDL → stay below → NO TRADE
    S3_2 = "3.2"  # Open below PDL → bounce above PDL → re-break PDL → SHORT
    S3_3 = "3.3"  # Open below PDL → move above PDL → stay in range → NO TRADE
    S3_4 = "3.4"  # Open below PDL → move above PDL → break PDH → LONG

# ─────────────────────────────────────────────
# STRATEGY CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class StrategyConfig:
    # --- Credentials ---
    client_id:      str = os.getenv("TS_CLIENT_ID", "YOUR_CLIENT_ID")
    client_secret:  str = os.getenv("TS_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
    refresh_token:  str = os.getenv("TS_REFRESH_TOKEN", "YOUR_REFRESH_TOKEN")
    account_id:     str = os.getenv("TS_ACCOUNT_ID", "YOUR_ACCOUNT_ID")
    paper_trading:  bool = True    # ALWAYS start with paper/simulation

    # --- Instrument ---
    symbol:         str = "ESH25"  # Front-month ES contract (update each roll)
    contracts:      int = 1        # Number of contracts

    # --- Stop Loss ---
    sl_type:        StopLossType = StopLossType.FIXED_PTS
    sl_pct:         float = 0.005          # 0.5% of notional (FIXED_PCT)
    sl_pts:         float = 10.0           # 10 ES points (FIXED_PTS)
    sl_atr_mult:    float = 1.5            # ATR multiplier (DYNAMIC_ATR)
    sl_fibo_lvl:    float = 0.618          # Fibo retracement level (DYNAMIC_FIBO)
    atr_period:     int   = 14             # ATR lookback period

    # --- Entry ---
    entry_type:     EntryType = EntryType.CONFIRM_LVL
    vol_filter:     bool  = True           # Require volume > 1.5x avg
    vol_filter_mult:float = 1.5            # Volume multiplier for confirmation
    vol_avg_period: int   = 20             # Volume average lookback
    rsi_filter:     bool  = True           # Enable RSI momentum filter
    rsi_period:     int   = 14
    rsi_long_min:   float = 50.0           # RSI must be > this for longs
    rsi_short_max:  float = 50.0           # RSI must be < this for shorts
    vwap_filter:    bool  = True           # VWAP alignment filter (price above/below)
    # Fix 2: VWAP slope filter — instead of (or in addition to) checking
    # price vs VWAP, check that VWAP itself is trending in the trade direction.
    # vwap[now] > vwap[vwap_slope_period bars ago] = upward slope (long bias).
    # This eliminates the "flicker" problem in choppy markets where price
    # oscillates across a flat VWAP line.
    vwap_slope_filter: bool  = False          # Enable VWAP slope check
    vwap_slope_period: int   = 5              # Look-back bars for slope (default 5 min)
    max_entries_per_day: int = 2           # Max trades per session

    # --- Time Windows (ET) ---
    # "ALL"        -> 09:30-16:00 (no restriction)
    # "OPEN_DRIVE" -> 09:30-10:30 only  (highest follow-through per doc)
    # "MORNING"    -> 09:30-12:00 only
    # "POWER_HOUR" -> 14:00-16:00 only
    # Fix 1: enforced with precise hh:mm boundaries so e.g. scenario 1.2
    # cannot fire at 2 PM during an OPEN_DRIVE run.
    time_window:    str   = "ALL"

    # --- Seasonality Filters (Section 7.8) ---
    # Fix 4: Friday early exit — force-close all open trades at
    # friday_exit_hour:friday_exit_minute ET to avoid the weekly close
    # where profit-taking / mean-reversion dominates (doc Section 7.8).
    friday_early_exit:   bool  = True
    friday_exit_hour:    int   = 15   # default 15:45 ET
    friday_exit_minute:  int   = 45

    # Month-end size reduction: trade at month_end_size_pct of normal
    # contract count on the last 2 and first 2 trading days of the month
    # to account for institutional rebalancing flows (Section 7.8).
    month_end_filter:    bool  = True
    month_end_size_pct:  float = 0.5  # 50% normal size on flagged days

    # --- Profit Target ---
    pt_type:        ProfitTargetType = ProfitTargetType.FIXED_RR
    pt_rr_ratio:    float = 3.0            # Minimum 1:3 R:R
    pt_atr_mult:    float = 3.0            # ATR multiplier for target
    trail_pts:      float = 10.0           # Trailing stop distance (points)
    trail_atr_mult: float = 1.5            # Trailing stop ATR multiplier
    pt_fibo_ext:    float = 1.618          # Fibonacci extension level

    # Fix 4: ATR target refresh.
    # When True, the ATR_MULT target is re-evaluated on every bar using
    # the current live ATR, so that expanding volatility after entry
    # extends the target rather than locking you out of a big move.
    # When False (original behaviour), the target is frozen at entry.
    # Recommended: True for trending sessions, False for mean-reversion.
    atr_target_dynamic: bool = True
    # Partial profit: book 50% at 1:2, trail rest
    partial_enabled:bool  = False
    partial_rr:     float = 2.0            # R:R for first partial target
    partial_pct:    float = 0.50           # % of position to close at first target

    # --- Breakeven Stop ---
    be_trigger:     float = 1.0            # Move to BE when trade is 1:1 in profit (0 = off)

    # --- Risk Management ---
    daily_loss_limit_pct: float = 0.02    # 2% of account capital daily loss limit
    daily_profit_lock_pct:float = 0.04    # Stop trading once 4% daily gain hit
    account_capital: float = 50000.0      # Account capital for position sizing

    # --- PDC Bias Filter (Fix 1) ---
    # FIX 1: PDC is now used as a directional bias confirmation filter.
    # For LONG entries: current price must be above PDC.
    # For SHORT entries: current price must be below PDC.
    pdc_filter:     bool  = True           # Enable Previous Day Close bias filter

    # --- Market Regime Filters ---
    adx_filter:     bool  = False          # Enable ADX trend filter
    adx_period:     int   = 14
    adx_min:        float = 20.0           # Min ADX to take trades
    sma_200_filter: bool  = False          # 200 SMA directional filter

    # --- Slippage Model ---
    # Toggle: False = zero-slippage theoretical run, True = realistic
    include_slippage:  bool  = True
    # Fix 3: Per-contract slippage model.
    # The first contract fills at slippage_ticks_base ticks.
    # Each additional contract adds slippage_ticks_increment ticks
    # to model ES order book depth — larger size walks the book further.
    # Example with 5 contracts, base=1, incr=0.5:
    #   contract 1 → 1.0 tick, contract 2 → 1.5 ticks, ...
    #   average fill = 2.0 ticks (vs 1.0 flat — a material difference)
    slippage_ticks_base:      float = 1.0   # ticks for 1st contract
    slippage_ticks_increment: float = 0.5   # additional ticks per extra contract
    commission_rt:            float = 4.50  # round-trip commission per contract ($)

    # --- Scenarios to trade (set False to deactivate) ---
    active_scenarios: dict = field(default_factory=lambda: {
        "1.2": True, "1.4": True,
        "2.1": True, "2.3": True,
        "3.2": True, "3.4": True,
    })

    @property
    def tick_size(self):
        return 0.25  # ES tick size

    @property
    def point_value(self):
        return 50.0  # $50 per point for ES

    @property
    def slippage_pts(self):
        """Entry signal offset for 1 contract (base ticks only)."""
        return self.slippage_ticks_base * self.tick_size

    def effective_slippage_pts(self, qty: int) -> float:
        """
        Fix 3: Average slippage per contract for a given order size.
        Models partial fills at different price levels as order size
        walks the ES order book — larger orders incur more slippage.
        """
        if qty <= 1:
            return self.slippage_ticks_base * self.tick_size
        total_ticks = sum(
            self.slippage_ticks_base + self.slippage_ticks_increment * i
            for i in range(qty)
        )
        avg_ticks = total_ticks / qty
        return avg_ticks * self.tick_size


# ─────────────────────────────────────────────
# TRADESTATION API CLIENT
# ─────────────────────────────────────────────

class TradeStationClient:
    BASE_URL = "https://api.tradestation.com/v3"
    AUTH_URL = "https://signin.tradestation.com/oauth/token"
    SIM_BASE  = "https://sim-api.tradestation.com/v3"  # paper trading

    def __init__(self, config: StrategyConfig):
        self.cfg = config
        self.base = self.SIM_BASE if config.paper_trading else self.BASE_URL
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._authenticate()

    def _authenticate(self):
        """OAuth2 token exchange using refresh token."""
        resp = requests.post(self.AUTH_URL, data={
            "grant_type":    "refresh_token",
            "client_id":     self.cfg.client_id,
            "client_secret": self.cfg.client_secret,
            "refresh_token": self.cfg.refresh_token,
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["access_token"]
        self._token_expiry = datetime.now() + timedelta(seconds=data.get("expires_in", 1200) - 60)
        log.info("Authenticated with TradeStation API (%s)", "PAPER" if self.cfg.paper_trading else "LIVE")

    def _ensure_token(self):
        if not self._token_expiry or datetime.now() >= self._token_expiry:
            log.info("Token expired — re-authenticating")
            self._authenticate()

    @property
    def _headers(self) -> dict:
        self._ensure_token()
        return {"Authorization": f"Bearer {self._access_token}", "Content-Type": "application/json"}

    # ── Market Data ─────────────────────────────────

    def get_bars(self, symbol: str, unit: str = "Daily", interval: int = 1,
                 bars_back: int = 2) -> list[dict]:
        """Fetch historical OHLCV bars."""
        url = f"{self.BASE_URL}/marketdata/barcharts/{symbol}"  # always use live for market data
        params = {"unit": unit, "interval": interval, "barsback": bars_back}
        r = requests.get(url, headers=self._headers, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("Bars", [])

    def get_quote(self, symbol: str) -> dict:
        """Get latest quote snapshot."""
        url = f"{self.BASE_URL}/marketdata/quotes/{symbol}"
        r = requests.get(url, headers=self._headers, timeout=10)
        r.raise_for_status()
        quotes = r.json().get("Quotes", [])
        return quotes[0] if quotes else {}

    def stream_bars(self, symbol: str, unit: str = "Minute", interval: int = 1):
        """Generator that yields streaming bar updates."""
        url = f"{self.BASE_URL}/marketdata/stream/barcharts/{symbol}"
        params = {"unit": unit, "interval": interval}
        with requests.get(url, headers=self._headers, params=params,
                          stream=True, timeout=30) as r:
            r.raise_for_status()
            buffer = ""
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    buffer += chunk.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                pass

    def get_account_balance(self) -> dict:
        """Fetch account balance/equity."""
        url = f"{self.base}/brokerage/accounts/{self.cfg.account_id}/balances"
        r = requests.get(url, headers=self._headers, timeout=10)
        r.raise_for_status()
        balances = r.json().get("Balances", [])
        return balances[0] if balances else {}

    def get_positions(self) -> list[dict]:
        """Fetch open positions."""
        url = f"{self.base}/brokerage/accounts/{self.cfg.account_id}/positions"
        r = requests.get(url, headers=self._headers, timeout=10)
        r.raise_for_status()
        return r.json().get("Positions", [])

    # ── Order Management ────────────────────────────

    def place_order(self, side: str, quantity: int, order_type: str = "Market",
                    limit_price: float = None, stop_price: float = None,
                    time_in_force: str = "DAY", oso_orders: list = None) -> dict:
        """
        Place an order via the TradeStation API.

        side        : "BUY" or "SELL" (for short: "SELLSHORT", "BUYTOCOVER")
        order_type  : "Market", "Limit", "StopMarket", "StopLimit"
        """
        body = {
            "AccountID":  self.cfg.account_id,
            "Symbol":     self.cfg.symbol,
            "Quantity":   str(quantity),
            "OrderType":  order_type,
            "TradeAction": side,
            "TimeInForce": {"Duration": time_in_force},
            "Route":      "Intelligent",
        }
        if limit_price:
            body["LimitPrice"] = str(limit_price)
        if stop_price:
            body["StopPrice"] = str(stop_price)
        if oso_orders:
            body["OSOs"] = oso_orders  # One-Sends-Other bracket orders

        url = f"{self.base}/orderexecution/orders"
        r = requests.post(url, headers=self._headers, json=body, timeout=10)
        r.raise_for_status()
        result = r.json()
        log.info("Order placed → %s %s %s @ %s | Response: %s",
                 side, quantity, self.cfg.symbol, order_type, result)
        return result

    def cancel_order(self, order_id: str) -> dict:
        url = f"{self.base}/orderexecution/orders/{order_id}"
        r = requests.delete(url, headers=self._headers, timeout=10)
        r.raise_for_status()
        log.info("Order cancelled: %s", order_id)
        return r.json()

    def replace_stop(self, order_id: str, new_stop: float) -> dict:
        """Modify an existing stop order price."""
        url = f"{self.base}/orderexecution/orders/{order_id}"
        body = {"StopPrice": str(new_stop)}
        r = requests.put(url, headers=self._headers, json=body, timeout=10)
        r.raise_for_status()
        log.info("Stop updated → order %s new stop %.2f", order_id, new_stop)
        return r.json()


# ─────────────────────────────────────────────
# INDICATOR CALCULATIONS
# ─────────────────────────────────────────────

def calc_atr(bars: list[dict], period: int = 14) -> float:
    """Calculate ATR from a list of bar dicts with High/Low/Close."""
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(bars)):
        h = float(bars[i]["High"])
        l = float(bars[i]["Low"])
        pc = float(bars[i-1]["Close"])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-period:]) / period


def calc_rsi(closes: list[float], period: int = 14) -> float:
    """Wilder RSI calculation."""
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


def calc_vwap(bars: list[dict]) -> float:
    """Intraday VWAP from minute bars (reset each RTH session)."""
    cum_vol = cum_pv = 0.0
    for b in bars:
        typical = (float(b["High"]) + float(b["Low"]) + float(b["Close"])) / 3.0
        vol = float(b.get("TotalVolume", 0))
        cum_pv  += typical * vol
        cum_vol += vol
    return cum_pv / cum_vol if cum_vol > 0 else 0.0


def calc_adx(bars: list[dict], period: int = 14) -> float:
    """Simplified ADX (non-Wilder smoothed) for regime filter."""
    if len(bars) < period + 1:
        return 0.0
    plus_dm, minus_dm, trs = [], [], []
    for i in range(1, len(bars)):
        h, l, pc = float(bars[i]["High"]), float(bars[i]["Low"]), float(bars[i-1]["Close"])
        ph, pl   = float(bars[i-1]["High"]), float(bars[i-1]["Low"])
        up_move   = h - ph
        down_move = pl - l
        plus_dm.append(up_move  if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr_s = sum(trs[-period:]) / period
    if atr_s == 0:
        return 0.0
    di_plus  = 100 * (sum(plus_dm[-period:]) / period) / atr_s
    di_minus = 100 * (sum(minus_dm[-period:]) / period) / atr_s
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9)
    return dx


# ─────────────────────────────────────────────
# STRATEGY STATE
# ─────────────────────────────────────────────

class StrategyState:
    def __init__(self):
        self.reset_daily()

    def reset_daily(self):
        self.pdh: float = 0.0
        self.pdl: float = 0.0
        self.pdo: float = 0.0
        self.pdc: float = 0.0

        # Opening bias tracking
        self.open_above_pdh: bool = False
        self.open_below_pdl: bool = False
        self.open_inside:    bool = False

        # Intraday tracking
        self.touched_below_pdh: bool = False
        self.touched_above_pdl: bool = False
        self.crossed_pdh_up:    bool = False
        self.crossed_pdl_down:  bool = False

        # Fix 1: Open Drive classification
        # True  = price opened outside range and has NEVER retraced back
        #         through the breakout level → pure Scenario 1.1 / 3.1
        # False = price has tested back through the level at least once
        #         → qualifies for Scenario 1.2 / 3.2 retest-and-drive
        self.open_drive_pure:   bool = True   # resets to False on first retest

        # Trade state
        self.in_trade:      bool  = False
        self.direction:     TradeDirection = TradeDirection.NONE
        self.entry_price:   float = 0.0
        self.stop_price:    float = 0.0
        self.target_price:  float = 0.0
        self.entry_order_id: str = ""
        self.stop_order_id:  str = ""
        self.target_order_id:str = ""
        self.scenario_id:   str  = ""
        self.partial_done:  bool = False
        self.be_triggered:  bool = False

        # Session P&L tracking
        self.entries_today:  int   = 0
        self.daily_pnl:      float = 0.0

        # Indicator data (intraday 1-min bars)
        self.intraday_bars:  list[dict] = []
        self.session_vwap:   float = 0.0
        # Fix 2: VWAP slope — rolling history of VWAP values (one per bar)
        # so FilterEngine can check vwap[now] > vwap[n bars ago] rather
        # than just price > vwap (which flickers in choppy conditions).
        self.vwap_history:   list[float] = []

        log.info("Daily state reset")


# ─────────────────────────────────────────────
# SCENARIO DETECTION ENGINE
# ─────────────────────────────────────────────

class ScenarioEngine:
    """Determines which trade scenario applies given the current state."""

    def __init__(self, cfg: StrategyConfig, state: StrategyState):
        self.cfg = cfg
        self.st  = state

    def update(self, price: float) -> tuple[str, TradeDirection]:
        """
        Feed current price and return (scenario_id, direction).
        Called on each tick/bar close.
        """
        st = self.st
        pdh, pdl = st.pdh, st.pdl

        # ── MASTER SECTION 1: Open ABOVE PDH ──────────────────────
        if st.open_above_pdh:
            # Fix 1: Track whether the market has retraced back below PDH
            # AFTER the open.  The first time it does, open_drive_pure
            # becomes False — this is the definitive gate separating
            # Scenario 1.1 (pure open drive, no retest) from
            # Scenario 1.2 (open → dip → recross = test-and-drive).
            if price < pdh:
                st.touched_below_pdh = True
                st.open_drive_pure   = False   # Fix 1: mark as retest, not pure drive

            if st.touched_below_pdh:
                # 1.2 — retraced below PDH, now recrossing upward → LONG
                # Only valid if this is a retest (open_drive_pure == False)
                if price >= pdh and not st.open_drive_pure:
                    if self.cfg.active_scenarios.get("1.2", True):
                        return "1.2", TradeDirection.LONG
                # 1.4 — dipped below PDH and now breaks PDL → SHORT
                if price < pdl:
                    if self.cfg.active_scenarios.get("1.4", True):
                        return "1.4", TradeDirection.SHORT
                # 1.3 — dipped below PDH, stays between PDH and PDL → NO TRADE

            # 1.1 — opened above PDH, open_drive_pure still True,
            # price has never retraced below PDH → pure open drive → NO TRADE
            # (the market may continue up, but there is no confirmed retest setup)
            return "", TradeDirection.NONE

        # ── MASTER SECTION 2: Open INSIDE range ────────────────────
        elif st.open_inside:
            if price > pdh:
                if self.cfg.active_scenarios.get("2.1", True):
                    return "2.1", TradeDirection.LONG
            elif price < pdl:
                if self.cfg.active_scenarios.get("2.3", True):
                    return "2.3", TradeDirection.SHORT
            return "", TradeDirection.NONE

        # ── MASTER SECTION 3: Open BELOW PDL ───────────────────────
        elif st.open_below_pdl:
            # Fix 1 (mirror of Section 1): open_drive_pure tracks whether
            # the market has bounced back above PDL after opening below it.
            # If it has, open_drive_pure = False → eligible for 3.2 retest.
            if price > pdl:
                st.touched_above_pdl = True
                st.open_drive_pure   = False  # Fix 1: retest detected

            if st.touched_above_pdl:
                # 3.2 — bounced above PDL, now re-breaks PDL downward → SHORT
                # Only valid after confirmed retest (open_drive_pure == False)
                if price < pdl and not st.open_drive_pure:
                    if self.cfg.active_scenarios.get("3.2", True):
                        return "3.2", TradeDirection.SHORT
                # 3.4 — bounced above PDL and keeps going to break PDH → LONG
                if price > pdh:
                    if self.cfg.active_scenarios.get("3.4", True):
                        return "3.4", TradeDirection.LONG
                # 3.3 — moved above PDL but stayed inside range → NO TRADE

            # 3.1 — opened below PDL, never bounced above it → pure drive → NO TRADE
            return "", TradeDirection.NONE

        return "", TradeDirection.NONE


# ─────────────────────────────────────────────
# FILTER ENGINE
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# SEASONALITY HELPERS  (Fix 4)
# ─────────────────────────────────────────────

def _is_month_end_day(dt: datetime, n: int = 2) -> bool:
    """
    Returns True if dt is within the first or last n trading days of
    the calendar month. Uses a simple calendar-day heuristic: last n
    calendar days of the month, or first n calendar days.
    A proper implementation would use a trading-calendar library, but
    this approximation covers the vast majority of rebalancing windows.
    """
    import calendar
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    # last n calendar days of month
    if dt.day >= last_day - n:
        return True
    # first n calendar days of month
    if dt.day <= n:
        return True
    return False


def _effective_contracts(cfg: StrategyConfig, now_et: datetime) -> int:
    """
    Fix 4: Returns the contract count adjusted for month-end sizing.
    On flagged days the position size is scaled by month_end_size_pct.
    """
    if cfg.month_end_filter and _is_month_end_day(now_et):
        scaled = max(1, int(cfg.contracts * cfg.month_end_size_pct))
        log.info("Month-end day detected — sizing reduced to %d contract(s)", scaled)
        return scaled
    return cfg.contracts


class FilterEngine:
    """
    Applies all entry filters:
      - Fix 1 (Time Window): precise hh:mm boundary enforcement per window
      - Fix 2 (ADX):         properly wired regime filter
      - PDC bias, RSI, VWAP, Volume, daily P&L limits
      - Fix 4 (Seasonality): Friday no-new-entries gate
    """

    # Fix 1: canonical window definitions (start_hhmm, end_hhmm) in ET
    TIME_WINDOWS: dict = {
        "OPEN_DRIVE":  ((9, 30), (10, 30)),
        "MORNING":     ((9, 30), (12,  0)),
        "POWER_HOUR":  ((14, 0), (16,  0)),
        "ALL":         ((9, 30), (16,  0)),
    }

    def __init__(self, cfg: StrategyConfig, state: StrategyState):
        self.cfg = cfg
        self.st  = state

    # ── Public entry point ───────────────────────────────────────────

    def check_all(self, direction: TradeDirection, last_bar: dict,
                  scenario_id: str = "") -> tuple[bool, str]:
        """Returns (pass, reason_string)."""
        now_et = datetime.now(ET)
        cur_min = now_et.hour * 60 + now_et.minute

        # ── Fix 1: Precise Time Window Enforcement ───────────────────
        # Look up the configured window; fall back to ALL if unknown.
        window = self.TIME_WINDOWS.get(self.cfg.time_window, self.TIME_WINDOWS["ALL"])
        start_min = window[0][0] * 60 + window[0][1]
        end_min   = window[1][0] * 60 + window[1][1]
        if not (start_min <= cur_min < end_min):
            return False, (
                f"Outside {self.cfg.time_window} window "
                f"({window[0][0]:02d}:{window[0][1]:02d}–"
                f"{window[1][0]:02d}:{window[1][1]:02d} ET)"
            )

        # ── Fix 4: Seasonality — Friday no-new-entries after cutoff ──
        # Scenario 1.2 / 3.2 retests can still trigger late on Fridays;
        # this gate prevents opening NEW trades after the cutoff time.
        if self.cfg.friday_early_exit and now_et.weekday() == 4:  # 4 = Friday
            cutoff = self.cfg.friday_exit_hour * 60 + self.cfg.friday_exit_minute
            if cur_min >= cutoff:
                return False, (
                    f"Friday early-exit gate: no new entries after "
                    f"{self.cfg.friday_exit_hour:02d}:{self.cfg.friday_exit_minute:02d} ET"
                )

        # ── Fix 4: Month-end — flag day, but allow trades (sizing handled at entry)
        if self.cfg.month_end_filter and _is_month_end_day(now_et):
            log.info("Month-end / month-start day — reduced sizing will apply at entry")

        # ── Max Entries Per Day ──────────────────────────────────────
        if self.st.entries_today >= self.cfg.max_entries_per_day:
            return False, f"Max entries reached ({self.cfg.max_entries_per_day})"

        # ── Daily Loss Limit ──────────────────────────────────────────
        limit = -self.cfg.account_capital * self.cfg.daily_loss_limit_pct
        if self.st.daily_pnl <= limit:
            return False, f"Daily loss limit hit (P&L: {self.st.daily_pnl:.2f})"

        # ── Daily Profit Lock ─────────────────────────────────────────
        profit_lock = self.cfg.account_capital * self.cfg.daily_profit_lock_pct
        if self.st.daily_pnl >= profit_lock:
            return False, f"Daily profit lock hit (P&L: {self.st.daily_pnl:.2f})"

        # ── PDC Bias Filter ───────────────────────────────────────────
        if self.cfg.pdc_filter and self.st.pdc > 0:
            cur_price = float(last_bar["Close"])
            if direction == TradeDirection.LONG and cur_price < self.st.pdc:
                return False, f"Price {cur_price:.2f} below PDC {self.st.pdc:.2f} — no long bias"
            if direction == TradeDirection.SHORT and cur_price > self.st.pdc:
                return False, f"Price {cur_price:.2f} above PDC {self.st.pdc:.2f} — no short bias"

        bars = self.st.intraday_bars
        if not bars:
            return True, "OK"

        closes = [float(b["Close"]) for b in bars]

        # ── Fix 2: ADX Market Regime Filter (now runs BEFORE momentum filters)
        # Doc Section 7.1: skip all signals when ADX < threshold (ranging market).
        # Previously this check existed but was unreachable due to an early return.
        if self.cfg.adx_filter and len(bars) > self.cfg.adx_period + 1:
            adx = calc_adx(bars, self.cfg.adx_period)
            if adx < self.cfg.adx_min:
                return False, (
                    f"ADX {adx:.1f} < {self.cfg.adx_min} — "
                    f"ranging market, trend continuation trades suppressed"
                )

        # ── RSI Momentum Filter ───────────────────────────────────────
        if self.cfg.rsi_filter and len(closes) > self.cfg.rsi_period:
            rsi = calc_rsi(closes, self.cfg.rsi_period)
            if direction == TradeDirection.LONG and rsi < self.cfg.rsi_long_min:
                return False, f"RSI {rsi:.1f} < {self.cfg.rsi_long_min} for LONG"
            if direction == TradeDirection.SHORT and rsi > self.cfg.rsi_short_max:
                return False, f"RSI {rsi:.1f} > {self.cfg.rsi_short_max} for SHORT"

        # ── Fix 2: VWAP Alignment + Slope Filter ──────────────────────
        # Two independently-toggled modes:
        #
        # vwap_filter (position-based): price above VWAP for longs,
        # below for shorts. Fast but flickers on choppy flat-VWAP days.
        #
        # vwap_slope_filter (slope-based): VWAP[now] > VWAP[n bars ago]
        # for longs (trending up), VWAP[now] < VWAP[n bars ago] for shorts.
        # Stable gate — VWAP is cumulative so its slope reverses slowly,
        # eliminating the flicker caused by price oscillating across a
        # flat VWAP line. Use alone or in combination with vwap_filter.
        if self.cfg.vwap_filter and self.st.session_vwap > 0:
            cur_price = float(last_bar["Close"])
            if direction == TradeDirection.LONG and cur_price < self.st.session_vwap:
                return False, f"Price {cur_price:.2f} below VWAP {self.st.session_vwap:.2f} for LONG"
            if direction == TradeDirection.SHORT and cur_price > self.st.session_vwap:
                return False, f"Price {cur_price:.2f} above VWAP {self.st.session_vwap:.2f} for SHORT"

        if self.cfg.vwap_slope_filter:
            history = self.st.vwap_history
            n = self.cfg.vwap_slope_period
            if len(history) >= n + 1:
                vwap_now  = history[-1]
                vwap_prev = history[-(n + 1)]
                slope_up   = vwap_now > vwap_prev
                slope_down = vwap_now < vwap_prev
                if direction == TradeDirection.LONG and not slope_up:
                    return False, (
                        f"VWAP slope flat/down ({vwap_prev:.2f}→{vwap_now:.2f} "
                        f"over {n} bars) — no upward trend for LONG"
                    )
                if direction == TradeDirection.SHORT and not slope_down:
                    return False, (
                        f"VWAP slope flat/up ({vwap_prev:.2f}→{vwap_now:.2f} "
                        f"over {n} bars) — no downward trend for SHORT"
                    )

        # ── Volume Confirmation Filter ────────────────────────────────
        if self.cfg.vol_filter and len(bars) >= self.cfg.vol_avg_period:
            vols = [float(b.get("TotalVolume", 0)) for b in bars]
            avg_vol = sum(vols[-self.cfg.vol_avg_period:]) / self.cfg.vol_avg_period
            cur_vol = float(last_bar.get("TotalVolume", 0))
            if cur_vol < avg_vol * self.cfg.vol_filter_mult:
                return False, f"Volume {cur_vol:.0f} < {self.cfg.vol_filter_mult}x avg {avg_vol:.0f}"

        return True, "OK"


# ─────────────────────────────────────────────
# STOP LOSS CALCULATOR
# ─────────────────────────────────────────────

class StopCalculator:
    def __init__(self, cfg: StrategyConfig, state: StrategyState):
        self.cfg = cfg
        self.st  = state

    def calculate(self, direction: TradeDirection, entry_price: float,
                  scenario_id: str) -> float:
        cfg = self.cfg
        bars = self.st.intraday_bars
        pdh, pdl = self.st.pdh, self.st.pdl

        sl_type = cfg.sl_type
        sign = 1 if direction == TradeDirection.LONG else -1  # direction of stop from entry

        if sl_type == StopLossType.FIXED_PTS:
            stop = entry_price - sign * cfg.sl_pts
        elif sl_type == StopLossType.FIXED_PCT:
            stop = entry_price * (1 - sign * cfg.sl_pct)
        elif sl_type == StopLossType.DYNAMIC_ATR:
            atr = calc_atr(bars, cfg.atr_period) if bars else cfg.sl_pts
            stop = entry_price - sign * (cfg.sl_atr_mult * atr)
        elif sl_type == StopLossType.DYNAMIC_FIBO:
            # Stop beyond the fibo retracement of the prior day's range
            day_range = pdh - pdl
            fibo_dist = day_range * cfg.sl_fibo_lvl
            # For longs, stop = entry - distance; for shorts, entry + distance
            stop = entry_price - sign * fibo_dist
        elif sl_type == StopLossType.LEVEL_REBREAK:
            # Stop is just below/above the entry reference level
            if direction == TradeDirection.LONG:
                stop = pdh - cfg.tick_size  # 1 tick below PDH breakout
            else:
                stop = pdl + cfg.tick_size  # 1 tick above PDL breakdown
        else:
            stop = entry_price - sign * cfg.sl_pts

        log.info("Stop calculated → type=%s direction=%s entry=%.2f stop=%.2f",
                 sl_type.value, direction.value, entry_price, stop)
        return round(stop, 2)


# ─────────────────────────────────────────────
# PROFIT TARGET CALCULATOR
# ─────────────────────────────────────────────

class TargetCalculator:
    def __init__(self, cfg: StrategyConfig, state: StrategyState):
        self.cfg = cfg
        self.st  = state

    def calculate(self, direction: TradeDirection, entry_price: float,
                  stop_price: float) -> float:
        cfg = self.cfg
        bars = self.st.intraday_bars
        sign = 1 if direction == TradeDirection.LONG else -1
        risk = abs(entry_price - stop_price)

        pt_type = cfg.pt_type

        if pt_type == ProfitTargetType.FIXED_RR:
            target = entry_price + sign * (risk * cfg.pt_rr_ratio)
        elif pt_type == ProfitTargetType.FIXED_PTS:
            target = entry_price + sign * cfg.sl_pts * cfg.pt_rr_ratio
        elif pt_type == ProfitTargetType.ATR_MULT:
            atr = calc_atr(bars, cfg.atr_period) if bars else risk
            target = entry_price + sign * (cfg.pt_atr_mult * atr)
        elif pt_type == ProfitTargetType.FIBO_EXT:
            day_range = self.st.pdh - self.st.pdl
            target = entry_price + sign * (day_range * cfg.pt_fibo_ext)
        elif pt_type in (ProfitTargetType.TRAILING_PTS, ProfitTargetType.TRAILING_ATR,
                         ProfitTargetType.PARTIAL):
            # No fixed target; return a very far level and manage dynamically
            target = entry_price + sign * 9999
        else:
            target = entry_price + sign * (risk * cfg.pt_rr_ratio)

        log.info("Target calculated → type=%s entry=%.2f target=%.2f (risk=%.2f)",
                 pt_type.value, entry_price, target, risk)
        return round(target, 2)


# ─────────────────────────────────────────────
# TRADE MANAGER
# ─────────────────────────────────────────────

class TradeManager:
    """Manages open trade lifecycle: trailing stops, BE, partial exits."""

    def __init__(self, cfg: StrategyConfig, state: StrategyState, client: TradeStationClient):
        self.cfg = cfg
        self.st  = state
        self.api = client
        self._best_price: float = 0.0  # Best price seen since entry (for trailing)

    def on_new_bar(self, bar: dict):
        """Call this on each new 1-minute bar close while in a trade."""
        st = self.st
        if not st.in_trade:
            return

        price   = float(bar["Close"])
        high    = float(bar["High"])
        low     = float(bar["Low"])
        direction = st.direction
        sign = 1 if direction == TradeDirection.LONG else -1
        cfg = self.cfg

        # ── Fix 4: Friday forced exit ─────────────────────────────────
        # Close all open trades at the configured ET cutoff on Fridays.
        # Doc Section 7.8: "Fridays often exhibit profit-taking /
        # mean-reversion behaviour — avoid holding trend trades into
        # Friday close."
        if cfg.friday_early_exit:
            now_et = datetime.now(ET)
            if now_et.weekday() == 4:  # Friday
                cutoff_min = cfg.friday_exit_hour * 60 + cfg.friday_exit_minute
                if now_et.hour * 60 + now_et.minute >= cutoff_min:
                    log.info(
                        "Fix 4: Friday forced exit at %02d:%02d ET — closing %s",
                        now_et.hour, now_et.minute, direction.value
                    )
                    self._exit_trade("Market", price)
                    return

        # Track best price achieved
        check_price = high if direction == TradeDirection.LONG else low
        if self._best_price == 0:
            self._best_price = st.entry_price
        if direction == TradeDirection.LONG and check_price > self._best_price:
            self._best_price = check_price
        elif direction == TradeDirection.SHORT and check_price < self._best_price:
            self._best_price = check_price

        profit_pts = sign * (price - st.entry_price)
        risk_pts   = abs(st.entry_price - st.stop_price)

        # ── Breakeven Stop ──────────────────────────────────────────
        if cfg.be_trigger > 0 and not st.be_triggered:
            if profit_pts >= risk_pts * cfg.be_trigger:
                new_stop = st.entry_price + sign * cfg.slippage_pts
                if (direction == TradeDirection.LONG and new_stop > st.stop_price) or \
                   (direction == TradeDirection.SHORT and new_stop < st.stop_price):
                    self._move_stop(new_stop)
                    st.be_triggered = True
                    log.info("Breakeven stop triggered → stop moved to %.2f", new_stop)

        # ── Partial Profit Exit ──────────────────────────────────────
        if cfg.partial_enabled and not st.partial_done:
            if profit_pts >= risk_pts * cfg.partial_rr:
                partial_qty = max(1, int(cfg.contracts * cfg.partial_pct))
                action = "SELL" if direction == TradeDirection.LONG else "BUYTOCOVER"
                self.api.place_order(action, partial_qty, order_type="Market")
                st.partial_done = True
                log.info("Partial profit taken → %d contracts @ market", partial_qty)

        # ── Trailing Stop ────────────────────────────────────────────
        pt_type = cfg.pt_type
        if pt_type in (ProfitTargetType.TRAILING_PTS, ProfitTargetType.PARTIAL):
            trail_dist = cfg.trail_pts
            new_stop = self._best_price - sign * trail_dist
            self._maybe_tighten_stop(new_stop, direction)
        elif pt_type == ProfitTargetType.TRAILING_ATR:
            bars = st.intraday_bars
            atr = calc_atr(bars, cfg.atr_period) if len(bars) > cfg.atr_period else cfg.trail_pts
            trail_dist = atr * cfg.trail_atr_mult
            new_stop = self._best_price - sign * trail_dist
            self._maybe_tighten_stop(new_stop, direction)

        # ── Fix 4: ATR Target Refresh ────────────────────────────────
        # When atr_target_dynamic=True and pt_type=ATR_MULT, recalculate
        # the profit target on every bar using the current ATR.
        # This prevents the "frozen target too close" problem when
        # volatility expands after entry — the target moves out with it.
        # Only widens the target (never tightens it against the trade).
        if cfg.atr_target_dynamic and cfg.pt_type == ProfitTargetType.ATR_MULT:
            bars = st.intraday_bars
            if len(bars) > cfg.atr_period:
                live_atr     = calc_atr(bars, cfg.atr_period)
                new_target   = round(st.entry_price + sign * cfg.pt_atr_mult * live_atr, 2)
                # Only update if new target is further from entry (never pull it closer)
                target_moved = (
                    (direction == TradeDirection.LONG  and new_target > st.target_price) or
                    (direction == TradeDirection.SHORT and new_target < st.target_price)
                )
                if target_moved:
                    old_target = st.target_price
                    st.target_price = new_target
                    # Cancel and replace live target limit order if one exists
                    if st.target_order_id:
                        try:
                            self.api.replace_stop(st.target_order_id, new_target)
                        except Exception as e:
                            log.warning("Could not refresh ATR target order: %s", e)
                    log.info(
                        "Fix 4: ATR target refreshed %.2f → %.2f (live ATR=%.2f)",
                        old_target, new_target, live_atr
                    )

        # ── Fix 3: LEVEL_REBREAK — reactive close-basis exit ────────
        # The document defines Level Rebreak as a DYNAMIC exit:
        # "Exit as soon as a trade occurs back below/above the entry
        # reference level."  The original code referenced st.st (typo)
        # and used a high/low wick trigger. This version uses the bar
        # CLOSE so that brief wicks through the level (noise) do not
        # prematurely exit the trade — only a sustained close back
        # through the breakout level invalidates the setup.
        if cfg.sl_type == StopLossType.LEVEL_REBREAK:
            # Reference level: PDH for longs (breakout level), PDL for shorts
            ref_level = st.pdh if direction == TradeDirection.LONG else st.pdl
            level_reclaimed = (
                (direction == TradeDirection.LONG  and price < ref_level) or
                (direction == TradeDirection.SHORT and price > ref_level)
            )
            if level_reclaimed:
                log.info(
                    "LEVEL_REBREAK exit: close %.2f reclaimed ref %.2f — exiting %s",
                    price, ref_level, direction.value
                )
                self._exit_trade("Market", price)
                return  # trade closed; skip further processing this bar

    def _maybe_tighten_stop(self, new_stop: float, direction: TradeDirection):
        st = self.st
        if direction == TradeDirection.LONG and new_stop > st.stop_price:
            self._move_stop(new_stop)
        elif direction == TradeDirection.SHORT and new_stop < st.stop_price:
            self._move_stop(new_stop)

    def _move_stop(self, new_stop: float):
        st = self.st
        new_stop = round(new_stop, 2)
        if st.stop_order_id:
            try:
                self.api.replace_stop(st.stop_order_id, new_stop)
                st.stop_price = new_stop
            except Exception as e:
                log.error("Failed to update stop order %s: %s", st.stop_order_id, e)
        else:
            st.stop_price = new_stop

    def _exit_trade(self, order_type: str = "Market", price: float = None):
        """Close the open position."""
        st = self.st
        if not st.in_trade:
            return
        action = "SELL" if st.direction == TradeDirection.LONG else "BUYTOCOVER"
        self.api.place_order(action, self.cfg.contracts, order_type=order_type)
        pnl = self._calculate_pnl(price or st.entry_price)
        st.daily_pnl += pnl
        log.info("Trade closed → direction=%s pnl=%.2f daily_pnl=%.2f",
                 st.direction.value, pnl, st.daily_pnl)
        st.in_trade = False
        st.direction = TradeDirection.NONE
        self._best_price = 0.0

    def _calculate_pnl(self, exit_price: float) -> float:
        st  = self.st
        cfg = self.cfg
        qty = cfg.contracts
        sign = 1 if st.direction == TradeDirection.LONG else -1
        gross = sign * (exit_price - st.entry_price) * cfg.point_value * qty
        # Fix 3: Per-contract slippage — entry and exit each walk the book
        # so we apply effective_slippage_pts(qty) for both sides (x2).
        if cfg.include_slippage:
            avg_slip_pts  = cfg.effective_slippage_pts(qty)
            slippage_cost = avg_slip_pts * cfg.point_value * qty * 2  # entry + exit
            return gross - slippage_cost - cfg.commission_rt * qty
        return gross


# ─────────────────────────────────────────────
# MAIN STRATEGY ENGINE
# ─────────────────────────────────────────────

class ESStrategy:
    """
    Main orchestrator. Wires together all components and drives
    the real-time decision loop.
    """

    def __init__(self, config: StrategyConfig):
        self.cfg    = config
        self.api    = TradeStationClient(config)
        self.state  = StrategyState()
        self.scenario_eng = ScenarioEngine(config, self.state)
        self.filter_eng   = FilterEngine(config, self.state)
        self.stop_calc    = StopCalculator(config, self.state)
        self.target_calc  = TargetCalculator(config, self.state)
        self.trade_mgr    = TradeManager(config, self.state, self.api)

    # ── Startup ──────────────────────────────────────────────────────

    def initialize(self):
        """Fetch previous day's OHLC and set up reference levels."""
        log.info("Initializing strategy for %s", self.cfg.symbol)
        daily_bars = self.api.get_bars(self.cfg.symbol, unit="Daily", interval=1, bars_back=2)
        if len(daily_bars) < 2:
            raise RuntimeError("Could not fetch previous day's bars")

        prev = daily_bars[-2]  # Second-to-last = previous complete session
        self.state.pdh = float(prev["High"])
        self.state.pdl = float(prev["Low"])
        self.state.pdo = float(prev["Open"])
        self.state.pdc = float(prev["Close"])
        log.info("Reference levels → PDH=%.2f PDL=%.2f PDO=%.2f PDC=%.2f",
                 self.state.pdh, self.state.pdl, self.state.pdo, self.state.pdc)

    def set_opening_bias(self, open_price: float):
        """Classify the opening price relative to PDH/PDL."""
        st = self.state
        if open_price > st.pdh:
            st.open_above_pdh = True
            log.info("Opening bias: ABOVE PDH (Section 1)")
        elif open_price < st.pdl:
            st.open_below_pdl = True
            log.info("Opening bias: BELOW PDL (Section 3)")
        else:
            st.open_inside = True
            log.info("Opening bias: INSIDE range (Section 2)")

    # ── Main Loop ────────────────────────────────────────────────────

    def run(self):
        """Streaming main loop — connects to TradeStation and processes bars."""
        self.initialize()

        log.info("Entering streaming bar loop for %s", self.cfg.symbol)
        first_bar = True

        for bar in self.api.stream_bars(self.cfg.symbol, unit="Minute", interval=1):
            # Skip historical backfill bars from stream
            if bar.get("IsEndOfHistory") is False and bar.get("IsRealtime") is False:
                continue

            # New trading day: reset state
            bar_time = datetime.fromisoformat(bar["TimeStamp"].replace("Z", "+00:00"))
            bar_time_et = bar_time.astimezone(ET)
            if bar_time_et.hour == 9 and bar_time_et.minute == 30:
                self.state.reset_daily()
                self.initialize()
                first_bar = True

            self.state.intraday_bars.append(bar)

            # Update VWAP and Fix 2: maintain rolling VWAP history for slope check
            self.state.session_vwap = calc_vwap(self.state.intraday_bars)
            self.state.vwap_history.append(self.state.session_vwap)

            # Set opening bias on the first RTH bar
            if first_bar:
                self.set_opening_bias(float(bar["Open"]))
                first_bar = False

            # Manage open trade
            if self.state.in_trade:
                self.trade_mgr.on_new_bar(bar)
                continue

            # Check for new trade signal
            price = float(bar["Close"])
            scenario_id, direction = self.scenario_eng.update(price)

            if direction == TradeDirection.NONE or not scenario_id:
                continue

            # Apply filters
            passed, reason = self.filter_eng.check_all(direction, bar, scenario_id)
            if not passed:
                log.info("Signal %s %s FILTERED → %s", scenario_id, direction.value, reason)
                continue

            log.info("SIGNAL → Scenario %s | %s", scenario_id, direction.value)
            self._enter_trade(direction, price, scenario_id)

    # ── Trade Entry ──────────────────────────────────────────────────

    def _enter_trade(self, direction: TradeDirection, signal_price: float, scenario_id: str):
        cfg = self.cfg
        st  = self.state

        # Determine entry price with slippage
        sign = 1 if direction == TradeDirection.LONG else -1
        entry_price = signal_price + sign * cfg.slippage_pts

        # Calculate stop and target
        stop_price   = self.stop_calc.calculate(direction, entry_price, scenario_id)
        target_price = self.target_calc.calculate(direction, entry_price, stop_price)

        # Validate minimum R:R
        risk   = abs(entry_price - stop_price)
        reward = abs(target_price - entry_price)
        if risk > 0 and reward / risk < cfg.pt_rr_ratio:
            log.info("Trade rejected: R:R %.2f below minimum %.2f", reward/risk, cfg.pt_rr_ratio)
            return

        # --- Fix 4: Month-end adjusted position size ---
        effective_qty = _effective_contracts(cfg, datetime.now(ET))

        # --- Place Entry Order ---
        action = "BUY" if direction == TradeDirection.LONG else "SELLSHORT"
        result = self.api.place_order(action, effective_qty, order_type="Market")
        st.entry_order_id = result.get("OrderID", "")

        # --- Place Stop Order ---
        stop_action = "SELL" if direction == TradeDirection.LONG else "BUYTOCOVER"
        stop_result = self.api.place_order(
            stop_action, effective_qty,
            order_type="StopMarket", stop_price=stop_price
        )
        st.stop_order_id = stop_result.get("OrderID", "")

        # --- Place Target Order (if fixed target) ---
        if cfg.pt_type in (ProfitTargetType.FIXED_RR, ProfitTargetType.FIXED_PTS,
                           ProfitTargetType.ATR_MULT, ProfitTargetType.FIBO_EXT):
            target_result = self.api.place_order(
                stop_action, effective_qty,
                order_type="Limit", limit_price=target_price
            )
            st.target_order_id = target_result.get("OrderID", "")

        # Update state
        st.in_trade     = True
        st.direction    = direction
        st.entry_price  = entry_price
        st.stop_price   = stop_price
        st.target_price = target_price
        st.scenario_id  = scenario_id
        st.partial_done = False
        st.be_triggered = False
        st.entries_today += 1

        log.info(
            "TRADE ENTERED → %s | Scenario %s | Entry: %.2f | Stop: %.2f | "
            "Target: %.2f | Risk: %.2f pts | R:R: 1:%.1f",
            direction.value, scenario_id, entry_price,
            stop_price, target_price, risk, reward / risk
        )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cfg = StrategyConfig(
        # ── SET YOUR CREDENTIALS ──
        # Either set here or via environment variables:
        #   TS_CLIENT_ID, TS_CLIENT_SECRET, TS_REFRESH_TOKEN, TS_ACCOUNT_ID
        paper_trading = True,           # ← ALWAYS start on paper!

        # ── STRATEGY PARAMETERS (customize for your backtest findings) ──
        symbol         = "ESH25",
        contracts      = 1,
        sl_type        = StopLossType.FIXED_PTS,
        sl_pts         = 10.0,
        entry_type     = EntryType.CONFIRM_LVL,
        vol_filter     = True,
        rsi_filter     = True,
        vwap_filter    = True,
        pt_type        = ProfitTargetType.FIXED_RR,
        pt_rr_ratio    = 3.0,
        max_entries_per_day = 2,
        daily_loss_limit_pct = 0.02,
        account_capital = 50000.0,
    )

    strategy = ESStrategy(cfg)
    strategy.run()
