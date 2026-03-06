"""
ES Strategy Backtester
======================
Runs parameter-sweep backtests over historical ES bar data fetched
from TradeStation API v3.

USAGE:
    python backtest.py

OUTPUT:
    backtest_results.csv — all parameter combinations and their metrics
    best_config.json      — top configuration by Expectancy
"""

import csv
import json
import logging
import os
import itertools
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
# ENVIRONMENT — load .env before anything else
# ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env from the current working directory
except ImportError:
    pass  # dotenv not installed — fall back to real environment variables

from es_strategy import (
    StrategyConfig, TradeStationClient, StopCalculator, TargetCalculator,
    ScenarioEngine, FilterEngine, StrategyState, TradeDirection,
    StopLossType, EntryType, ProfitTargetType,
    calc_atr, calc_rsi, calc_vwap, calc_adx,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────────────────────
# BACKTEST RESULT
# ─────────────────────────────────────────────

@dataclass
class BacktestResult:
    params: dict
    net_pnl:        float = 0.0
    win_rate:       float = 0.0
    avg_win:        float = 0.0
    avg_loss:       float = 0.0
    expectancy:     float = 0.0
    max_drawdown:   float = 0.0
    profit_factor:  float = 0.0
    sharpe:         float = 0.0
    num_trades:     int   = 0
    avg_trades_day: float = 0.0


# ─────────────────────────────────────────────
# PARAMETER GRID
# ─────────────────────────────────────────────

PARAM_GRID = {
    "sl_type":    [StopLossType.FIXED_PTS, StopLossType.DYNAMIC_ATR, StopLossType.LEVEL_REBREAK],
    "sl_pts":     [5, 8, 10, 12, 15],
    "sl_atr_mult":[1.0, 1.5, 2.0, 2.5],
    "entry_type": [EntryType.MARKET_FIRST, EntryType.CONFIRM_LVL, EntryType.CONFIRM_LAST],
    "vol_filter": [False, True],
    "rsi_filter": [False, True],
    "vwap_filter":[False, True],
    "pdc_filter": [False, True],           # FIX 1: PDC bias filter sweep
    "pt_type":    [ProfitTargetType.FIXED_RR, ProfitTargetType.TRAILING_PTS, ProfitTargetType.ATR_MULT],
    "pt_rr_ratio":[2.0, 3.0, 4.0, 5.0],
    "pt_atr_mult":[2.0, 3.0, 4.0],
    "trail_pts":  [5, 8, 10, 15],
    "max_entries_per_day": [1, 2, 3],
    "time_window":["ALL", "OPEN_DRIVE", "MORNING", "POWER_HOUR"],
    "include_slippage": [False, True],     # FIX 3: theoretical vs realistic
}

# To run a quick test, use a minimal grid:
QUICK_GRID = {
    "sl_type":    [StopLossType.FIXED_PTS],
    "sl_pts":     [10],
    "sl_atr_mult":[1.5],
    "entry_type": [EntryType.CONFIRM_LVL],
    "vol_filter": [True],
    "rsi_filter": [True],
    "vwap_filter":[True],
    "pdc_filter": [True],                  # FIX 1
    "pt_type":    [ProfitTargetType.FIXED_RR],
    "pt_rr_ratio":[3.0],
    "pt_atr_mult":[3.0],
    "trail_pts":  [10],
    "max_entries_per_day": [2],
    "time_window":["ALL"],
    "include_slippage": [True],            # FIX 3
}


# ─────────────────────────────────────────────
# FIX 2: IQR STATISTICAL STOP LOSS ANALYSER
# ─────────────────────────────────────────────

class StatisticalStopAnalyser:
    """
    FIX 2 — Implements the document's "Statistically Derived Stop Loss" method.

    Workflow (as specified in Section 3.1 Option A):
      1. Run a first-pass backtest with a generous fixed stop to capture all losses.
      2. Collect the raw loss distribution (in ES points).
      3. Remove outliers using the IQR method (Q3 + 1.5 * IQR).
      4. Calculate mean, Q1, Q2 (median), Q3 of the cleaned distribution.
      5. Return these as candidate stop loss values for the main sweep.

    Usage:
        analyser = StatisticalStopAnalyser(base_cfg, bars_1min, bars_daily)
        candidates = analyser.derive_stops()
        # candidates = {"mean": 9.2, "q1": 6.1, "q2": 8.4, "q3": 12.3}
        # Feed these into PARAM_GRID["sl_pts"] for the main sweep.
    """

    # Wide initial stop to capture the full loss distribution
    DISCOVERY_STOP_PTS = 50.0

    def __init__(self, base_cfg: StrategyConfig, bars_1min: list[dict], bars_daily: list[dict]):
        self.base_cfg   = base_cfg
        self.bars_1min  = bars_1min
        self.bars_daily = bars_daily

    def derive_stops(self) -> dict:
        """
        Returns a dict of statistically derived stop candidates (in ES points).
        Keys: "mean", "q1", "q2" (median), "q3"
        """
        import statistics

        # First pass: run with wide stop to capture all losing trades
        discovery_cfg = StrategyConfig(
            **{**self.base_cfg.__dict__,
               "sl_type":          StopLossType.FIXED_PTS,
               "sl_pts":           self.DISCOVERY_STOP_PTS,
               "include_slippage": False,   # gross losses only for distribution
               "vol_filter":       False,
               "rsi_filter":       False,
               "vwap_filter":      False,
               "pdc_filter":       False,
            }
        )

        bt = SingleBacktest(discovery_cfg, self.bars_1min, self.bars_daily)
        result = bt.run()

        # We need the raw trade list — re-run capturing individual trades
        raw_losses_pts = self._collect_raw_losses(discovery_cfg)

        if len(raw_losses_pts) < 4:
            log.warning("StatisticalStopAnalyser: fewer than 4 losing trades found. "
                        "Cannot derive reliable stops. Using document defaults.")
            return {"mean": 10.0, "q1": 6.0, "q2": 10.0, "q3": 13.0}

        # Remove outliers: filter beyond Q3 + 1.5 * IQR (document spec)
        raw_losses_pts.sort()
        n = len(raw_losses_pts)
        q1_raw = raw_losses_pts[n // 4]
        q3_raw = raw_losses_pts[(3 * n) // 4]
        iqr    = q3_raw - q1_raw
        upper_fence = q3_raw + 1.5 * iqr

        cleaned = [x for x in raw_losses_pts if x <= upper_fence]
        removed = len(raw_losses_pts) - len(cleaned)
        log.info("StatisticalStopAnalyser: %d raw losses, %d outliers removed (fence=%.2f pts)",
                 len(raw_losses_pts), removed, upper_fence)

        if len(cleaned) < 4:
            cleaned = raw_losses_pts  # fall back if over-filtered

        cleaned.sort()
        nc = len(cleaned)
        mean_sl = statistics.mean(cleaned)
        q1_sl   = cleaned[nc // 4]
        q2_sl   = statistics.median(cleaned)
        q3_sl   = cleaned[(3 * nc) // 4]

        candidates = {
            "mean": round(mean_sl, 2),
            "q1":   round(q1_sl,   2),
            "q2":   round(q2_sl,   2),
            "q3":   round(q3_sl,   2),
        }

        log.info(
            "Statistically derived stop levels (ES pts) → "
            "Mean=%.2f | Q1=%.2f | Q2=%.2f | Q3=%.2f",
            candidates["mean"], candidates["q1"],
            candidates["q2"],   candidates["q3"]
        )
        return candidates

    def _collect_raw_losses(self, cfg: StrategyConfig) -> list[float]:
        """Re-runs the backtest and returns a list of losing trade sizes in ES points."""
        raw_losses = []
        sessions = self._group_by_session(self.bars_1min)

        for date_str, day_bars in sessions.items():
            prev = self._get_prev_day(date_str)
            if not prev:
                continue

            state = StrategyState()
            state.pdh = float(prev["High"])
            state.pdl = float(prev["Low"])
            state.pdo = float(prev["Open"])
            state.pdc = float(prev["Close"])

            scen_eng  = ScenarioEngine(cfg, state)
            stop_calc = StopCalculator(cfg, state)
            target_calc = TargetCalculator(cfg, state)

            in_trade = False
            direction = TradeDirection.NONE
            entry_price = stop_price = target_price = 0.0
            first_bar = True
            entries = 0

            for bar in day_bars:
                state.intraday_bars.append(bar)
                state.session_vwap = calc_vwap(state.intraday_bars)
                price = float(bar["Close"])
                high  = float(bar["High"])
                low   = float(bar["Low"])

                if first_bar:
                    op = float(bar["Open"])
                    if op > state.pdh:      state.open_above_pdh = True
                    elif op < state.pdl:    state.open_below_pdl = True
                    else:                   state.open_inside    = True
                    first_bar = False

                if in_trade:
                    sign = 1 if direction == TradeDirection.LONG else -1
                    stop_hit   = (direction == TradeDirection.LONG  and low  <= stop_price) or \
                                 (direction == TradeDirection.SHORT and high >= stop_price)
                    target_hit = (direction == TradeDirection.LONG  and high >= target_price) or \
                                 (direction == TradeDirection.SHORT and low  <= target_price)

                    if stop_hit:
                        loss_pts = abs(entry_price - stop_price)
                        raw_losses.append(loss_pts)
                        in_trade = False
                        direction = TradeDirection.NONE
                    elif target_hit and target_price < 9000:
                        in_trade = False
                        direction = TradeDirection.NONE
                    continue

                if entries >= cfg.max_entries_per_day:
                    continue

                scenario_id, sig_dir = scen_eng.update(price)
                if sig_dir == TradeDirection.NONE or not scenario_id:
                    continue

                sign = 1 if sig_dir == TradeDirection.LONG else -1
                entry_price  = price + sign * cfg.slippage_pts
                stop_price   = stop_calc.calculate(sig_dir, entry_price, scenario_id)
                target_price = target_calc.calculate(sig_dir, entry_price, stop_price)
                in_trade = True
                direction = sig_dir
                entries += 1
                state.entries_today = entries

        return raw_losses

    def _group_by_session(self, bars):
        sessions: dict = {}
        for b in bars:
            ts = b["TimeStamp"][:10]
            sessions.setdefault(ts, []).append(b)
        return dict(sorted(sessions.items()))

    def _get_prev_day(self, date_str):
        for bar in reversed(self.bars_daily):
            if bar["TimeStamp"][:10] < date_str:
                return bar
        return None




class SingleBacktest:
    """
    Runs the strategy logic over a list of 1-minute historical bars
    WITHOUT placing real orders, computing P&L on each trade.
    """

    def __init__(self, cfg: StrategyConfig, bars_1min: list[dict], bars_daily: list[dict]):
        self.cfg = cfg
        self.bars_1min  = bars_1min
        self.bars_daily = bars_daily

    def run(self) -> BacktestResult:
        cfg = self.cfg
        trades = []          # list of (entry_price, exit_price, direction)
        daily_pnls = {}      # date -> pnl

        # Group 1-min bars by session date
        sessions = self._group_by_session(self.bars_1min)

        for date_str, day_bars in sessions.items():
            # Find previous day's OHLC
            prev = self._get_prev_day(date_str)
            if not prev:
                continue

            state = StrategyState()
            state.pdh = float(prev["High"])
            state.pdl = float(prev["Low"])
            state.pdo = float(prev["Open"])
            state.pdc = float(prev["Close"])

            scen_eng   = ScenarioEngine(cfg, state)
            filter_eng = FilterEngine(cfg, state)
            stop_calc  = StopCalculator(cfg, state)
            target_calc= TargetCalculator(cfg, state)

            in_trade = False
            direction = TradeDirection.NONE
            entry_price = stop_price = target_price = 0.0
            best_price = 0.0
            first_bar = True
            daily_pnl = 0.0
            entries = 0

            for bar in day_bars:
                state.intraday_bars.append(bar)
                state.session_vwap = calc_vwap(state.intraday_bars)
                price = float(bar["Close"])
                high  = float(bar["High"])
                low   = float(bar["Low"])

                if first_bar:
                    open_price = float(bar["Open"])
                    if open_price > state.pdh:
                        state.open_above_pdh = True
                    elif open_price < state.pdl:
                        state.open_below_pdl = True
                    else:
                        state.open_inside = True
                    first_bar = False

                # ── Manage open trade ──────────────────────────────────
                if in_trade:
                    sign = 1 if direction == TradeDirection.LONG else -1

                    # Track best price
                    check = high if direction == TradeDirection.LONG else low
                    if sign > 0 and check > best_price:
                        best_price = check
                    elif sign < 0 and check < best_price:
                        best_price = check

                    # Trailing stop update
                    if cfg.pt_type in (ProfitTargetType.TRAILING_PTS, ProfitTargetType.PARTIAL):
                        new_stop = best_price - sign * cfg.trail_pts
                        if sign > 0 and new_stop > stop_price:
                            stop_price = new_stop
                        elif sign < 0 and new_stop < stop_price:
                            stop_price = new_stop
                    elif cfg.pt_type == ProfitTargetType.TRAILING_ATR:
                        atr = calc_atr(state.intraday_bars, cfg.atr_period)
                        new_stop = best_price - sign * (atr * cfg.trail_atr_mult)
                        if sign > 0 and new_stop > stop_price:
                            stop_price = new_stop
                        elif sign < 0 and new_stop < stop_price:
                            stop_price = new_stop

                    # Fix 4: Friday forced exit in backtest
                    bar_dt = datetime.fromisoformat(
                        bar["TimeStamp"].replace("Z", "+00:00")
                    ).astimezone(ET)
                    friday_force_exit = False
                    if cfg.friday_early_exit and bar_dt.weekday() == 4:
                        cutoff = cfg.friday_exit_hour * 60 + cfg.friday_exit_minute
                        if bar_dt.hour * 60 + bar_dt.minute >= cutoff:
                            friday_force_exit = True

                    # Check stop hit
                    stop_hit = (direction == TradeDirection.LONG and low <= stop_price) or \
                               (direction == TradeDirection.SHORT and high >= stop_price)
                    # Check target hit
                    target_hit = (direction == TradeDirection.LONG and high >= target_price) or \
                                 (direction == TradeDirection.SHORT and low <= target_price)
                    # Fix 3: Level Rebreak — close-basis check
                    level_rebreak_exit = False
                    if cfg.sl_type == StopLossType.LEVEL_REBREAK:
                        ref = state.pdh if direction == TradeDirection.LONG else state.pdl
                        level_rebreak_exit = (
                            (direction == TradeDirection.LONG  and price < ref) or
                            (direction == TradeDirection.SHORT and price > ref)
                        )

                    exit_price = None
                    if friday_force_exit:
                        exit_price = price
                    elif stop_hit:
                        exit_price = stop_price
                    elif target_hit and target_price < 9000:
                        exit_price = target_price
                    elif level_rebreak_exit:
                        exit_price = price

                    if exit_price:
                        pnl = self._calc_pnl(direction, entry_price, exit_price)
                        trades.append({"pnl": pnl, "entry": entry_price, "exit": exit_price,
                                       "direction": direction.value, "date": date_str})
                        daily_pnl += pnl
                        in_trade = False
                        direction = TradeDirection.NONE
                    continue

                # ── Check for new signal ───────────────────────────────
                if entries >= cfg.max_entries_per_day:
                    continue

                scenario_id, sig_dir = scen_eng.update(price)
                if sig_dir == TradeDirection.NONE or not scenario_id:
                    continue

                passed, _ = filter_eng.check_all(sig_dir, bar)
                if not passed:
                    continue

                # Enter
                sign = 1 if sig_dir == TradeDirection.LONG else -1

                # Fix 4: Friday no-new-entries gate in backtest
                bar_dt_entry = datetime.fromisoformat(
                    bar["TimeStamp"].replace("Z", "+00:00")
                ).astimezone(ET)
                if cfg.friday_early_exit and bar_dt_entry.weekday() == 4:
                    cutoff = cfg.friday_exit_hour * 60 + cfg.friday_exit_minute
                    if bar_dt_entry.hour * 60 + bar_dt_entry.minute >= cutoff:
                        continue  # no new entries after Friday cutoff

                entry_price = price + sign * cfg.slippage_pts
                stop_price  = stop_calc.calculate(sig_dir, entry_price, scenario_id)
                target_price= target_calc.calculate(sig_dir, entry_price, stop_price)

                risk = abs(entry_price - stop_price)
                reward = abs(target_price - entry_price)
                if risk > 0 and reward / risk < cfg.pt_rr_ratio:
                    continue

                in_trade = True
                direction = sig_dir
                best_price = entry_price
                entries += 1
                state.entries_today = entries

            daily_pnls[date_str] = daily_pnl

        return self._compile_result(trades, daily_pnls, cfg)

    # ── Helpers ───────────────────────────────────────────────────────

    def _calc_pnl(self, direction: TradeDirection, entry: float, exit_p: float) -> float:
        sign = 1 if direction == TradeDirection.LONG else -1
        gross = sign * (exit_p - entry) * self.cfg.point_value * self.cfg.contracts
        # FIX 3: Respect slippage toggle for theoretical vs realistic comparison
        if self.cfg.include_slippage:
            return gross \
                   - self.cfg.slippage_pts * self.cfg.point_value * self.cfg.contracts * 2 \
                   - self.cfg.commission_rt * self.cfg.contracts
        return gross

    def _group_by_session(self, bars: list[dict]) -> dict:
        sessions: dict = {}
        for b in bars:
            ts = b["TimeStamp"][:10]  # YYYY-MM-DD
            sessions.setdefault(ts, []).append(b)
        return dict(sorted(sessions.items()))

    def _get_prev_day(self, date_str: str) -> Optional[dict]:
        for bar in reversed(self.bars_daily):
            bar_date = bar["TimeStamp"][:10]
            if bar_date < date_str:
                return bar
        return None

    def _compile_result(self, trades: list[dict], daily_pnls: dict, cfg: StrategyConfig) -> BacktestResult:
        if not trades:
            return BacktestResult(params=cfg.__dict__)

        pnls = [t["pnl"] for t in trades]
        wins  = [p for p in pnls if p > 0]
        losses= [p for p in pnls if p <= 0]

        net_pnl       = sum(pnls)
        win_rate      = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win       = sum(wins) / len(wins) if wins else 0
        avg_loss      = sum(losses) / len(losses) if losses else 0
        expectancy    = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss)
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

        # Max Drawdown
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cum += p
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd

        # Sharpe (daily P&L)
        daily_vals = list(daily_pnls.values())
        if len(daily_vals) > 1:
            import statistics
            mean_d = statistics.mean(daily_vals)
            std_d  = statistics.stdev(daily_vals)
            sharpe = (mean_d / std_d * (252**0.5)) if std_d else 0
        else:
            sharpe = 0

        num_days = max(len(daily_pnls), 1)
        params_summary = {
            "sl_type":         cfg.sl_type.value,
            "sl_pts":          cfg.sl_pts,
            "sl_atr_mult":     cfg.sl_atr_mult,
            "entry_type":      cfg.entry_type.value,
            "vol_filter":      cfg.vol_filter,
            "rsi_filter":      cfg.rsi_filter,
            "vwap_filter":     cfg.vwap_filter,
            "pdc_filter":      cfg.pdc_filter,        # FIX 1
            "pt_type":         cfg.pt_type.value,
            "pt_rr_ratio":     cfg.pt_rr_ratio,
            "trail_pts":       cfg.trail_pts,
            "max_entries":     cfg.max_entries_per_day,
            "time_window":     cfg.time_window,
            "include_slippage":cfg.include_slippage,  # FIX 3
        }

        return BacktestResult(
            params        = params_summary,
            net_pnl       = round(net_pnl, 2),
            win_rate      = round(win_rate, 2),
            avg_win       = round(avg_win, 2),
            avg_loss      = round(avg_loss, 2),
            expectancy    = round(expectancy, 2),
            max_drawdown  = round(max_dd, 2),
            profit_factor = round(profit_factor, 3),
            sharpe        = round(sharpe, 3),
            num_trades    = len(trades),
            avg_trades_day= round(len(trades) / num_days, 2),
        )


# ─────────────────────────────────────────────
# BACKTESTER RUNNER
# ─────────────────────────────────────────────

class BacktestRunner:
    """Fetches data once, then iterates over parameter grid."""

    OUTPUT_CSV = "backtest_results.csv"
    BEST_JSON  = "best_config.json"

    def __init__(self, base_cfg: StrategyConfig, use_quick_grid: bool = True):
        self.base_cfg = base_cfg
        self.grid = QUICK_GRID if use_quick_grid else PARAM_GRID
        self.api  = TradeStationClient(base_cfg)

    def fetch_data(self, bars_back_daily: int = 252, bars_back_minute: int = 50000):
        """Pull historical data once before the parameter sweep."""
        log.info("Fetching %d daily bars for %s", bars_back_daily, self.base_cfg.symbol)
        self.bars_daily = self.api.get_bars(
            self.base_cfg.symbol, unit="Daily", interval=1, bars_back=bars_back_daily
        )
        log.info("Fetching 1-minute bars (%d bars back)...", bars_back_minute)
        self.bars_1min = self.api.get_bars(
            self.base_cfg.symbol, unit="Minute", interval=1, bars_back=bars_back_minute
        )
        log.info("Data fetched: %d daily bars, %d minute bars",
                 len(self.bars_daily), len(self.bars_1min))

    def run_sweep(self, derive_statistical_stops: bool = True):
        """
        Run the full parameter sweep.

        FIX 2: If derive_statistical_stops=True, runs a first-pass backtest
        to auto-derive IQR-based stop loss candidates, then injects them
        into the sweep's sl_pts values alongside the manual ones.

        FIX 3: Results CSV now includes a 'include_slippage' column so you
        can directly compare theoretical-max vs realistic performance rows.
        """
        grid = dict(self.grid)  # copy so we don't mutate the original

        # ── FIX 2: Auto-derive statistical stop levels ──────────────────
        if derive_statistical_stops and StopLossType.FIXED_PTS in grid.get("sl_type", []):
            log.info("Running statistical stop derivation (IQR method)...")
            analyser = StatisticalStopAnalyser(self.base_cfg, self.bars_1min, self.bars_daily)
            stat_stops = analyser.derive_stops()
            # Merge auto-derived values into sl_pts, deduplicated
            existing = set(grid.get("sl_pts", []))
            for label, val in stat_stops.items():
                rounded = round(val * 4) / 4  # round to nearest 0.25 (ES tick)
                if rounded not in existing:
                    existing.add(rounded)
                    log.info("  Adding stat-derived sl_pts (%s): %.2f pts", label, rounded)
            grid["sl_pts"] = sorted(existing)
            log.info("Final sl_pts to sweep: %s", grid["sl_pts"])

        results = []
        keys   = list(grid.keys())
        values = list(grid.values())
        combos = list(itertools.product(*values))
        log.info("Running %d parameter combinations...", len(combos))

        for i, combo in enumerate(combos, 1):
            params = dict(zip(keys, combo))
            cfg = StrategyConfig(**{**self.base_cfg.__dict__, **params})
            bt = SingleBacktest(cfg, self.bars_1min, self.bars_daily)
            result = bt.run()
            results.append(result)
            if i % 10 == 0 or i == len(combos):
                log.info("[%d/%d] Net P&L: $%.2f | Win Rate: %.1f%% | Expectancy: $%.2f | Slippage: %s",
                         i, len(combos), result.net_pnl, result.win_rate, result.expectancy,
                         params.get("include_slippage", True))

        self._save_results(results)

        # ── FIX 3: Print slippage sensitivity summary ───────────────────
        self._print_slippage_sensitivity(results)

        return results

    def _print_slippage_sensitivity(self, results: list):
        """
        FIX 3: Compare theoretical-max vs realistic P&L across matched
        parameter sets, showing how execution-sensitive the strategy is.
        """
        import statistics as stats
        with_slip    = [r for r in results if r.params.get("include_slippage") is True]
        without_slip = [r for r in results if r.params.get("include_slippage") is False]

        if not with_slip or not without_slip:
            return

        avg_realistic    = stats.mean(r.net_pnl for r in with_slip)
        avg_theoretical  = stats.mean(r.net_pnl for r in without_slip)
        avg_gap          = avg_theoretical - avg_realistic
        pct_gap          = (avg_gap / abs(avg_theoretical) * 100) if avg_theoretical else 0

        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log.info("  SLIPPAGE SENSITIVITY ANALYSIS (Fix 3)")
        log.info("  Avg theoretical P&L (no slippage): $%.2f", avg_theoretical)
        log.info("  Avg realistic P&L (with slippage): $%.2f", avg_realistic)
        log.info("  Avg execution cost drag:           $%.2f (%.1f%%)", avg_gap, pct_gap)
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def _save_results(self, results: list[BacktestResult]):
        if not results:
            return

        # CSV
        fieldnames = list(results[0].params.keys()) + [
            "net_pnl", "win_rate", "avg_win", "avg_loss", "expectancy",
            "max_drawdown", "profit_factor", "sharpe", "num_trades", "avg_trades_day"
        ]
        with open(self.OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {**r.params,
                       "net_pnl": r.net_pnl, "win_rate": r.win_rate,
                       "avg_win": r.avg_win, "avg_loss": r.avg_loss,
                       "expectancy": r.expectancy, "max_drawdown": r.max_drawdown,
                       "profit_factor": r.profit_factor, "sharpe": r.sharpe,
                       "num_trades": r.num_trades, "avg_trades_day": r.avg_trades_day}
                writer.writerow(row)
        log.info("Results saved to %s", self.OUTPUT_CSV)

        # Best config
        best = max(results, key=lambda r: r.expectancy)
        with open(self.BEST_JSON, "w") as f:
            json.dump({"params": best.params, "metrics": {
                "net_pnl": best.net_pnl, "win_rate": best.win_rate,
                "expectancy": best.expectancy, "max_drawdown": best.max_drawdown,
                "profit_factor": best.profit_factor, "sharpe": best.sharpe,
                "num_trades": best.num_trades,
            }}, f, indent=2)
        log.info("Best config saved to %s (Expectancy: $%.2f)", self.BEST_JSON, best.expectancy)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    base_cfg = StrategyConfig(
        paper_trading  = True,
        symbol         = "ESH25",
        contracts      = 1,
        account_capital= 50000.0,
    )

    runner = BacktestRunner(base_cfg, use_quick_grid=True)  # ← set False for full grid
    runner.fetch_data(bars_back_daily=252, bars_back_minute=50000)
    runner.run_sweep()
