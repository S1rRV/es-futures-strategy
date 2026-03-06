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
    "pt_type":    [ProfitTargetType.FIXED_RR, ProfitTargetType.TRAILING_PTS, ProfitTargetType.ATR_MULT],
    "pt_rr_ratio":[2.0, 3.0, 4.0, 5.0],
    "pt_atr_mult":[2.0, 3.0, 4.0],
    "trail_pts":  [5, 8, 10, 15],
    "max_entries_per_day": [1, 2, 3],
    "time_window":["ALL", "OPEN_DRIVE", "MORNING", "POWER_HOUR"],
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
    "pt_type":    [ProfitTargetType.FIXED_RR],
    "pt_rr_ratio":[3.0],
    "pt_atr_mult":[3.0],
    "trail_pts":  [10],
    "max_entries_per_day": [2],
    "time_window":["ALL"],
}


# ─────────────────────────────────────────────
# SINGLE BACKTEST RUN
# ─────────────────────────────────────────────

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

                    # Check stop hit
                    stop_hit = (direction == TradeDirection.LONG and low <= stop_price) or \
                               (direction == TradeDirection.SHORT and high >= stop_price)
                    # Check target hit
                    target_hit = (direction == TradeDirection.LONG and high >= target_price) or \
                                 (direction == TradeDirection.SHORT and low <= target_price)

                    exit_price = None
                    if stop_hit:
                        exit_price = stop_price
                    elif target_hit and target_price < 9000:
                        exit_price = target_price

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
        return gross - self.cfg.slippage_pts * self.cfg.point_value * self.cfg.contracts * 2 \
               - self.cfg.commission_rt * self.cfg.contracts

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
            "sl_type":    cfg.sl_type.value,
            "sl_pts":     cfg.sl_pts,
            "sl_atr_mult":cfg.sl_atr_mult,
            "entry_type": cfg.entry_type.value,
            "vol_filter": cfg.vol_filter,
            "rsi_filter": cfg.rsi_filter,
            "vwap_filter":cfg.vwap_filter,
            "pt_type":    cfg.pt_type.value,
            "pt_rr_ratio":cfg.pt_rr_ratio,
            "trail_pts":  cfg.trail_pts,
            "max_entries":cfg.max_entries_per_day,
            "time_window":cfg.time_window,
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

    def run_sweep(self):
        results = []
        keys = list(self.grid.keys())
        values = list(self.grid.values())
        combos = list(itertools.product(*values))
        log.info("Running %d parameter combinations...", len(combos))

        for i, combo in enumerate(combos, 1):
            params = dict(zip(keys, combo))
            cfg = StrategyConfig(**{**self.base_cfg.__dict__, **params})
            bt = SingleBacktest(cfg, self.bars_1min, self.bars_daily)
            result = bt.run()
            results.append(result)
            if i % 10 == 0 or i == len(combos):
                log.info("[%d/%d] Net P&L: $%.2f | Win Rate: %.1f%% | Expectancy: $%.2f",
                         i, len(combos), result.net_pnl, result.win_rate, result.expectancy)

        self._save_results(results)
        return results

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
