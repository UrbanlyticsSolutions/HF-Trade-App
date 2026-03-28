"""
Complete Strategy Optimizer — Parallel Optuna + OOS Validation

Optimizes 13 dimensions using Bayesian TPE search:
  - Asymmetric CALL/PUT profit targets & stop losses
  - RSI thresholds
  - Hold time, Kelly, risk controls
  - Entry window, min contracts, post-loss strategy

Architecture:
  - Optuna TPE sampler (Bayesian, efficient exploration)
  - Parallel workers via Optuna n_jobs (thread-based, shared data)
  - Data loaded once, shared across all trials
  - 2-phase: optimize on 2025, validate best on 2026

Usage:
  python scripts/optimize_full.py [--trials N] [--workers N]
"""
import sys
sys.path.insert(0, '.')

import io
import os
import json
import time
import contextlib
import argparse
import numpy as np
import pandas as pd
import optuna

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from backtest.engine import Backtest0DTE, TradeConfig
from core.risk_manager import RiskConfig
from config import defaults as cfg
from config.config_manager import save_optimization_run, apply_run

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# GLOBAL DATA CACHE (loaded once, shared across threads)
# ============================================================
_DATA_CACHE = {}


def load_data_cached(period='2025'):
    """Load and cache backtest data."""
    global _DATA_CACHE
    if period in _DATA_CACHE:
        return _DATA_CACHE[period]

    cap = cfg.initial_capital()
    # Need a base engine just to load data
    base_bt = Backtest0DTE(TradeConfig(), RiskConfig(), initial_capital=cap)

    if period == '2025':
        u, o, f = base_bt.load_data('2025-01-01', '2025-12-31')
    elif period == '2026':
        u, o, f = base_bt.load_data('2026-01-02', '2026-02-21')
    else:
        raise ValueError(f"Unknown period: {period}")

    v = base_bt.compute_historical_volatility(u)
    dt_idx = base_bt._opt_by_date_time
    td_idx = base_bt._opt_by_ticker_date

    _DATA_CACHE[period] = (u, o, f, v, dt_idx, td_idx)
    return _DATA_CACHE[period]


def _load_strategy_json():
    """Load strategy.json once and cache it."""
    if not hasattr(_load_strategy_json, '_cache'):
        with open('config/strategy.json') as f:
            _load_strategy_json._cache = json.load(f)['trade_config']
    return _load_strategy_json._cache


def run_single_config(params, period='2025'):
    """Run a single backtest config and return metrics."""
    u, o, f, v, dt_idx, td_idx = load_data_cached(period)
    cap = cfg.initial_capital()

    # Load ALL regime params from strategy.json (the optimized baseline)
    sj = _load_strategy_json()

    # Build trade config: optimizer controls core params, regime params from strategy.json
    tc = TradeConfig(
        strategy='momentum',
        trade_start_hour=params.get('trade_start_hour', 9),
        trade_start_minute=params.get('trade_start_minute', 35),
        trade_end_hour=params.get('trade_end_hour', 15),
        trade_end_minute=0,
        rsi_call_threshold=params.get('rsi_call', 70),
        rsi_put_threshold=params.get('rsi_put', 35),
        profit_target_pct=params.get('call_pt', 0.50),
        stop_loss_pct=params.get('call_sl', 0.35),
        call_profit_target_pct=params.get('call_pt', 0.50),
        put_profit_target_pct=params.get('put_pt', 0.50),
        call_stop_loss_pct=params.get('call_sl', 0.35),
        put_stop_loss_pct=params.get('put_sl', 0.35),
        max_hold_bars=params.get('call_hold', 16),
        call_max_hold_bars=params.get('call_hold', 16),
        put_max_hold_bars=params.get('put_hold', 16),
        min_option_price=0.50,
        max_option_price=2.00,
        use_adaptive_exits=False,
        use_trailing_stop=False,
        use_time_decay_exit=False,
        use_quick_exit=False,
        use_ml_filter=False,
        skip_day_filter=True,
        min_contracts_per_trade=params.get('min_contracts', 1),
        post_loss_strategy=params.get('post_loss', 'none'),
        post_loss_momentum_threshold=params.get('momentum_threshold', 0.10),
        # Regime classification params — from strategy.json (already optimized)
        use_regime_detection=True,
        regime_lookback_days=sj.get('regime_lookback_days', 7),
        regime_vol_percentile=sj.get('regime_vol_percentile', 0.45),
        regime_trend_percentile=sj.get('regime_trend_percentile', 0.1),
        regime_up_day_pct=sj.get('regime_up_day_pct', 0.85),
        regime_dn_day_pct=sj.get('regime_dn_day_pct', 0.8),
        regime_momentum_threshold=sj.get('regime_momentum_threshold', 0.006),
        regime_high_vol_percentile=sj.get('regime_high_vol_percentile', 0.7),
        regime_adx_trend_threshold=sj.get('regime_adx_trend_threshold', 30.0),
        regime_skip_first_bar=sj.get('regime_skip_first_bar', True),
        regime_rsi_buffer=params.get('regime_rsi_buffer', sj.get('regime_rsi_buffer', 10)),
        regime_size_reduction=params.get('regime_size_reduction', sj.get('regime_size_reduction', 0.1)),
        # Regime type adjustments — from strategy.json
        steady_up_size_reduction=sj.get('steady_up_size_reduction', 0.1),
        steady_up_skip_puts=sj.get('steady_up_skip_puts', True),
        steady_up_rsi_buffer=sj.get('steady_up_rsi_buffer', 0),
        steady_up_call_pt_override=sj.get('steady_up_call_pt_override'),
        steady_dn_size_reduction=sj.get('steady_dn_size_reduction', 0.3),
        steady_dn_skip_calls=sj.get('steady_dn_skip_calls', True),
        steady_dn_rsi_buffer=sj.get('steady_dn_rsi_buffer', 4),
        steady_dn_put_pt_override=sj.get('steady_dn_put_pt_override', 0.35),
        choppy_size_reduction=sj.get('choppy_size_reduction', 0.3),
        choppy_skip_first_bar=sj.get('choppy_skip_first_bar', False),
        choppy_rsi_buffer=sj.get('choppy_rsi_buffer', 2),
        choppy_tighter_stop_pct=sj.get('choppy_tighter_stop_pct', 0.2),
        volatile_size_reduction=sj.get('volatile_size_reduction', 0.0),
        volatile_stop_buffer_pct=sj.get('volatile_stop_buffer_pct', 0.15),
        volatile_pt_buffer_pct=sj.get('volatile_pt_buffer_pct', 0.2),
        trending_skip_counter=sj.get('trending_skip_counter', False),
        trending_hold_buffer=sj.get('trending_hold_buffer', 2),
        # PUT filter params
        put_adaptive_filter=True,
        put_loss_streak_threshold=params.get('put_loss_streak', 1),
        put_adaptive_cooldown=params.get('put_cooldown', 4),
        call_adaptive_filter=params.get('call_adaptive_filter', False),
        call_loss_streak_threshold=params.get('call_loss_streak', 2),
        call_adaptive_cooldown=params.get('call_cooldown', 3),
        put_min_rsi=params.get('put_min_rsi', 15),
        put_skip_days=[0],
        put_min_entry_minutes=615,
        put_filter_require_uptrend=True,
        # Direction-aware loss escalation
        use_direction_loss_escalation=params.get('use_direction_loss_escalation', False),
        direction_loss_window=params.get('direction_loss_window', 3),
        direction_loss_threshold=params.get('direction_loss_threshold', 2),
        direction_loss_cooldown=params.get('direction_loss_cooldown', 3),
        consec_loss_rsi_buffer=params.get('consec_loss_rsi_buffer', 0),
    )

    rc = RiskConfig(
        max_risk_per_trade_pct=params.get('max_risk', 0.02),
        max_position_pct=0.07,
        max_daily_losses=params.get('max_daily_losses', 999),
        max_consecutive_losses=params.get('max_consec_losses', 3),
        max_daily_loss_pct=params.get('max_daily_loss_pct', 0.008),
        max_trades_per_day=999,
        reduce_size_at_dd_pct=0.99,
    )

    bt = Backtest0DTE(tc, rc, initial_capital=cap)
    bt._opt_by_date_time = dt_idx
    bt._opt_by_ticker_date = td_idx
    bt.risk_manager.set_kelly(params.get('kelly', 0.06))

    # Suppress stdout from engine's internal prints
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        trades = bt.run_no_ml(u, o, f, v, verbose=False)
    finally:
        sys.stdout = old_stdout

    return compute_metrics(trades, cap)


def compute_metrics(trades, cap):
    """Compute comprehensive metrics from trade list."""
    if not trades or len(trades) < 5:
        return None

    total_pnl = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    n = len(trades)
    wr = wins / n * 100
    ret = total_pnl / cap * 100

    # Max drawdown
    peak = cap
    max_dd = 0
    for t in trades:
        c = t.capital
        if c > peak:
            peak = c
        dd = (peak - c) / peak
        if dd > max_dd:
            max_dd = dd

    # Sharpe / Sortino
    rets = [t.pnl / max(t.capital - t.pnl, 1) for t in trades]
    mu = np.mean(rets)
    sigma = np.std(rets) if len(rets) > 1 else 1
    sharpe = (mu * 252) / (sigma * np.sqrt(252)) if sigma > 0 else 0
    down = [r for r in rets if r < 0]
    ds = np.std(down) if len(down) > 1 else 1
    sortino = (mu * 252) / (ds * np.sqrt(252)) if ds > 0 else 0

    # Profit factor
    gp = sum(t.pnl for t in trades if t.pnl > 0)
    gl = abs(sum(t.pnl for t in trades if t.pnl <= 0)) or 0.01
    pf = gp / gl

    # Calmar
    calmar = ret / (max_dd * 100) if max_dd > 0.001 else 0

    # CALL / PUT breakdown
    call_trades = [t for t in trades if t.direction == 'CALL']
    put_trades = [t for t in trades if t.direction == 'PUT']
    call_pnl = sum(t.pnl for t in call_trades)
    put_pnl = sum(t.pnl for t in put_trades)
    call_wr = sum(1 for t in call_trades if t.pnl > 0) / max(len(call_trades), 1) * 100
    put_wr = sum(1 for t in put_trades if t.pnl > 0) / max(len(put_trades), 1) * 100

    # Exit reason distribution
    profit_exits = sum(1 for t in trades if t.exit_reason == 'PROFIT')
    stop_exits = sum(1 for t in trades if t.exit_reason == 'STOP')
    time_exits = sum(1 for t in trades if t.exit_reason == 'TIME')

    # Daily P&L stats
    daily_pnl = {}
    for t in trades:
        daily_pnl[t.date] = daily_pnl.get(t.date, 0) + t.pnl
    profitable_days = sum(1 for p in daily_pnl.values() if p > 0)
    total_days = len(daily_pnl)
    day_win_pct = profitable_days / total_days * 100 if total_days > 0 else 0

    return {
        'trades': n, 'wins': wins, 'wr': round(wr, 1),
        'pnl': round(total_pnl, 0), 'ret': round(ret, 1),
        'max_dd': round(max_dd * 100, 1),
        'sharpe': round(sharpe, 2), 'sortino': round(sortino, 2),
        'pf': round(pf, 2), 'calmar': round(calmar, 1),
        'call_trades': len(call_trades), 'put_trades': len(put_trades),
        'call_pnl': round(call_pnl, 0), 'put_pnl': round(put_pnl, 0),
        'call_wr': round(call_wr, 1), 'put_wr': round(put_wr, 1),
        'profit_exits': profit_exits, 'stop_exits': stop_exits, 'time_exits': time_exits,
        'profit_exit_pct': round(profit_exits / n * 100, 1),
        'days': total_days, 'day_win_pct': round(day_win_pct, 1),
    }


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def create_objective(period='2025'):
    """Create Optuna objective function."""

    def objective(trial):
        params = {
            # Asymmetric profit targets
            # Analysis: CALLs hit PT only 14.7% (too high?), PUTs hit 37.6% (working well)
            'call_pt': trial.suggest_float('call_pt', 0.25, 0.55, step=0.05),
            'put_pt': trial.suggest_float('put_pt', 0.30, 0.60, step=0.05),

            # Asymmetric stop losses
            'call_sl': trial.suggest_float('call_sl', 0.20, 0.45, step=0.05),
            'put_sl': trial.suggest_float('put_sl', 0.25, 0.45, step=0.05),

            # RSI thresholds
            'rsi_call': trial.suggest_int('rsi_call', 65, 80, step=5),
            'rsi_put': trial.suggest_int('rsi_put', 25, 40, step=5),

            # Hold time (separate for CALL / PUT)
            # Analysis: CALLs avg 14.7 bars (TIME 76.5%), PUTs avg 12.1 bars
            'call_hold': trial.suggest_int('call_hold', 10, 20, step=2),
            'put_hold': trial.suggest_int('put_hold', 8, 18, step=2),

            # Position sizing
            'kelly': trial.suggest_float('kelly', 0.04, 0.08, step=0.01),
            'max_risk': trial.suggest_float('max_risk', 0.02, 0.05, step=0.01),
            'min_contracts': trial.suggest_int('min_contracts', 1, 2),

            # Risk controls
            'max_daily_losses': trial.suggest_categorical('max_daily_losses', [1, 2, 3, 999]),
            'max_consec_losses': trial.suggest_int('max_consec_losses', 2, 5),
            'max_daily_loss_pct': trial.suggest_float('max_daily_loss_pct', 0.005, 0.03, step=0.005),

            # Post-loss strategy
            'post_loss': trial.suggest_categorical('post_loss', [
                'none', 'momentum_confirm', 'multi_confirm'
            ]),

            # Post-loss momentum threshold
            'momentum_threshold': trial.suggest_float('momentum_threshold', 0.02, 0.20, step=0.02),

            # Regime detection tuning
            'regime_rsi_buffer': trial.suggest_int('regime_rsi_buffer', 0, 15, step=5),
            'regime_size_reduction': trial.suggest_float('regime_size_reduction', 0.0, 0.30, step=0.05),

            # PUT adaptive filter tuning
            'put_loss_streak': trial.suggest_int('put_loss_streak', 1, 3),
            'put_cooldown': trial.suggest_int('put_cooldown', 2, 8, step=2),
            'put_min_rsi': trial.suggest_int('put_min_rsi', 10, 25, step=5),

            # Trading window
            'trade_start_hour': 9,
            'trade_start_minute': 35,
            'trade_end_hour': trial.suggest_int('trade_end_hour', 13, 15),
        }

        m = run_single_config(params, period=period)
        if m is None:
            return float('-inf')

        # Reject configs that are clearly too restrictive or too risky
        if m['trades'] < 100:
            return float('-inf')  # must generate enough trades to be robust
        if m['max_dd'] > 20:
            return float('-inf')  # hard DD cap

        # Composite score: RETURN-DOMINANT with risk guardrails
        ret_score = m['ret'] / 2100  # baseline ~2000%+, so 2100 normalizes
        sharpe_score = m['sharpe'] / 8
        pf_score = min(m['pf'] / 4, 1)
        wr_score = (m['wr'] - 50) / 25  # penalize below 50%
        trade_score = min(m['trades'] / 1200, 1.0)  # reward more trades (baseline ~1300)
        dd_penalty = max(0, (m['max_dd'] - 12) / 10)  # only penalize DD > 12%

        composite = (
            0.40 * ret_score +      # return is king
            0.20 * sharpe_score +    # risk-adjusted return
            0.15 * trade_score +     # reward trade volume (robustness)
            0.10 * pf_score +
            0.10 * wr_score +
            0.05 * (1 - dd_penalty)  # light DD penalty, only >10%
        )

        # Store all metrics as user attributes for analysis
        for k, v in m.items():
            trial.set_user_attr(k, v)
        for k, v in params.items():
            trial.set_user_attr(f'p_{k}', v)

        return composite

    return objective


# ============================================================
# MAIN OPTIMIZATION (single-process, in-memory — avoids SQLite contention)
# ============================================================

# ============================================================
# BASELINE CONFIG (the config to beat — from position sizing sweep)
# ============================================================
BASELINE_PARAMS = {
    'call_pt': 0.50,
    'put_pt': 0.50,
    'call_sl': 0.35,
    'put_sl': 0.35,
    'rsi_call': 70,
    'rsi_put': 35,
    'call_hold': 16,
    'put_hold': 16,
    'kelly': 0.06,
    'max_risk': 0.02,
    'min_contracts': 1,
    'max_daily_losses': 999,
    'max_consec_losses': 3,
    'max_daily_loss_pct': 0.008,
    'post_loss': 'none',
    'momentum_threshold': 0.10,
    # Regime & filter params
    'regime_rsi_buffer': 10,
    'regime_size_reduction': 0.10,
    'put_loss_streak': 1,
    'put_cooldown': 4,
    'put_min_rsi': 15,
    'trade_start_hour': 9,
    'trade_start_minute': 35,
    'trade_end_hour': 15,
}


def run_baseline(period='2025'):
    """Run the baseline config and return metrics."""
    m = run_single_config(BASELINE_PARAMS, period=period)
    return m


def run_optimization(n_trials=300, n_workers=6, period='2025'):
    """Run optimization with Optuna TPE sampler (thread-parallel, in-memory)."""
    cap = cfg.initial_capital()

    print('=' * 70)
    print(f'  COMPLETE STRATEGY OPTIMIZATION')
    print(f'  Optuna TPE | {n_trials} trials | {n_workers} workers')
    print(f'  Period: {period} | ${cap:,.0f} starting capital')
    print(f'  20 dimensions: PT(C/P), SL(C/P), RSI(C/P), Hold(C/P), kelly,')
    print(f'    risk, min_ct, daily_losses, consec_losses, dll%, post_loss,')
    print(f'    mom_thr, regime_rsi_buf, regime_size_red, put_streak/cool/rsi, end_hr')
    print('=' * 70)

    # Load data once (shared across all trials in-process)
    print('\n  Loading data...')
    t0 = time.time()
    load_data_cached(period)
    print(f'  Data loaded in {time.time() - t0:.1f}s')

    # Run baseline first for comparison
    print('\n  Running BASELINE config...')
    baseline_m = run_baseline(period=period)
    if baseline_m:
        print(f'  BASELINE: {baseline_m["trades"]} trades, '
              f'ret={baseline_m["ret"]:+.1f}%, dd={baseline_m["max_dd"]:.1f}%, '
              f'sharpe={baseline_m["sharpe"]:.2f}, wr={baseline_m["wr"]:.1f}%')
        print(f'    CALL: {baseline_m["call_trades"]} trades ${baseline_m["call_pnl"]:+,.0f} '
              f'| PUT: {baseline_m["put_trades"]} trades ${baseline_m["put_pnl"]:+,.0f}')
    else:
        print('  BASELINE: failed (no trades)')

    # In-memory study (no SQLite = no locking issues)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=30),
    )

    # Enqueue baseline as first trial so optimizer knows the bar to beat
    study.enqueue_trial(BASELINE_PARAMS)

    # Progress callback
    best_score = float('-inf')
    completed = [0]
    t_start = time.time()

    def progress_callback(study, trial):
        nonlocal best_score
        completed[0] += 1
        n = completed[0]
        elapsed = time.time() - t_start
        rate = n / elapsed if elapsed > 0 else 0
        remaining = (n_trials - n) / rate if rate > 0 else 0

        if trial.value is not None and trial.value > best_score:
            best_score = trial.value
            ua = trial.user_attrs
            print(f'  [{n:>3}/{n_trials}] NEW BEST {best_score:.4f} | '
                  f'ret={ua.get("ret",0):+.1f}% dd={ua.get("max_dd",0):.1f}% '
                  f'sharpe={ua.get("sharpe",0):.2f} trades={ua.get("trades",0)} '
                  f'| {elapsed:.0f}s elapsed ~{remaining:.0f}s left', flush=True)
        elif n % 25 == 0:
            print(f'  [{n:>3}/{n_trials}] best={best_score:.4f} | '
                  f'{elapsed:.0f}s elapsed ~{remaining:.0f}s left', flush=True)

    # Run optimization
    print(f'\n  Running {n_trials} trials...')
    objective = create_objective(period=period)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_workers,
                   show_progress_bar=False, callbacks=[progress_callback])

    elapsed = time.time() - t_start
    print(f'\n  Optimization completed in {elapsed:.0f}s ({elapsed/60:.1f}min)')
    print(f'  Rate: {n_trials / elapsed:.1f} trials/sec')

    return study


def analyze_results(study, top_n=20):
    """Analyze and display optimization results."""
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
              and t.value is not None and t.value > float('-inf')]

    if not trials:
        print('  No completed trials!')
        return None

    # Sort by objective value
    trials.sort(key=lambda t: t.value, reverse=True)

    print(f'\n  Total completed trials: {len(trials)}')
    print(f'  Best score: {trials[0].value:.4f}')
    print(f'  Worst score: {trials[-1].value:.4f}')

    # Build results table
    rows = []
    for t in trials[:top_n]:
        ua = t.user_attrs
        rows.append({
            'rank': len(rows) + 1,
            'score': round(t.value, 4),
            'ret': ua.get('ret', 0),
            'dd': ua.get('max_dd', 99),
            'sharpe': ua.get('sharpe', 0),
            'calmar': ua.get('calmar', 0),
            'pf': ua.get('pf', 0),
            'wr': ua.get('wr', 0),
            'trades': ua.get('trades', 0),
            'call_pt': ua.get('p_call_pt', 0),
            'put_pt': ua.get('p_put_pt', 0),
            'call_sl': ua.get('p_call_sl', 0),
            'put_sl': ua.get('p_put_sl', 0),
            'rsi_c': ua.get('p_rsi_call', 0),
            'rsi_p': ua.get('p_rsi_put', 0),
            'c_hold': ua.get('p_call_hold', 0),
            'p_hold': ua.get('p_put_hold', 0),
            'kelly': ua.get('p_kelly', 0),
            'min_ct': ua.get('p_min_contracts', 0),
            'dl': ua.get('p_max_daily_losses', 0),
            'dll': ua.get('p_max_daily_loss_pct', 0),

            'post_loss': ua.get('p_post_loss', ''),
            'call_pnl': ua.get('call_pnl', 0),
            'put_pnl': ua.get('put_pnl', 0),
            'call_wr': ua.get('call_wr', 0),
            'put_wr': ua.get('put_wr', 0),
            'pt_exit': ua.get('profit_exit_pct', 0),
            'day_wr': ua.get('day_win_pct', 0),
        })

    df = pd.DataFrame(rows)

    # Print top results
    print(f'\n  TOP {top_n} CONFIGURATIONS:')
    print(f'  {"Rk":>3} {"Score":>6} {"Ret%":>7} {"DD%":>5} {"Shrp":>5} {"Calm":>5} '
          f'{"PF":>5} {"WR%":>5} {"N":>4} | '
          f'{"CPT":>4} {"PPT":>4} {"CSL":>4} {"PSL":>4} {"RSIc":>4} {"RSIp":>4} '
          f'{"CHld":>4} {"PHld":>4} {"K%":>4} {"mCt":>3} {"DL":>2} {"DLL":>5} {"PostL":>8}')
    print(f'  {"─" * 140}')

    for _, r in df.iterrows():
        print(f'  {int(r["rank"]):>3} {r["score"]:>6.3f} {r["ret"]:>+6.1f}% {r["dd"]:>4.1f}% '
              f'{r["sharpe"]:>5.2f} {r["calmar"]:>5.1f} {r["pf"]:>5.2f} {r["wr"]:>4.1f}% '
              f'{int(r["trades"]):>4} | '
              f'{r["call_pt"]*100:>3.0f}% {r["put_pt"]*100:>3.0f}% '
              f'{r["call_sl"]*100:>3.0f}% {r["put_sl"]*100:>3.0f}% '
              f'{int(r["rsi_c"]):>4} {int(r["rsi_p"]):>4} '
              f'{int(r["c_hold"]):>4} {int(r["p_hold"]):>4} {r["kelly"]*100:>3.0f}% {int(r["min_ct"]):>3} '
              f'{int(r["dl"]):>2} {r["dll"]*100:>4.1f}% '
              f'{r["post_loss"]:>8}')

    # Print best config details
    best = trials[0]
    ua = best.user_attrs
    print(f'\n  BEST CONFIGURATION:')
    print(f'    CALL: PT={ua.get("p_call_pt",0)*100:.0f}%  SL={ua.get("p_call_sl",0)*100:.0f}%  '
          f'RSI>{ua.get("p_rsi_call",0)}  Hold={ua.get("p_call_hold",0)} bars')
    print(f'    PUT:  PT={ua.get("p_put_pt",0)*100:.0f}%  SL={ua.get("p_put_sl",0)*100:.0f}%  '
          f'RSI<{ua.get("p_rsi_put",0)}  Hold={ua.get("p_put_hold",0)} bars')
    print(f'    Kelly: {ua.get("p_kelly",0)*100:.0f}%  '
          f'MinCt: {ua.get("p_min_contracts",0)}')
    print(f'    DailyLosses: {ua.get("p_max_daily_losses",0)}  '
          f'DLL: {ua.get("p_max_daily_loss_pct",0)*100:.1f}%  '
          f'ConsecLosses: {ua.get("p_max_consec_losses",0)}')
    print(f'    PostLoss: {ua.get("p_post_loss","")}')
    print(f'    ---')
    print(f'    Return: {ua.get("ret",0):+.1f}%  DD: {ua.get("max_dd",0):.1f}%  '
          f'Sharpe: {ua.get("sharpe",0):.2f}  Calmar: {ua.get("calmar",0):.1f}')
    print(f'    Trades: {ua.get("trades",0)}  WR: {ua.get("wr",0):.1f}%  '
          f'PF: {ua.get("pf",0):.2f}')
    print(f'    CALL: {ua.get("call_trades",0)} trades, WR={ua.get("call_wr",0):.1f}%, '
          f'P&L=${ua.get("call_pnl",0):+,.0f}')
    print(f'    PUT:  {ua.get("put_trades",0)} trades, WR={ua.get("put_wr",0):.1f}%, '
          f'P&L=${ua.get("put_pnl",0):+,.0f}')
    print(f'    PT exits: {ua.get("profit_exit_pct",0):.1f}%  '
          f'Day WR: {ua.get("day_win_pct",0):.1f}%')

    # Save full results
    all_rows = []
    for t in trials:
        ua = t.user_attrs
        row = {'score': t.value}
        row.update({k: v for k, v in ua.items()})
        all_rows.append(row)
    pd.DataFrame(all_rows).to_csv('output/optimization_full_results.csv', index=False)
    print(f'\n  Full results saved to output/optimization_full_results.csv')

    return df, trials[0]


def validate_on_oos(best_trial, period='2026'):
    """Validate the best config on out-of-sample data."""
    ua = best_trial.user_attrs
    params = {k.replace('p_', ''): v for k, v in ua.items() if k.startswith('p_')}
    # Fix param names
    params['trade_end_hour'] = 11

    print(f'\n{"="*70}')
    print(f'  OUT-OF-SAMPLE VALIDATION: {period}')
    print(f'{"="*70}')

    # Load OOS data
    print(f'  Loading {period} data...')
    load_data_cached(period)

    # Run with best params
    m = run_single_config(params, period=period)
    if m is None:
        print('  FAILED: No trades on OOS data!')
        return None

    print(f'\n  OOS Results ({period}):')
    print(f'    Trades: {m["trades"]}  WR: {m["wr"]:.1f}%  Return: {m["ret"]:+.1f}%')
    print(f'    Max DD: {m["max_dd"]:.1f}%  Sharpe: {m["sharpe"]:.2f}  '
          f'Calmar: {m["calmar"]:.1f}  PF: {m["pf"]:.2f}')
    print(f'    CALL: {m["call_trades"]} trades, WR={m["call_wr"]:.1f}%, '
          f'P&L=${m["call_pnl"]:+,.0f}')
    print(f'    PUT:  {m["put_trades"]} trades, WR={m["put_wr"]:.1f}%, '
          f'P&L=${m["put_pnl"]:+,.0f}')
    print(f'    PT exits: {m["profit_exit_pct"]:.1f}%  Day WR: {m["day_win_pct"]:.1f}%')

    # Compare IS vs OOS
    is_ret = ua.get('ret', 0)
    is_dd = ua.get('max_dd', 0)
    is_sharpe = ua.get('sharpe', 0)

    print(f'\n  IS vs OOS Comparison:')
    print(f'    {"Metric":<15} {"In-Sample":>12} {"OOS":>12} {"Decay":>10}')
    print(f'    {"─"*50}')
    for name, is_v, oos_v in [
        ('Return %', is_ret, m['ret']),
        ('Max DD %', is_dd, m['max_dd']),
        ('Sharpe', is_sharpe, m['sharpe']),
        ('Win Rate %', ua.get('wr', 0), m['wr']),
        ('Profit Factor', ua.get('pf', 0), m['pf']),
    ]:
        if name == 'Max DD %':
            decay = f'{oos_v - is_v:+.1f}pp'
        elif is_v != 0:
            decay = f'{(oos_v / is_v - 1) * 100:+.0f}%'
        else:
            decay = 'N/A'
        print(f'    {name:<15} {is_v:>12.1f} {oos_v:>12.1f} {decay:>10}')

    return m


def generate_optimization_charts(study, period='2025'):
    """Generate optimization analysis charts."""
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE
              and t.value is not None and t.value > float('-inf')]

    if len(trials) < 10:
        print('  Not enough trials for charts')
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Strategy Optimization Results ({period}) — {len(trials)} Trials',
                 fontsize=16, fontweight='bold')

    # 1. Score over trials (convergence)
    ax = axes[0, 0]
    scores = [t.value for t in trials]
    best_so_far = [max(scores[:i+1]) for i in range(len(scores))]
    ax.plot(scores, alpha=0.3, color='gray', label='Trial score')
    ax.plot(best_so_far, color='red', lw=2, label='Best so far')
    ax.set_xlabel('Trial #')
    ax.set_ylabel('Composite Score')
    ax.set_title('Optimization Convergence')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Return vs DD scatter
    ax = axes[0, 1]
    rets = [t.user_attrs.get('ret', 0) for t in trials]
    dds = [t.user_attrs.get('max_dd', 99) for t in trials]
    colors = [t.value for t in trials]
    sc = ax.scatter(dds, rets, c=colors, cmap='RdYlGn', alpha=0.6, s=20)
    # Highlight top 10
    top_10 = sorted(trials, key=lambda t: t.value, reverse=True)[:10]
    for t in top_10:
        ax.scatter(t.user_attrs.get('max_dd', 0), t.user_attrs.get('ret', 0),
                   color='red', s=80, marker='*', zorder=5)
    ax.set_xlabel('Max Drawdown %')
    ax.set_ylabel('Return %')
    ax.set_title('Return vs Drawdown (top 10 = ★)')
    fig.colorbar(sc, ax=ax, label='Score')
    ax.grid(True, alpha=0.3)

    # 3. Parameter importance (use hyperparameter importance)
    ax = axes[0, 2]
    param_scores = {}
    for param_name in ['call_pt', 'put_pt', 'call_sl', 'put_sl', 'rsi_call',
                        'rsi_put', 'call_hold', 'put_hold', 'kelly', 'min_contracts',
                        'max_daily_losses', 'post_loss']:
        vals = []
        for t in trials:
            if param_name in t.params:
                vals.append((t.params[param_name], t.value))
        if vals:
            # Correlation between param value and score
            if isinstance(vals[0][0], str):
                # Categorical: variance of scores per category
                from collections import defaultdict
                cat_scores = defaultdict(list)
                for v, s in vals:
                    cat_scores[v].append(s)
                if len(cat_scores) > 1:
                    means = [np.mean(s) for s in cat_scores.values()]
                    param_scores[param_name] = np.std(means)
                else:
                    param_scores[param_name] = 0
            else:
                # Numerical: absolute correlation
                x = [v for v, _ in vals]
                y = [s for _, s in vals]
                if np.std(x) > 0 and np.std(y) > 0:
                    corr = abs(np.corrcoef(x, y)[0, 1])
                    param_scores[param_name] = corr
                else:
                    param_scores[param_name] = 0

    sorted_params = sorted(param_scores.items(), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in sorted_params]
    importances = [p[1] for p in sorted_params]
    bars = ax.barh(range(len(names)), importances, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Importance (|correlation| with score)')
    ax.set_title('Parameter Importance')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # 4. CALL PT distribution (top 50 vs rest)
    ax = axes[1, 0]
    top_50 = sorted(trials, key=lambda t: t.value, reverse=True)[:50]
    rest = sorted(trials, key=lambda t: t.value, reverse=True)[50:]
    top_cpt = [t.params.get('call_pt', 0.5) * 100 for t in top_50]
    rest_cpt = [t.params.get('call_pt', 0.5) * 100 for t in rest]
    ax.hist(rest_cpt, bins=15, alpha=0.4, color='gray', label='Rest', density=True)
    ax.hist(top_cpt, bins=15, alpha=0.7, color='green', label='Top 50', density=True)
    ax.set_xlabel('CALL Profit Target %')
    ax.set_ylabel('Density')
    ax.set_title('CALL PT: Top 50 vs Rest')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5. PUT PT distribution
    ax = axes[1, 1]
    top_ppt = [t.params.get('put_pt', 0.5) * 100 for t in top_50]
    rest_ppt = [t.params.get('put_pt', 0.5) * 100 for t in rest]
    ax.hist(rest_ppt, bins=15, alpha=0.4, color='gray', label='Rest', density=True)
    ax.hist(top_ppt, bins=15, alpha=0.7, color='green', label='Top 50', density=True)
    ax.set_xlabel('PUT Profit Target %')
    ax.set_ylabel('Density')
    ax.set_title('PUT PT: Top 50 vs Rest')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 6. Post-loss strategy comparison
    ax = axes[1, 2]
    from collections import defaultdict
    pl_scores = defaultdict(list)
    for t in trials:
        pl = t.params.get('post_loss', 'none')
        pl_scores[pl].append(t.value)

    pl_names = sorted(pl_scores.keys())
    pl_means = [np.mean(pl_scores[k]) for k in pl_names]
    pl_maxs = [max(pl_scores[k]) for k in pl_names]
    x = range(len(pl_names))
    ax.bar([i - 0.15 for i in x], pl_means, 0.3, color='steelblue', alpha=0.8, label='Mean')
    ax.bar([i + 0.15 for i in x], pl_maxs, 0.3, color='orange', alpha=0.8, label='Max')
    ax.set_xticks(list(x))
    ax.set_xticklabels(pl_names, fontsize=9)
    ax.set_ylabel('Composite Score')
    ax.set_title('Post-Loss Strategy Comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = 'output/optimization_charts.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'  Charts saved to {out}')


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Complete Strategy Optimizer')
    parser.add_argument('--trials', type=int, default=300,
                        help='Number of optimization trials (default: 300)')
    parser.add_argument('--period', type=str, default='2025',
                        help='Optimization period (default: 2025)')
    parser.add_argument('--oos', type=str, default='2026',
                        help='OOS validation period (default: 2026)')
    parser.add_argument('--workers', type=int, default=6,
                        help='Parallel workers (default: 6)')
    parser.add_argument('--apply', action='store_true',
                        help='Apply best config to strategy.json')
    args = parser.parse_args()

    t_start = time.time()

    # Phase 1: Run optimization
    study = run_optimization(
        n_trials=args.trials,
        n_workers=args.workers,
        period=args.period,
    )

    # Phase 2: Analyze results
    print('\n')
    result = analyze_results(study, top_n=20)
    if result is None:
        print('  Optimization failed!')
        return

    df, best_trial = result

    # Phase 3: OOS validation
    oos_metrics = validate_on_oos(best_trial, period=args.oos)

    # Phase 4: Generate charts
    print('\n  Generating optimization charts...')
    generate_optimization_charts(study, period=args.period)

    # Phase 5: Save best config
    ua = best_trial.user_attrs
    best_config = {
        'optimization_date': time.strftime('%Y-%m-%d'),
        'trials': len(study.trials),
        'is_period': args.period,
        'oos_period': args.oos,
        'trade_config': {
            'strategy': 'momentum',
            'call_profit_target_pct': ua.get('p_call_pt'),
            'put_profit_target_pct': ua.get('p_put_pt'),
            'call_stop_loss_pct': ua.get('p_call_sl'),
            'put_stop_loss_pct': ua.get('p_put_sl'),
            'profit_target_pct': ua.get('p_call_pt'),  # fallback
            'stop_loss_pct': ua.get('p_call_sl'),       # fallback
            'rsi_call_threshold': ua.get('p_rsi_call'),
            'rsi_put_threshold': ua.get('p_rsi_put'),
            'call_max_hold_bars': ua.get('p_call_hold'),
            'put_max_hold_bars': ua.get('p_put_hold'),
            'max_hold_bars': ua.get('p_call_hold'),
            'min_contracts_per_trade': ua.get('p_min_contracts'),
            'trade_start_hour': 9,
            'trade_start_minute': 35,
            'trade_end_hour': ua.get('p_trade_end_hour', 15),
            'trade_end_minute': 0,
            'post_loss_strategy': ua.get('p_post_loss'),
            'post_loss_momentum_threshold': ua.get('p_momentum_threshold'),
            'min_option_price': 0.50,
            'max_option_price': 2.00,
            'use_regime_detection': True,
            'regime_rsi_buffer': ua.get('p_regime_rsi_buffer', 10),
            'regime_size_reduction': ua.get('p_regime_size_reduction', 0.10),
            'put_adaptive_filter': True,
            'put_loss_streak_threshold': ua.get('p_put_loss_streak', 1),
            'put_adaptive_cooldown': ua.get('p_put_cooldown', 4),
            'put_min_rsi': ua.get('p_put_min_rsi', 15),
        },
        'risk_config': {
            'kelly_pct': ua.get('p_kelly'),
            'max_risk_per_trade_pct': ua.get('p_max_risk'),
            'max_daily_losses': ua.get('p_max_daily_losses'),
            'max_consecutive_losses': ua.get('p_max_consec_losses'),
            'max_daily_loss_pct': ua.get('p_max_daily_loss_pct'),
        },
        'is_results': {
            'return': ua.get('ret'),
            'max_dd': ua.get('max_dd'),
            'sharpe': ua.get('sharpe'),
            'calmar': ua.get('calmar'),
            'profit_factor': ua.get('pf'),
            'win_rate': ua.get('wr'),
            'trades': ua.get('trades'),
            'call_pnl': ua.get('call_pnl'),
            'put_pnl': ua.get('put_pnl'),
        },
        'oos_results': oos_metrics,
    }

    config_path = 'output/optimized_config.json'
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f'\n  Best config saved to {config_path}')

    # Save to centralized optimization history
    run_id = save_optimization_run(
        source='optimize_full',
        trade_config=best_config['trade_config'],
        risk_config=best_config['risk_config'],
        results={
            'is_results': best_config['is_results'],
            'oos_results': oos_metrics,
        },
        metadata={
            'trials': len(study.trials),
            'is_period': args.period,
            'oos_period': args.oos,
        },
    )

    if args.apply:
        apply_run(run_id)

    # ============================================================
    # SAVE RUN HISTORY (append to optimization_history.json)
    # ============================================================
    history_path = 'output/optimization_history.json'
    history = []
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)

    # Run baseline on OOS for comparison
    print('\n  Running BASELINE on OOS for comparison...')
    baseline_oos = run_single_config(BASELINE_PARAMS, period=args.oos)
    baseline_is = run_single_config(BASELINE_PARAMS, period=args.period)

    run_entry = {
        'run_id': len(history) + 1,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'trials': len(study.trials),
        'is_period': args.period,
        'oos_period': args.oos,
        'best_params': {k: v for k, v in best_config['trade_config'].items()},
        'best_risk': best_config['risk_config'],
        'is_results': best_config['is_results'],
        'oos_results': oos_metrics,
        'baseline_is': baseline_is,
        'baseline_oos': baseline_oos,
    }
    history.append(run_entry)

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Print comparison table
    print(f'\n  {"+" * 70}')
    print(f'  OPTIMIZED vs BASELINE COMPARISON')
    print(f'  {"+" * 70}')
    if baseline_oos and oos_metrics:
        print(f'    {"":20} {"Baseline":>12} {"Optimized":>12} {"Winner":>8}')
        print(f'    {"─"*55}')
        comparisons = [
            ('IS Return %', baseline_is.get('ret', 0) if baseline_is else 0, ua.get('ret', 0), 'max'),
            ('IS Sharpe', baseline_is.get('sharpe', 0) if baseline_is else 0, ua.get('sharpe', 0), 'max'),
            ('IS Trades', baseline_is.get('trades', 0) if baseline_is else 0, ua.get('trades', 0), 'max'),
            ('IS Max DD %', baseline_is.get('max_dd', 0) if baseline_is else 0, ua.get('max_dd', 0), 'min'),
            ('OOS Return %', baseline_oos.get('ret', 0), oos_metrics.get('ret', 0), 'max'),
            ('OOS Sharpe', baseline_oos.get('sharpe', 0), oos_metrics.get('sharpe', 0), 'max'),
            ('OOS Trades', baseline_oos.get('trades', 0), oos_metrics.get('trades', 0), 'max'),
            ('OOS Max DD %', baseline_oos.get('max_dd', 0), oos_metrics.get('max_dd', 0), 'min'),
            ('OOS WR %', baseline_oos.get('wr', 0), oos_metrics.get('wr', 0), 'max'),
            ('OOS PF', baseline_oos.get('pf', 0), oos_metrics.get('pf', 0), 'max'),
        ]
        for name, bv, ov, mode in comparisons:
            if mode == 'max':
                winner = 'BASE' if bv > ov else ('OPT' if ov > bv else 'TIE')
            else:
                winner = 'BASE' if bv < ov else ('OPT' if ov < bv else 'TIE')
            print(f'    {name:<20} {bv:>12.1f} {ov:>12.1f} {winner:>8}')

    print(f'\n  Run history saved to {history_path} (run #{len(history)})')

    total_time = time.time() - t_start
    print(f'\n{"="*70}')
    print(f'  OPTIMIZATION COMPLETE | {total_time:.0f}s ({total_time/60:.1f} min)')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
