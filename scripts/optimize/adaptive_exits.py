"""
Adaptive Exit Optimizer — Parallel Optuna TPE

Optimizes 11 adaptive-exit dimensions while keeping base strategy params FIXED
at their already-optimized values from strategy.json.

Search space:
  - VIX thresholds: low (5-18), high (20-35)
  - PT/SL per regime: low-vol, mid-vol, high-vol
  - Cheap-option adjustments
  - Max contracts per trade

Architecture:
  - Optuna TPE sampler (Bayesian, efficient exploration)
  - Thread-based parallel workers via n_jobs (shared data in-memory)
  - Data loaded once, shared across all trials
  - 2-phase: optimize on 2025, validate best on 2026

Usage:
  python scripts/optimize_adaptive_exits.py [--trials N] [--workers N]
"""
import sys
sys.path.insert(0, '.')

import io
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import optuna

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from copy import deepcopy
from backtest.engine import Backtest0DTE, TradeConfig
from core.risk_manager import RiskConfig
from config import defaults as cfg

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# GLOBAL DATA CACHE (loaded once, shared across threads)
# ============================================================
_DATA_CACHE = {}


# ============================================================
# FIXED BASE PARAMS — loaded from strategy.json (single source of truth)
# ============================================================
_strategy_json = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'config', 'strategy.json')))
BASE_TRADE_CFG = {k: v for k, v in _strategy_json['trade_config'].items()
                  if k in TradeConfig.__dataclass_fields__}
BASE_RISK_CFG = {k: v for k, v in _strategy_json['risk_config'].items()
                 if k in RiskConfig.__dataclass_fields__}
KELLY_PCT = _strategy_json['risk_config'].get('kelly_pct', 0.06)


def load_data_cached(period='2025'):
    """Load and cache backtest data + option indexes."""
    global _DATA_CACHE
    if period in _DATA_CACHE:
        return _DATA_CACHE[period]

    cap = cfg.initial_capital()
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


def run_adaptive_config(params, period='2025'):
    """Run backtest with adaptive exit parameters. Returns metrics dict or None."""
    u, o, f, v, dt_idx, td_idx = load_data_cached(period)
    cap = cfg.initial_capital()

    # Build TradeConfig: base params + adaptive overrides
    tc_kwargs = dict(BASE_TRADE_CFG)
    tc_kwargs['use_adaptive_exits'] = True
    tc_kwargs['vix_low_threshold'] = params['vix_low_threshold']
    tc_kwargs['vix_high_threshold'] = params['vix_high_threshold']
    tc_kwargs['profit_low_vol'] = params['profit_low_vol']
    tc_kwargs['stop_low_vol'] = params['stop_low_vol']
    tc_kwargs['profit_mid_vol'] = params['profit_mid_vol']
    tc_kwargs['stop_mid_vol'] = params['stop_mid_vol']
    tc_kwargs['profit_high_vol'] = params['profit_high_vol']
    tc_kwargs['stop_high_vol'] = params['stop_high_vol']
    tc_kwargs['cheap_option_threshold'] = params['cheap_option_threshold']
    tc_kwargs['cheap_option_bonus'] = params['cheap_option_bonus']
    tc_kwargs['max_contracts_per_trade'] = params['max_contracts_per_trade']

    tc = TradeConfig(**tc_kwargs)
    rc = RiskConfig(**BASE_RISK_CFG)

    bt = Backtest0DTE(tc, rc, initial_capital=cap)
    bt._opt_by_date_time = dt_idx
    bt._opt_by_ticker_date = td_idx
    bt.risk_manager.set_kelly(KELLY_PCT)

    old_out = sys.stdout
    sys.stdout = io.StringIO()
    try:
        trades = bt.run_no_ml(u, o, f, v, verbose=False)
    finally:
        sys.stdout = old_out

    return compute_metrics(trades, cap)


def run_baseline(period='2025'):
    """Run baseline (no adaptive exits) for comparison."""
    u, o, f, v, dt_idx, td_idx = load_data_cached(period)
    cap = cfg.initial_capital()

    tc = TradeConfig(**BASE_TRADE_CFG)
    rc = RiskConfig(**BASE_RISK_CFG)

    bt = Backtest0DTE(tc, rc, initial_capital=cap)
    bt._opt_by_date_time = dt_idx
    bt._opt_by_ticker_date = td_idx
    bt.risk_manager.set_kelly(KELLY_PCT)

    old_out = sys.stdout
    sys.stdout = io.StringIO()
    try:
        trades = bt.run_no_ml(u, o, f, v, verbose=False)
    finally:
        sys.stdout = old_out

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

    # Monthly breakdown for robustness scoring
    monthly_pnl = {}
    for t in trades:
        m = t.date[:7]  # YYYY-MM
        monthly_pnl[m] = monthly_pnl.get(m, 0) + t.pnl
    profitable_months = sum(1 for p in monthly_pnl.values() if p > 0)
    total_months = len(monthly_pnl)
    month_win_pct = profitable_months / total_months * 100 if total_months > 0 else 0
    worst_month_pnl = min(monthly_pnl.values()) if monthly_pnl else 0

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
        'month_win_pct': round(month_win_pct, 1),
        'worst_month_pnl': round(worst_month_pnl, 0),
        'profitable_months': profitable_months, 'total_months': total_months,
    }


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

# Baseline params (adaptive disabled) — enqueue as trial 0
BASELINE_ADAPTIVE_PARAMS = {
    'vix_low_threshold': 10.0,
    'vix_high_threshold': 25.0,
    'profit_low_vol': 0.15,
    'stop_low_vol': 0.20,
    'profit_mid_vol': 0.55,
    'stop_mid_vol': 0.40,
    'profit_high_vol': 0.55,
    'stop_high_vol': 0.40,
    'cheap_option_threshold': 1.00,
    'cheap_option_bonus': 0.05,
    'max_contracts_per_trade': 0,
}


def create_objective(period='2025', baseline_ret=0):
    """Create Optuna objective function for adaptive exit optimization."""

    def objective(trial):
        params = {
            # VIX regime thresholds
            'vix_low_threshold': trial.suggest_float(
                'vix_low_threshold', 5.0, 18.0, step=1.0),
            'vix_high_threshold': trial.suggest_float(
                'vix_high_threshold', 20.0, 35.0, step=1.0),

            # Low-vol regime: PT and SL (the key lever for August fix)
            'profit_low_vol': trial.suggest_float(
                'profit_low_vol', 0.05, 0.35, step=0.01),
            'stop_low_vol': trial.suggest_float(
                'stop_low_vol', 0.10, 0.35, step=0.01),

            # Mid-vol regime: PT and SL (normal days — preserve baseline)
            'profit_mid_vol': trial.suggest_float(
                'profit_mid_vol', 0.35, 0.65, step=0.05),
            'stop_mid_vol': trial.suggest_float(
                'stop_mid_vol', 0.25, 0.50, step=0.05),

            # High-vol regime: PT and SL
            'profit_high_vol': trial.suggest_float(
                'profit_high_vol', 0.35, 0.70, step=0.05),
            'stop_high_vol': trial.suggest_float(
                'stop_high_vol', 0.25, 0.50, step=0.05),

            # Cheap option adjustments
            'cheap_option_threshold': trial.suggest_float(
                'cheap_option_threshold', 0.50, 1.50, step=0.25),
            'cheap_option_bonus': trial.suggest_float(
                'cheap_option_bonus', 0.00, 0.10, step=0.01),

            # Contract cap
            'max_contracts_per_trade': trial.suggest_categorical(
                'max_contracts_per_trade', [0, 5, 10, 15, 20]),
        }

        m = run_adaptive_config(params, period=period)
        if m is None:
            return float('-inf')

        # Reject configs with too few trades or excessive DD
        if m['trades'] < 100:
            return float('-inf')
        if m['max_dd'] > 20:
            return float('-inf')

        # Composite score: Return-dominant + robustness
        # Normalize so baseline scores ~0.5
        baseline_norm = max(baseline_ret, 300)

        ret_score = m['ret'] / baseline_norm
        sharpe_score = m['sharpe'] / 8
        pf_score = min(m['pf'] / 4, 1)
        wr_score = (m['wr'] - 55) / 25
        trade_score = min(m['trades'] / 350, 1.0)
        dd_penalty = max(0, (m['max_dd'] - 5) / 15)  # Penalize DD > 5%
        month_score = m['month_win_pct'] / 100

        # Bonus: reward improving worst month (August fix)
        worst_month_bonus = max(0, (m['worst_month_pnl'] + 1500) / 3000)

        composite = (
            0.35 * ret_score +          # Return dominates
            0.15 * sharpe_score +       # Risk-adjusted
            0.10 * trade_score +        # Robustness via trade count
            0.10 * pf_score +           # Profit quality
            0.10 * wr_score +           # Win rate
            0.10 * month_score +        # Monthly consistency
            0.05 * worst_month_bonus +  # Worst month improvement
            0.05 * (1 - dd_penalty)     # DD control
        )

        # Store all metrics for analysis
        for k, v in m.items():
            trial.set_user_attr(k, v)
        for k, v in params.items():
            trial.set_user_attr(f'p_{k}', v)

        return composite

    return objective


# ============================================================
# MAIN OPTIMIZATION
# ============================================================

def run_optimization(n_trials=300, n_workers=1, period='2025'):
    """Run parallel Optuna optimization for adaptive exit parameters."""
    cap = cfg.initial_capital()

    print('=' * 70)
    print('  ADAPTIVE EXIT OPTIMIZATION — Parallel Optuna TPE')
    print(f'  {n_trials} trials | {n_workers} workers | period: {period}')
    print(f'  ${cap:,.0f} starting capital')
    print(f'  11 dimensions: VIX thresholds, PT/SL per regime,')
    print(f'    cheap option adjust, max contracts')
    print('=' * 70)

    # Load data once (shared across all threads)
    print('\n  Loading data...')
    t0 = time.time()
    load_data_cached(period)
    print(f'  Data loaded in {time.time() - t0:.1f}s')

    # Run baseline (no adaptive exits) for comparison
    print('\n  Running BASELINE (no adaptive exits)...')
    baseline_m = run_baseline(period=period)
    if baseline_m:
        print(f'  BASELINE: {baseline_m["trades"]} trades, '
              f'ret={baseline_m["ret"]:+.1f}%, dd={baseline_m["max_dd"]:.1f}%, '
              f'sharpe={baseline_m["sharpe"]:.2f}, wr={baseline_m["wr"]:.1f}%')
        print(f'    CALL: {baseline_m["call_trades"]} trades ${baseline_m["call_pnl"]:+,.0f} '
              f'| PUT: {baseline_m["put_trades"]} trades ${baseline_m["put_pnl"]:+,.0f}')
        print(f'    PT exits: {baseline_m["profit_exits"]} | '
              f'STOP: {baseline_m["stop_exits"]} | TIME: {baseline_m["time_exits"]}')
        print(f'    Month WR: {baseline_m["month_win_pct"]:.0f}% '
              f'| Worst month: ${baseline_m["worst_month_pnl"]:+,.0f}')
        baseline_ret = baseline_m['ret']
    else:
        print('  BASELINE: failed (no trades)')
        baseline_ret = 300

    # In-memory study (no SQLite = no locking issues with threads)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=30),
    )

    # Enqueue baseline adaptive config as first trial
    study.enqueue_trial(BASELINE_ADAPTIVE_PARAMS)

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
                  f'worst_mo=${ua.get("worst_month_pnl",0):+,.0f} '
                  f'pt_exits={ua.get("profit_exits",0)} '
                  f'| {elapsed:.0f}s ~{remaining:.0f}s left', flush=True)
        elif n % 25 == 0:
            print(f'  [{n:>3}/{n_trials}] best={best_score:.4f} | '
                  f'{elapsed:.0f}s ~{remaining:.0f}s left', flush=True)

    # Run optimization
    print(f'\n  Running {n_trials} trials with {n_workers} parallel workers...')
    objective = create_objective(period=period, baseline_ret=baseline_ret)
    study.optimize(
        objective, n_trials=n_trials, n_jobs=n_workers,
        show_progress_bar=False, callbacks=[progress_callback],
    )

    elapsed = time.time() - t_start
    print(f'\n  Optimization completed in {elapsed:.0f}s ({elapsed/60:.1f}min)')
    print(f'  Rate: {n_trials / elapsed:.1f} trials/sec')

    return study, baseline_m


# ============================================================
# ANALYSIS
# ============================================================

def analyze_results(study, baseline_m, top_n=20):
    """Analyze and display optimization results."""
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE
              and t.value is not None and t.value > float('-inf')]

    if not trials:
        print('  No completed trials!')
        return None, None

    trials.sort(key=lambda t: t.value, reverse=True)

    print(f'\n{"="*70}')
    print(f'  OPTIMIZATION RESULTS — {len(trials)} completed trials')
    print(f'{"="*70}')

    # Top configs table
    print(f'\n  TOP {min(top_n, len(trials))} CONFIGURATIONS:')
    print(f'  {"Rk":>3} {"Score":>6} {"Ret%":>7} {"DD%":>5} {"Shrp":>5} '
          f'{"WR%":>5} {"PF":>5} {"N":>4} {"PT":>3} {"SL":>3} {"TM":>3} '
          f'| {"VIXlo":>5} {"VIXhi":>5} {"PLv":>4} {"SLv":>4} '
          f'{"PMv":>4} {"SMv":>4} {"PHv":>4} {"SHv":>4} '
          f'{"CheapT":>6} {"CheapB":>6} {"MaxCt":>5}')
    print(f'  {"─" * 145}')

    rows = []
    for i, t in enumerate(trials[:top_n]):
        ua = t.user_attrs
        row = {
            'rank': i + 1,
            'score': round(t.value, 4),
            'ret': ua.get('ret', 0),
            'dd': ua.get('max_dd', 99),
            'sharpe': ua.get('sharpe', 0),
            'wr': ua.get('wr', 0),
            'pf': ua.get('pf', 0),
            'trades': ua.get('trades', 0),
            'pt_exits': ua.get('profit_exits', 0),
            'sl_exits': ua.get('stop_exits', 0),
            'tm_exits': ua.get('time_exits', 0),
            'vix_lo': ua.get('p_vix_low_threshold', 0),
            'vix_hi': ua.get('p_vix_high_threshold', 0),
            'p_lv': ua.get('p_profit_low_vol', 0),
            's_lv': ua.get('p_stop_low_vol', 0),
            'p_mv': ua.get('p_profit_mid_vol', 0),
            's_mv': ua.get('p_stop_mid_vol', 0),
            'p_hv': ua.get('p_profit_high_vol', 0),
            's_hv': ua.get('p_stop_high_vol', 0),
            'cheap_t': ua.get('p_cheap_option_threshold', 0),
            'cheap_b': ua.get('p_cheap_option_bonus', 0),
            'max_ct': ua.get('p_max_contracts_per_trade', 0),
            'worst_mo': ua.get('worst_month_pnl', 0),
            'mo_wr': ua.get('month_win_pct', 0),
            'call_pnl': ua.get('call_pnl', 0),
            'put_pnl': ua.get('put_pnl', 0),
        }
        rows.append(row)

        print(f'  {row["rank"]:>3} {row["score"]:>6.3f} {row["ret"]:>+6.1f}% '
              f'{row["dd"]:>4.1f}% {row["sharpe"]:>5.2f} {row["wr"]:>4.1f}% '
              f'{row["pf"]:>4.2f} {row["trades"]:>4} '
              f'{row["pt_exits"]:>3} {row["sl_exits"]:>3} {row["tm_exits"]:>3} '
              f'| {row["vix_lo"]:>5.0f} {row["vix_hi"]:>5.0f} '
              f'{row["p_lv"]*100:>3.0f}% {row["s_lv"]*100:>3.0f}% '
              f'{row["p_mv"]*100:>3.0f}% {row["s_mv"]*100:>3.0f}% '
              f'{row["p_hv"]*100:>3.0f}% {row["s_hv"]*100:>3.0f}% '
              f'{row["cheap_t"]:>6.2f} {row["cheap_b"]*100:>5.0f}% '
              f'{row["max_ct"]:>5}')

    # Best config detail
    best = trials[0]
    ua = best.user_attrs
    print(f'\n  {"="*70}')
    print(f'  BEST CONFIGURATION (score={best.value:.4f}):')
    print(f'  {"="*70}')
    print(f'    VIX thresholds:  low={ua.get("p_vix_low_threshold",0):.0f}  '
          f'high={ua.get("p_vix_high_threshold",0):.0f}')
    print(f'    Low-vol regime:  PT={ua.get("p_profit_low_vol",0)*100:.0f}%  '
          f'SL={ua.get("p_stop_low_vol",0)*100:.0f}%')
    print(f'    Mid-vol regime:  PT={ua.get("p_profit_mid_vol",0)*100:.0f}%  '
          f'SL={ua.get("p_stop_mid_vol",0)*100:.0f}%')
    print(f'    High-vol regime: PT={ua.get("p_profit_high_vol",0)*100:.0f}%  '
          f'SL={ua.get("p_stop_high_vol",0)*100:.0f}%')
    print(f'    Cheap options:   threshold=${ua.get("p_cheap_option_threshold",0):.2f}  '
          f'bonus={ua.get("p_cheap_option_bonus",0)*100:.0f}%')
    print(f'    Max contracts:   {ua.get("p_max_contracts_per_trade",0)}')
    print(f'    ---')
    print(f'    Return: {ua.get("ret",0):+.1f}%  DD: {ua.get("max_dd",0):.1f}%  '
          f'Sharpe: {ua.get("sharpe",0):.2f}  Calmar: {ua.get("calmar",0):.1f}')
    print(f'    Trades: {ua.get("trades",0)}  WR: {ua.get("wr",0):.1f}%  '
          f'PF: {ua.get("pf",0):.2f}')
    print(f'    CALL: {ua.get("call_trades",0)} trades ${ua.get("call_pnl",0):+,.0f}  '
          f'| PUT: {ua.get("put_trades",0)} trades ${ua.get("put_pnl",0):+,.0f}')
    print(f'    PT exits: {ua.get("profit_exits",0)} ({ua.get("profit_exit_pct",0):.1f}%)  '
          f'STOP: {ua.get("stop_exits",0)}  TIME: {ua.get("time_exits",0)}')
    print(f'    Month WR: {ua.get("month_win_pct",0):.0f}%  '
          f'Worst month: ${ua.get("worst_month_pnl",0):+,.0f}')

    # Compare best vs baseline
    if baseline_m:
        print(f'\n  {"="*70}')
        print(f'  BEST vs BASELINE')
        print(f'  {"="*70}')
        print(f'    {"Metric":<20} {"Baseline":>12} {"Best Adaptive":>14} {"Delta":>10}')
        print(f'    {"─"*58}')
        for name, bv, av in [
            ('Return %', baseline_m['ret'], ua.get('ret', 0)),
            ('Max DD %', baseline_m['max_dd'], ua.get('max_dd', 0)),
            ('Sharpe', baseline_m['sharpe'], ua.get('sharpe', 0)),
            ('Win Rate %', baseline_m['wr'], ua.get('wr', 0)),
            ('Profit Factor', baseline_m['pf'], ua.get('pf', 0)),
            ('Trades', baseline_m['trades'], ua.get('trades', 0)),
            ('PT exits', baseline_m['profit_exits'], ua.get('profit_exits', 0)),
            ('Worst Month $', baseline_m['worst_month_pnl'], ua.get('worst_month_pnl', 0)),
        ]:
            if name in ('Max DD %',):
                delta = f'{av - bv:+.1f}pp'
            elif name in ('Trades', 'PT exits'):
                delta = f'{int(av - bv):+d}'
            elif bv != 0:
                delta = f'{(av / bv - 1) * 100:+.1f}%'
            else:
                delta = 'N/A'
            if name in ('Trades', 'PT exits'):
                print(f'    {name:<20} {int(bv):>12} {int(av):>14} {delta:>10}')
            elif name in ('Worst Month $',):
                print(f'    {name:<20} ${bv:>+11,.0f} ${av:>+13,.0f} {delta:>10}')
            else:
                print(f'    {name:<20} {bv:>12.1f} {av:>14.1f} {delta:>10}')

    # Save full results
    all_rows = []
    for t in trials:
        ua = t.user_attrs
        row = {'score': t.value}
        row.update({k: v for k, v in ua.items()})
        all_rows.append(row)
    pd.DataFrame(all_rows).to_csv(
        'output/optimization_adaptive_exits.csv', index=False)
    print(f'\n  Full results: output/optimization_adaptive_exits.csv')

    return pd.DataFrame(rows), trials[0]


# ============================================================
# OOS VALIDATION
# ============================================================

def validate_on_oos(best_trial, period='2026'):
    """Validate best adaptive config on out-of-sample data."""
    ua = best_trial.user_attrs
    params = {k[2:]: v for k, v in ua.items() if k.startswith('p_')}

    print(f'\n{"="*70}')
    print(f'  OUT-OF-SAMPLE VALIDATION: {period}')
    print(f'{"="*70}')

    print(f'  Loading {period} data...')
    load_data_cached(period)

    # Run adaptive config on OOS
    m_adaptive = run_adaptive_config(params, period=period)
    # Run baseline on OOS
    m_baseline = run_baseline(period=period)

    for label, m in [('BASELINE (no adaptive)', m_baseline),
                     ('BEST ADAPTIVE', m_adaptive)]:
        if m is None:
            print(f'  [{label}] FAILED: No trades!')
            continue
        print(f'\n  [{label}] OOS {period}:')
        print(f'    Trades: {m["trades"]}  WR: {m["wr"]:.1f}%  '
              f'Return: {m["ret"]:+.1f}%  DD: {m["max_dd"]:.1f}%')
        print(f'    Sharpe: {m["sharpe"]:.2f}  PF: {m["pf"]:.2f}  '
              f'Calmar: {m["calmar"]:.1f}')
        print(f'    CALL: {m["call_trades"]} trades ${m["call_pnl"]:+,.0f}  '
              f'| PUT: {m["put_trades"]} trades ${m["put_pnl"]:+,.0f}')
        print(f'    PT exits: {m["profit_exits"]} ({m["profit_exit_pct"]:.1f}%)')

    if m_adaptive and m_baseline:
        print(f'\n  {"Metric":<20} {"Baseline":>12} {"Adaptive":>12} {"Delta":>10}')
        print(f'  {"─"*56}')
        for name, bv, av in [
            ('Return %', m_baseline['ret'], m_adaptive['ret']),
            ('Max DD %', m_baseline['max_dd'], m_adaptive['max_dd']),
            ('Sharpe', m_baseline['sharpe'], m_adaptive['sharpe']),
            ('Win Rate %', m_baseline['wr'], m_adaptive['wr']),
            ('PF', m_baseline['pf'], m_adaptive['pf']),
        ]:
            if name == 'Max DD %':
                delta = f'{av - bv:+.1f}pp'
            elif bv != 0:
                delta = f'{(av / bv - 1) * 100:+.1f}%'
            else:
                delta = 'N/A'
            print(f'  {name:<20} {bv:>12.1f} {av:>12.1f} {delta:>10}')

    return m_adaptive


# ============================================================
# CHARTS
# ============================================================

def generate_charts(study):
    """Generate optimization analysis charts."""
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE
              and t.value is not None and t.value > float('-inf')]

    if len(trials) < 10:
        print('  Not enough trials for charts')
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Adaptive Exit Optimization — {len(trials)} Trials',
                 fontsize=16, fontweight='bold')

    # 1. Convergence
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
    top_10 = sorted(trials, key=lambda t: t.value, reverse=True)[:10]
    for t in top_10:
        ax.scatter(t.user_attrs.get('max_dd', 0), t.user_attrs.get('ret', 0),
                   color='red', s=80, marker='*', zorder=5)
    ax.set_xlabel('Max Drawdown %')
    ax.set_ylabel('Return %')
    ax.set_title('Return vs Drawdown (top 10 = ★)')
    fig.colorbar(sc, ax=ax, label='Score')
    ax.grid(True, alpha=0.3)

    # 3. Parameter importance
    ax = axes[0, 2]
    param_scores = {}
    for param_name in ['vix_low_threshold', 'vix_high_threshold',
                        'profit_low_vol', 'stop_low_vol',
                        'profit_mid_vol', 'stop_mid_vol',
                        'profit_high_vol', 'stop_high_vol',
                        'cheap_option_threshold', 'cheap_option_bonus']:
        vals = []
        for t in trials:
            if param_name in t.params:
                vals.append((t.params[param_name], t.value))
        if vals:
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
    ax.barh(range(len(names)), importances, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Importance (|correlation| with score)')
    ax.set_title('Parameter Importance')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # 4. Low-vol PT distribution (top 50 vs rest) — the key lever
    ax = axes[1, 0]
    top_50 = sorted(trials, key=lambda t: t.value, reverse=True)[:50]
    rest = sorted(trials, key=lambda t: t.value, reverse=True)[50:]
    top_vals = [t.params.get('profit_low_vol', 0.15) * 100 for t in top_50]
    rest_vals = [t.params.get('profit_low_vol', 0.15) * 100 for t in rest]
    ax.hist(rest_vals, bins=15, alpha=0.4, color='gray', label='Rest', density=True)
    ax.hist(top_vals, bins=15, alpha=0.7, color='green', label='Top 50', density=True)
    ax.set_xlabel('Low-Vol Profit Target %')
    ax.set_ylabel('Density')
    ax.set_title('Low-Vol PT: Top 50 vs Rest')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5. VIX low threshold distribution
    ax = axes[1, 1]
    top_vix = [t.params.get('vix_low_threshold', 10) for t in top_50]
    rest_vix = [t.params.get('vix_low_threshold', 10) for t in rest]
    ax.hist(rest_vix, bins=13, alpha=0.4, color='gray', label='Rest', density=True)
    ax.hist(top_vix, bins=13, alpha=0.7, color='green', label='Top 50', density=True)
    ax.set_xlabel('VIX Low Threshold')
    ax.set_ylabel('Density')
    ax.set_title('VIX Low Threshold: Top 50 vs Rest')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 6. PT exits vs Return
    ax = axes[1, 2]
    pt_exits = [t.user_attrs.get('profit_exits', 0) for t in trials]
    rets_all = [t.user_attrs.get('ret', 0) for t in trials]
    ax.scatter(pt_exits, rets_all, alpha=0.4, s=15, c='steelblue')
    for t in top_10:
        ax.scatter(t.user_attrs.get('profit_exits', 0),
                   t.user_attrs.get('ret', 0),
                   color='red', s=80, marker='*', zorder=5)
    ax.set_xlabel('Profit Target Exits')
    ax.set_ylabel('Return %')
    ax.set_title('PT Exits vs Return (top 10 = ★)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = 'output/optimization_adaptive_exits.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Charts saved: {chart_path}')


# ============================================================
# SAVE BEST CONFIG
# ============================================================

def save_best_config(best_trial):
    """Save the best adaptive exit config to JSON."""
    ua = best_trial.user_attrs
    config = {
        'adaptive_exits': {
            'use_adaptive_exits': True,
            'vix_low_threshold': ua.get('p_vix_low_threshold'),
            'vix_high_threshold': ua.get('p_vix_high_threshold'),
            'profit_low_vol': ua.get('p_profit_low_vol'),
            'stop_low_vol': ua.get('p_stop_low_vol'),
            'profit_mid_vol': ua.get('p_profit_mid_vol'),
            'stop_mid_vol': ua.get('p_stop_mid_vol'),
            'profit_high_vol': ua.get('p_profit_high_vol'),
            'stop_high_vol': ua.get('p_stop_high_vol'),
            'cheap_option_threshold': ua.get('p_cheap_option_threshold'),
            'cheap_option_bonus': ua.get('p_cheap_option_bonus'),
            'max_contracts_per_trade': ua.get('p_max_contracts_per_trade'),
        },
        'metrics': {
            'score': best_trial.value,
            'return_pct': ua.get('ret'),
            'max_dd_pct': ua.get('max_dd'),
            'sharpe': ua.get('sharpe'),
            'win_rate': ua.get('wr'),
            'profit_factor': ua.get('pf'),
            'trades': ua.get('trades'),
            'profit_exits': ua.get('profit_exits'),
            'worst_month_pnl': ua.get('worst_month_pnl'),
            'month_win_pct': ua.get('month_win_pct'),
        },
    }
    path = 'output/optimized_adaptive_exits.json'
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'  Best config saved: {path}')

    return config


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Optimize adaptive exit parameters')
    parser.add_argument('--trials', type=int, default=300,
                        help='Number of Optuna trials (default: 300)')
    parser.add_argument('--workers', type=int, default=6,
                        help='Parallel workers (default: 6)')
    parser.add_argument('--period', type=str, default='2025',
                        help='Training period (default: 2025)')
    args = parser.parse_args()

    t_total = time.time()

    # Phase 1: Optimize
    study, baseline_m = run_optimization(
        n_trials=args.trials, n_workers=args.workers, period=args.period)

    # Phase 2: Analyze
    df, best_trial = analyze_results(study, baseline_m)

    if best_trial is None:
        print('\n  FAILED: No valid trials!')
        return

    # Phase 3: OOS Validation
    validate_on_oos(best_trial)

    # Phase 4: Charts + Save
    generate_charts(study)
    save_best_config(best_trial)

    elapsed = time.time() - t_total
    print(f'\n{"="*70}')
    print(f'  COMPLETE — Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
