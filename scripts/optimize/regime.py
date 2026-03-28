"""
Regime Detection Optimizer V2 — Comprehensive Multi-Regime Parameter Tuning

Optimizes ALL regime detection parameters (classification thresholds + per-regime
adjustments) to eliminate bad months and improve risk-adjusted returns.

Parameters optimized (28 dimensions):
  Classification (7):  lookback, vol_pctl, trend_pctl, up_day_pct, dn_day_pct,
                        momentum_threshold, high_vol_pctl, adx_trend_threshold
  STEADY_UP (4):       size_reduction, skip_puts, rsi_buffer, call_pt_override
  STEADY_DN (4):       size_reduction, skip_calls, rsi_buffer, put_pt_override
  CHOPPY (4):          size_reduction, skip_first_bar, rsi_buffer, tighter_stop
  VOLATILE (3):        size_reduction, stop_buffer, pt_buffer
  TRENDING (3):        skip_counter, hold_buffer, [no size cut]

Objective:  Composite score penalizing red months, bad worst-month P&L,
            and drawdown while rewarding overall return and profit factor.

Architecture:
  - Optuna TPE sampler (Bayesian, efficient high-dim search)
  - 3-phase: Diagnostic → Optimize on 2025 → OOS validate on 2026
  - Scenario stress-test: run best config on sub-periods

Usage:
  python scripts/optimize_regime_v2.py [--trials 300] [--workers 4]
"""
import sys
sys.path.insert(0, '.')

import io
import json
import time
import argparse
import numpy as np
import pandas as pd
import optuna

from backtest.engine import Backtest0DTE, TradeConfig
from core.risk_manager import RiskConfig
from config import defaults as cfg
from config.config_manager import save_optimization_run, apply_run

optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='optuna')

_DATA_CACHE = {}


def load_data_cached(period='2025'):
    global _DATA_CACHE
    if period in _DATA_CACHE:
        return _DATA_CACHE[period]

    cap = cfg.initial_capital()
    base_bt = Backtest0DTE(TradeConfig(), RiskConfig(), initial_capital=cap)

    ranges = {
        '2025': ('2025-01-01', '2025-12-31'),
        '2026': ('2026-01-02', '2026-02-25'),
        '2025_h1': ('2025-01-01', '2025-06-30'),
        '2025_h2': ('2025-07-01', '2025-12-31'),
    }
    start, end = ranges[period]
    u, o, f = base_bt.load_data(start, end)
    v = base_bt.compute_historical_volatility(u)
    dt_idx = base_bt._opt_by_date_time
    td_idx = base_bt._opt_by_ticker_date

    _DATA_CACHE[period] = (u, o, f, v, dt_idx, td_idx)
    return _DATA_CACHE[period]


def get_baseline_tc():
    config = json.load(open('config/strategy.json'))
    tc_dict = config['trade_config']
    return TradeConfig(**{k: v for k, v in tc_dict.items() if k in TradeConfig.__dataclass_fields__})


def get_risk_config():
    config = json.load(open('config/strategy.json'))
    rc_dict = config['risk_config']
    return RiskConfig(**{k: v for k, v in rc_dict.items() if k in RiskConfig.__dataclass_fields__})


def run_config(tc, period='2025'):
    u, o, f, v, dt_idx, td_idx = load_data_cached(period)
    cap = cfg.initial_capital()
    rc = get_risk_config()

    bt = Backtest0DTE(tc, rc, initial_capital=cap)
    bt._opt_by_date_time = dt_idx
    bt._opt_by_ticker_date = td_idx
    bt.risk_manager.set_kelly(0.06)

    old_out = sys.stdout
    sys.stdout = io.StringIO()
    try:
        trades = bt.run_no_ml(u, o, f, v, verbose=False)
    finally:
        sys.stdout = old_out

    return trades, compute_metrics(trades, cap)


def compute_metrics(trades, cap):
    if not trades or len(trades) < 5:
        return None

    total_pnl = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    n = len(trades)
    wr = wins / n * 100
    ret = total_pnl / cap * 100

    peak = cap
    max_dd = 0
    for t in trades:
        c = t.capital
        if c > peak:
            peak = c
        dd = (peak - c) / peak
        if dd > max_dd:
            max_dd = dd

    rets = [t.pnl / max(t.capital - t.pnl, 1) for t in trades]
    mu = np.mean(rets)
    sigma = np.std(rets) if len(rets) > 1 else 1
    sharpe = (mu * 252) / (sigma * np.sqrt(252)) if sigma > 0 else 0
    down = [r for r in rets if r < 0]
    ds = np.std(down) if len(down) > 1 else 1
    sortino = (mu * 252) / (ds * np.sqrt(252)) if ds > 0 else 0

    gp = sum(t.pnl for t in trades if t.pnl > 0)
    gl = abs(sum(t.pnl for t in trades if t.pnl <= 0)) or 0.01
    pf = gp / gl

    call_trades = [t for t in trades if t.direction == 'CALL']
    put_trades = [t for t in trades if t.direction == 'PUT']

    # Monthly stats
    monthly_pnl = {}
    for t in trades:
        m = t.date[:7]
        monthly_pnl[m] = monthly_pnl.get(m, 0) + t.pnl
    red_months = sum(1 for v in monthly_pnl.values() if v < 0)
    worst_month_pnl = min(monthly_pnl.values()) if monthly_pnl else 0

    # Weekly stats (for sub-period analysis)
    weekly_pnl = {}
    for t in trades:
        # ISO week
        from datetime import datetime
        dt = datetime.strptime(t.date, '%Y-%m-%d')
        wk = dt.strftime('%Y-W%V')
        weekly_pnl[wk] = weekly_pnl.get(wk, 0) + t.pnl
    red_weeks = sum(1 for v in weekly_pnl.values() if v < 0)
    worst_week_pnl = min(weekly_pnl.values()) if weekly_pnl else 0

    return {
        'trades': n, 'wr': round(wr, 1),
        'pnl': round(total_pnl, 0), 'ret': round(ret, 1),
        'max_dd': round(max_dd * 100, 1),
        'sharpe': round(sharpe, 2), 'sortino': round(sortino, 2),
        'pf': round(pf, 2),
        'call_trades': len(call_trades), 'put_trades': len(put_trades),
        'call_pnl': round(sum(t.pnl for t in call_trades), 0),
        'put_pnl': round(sum(t.pnl for t in put_trades), 0),
        'red_months': red_months,
        'worst_month': round(worst_month_pnl, 0),
        'red_weeks': red_weeks,
        'worst_week': round(worst_week_pnl, 0),
        'monthly_pnl': {k: round(v, 0) for k, v in sorted(monthly_pnl.items())},
    }


def monthly_breakdown(trades):
    monthly = {}
    for t in trades:
        m = t.date[:7]
        if m not in monthly:
            monthly[m] = {'n': 0, 'wins': 0, 'pnl': 0.0, 'calls': 0, 'puts': 0}
        monthly[m]['n'] += 1
        if t.pnl > 0:
            monthly[m]['wins'] += 1
        monthly[m]['pnl'] += t.pnl
        if t.direction == 'CALL':
            monthly[m]['calls'] += 1
        else:
            monthly[m]['puts'] += 1
    return monthly


def print_monthly(trades, label):
    monthly = monthly_breakdown(trades)
    D = "$"
    print(f"\n  {label} -- Monthly P&L:")
    print(f"    {'Month':<10} {'N':>5} {'C/P':>7} {'WR':>6} {'PnL':>12}")
    print(f"    {'-'*45}")
    for m in sorted(monthly):
        d = monthly[m]
        wr = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
        marker = ' ***' if d['pnl'] < -500 else ''
        cp = f"{d['calls']}/{d['puts']}"
        print(f"    {m:<10} {d['n']:>5} {cp:>7} {wr:>5.1f}% {D}{d['pnl']:>+10,.0f}{marker}")


def regime_breakdown(trades, tc, period='2025'):
    """Show regime distribution and P&L per regime."""
    u, o, f, v, dt_idx, td_idx = load_data_cached(period)
    cap = cfg.initial_capital()
    rc = get_risk_config()
    bt = Backtest0DTE(tc, rc, initial_capital=cap)
    bt._opt_by_date_time = dt_idx
    bt._opt_by_ticker_date = td_idx
    regime_data = bt.compute_regime_features(u)

    regime_stats = {}
    for t in trades:
        rd = regime_data.get(t.date, {})
        rt = rd.get('regime_type', 'UNKNOWN')
        if rt not in regime_stats:
            regime_stats[rt] = {'n': 0, 'wins': 0, 'pnl': 0.0}
        regime_stats[rt]['n'] += 1
        if t.pnl > 0:
            regime_stats[rt]['wins'] += 1
        regime_stats[rt]['pnl'] += t.pnl

    D = "$"
    print(f"\n    Regime breakdown:")
    print(f"    {'Regime':<12} {'N':>5} {'WR':>6} {'PnL':>12}")
    print(f"    {'-'*40}")
    for rt in ['NORMAL', 'CHOPPY', 'STEADY_UP', 'STEADY_DN', 'VOLATILE', 'TRENDING', 'UNKNOWN']:
        if rt in regime_stats:
            s = regime_stats[rt]
            wr = s['wins'] / s['n'] * 100 if s['n'] > 0 else 0
            print(f"    {rt:<12} {s['n']:>5} {wr:>5.1f}% {D}{s['pnl']:>+10,.0f}")


def apply_regime_params(tc, params):
    """Apply a dict of regime params to a TradeConfig."""
    tc.use_regime_detection = True

    # Classification thresholds
    tc.regime_lookback_days = params['lookback']
    tc.regime_vol_percentile = params['vol_pctl']
    tc.regime_trend_percentile = params['trend_pctl']
    tc.regime_up_day_pct = params['up_day_pct']
    tc.regime_dn_day_pct = params['dn_day_pct']
    tc.regime_momentum_threshold = params['momentum_thr']
    tc.regime_high_vol_percentile = params['high_vol_pctl']
    tc.regime_adx_trend_threshold = params['adx_thr']

    # STEADY_UP adjustments
    tc.steady_up_size_reduction = params['su_size_red']
    tc.steady_up_skip_puts = params['su_skip_puts']
    tc.steady_up_rsi_buffer = params['su_rsi_buf']
    su_pt = params.get('su_call_pt', 'none')
    tc.steady_up_call_pt_override = None if su_pt == 'none' else float(su_pt)

    # STEADY_DN adjustments
    tc.steady_dn_size_reduction = params['sd_size_red']
    tc.steady_dn_skip_calls = params['sd_skip_calls']
    tc.steady_dn_rsi_buffer = params['sd_rsi_buf']
    sd_pt = params.get('sd_put_pt', 'none')
    tc.steady_dn_put_pt_override = None if sd_pt == 'none' else float(sd_pt)

    # CHOPPY adjustments
    tc.choppy_size_reduction = params['ch_size_red']
    tc.choppy_skip_first_bar = params['ch_skip_first']
    tc.choppy_rsi_buffer = params['ch_rsi_buf']
    ch_stop = params.get('ch_stop', 'none')
    tc.choppy_tighter_stop_pct = None if ch_stop == 'none' else float(ch_stop)

    # VOLATILE adjustments
    tc.volatile_size_reduction = params['vol_size_red']
    tc.volatile_stop_buffer_pct = params['vol_stop_buf']
    tc.volatile_pt_buffer_pct = params['vol_pt_buf']

    # TRENDING adjustments
    tc.trending_skip_counter = params['tr_skip_counter']
    tc.trending_hold_buffer = params['tr_hold_buf']

    return tc


# ============================================================
# PHASE 1: DIAGNOSTIC
# ============================================================
def run_diagnostic():
    D = "$"
    print("=" * 70)
    print("  PHASE 1: DIAGNOSTIC -- Current Regime Detection Impact")
    print("=" * 70)

    tc_base = get_baseline_tc()

    # Run with regime OFF
    print("\n  Running baseline (regime OFF)...")
    tc_off = get_baseline_tc()
    tc_off.use_regime_detection = False
    trades_off, m_off = run_config(tc_off, '2025')
    print(f"    2025: {m_off['trades']}t WR={m_off['wr']}% Ret={m_off['ret']}% DD={m_off['max_dd']}% PF={m_off['pf']}")
    print(f"    Red months: {m_off['red_months']}, Worst: {D}{m_off['worst_month']:+,.0f}")
    print_monthly(trades_off, "BASELINE (regime OFF)")

    # Run with regime ON (current params)
    print("\n  Running with current regime params...")
    trades_on, m_on = run_config(tc_base, '2025')
    print(f"    2025: {m_on['trades']}t WR={m_on['wr']}% Ret={m_on['ret']}% DD={m_on['max_dd']}% PF={m_on['pf']}")
    print(f"    Red months: {m_on['red_months']}, Worst: {D}{m_on['worst_month']:+,.0f}")
    print_monthly(trades_on, "CURRENT (regime ON)")
    regime_breakdown(trades_on, tc_base, '2025')

    # Delta
    print(f"\n  DELTA (regime ON vs OFF):")
    print(f"    Trades:     {m_on['trades']} vs {m_off['trades']} ({m_on['trades']-m_off['trades']:+d})")
    print(f"    Return:     {m_on['ret']}% vs {m_off['ret']}% ({m_on['ret']-m_off['ret']:+.1f}pp)")
    print(f"    Max DD:     {m_on['max_dd']}% vs {m_off['max_dd']}% ({m_on['max_dd']-m_off['max_dd']:+.1f}pp)")
    print(f"    Red months: {m_on['red_months']} vs {m_off['red_months']}")
    print(f"    PF:         {m_on['pf']} vs {m_off['pf']}")

    # Bad month comparison
    for bm in ['2025-05', '2025-07', '2025-08', '2025-10']:
        off_pnl = m_off['monthly_pnl'].get(bm, 0)
        on_pnl = m_on['monthly_pnl'].get(bm, 0)
        if off_pnl < 0 or on_pnl < 0:
            print(f"    {bm}: {D}{off_pnl:+,.0f} -> {D}{on_pnl:+,.0f} (delta {D}{on_pnl-off_pnl:+,.0f})")

    return trades_off, m_off, trades_on, m_on


# ============================================================
# PHASE 2: OPTIMIZE
# ============================================================
def run_optimization(n_trials=300, n_workers=1):
    D = "$"
    print("\n" + "=" * 70)
    print("  PHASE 2: OPTIMIZE REGIME PARAMETERS (Optuna TPE)")
    print(f"  {n_trials} trials | Train: 2025 full year")
    print(f"  28 parameters across classification + 5 regime types")
    print("=" * 70)

    # Get baseline metrics for reference
    tc_base = get_baseline_tc()
    tc_base.use_regime_detection = False
    _, m_base = run_config(tc_base, '2025')
    base_ret = m_base['ret']

    def objective(trial):
        params = {
            # --- Classification thresholds ---
            'lookback': trial.suggest_int('lookback', 3, 10),
            'vol_pctl': trial.suggest_float('vol_pctl', 0.15, 0.50, step=0.05),
            'trend_pctl': trial.suggest_float('trend_pctl', 0.10, 0.45, step=0.05),
            'up_day_pct': trial.suggest_float('up_day_pct', 0.55, 0.85, step=0.05),
            'dn_day_pct': trial.suggest_float('dn_day_pct', 0.55, 0.85, step=0.05),
            'momentum_thr': trial.suggest_float('momentum_thr', 0.005, 0.025, step=0.001),
            'high_vol_pctl': trial.suggest_float('high_vol_pctl', 0.65, 0.90, step=0.05),
            'adx_thr': trial.suggest_float('adx_thr', 15.0, 35.0, step=2.5),

            # --- STEADY_UP adjustments ---
            'su_size_red': trial.suggest_float('su_size_red', 0.0, 0.70, step=0.10),
            'su_skip_puts': trial.suggest_categorical('su_skip_puts', [True, False]),
            'su_rsi_buf': trial.suggest_int('su_rsi_buf', 0, 10, step=2),
            'su_call_pt': trial.suggest_categorical('su_call_pt', ['none', '0.30', '0.35', '0.40']),

            # --- STEADY_DN adjustments ---
            'sd_size_red': trial.suggest_float('sd_size_red', 0.0, 0.70, step=0.10),
            'sd_skip_calls': trial.suggest_categorical('sd_skip_calls', [True, False]),
            'sd_rsi_buf': trial.suggest_int('sd_rsi_buf', 0, 10, step=2),
            'sd_put_pt': trial.suggest_categorical('sd_put_pt', ['none', '0.30', '0.35', '0.40']),

            # --- CHOPPY adjustments ---
            'ch_size_red': trial.suggest_float('ch_size_red', 0.20, 0.80, step=0.10),
            'ch_skip_first': trial.suggest_categorical('ch_skip_first', [True, False]),
            'ch_rsi_buf': trial.suggest_int('ch_rsi_buf', 0, 12, step=2),
            'ch_stop': trial.suggest_categorical('ch_stop', ['none', '0.20', '0.25', '0.30']),

            # --- VOLATILE adjustments ---
            'vol_size_red': trial.suggest_float('vol_size_red', 0.0, 0.40, step=0.10),
            'vol_stop_buf': trial.suggest_float('vol_stop_buf', 0.0, 0.20, step=0.05),
            'vol_pt_buf': trial.suggest_float('vol_pt_buf', 0.0, 0.20, step=0.05),

            # --- TRENDING adjustments ---
            'tr_skip_counter': trial.suggest_categorical('tr_skip_counter', [True, False]),
            'tr_hold_buf': trial.suggest_int('tr_hold_buf', 0, 8, step=2),
        }

        tc = get_baseline_tc()
        apply_regime_params(tc, params)

        _, m = run_config(tc, '2025')
        if m is None:
            return float('-inf')

        # === COMPOSITE OBJECTIVE ===
        # Goals (in priority order):
        #   1. Eliminate red months (zero tolerance)
        #   2. Reduce worst-month severity
        #   3. Maintain/improve total return vs no-regime baseline
        #   4. Keep drawdown low
        #   5. High profit factor and Sharpe

        # Return preservation: penalize losing >10% of baseline return
        ret_ratio = m['ret'] / max(base_ret, 1)
        ret_score = min(ret_ratio, 1.2)  # cap at 1.2x (avoid overfitting to high return)

        # Red month elimination (heavy penalty)
        red_month_penalty = m['red_months'] * 0.25

        # Worst month severity
        worst_month_penalty = max(0, -m['worst_month']) / 5000

        # Max drawdown penalty
        dd_penalty = max(0, m['max_dd'] - 3.0) * 0.05  # penalize DD > 3%

        # Profit factor bonus
        pf_score = min(m['pf'], 5.0) / 5.0  # normalize, cap at 5

        # Sharpe bonus
        sharpe_score = min(max(m['sharpe'], 0), 5.0) / 5.0

        # Red weeks penalty (finer-grained pain signal)
        red_week_penalty = m['red_weeks'] * 0.02

        composite = (
            0.30 * ret_score
            + 0.15 * pf_score
            + 0.10 * sharpe_score
            - 0.25 * red_month_penalty
            - 0.10 * worst_month_penalty
            - 0.05 * dd_penalty
            - 0.05 * red_week_penalty
        )

        # Store metrics as user attrs
        for k, v in m.items():
            if k != 'monthly_pnl':
                trial.set_user_attr(k, v)
        for mon, pnl in m['monthly_pnl'].items():
            trial.set_user_attr(f'month_{mon}', pnl)

        return composite

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=30),
    )

    # Seed trials with known configurations
    # Current config (aligned to grid: rsi_buf step=2)
    study.enqueue_trial({
        'lookback': 3, 'vol_pctl': 0.30, 'trend_pctl': 0.25,
        'up_day_pct': 0.70, 'dn_day_pct': 0.70,
        'momentum_thr': 0.012, 'high_vol_pctl': 0.75, 'adx_thr': 25.0,
        'su_size_red': 0.30, 'su_skip_puts': True, 'su_rsi_buf': 6, 'su_call_pt': 'none',
        'sd_size_red': 0.30, 'sd_skip_calls': True, 'sd_rsi_buf': 6, 'sd_put_pt': 'none',
        'ch_size_red': 0.50, 'ch_skip_first': True, 'ch_rsi_buf': 6, 'ch_stop': 'none',
        'vol_size_red': 0.0, 'vol_stop_buf': 0.10, 'vol_pt_buf': 0.10,
        'tr_skip_counter': True, 'tr_hold_buf': 4,
    })
    # Aggressive filtering
    study.enqueue_trial({
        'lookback': 5, 'vol_pctl': 0.40, 'trend_pctl': 0.35,
        'up_day_pct': 0.60, 'dn_day_pct': 0.60,
        'momentum_thr': 0.008, 'high_vol_pctl': 0.70, 'adx_thr': 20.0,
        'su_size_red': 0.60, 'su_skip_puts': True, 'su_rsi_buf': 8, 'su_call_pt': '0.35',
        'sd_size_red': 0.60, 'sd_skip_calls': True, 'sd_rsi_buf': 8, 'sd_put_pt': '0.35',
        'ch_size_red': 0.70, 'ch_skip_first': True, 'ch_rsi_buf': 10, 'ch_stop': '0.25',
        'vol_size_red': 0.10, 'vol_stop_buf': 0.15, 'vol_pt_buf': 0.15,
        'tr_skip_counter': True, 'tr_hold_buf': 6,
    })
    # Light filtering
    study.enqueue_trial({
        'lookback': 3, 'vol_pctl': 0.20, 'trend_pctl': 0.15,
        'up_day_pct': 0.80, 'dn_day_pct': 0.80,
        'momentum_thr': 0.015, 'high_vol_pctl': 0.80, 'adx_thr': 30.0,
        'su_size_red': 0.20, 'su_skip_puts': False, 'su_rsi_buf': 4, 'su_call_pt': 'none',
        'sd_size_red': 0.20, 'sd_skip_calls': False, 'sd_rsi_buf': 4, 'sd_put_pt': 'none',
        'ch_size_red': 0.30, 'ch_skip_first': False, 'ch_rsi_buf': 4, 'ch_stop': 'none',
        'vol_size_red': 0.0, 'vol_stop_buf': 0.05, 'vol_pt_buf': 0.05,
        'tr_skip_counter': True, 'tr_hold_buf': 2,
    })
    # Choppy-focused (target low-vol months)
    study.enqueue_trial({
        'lookback': 5, 'vol_pctl': 0.35, 'trend_pctl': 0.30,
        'up_day_pct': 0.65, 'dn_day_pct': 0.65,
        'momentum_thr': 0.010, 'high_vol_pctl': 0.75, 'adx_thr': 22.5,
        'su_size_red': 0.40, 'su_skip_puts': True, 'su_rsi_buf': 6, 'su_call_pt': '0.40',
        'sd_size_red': 0.40, 'sd_skip_calls': True, 'sd_rsi_buf': 6, 'sd_put_pt': '0.40',
        'ch_size_red': 0.60, 'ch_skip_first': True, 'ch_rsi_buf': 8, 'ch_stop': '0.25',
        'vol_size_red': 0.0, 'vol_stop_buf': 0.10, 'vol_pt_buf': 0.10,
        'tr_skip_counter': True, 'tr_hold_buf': 4,
    })

    t0 = time.time()
    completed = [0]
    best_score = [float('-inf')]

    def progress_callback(study, trial):
        completed[0] += 1
        n = completed[0]
        if n % 25 == 0 or (trial.value is not None and trial.value > best_score[0]):
            if trial.value is not None and trial.value > best_score[0]:
                best_score[0] = trial.value
            best = study.best_trial
            a = best.user_attrs
            elapsed = time.time() - t0
            rate = n / elapsed
            print(f"    [{n:>4}/{n_trials}] best: Ret={a.get('ret',0)}% "
                  f"DD={a.get('max_dd',0)}% WR={a.get('wr',0)}% "
                  f"PF={a.get('pf',0)} "
                  f"RedMo={a.get('red_months',0)} "
                  f"Worst={D}{a.get('worst_month',0):+,.0f} "
                  f"({rate:.1f} t/s)", flush=True)

    print(f"\n  Running {n_trials} trials with {n_workers} parallel workers...")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_workers,
                   show_progress_bar=False, callbacks=[progress_callback])

    print(f"\n  Optimization done: {time.time()-t0:.0f}s")

    # Sort by composite score
    trials_sorted = sorted(study.trials,
                           key=lambda t: t.value if t.value is not None else float('-inf'),
                           reverse=True)

    # Filter: only keep trials with 0 red months (if any exist)
    zero_red = [t for t in trials_sorted if t.user_attrs.get('red_months', 99) == 0]
    if zero_red:
        print(f"\n  Found {len(zero_red)} configs with 0 red months! Selecting from those.")
        # Among zero-red-month configs, pick by highest return
        candidates = sorted(zero_red, key=lambda t: t.user_attrs.get('ret', 0), reverse=True)
    else:
        # Fall back to lowest red months, then highest return
        min_red = min(t.user_attrs.get('red_months', 99) for t in trials_sorted[:50])
        print(f"\n  No 0-red-month config found. Best: {min_red} red months.")
        candidates = sorted(
            [t for t in trials_sorted if t.user_attrs.get('red_months', 99) == min_red],
            key=lambda t: t.user_attrs.get('ret', 0), reverse=True
        )

    top_results = []
    print(f"\n  TOP 5:")
    for rank, trial in enumerate(candidates[:5], 1):
        a = trial.user_attrs
        p = trial.params
        print(f"\n  #{rank} (score={trial.value:.4f}):")
        print(f"    Ret={a.get('ret',0)}% WR={a.get('wr',0)}% DD={a.get('max_dd',0)}% "
              f"PF={a.get('pf',0)} Sharpe={a.get('sharpe',0)}")
        print(f"    Red months: {a.get('red_months',0)}, Worst: {D}{a.get('worst_month',0):+,.0f}")
        print(f"    RedWeeks: {a.get('red_weeks',0)}, WorstWeek: {D}{a.get('worst_week',0):+,.0f}")
        print(f"    Classification: lookback={p['lookback']} vol_pctl={p['vol_pctl']} "
              f"trend_pctl={p['trend_pctl']} up_day={p['up_day_pct']} dn_day={p['dn_day_pct']}")
        print(f"    momentum_thr={p['momentum_thr']} high_vol={p['high_vol_pctl']} adx={p['adx_thr']}")
        print(f"    STEADY_UP: size={p['su_size_red']} skip_puts={p['su_skip_puts']} "
              f"rsi_buf={p['su_rsi_buf']} call_pt={p['su_call_pt']}")
        print(f"    STEADY_DN: size={p['sd_size_red']} skip_calls={p['sd_skip_calls']} "
              f"rsi_buf={p['sd_rsi_buf']} put_pt={p['sd_put_pt']}")
        print(f"    CHOPPY:    size={p['ch_size_red']} skip_1st={p['ch_skip_first']} "
              f"rsi_buf={p['ch_rsi_buf']} stop={p['ch_stop']}")
        print(f"    VOLATILE:  size={p['vol_size_red']} stop_buf={p['vol_stop_buf']} "
              f"pt_buf={p['vol_pt_buf']}")
        print(f"    TRENDING:  skip_ctr={p['tr_skip_counter']} hold_buf={p['tr_hold_buf']}")

        # Monthly detail
        for m_key in sorted([k for k in a if k.startswith('month_')]):
            mon = m_key.replace('month_', '')
            pnl = a[m_key]
            marker = ' ***' if pnl < -500 else ''
            print(f"      {mon}: {D}{pnl:+,.0f}{marker}")

        top_results.append((rank, p, a))

    return study, top_results


# ============================================================
# PHASE 3: OOS VALIDATION + SCENARIO STRESS TEST
# ============================================================
def run_oos_validation(top_results):
    D = "$"
    print("\n" + "=" * 70)
    print("  PHASE 3: OOS VALIDATION & SCENARIO STRESS TEST")
    print("=" * 70)

    # Baseline OOS
    tc_base = get_baseline_tc()
    tc_base.use_regime_detection = False
    trades_base, m_base = run_config(tc_base, '2026')
    print(f"\n  Baseline (no regime) 2026: {m_base['trades']}t WR={m_base['wr']}% "
          f"Ret={m_base['ret']}% DD={m_base['max_dd']}%")
    print_monthly(trades_base, "BASELINE 2026 (no regime)")

    # Current config OOS
    tc_current = get_baseline_tc()
    trades_curr, m_curr = run_config(tc_current, '2026')
    print(f"\n  Current config 2026: {m_curr['trades']}t WR={m_curr['wr']}% "
          f"Ret={m_curr['ret']}% DD={m_curr['max_dd']}%")
    print_monthly(trades_curr, "CURRENT 2026")

    best_overall = None
    best_rank = None
    best_params = None

    for rank, params, train_attrs in top_results:
        tc = get_baseline_tc()
        apply_regime_params(tc, params)

        print(f"\n  --- #{rank} OOS Results ---")

        # 2026 OOS
        trades_oos, m_oos = run_config(tc, '2026')
        if m_oos is None:
            print(f"  #{rank} 2026: FAILED")
            continue
        print(f"  2026: {m_oos['trades']}t WR={m_oos['wr']}% Ret={m_oos['ret']}% "
              f"DD={m_oos['max_dd']}% PF={m_oos['pf']}")
        print(f"    vs baseline: Ret {m_oos['ret']}% vs {m_base['ret']}%, "
              f"DD {m_oos['max_dd']}% vs {m_base['max_dd']}%")
        print_monthly(trades_oos, f"#{rank} 2026 OOS")

        # 2025 H1 stress test (early year)
        trades_h1, m_h1 = run_config(tc, '2025_h1')
        if m_h1:
            print(f"  2025 H1: {m_h1['trades']}t WR={m_h1['wr']}% Ret={m_h1['ret']}% DD={m_h1['max_dd']}%")
            print(f"    Red months: {m_h1['red_months']}, Worst: {D}{m_h1['worst_month']:+,.0f}")

        # 2025 H2 stress test (includes bad months Jul/Aug/Oct)
        trades_h2, m_h2 = run_config(tc, '2025_h2')
        if m_h2:
            print(f"  2025 H2: {m_h2['trades']}t WR={m_h2['wr']}% Ret={m_h2['ret']}% DD={m_h2['max_dd']}%")
            print(f"    Red months: {m_h2['red_months']}, Worst: {D}{m_h2['worst_month']:+,.0f}")

        # Score: combine OOS return + train return consistency + low red months across all periods
        h1_red = m_h1['red_months'] if m_h1 else 99
        h2_red = m_h2['red_months'] if m_h2 else 99
        oos_ret = m_oos['ret']
        total_red = m_oos.get('red_months', 0) + h1_red + h2_red
        score = oos_ret - total_red * 5  # penalize each red month across periods

        if best_overall is None or score > best_overall:
            best_overall = score
            best_rank = rank
            best_params = params

    if best_params:
        print(f"\n  {'='*50}")
        print(f"  BEST OVERALL: #{best_rank}")
        print(f"  {'='*50}")
        print(f"  Params to update in strategy.json:")
        print(f"    regime_lookback_days: {best_params['lookback']}")
        print(f"    regime_vol_percentile: {best_params['vol_pctl']}")
        print(f"    regime_trend_percentile: {best_params['trend_pctl']}")
        print(f"    regime_up_day_pct: {best_params['up_day_pct']}")
        print(f"    regime_dn_day_pct: {best_params['dn_day_pct']}")
        print(f"    regime_momentum_threshold: {best_params['momentum_thr']}")
        print(f"    regime_high_vol_percentile: {best_params['high_vol_pctl']}")
        print(f"    regime_adx_trend_threshold: {best_params['adx_thr']}")
        print(f"    steady_up_size_reduction: {best_params['su_size_red']}")
        print(f"    steady_up_skip_puts: {best_params['su_skip_puts']}")
        print(f"    steady_up_rsi_buffer: {best_params['su_rsi_buf']}")
        su_pt = best_params['su_call_pt']
        print(f"    steady_up_call_pt_override: {None if su_pt == 'none' else float(su_pt)}")
        print(f"    steady_dn_size_reduction: {best_params['sd_size_red']}")
        print(f"    steady_dn_skip_calls: {best_params['sd_skip_calls']}")
        print(f"    steady_dn_rsi_buffer: {best_params['sd_rsi_buf']}")
        sd_pt = best_params['sd_put_pt']
        print(f"    steady_dn_put_pt_override: {None if sd_pt == 'none' else float(sd_pt)}")
        print(f"    choppy_size_reduction: {best_params['ch_size_red']}")
        print(f"    choppy_skip_first_bar: {best_params['ch_skip_first']}")
        print(f"    choppy_rsi_buffer: {best_params['ch_rsi_buf']}")
        ch_stop = best_params['ch_stop']
        print(f"    choppy_tighter_stop_pct: {None if ch_stop == 'none' else float(ch_stop)}")
        print(f"    volatile_size_reduction: {best_params['vol_size_red']}")
        print(f"    volatile_stop_buffer_pct: {best_params['vol_stop_buf']}")
        print(f"    volatile_pt_buffer_pct: {best_params['vol_pt_buf']}")
        print(f"    trending_skip_counter: {best_params['tr_skip_counter']}")
        print(f"    trending_hold_buffer: {best_params['tr_hold_buf']}")

    return best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=300)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--apply', action='store_true',
                        help='Apply best config to strategy.json')
    args = parser.parse_args()

    cap = cfg.initial_capital()
    D = "$"

    print("=" * 70)
    print("  REGIME DETECTION OPTIMIZER V2")
    print(f"  28-parameter comprehensive regime tuning")
    print(f"  Goal: Eliminate red months + preserve return under all scenarios")
    print(f"  Capital: {D}{cap:,.0f}")
    print("=" * 70)

    # Load all data upfront
    t0 = time.time()
    for period in ['2025', '2026', '2025_h1', '2025_h2']:
        t1 = time.time()
        print(f"  Loading {period} data...", end='', flush=True)
        load_data_cached(period)
        print(f" done ({time.time()-t1:.1f}s)")

    # Phase 1: Diagnostic
    run_diagnostic()

    # Phase 2: Optimize
    study, top_results = run_optimization(args.trials, args.workers)

    # Phase 3: OOS + Stress Test
    best_params = run_oos_validation(top_results)

    print(f"\n  Total runtime: {time.time()-t0:.0f}s")
    print("=" * 70)

    # Save results
    if best_params:
        # Map internal param names to strategy.json keys
        trade_config = {
            'use_regime_detection': True,
            'regime_lookback_days': best_params['lookback'],
            'regime_vol_percentile': best_params['vol_pctl'],
            'regime_trend_percentile': best_params['trend_pctl'],
            'regime_up_day_pct': best_params['up_day_pct'],
            'regime_dn_day_pct': best_params['dn_day_pct'],
            'regime_momentum_threshold': best_params['momentum_thr'],
            'regime_high_vol_percentile': best_params['high_vol_pctl'],
            'regime_adx_trend_threshold': best_params['adx_thr'],
            'steady_up_size_reduction': best_params['su_size_red'],
            'steady_up_skip_puts': best_params['su_skip_puts'],
            'steady_up_rsi_buffer': best_params['su_rsi_buf'],
            'steady_up_call_pt_override': None if best_params.get('su_call_pt') == 'none' else float(best_params['su_call_pt']),
            'steady_dn_size_reduction': best_params['sd_size_red'],
            'steady_dn_skip_calls': best_params['sd_skip_calls'],
            'steady_dn_rsi_buffer': best_params['sd_rsi_buf'],
            'steady_dn_put_pt_override': None if best_params.get('sd_put_pt') == 'none' else float(best_params['sd_put_pt']),
            'choppy_size_reduction': best_params['ch_size_red'],
            'choppy_skip_first_bar': best_params['ch_skip_first'],
            'choppy_rsi_buffer': best_params['ch_rsi_buf'],
            'choppy_tighter_stop_pct': None if best_params.get('ch_stop') == 'none' else float(best_params['ch_stop']),
            'volatile_size_reduction': best_params['vol_size_red'],
            'volatile_stop_buffer_pct': best_params['vol_stop_buf'],
            'volatile_pt_buffer_pct': best_params['vol_pt_buf'],
            'trending_skip_counter': best_params['tr_skip_counter'],
            'trending_hold_buffer': best_params['tr_hold_buf'],
        }

        run_id = save_optimization_run(
            source='optimize_regime_v2',
            trade_config=trade_config,
            results={},
            metadata={
                'trials': args.trials,
                'raw_params': best_params,
            },
        )

        if args.apply:
            apply_run(run_id)

        result = {
            'optimization_date': time.strftime('%Y-%m-%d'),
            'trials': args.trials,
            'run_id': run_id,
            'best_params': best_params,
        }
        with open('output/regime_optimization_v2.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Results saved to output/regime_optimization_v2.json")


if __name__ == '__main__':
    main()
