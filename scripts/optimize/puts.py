"""
PUT Option Optimizer — Bayesian search for PUT-specific parameters

Optimizes PUT-specific dimensions while keeping CALL config fixed at Phase 8 values.
Training on 2025, validating on 2026 OOS.

Dimensions (10):
  1. put_profit_target_pct     (PT for PUTs)
  2. put_stop_loss_pct         (SL for PUTs)
  3. put_max_hold_bars         (hold time for PUTs)
  4. rsi_put_threshold         (RSI entry threshold)
  5. put_min_rsi               (RSI floor filter)
  6. put_skip_days             (weekday filter)
  7. put_min_entry_minutes     (time-of-day filter)
  8. put_loss_streak_threshold (adaptive streak trigger)
  9. put_adaptive_cooldown     (cooldown after streak)
 10. put_filter_require_uptrend (gate filters on uptrend only)

Usage:
  python scripts/optimize_puts.py [--trials 200]
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

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# GLOBAL DATA CACHE
# ============================================================
_DATA_CACHE = {}


def load_data_cached(period='2025'):
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


def run_single_config(params, period='2025'):
    """Run a single backtest config and return metrics."""
    u, o, f, v, dt_idx, td_idx = load_data_cached(period)
    cap = cfg.initial_capital()

    # CALL config fixed at Phase 8 values, PUT config varies
    tc = TradeConfig(
        strategy='momentum',
        rsi_call_threshold=70,
        rsi_put_threshold=params['rsi_put'],
        # Shared defaults (CALL uses these)
        profit_target_pct=0.50,
        stop_loss_pct=0.35,
        max_hold_bars=16,
        # Asymmetric: CALL fixed, PUT varies
        call_profit_target_pct=0.50,
        put_profit_target_pct=params['put_pt'],
        call_stop_loss_pct=0.35,
        put_stop_loss_pct=params['put_sl'],
        call_max_hold_bars=16,
        put_max_hold_bars=params['put_hold'],
        # Option selection
        min_option_price=0.50,
        max_option_price=2.00,
        # Disable everything not needed
        use_adaptive_exits=False,
        use_trailing_stop=False,
        use_time_decay_exits=False,
        use_time_decay_exit=False,
        use_quick_exit=False,
        use_ml_filter=False,
        skip_day_filter=True,
        # PUT filters
        put_min_rsi=params['put_min_rsi'],
        put_skip_days=params['put_skip_days'],
        put_min_entry_minutes=params['put_min_entry_minutes'],
        put_filter_require_uptrend=params['put_filter_require_uptrend'],
        put_adaptive_filter=params['put_adaptive_filter'],
        put_loss_streak_threshold=params['put_loss_streak_threshold'],
        put_adaptive_cooldown=params['put_adaptive_cooldown'],
        # Post-loss: keep at none (Phase 8)
        post_loss_strategy='none',
    )

    rc = RiskConfig(
        kelly_fraction=0.2,
        max_risk_per_trade_pct=0.02,
        max_position_pct=0.07,
        max_position_value=5000,
        max_daily_losses=999,
        max_daily_loss_pct=0.008,
        max_consecutive_losses=3,
        consec_loss_reduction=0.5,
        wins_to_reset_streak=2,
        reduce_size_at_dd_pct=0.99,
        max_dd_reduction=0.5,
        max_trades_per_day=999,
    )

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

    return compute_metrics(trades, cap)


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

    gp = sum(t.pnl for t in trades if t.pnl > 0)
    gl = abs(sum(t.pnl for t in trades if t.pnl <= 0)) or 0.01
    pf = gp / gl

    # PUT-specific metrics
    call_trades = [t for t in trades if t.direction == 'CALL']
    put_trades = [t for t in trades if t.direction == 'PUT']
    call_pnl = sum(t.pnl for t in call_trades)
    put_pnl = sum(t.pnl for t in put_trades)
    call_wr = sum(1 for t in call_trades if t.pnl > 0) / max(len(call_trades), 1) * 100
    put_wr = sum(1 for t in put_trades if t.pnl > 0) / max(len(put_trades), 1) * 100

    put_gp = sum(t.pnl for t in put_trades if t.pnl > 0)
    put_gl = abs(sum(t.pnl for t in put_trades if t.pnl <= 0)) or 0.01
    put_pf = put_gp / put_gl

    # PUT stop rate
    put_stops = sum(1 for t in put_trades if t.exit_reason == 'STOP')
    put_stop_rate = put_stops / max(len(put_trades), 1) * 100

    return {
        'trades': n, 'wr': round(wr, 1),
        'pnl': round(total_pnl, 0), 'ret': round(ret, 1),
        'max_dd': round(max_dd * 100, 1),
        'sharpe': round(sharpe, 2), 'pf': round(pf, 2),
        'call_trades': len(call_trades), 'put_trades': len(put_trades),
        'call_pnl': round(call_pnl, 0), 'put_pnl': round(put_pnl, 0),
        'call_wr': round(call_wr, 1), 'put_wr': round(put_wr, 1),
        'put_pf': round(put_pf, 2), 'put_stop_rate': round(put_stop_rate, 1),
    }


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

# Weekday skip options (combinations to try)
SKIP_DAY_OPTIONS = {
    'none': None,
    'mon': [0],
    'fri': [4],
    'mon_fri': [0, 4],
    'mon_tue': [0, 1],
}


def create_objective(period='2025'):
    def objective(trial):
        # PUT exit parameters
        put_pt = trial.suggest_float('put_pt', 0.25, 0.70, step=0.05)
        put_sl = trial.suggest_float('put_sl', 0.20, 0.45, step=0.05)
        put_hold = trial.suggest_int('put_hold', 4, 20, step=2)

        # RSI thresholds
        rsi_put = trial.suggest_int('rsi_put', 20, 35, step=5)
        put_min_rsi = trial.suggest_float('put_min_rsi', 0, 25, step=5)

        # Time filters
        skip_key = trial.suggest_categorical('put_skip_days', list(SKIP_DAY_OPTIONS.keys()))
        # Minutes since midnight: 0=no filter, 585=9:45, 600=10:00, 615=10:15, 630=10:30
        put_min_entry_min = trial.suggest_categorical('put_min_entry_min', [0, 585, 600, 615, 630])

        # Adaptive filter
        put_adaptive = trial.suggest_categorical('put_adaptive', [True, False])
        streak_thr = trial.suggest_int('put_streak_thr', 1, 4) if put_adaptive else 2
        cooldown = trial.suggest_int('put_cooldown', 1, 8) if put_adaptive else 3

        # Uptrend gating
        require_uptrend = trial.suggest_categorical('require_uptrend', [True, False])

        params = {
            'put_pt': put_pt,
            'put_sl': put_sl,
            'put_hold': put_hold,
            'rsi_put': rsi_put,
            'put_min_rsi': put_min_rsi,
            'put_skip_days': SKIP_DAY_OPTIONS[skip_key],
            'put_min_entry_minutes': put_min_entry_min,
            'put_filter_require_uptrend': require_uptrend,
            'put_adaptive_filter': put_adaptive,
            'put_loss_streak_threshold': streak_thr,
            'put_adaptive_cooldown': cooldown,
        }

        m = run_single_config(params, period=period)
        if m is None:
            return float('-inf')

        # Reject configs where PUTs lose money on 2025 training data
        if m['put_pnl'] < 0:
            return float('-inf')

        # Multi-objective composite focused on PUT quality + overall return
        # We want: good total return, good PUT WR, good PUT PF, low DD
        ret_score = m['ret'] / 700  # normalize to ~1.0 for baseline
        put_wr_score = (m['put_wr'] - 40) / 30  # reward PUT WR above 40%
        put_pf_score = min(m['put_pf'] / 3, 1)  # reward PUT profit factor
        overall_pf_score = min(m['pf'] / 2, 1)
        dd_penalty = max(0, (m['max_dd'] - 15) / 10)  # penalize DD > 15%
        # Penalize configs that kill PUT trade count too much
        put_trade_score = min(m['put_trades'] / 200, 1.0)

        composite = (
            0.25 * ret_score +       # total return matters
            0.25 * put_wr_score +    # PUT win rate is critical
            0.15 * put_pf_score +    # PUT profit factor
            0.15 * overall_pf_score + # overall quality
            0.10 * put_trade_score + # don't over-filter PUTs
            0.10 * (1 - dd_penalty)  # risk control
        )

        for k, v in m.items():
            trial.set_user_attr(k, v)
        for k, v in params.items():
            if isinstance(v, list):
                trial.set_user_attr(f'p_{k}', str(v))
            elif v is None:
                trial.set_user_attr(f'p_{k}', 'None')
            else:
                trial.set_user_attr(f'p_{k}', v)

        return composite

    return objective


# ============================================================
# BASELINE
# ============================================================
BASELINE_PARAMS = {
    'put_pt': 0.50,
    'put_sl': 0.35,
    'put_hold': 16,
    'rsi_put': 30,
    'put_min_rsi': 25,
    'put_skip_days': [0],
    'put_min_entry_minutes': 610,
    'put_filter_require_uptrend': True,
    'put_adaptive_filter': True,
    'put_loss_streak_threshold': 2,
    'put_adaptive_cooldown': 3,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--workers', type=int, default=6,
                        help='Parallel workers (default: 6)')
    args = parser.parse_args()
    n_trials = args.trials
    n_workers = args.workers

    cap = cfg.initial_capital()
    D = "$"

    print('=' * 70)
    print('  PUT OPTION OPTIMIZER — Bayesian (Optuna TPE)')
    print(f'  {n_trials} trials | 10 PUT-specific dimensions')
    print(f'  Train: 2025 | Validate: 2026 OOS | {D}{cap:,.0f} capital')
    print('=' * 70)

    # Load data
    t0 = time.time()
    print('\n  Loading 2025 data...', end='', flush=True)
    load_data_cached('2025')
    print(f' done ({time.time()-t0:.1f}s)')
    t1 = time.time()
    print('  Loading 2026 data...', end='', flush=True)
    load_data_cached('2026')
    print(f' done ({time.time()-t1:.1f}s)')

    # Baseline
    print('\n  Running baseline (current Phase 8 config)...')
    base_2025 = run_single_config(BASELINE_PARAMS, '2025')
    base_2026 = run_single_config(BASELINE_PARAMS, '2026')
    print(f'    2025: {base_2025["trades"]}t, WR={base_2025["wr"]}%, Ret={base_2025["ret"]}%, '
          f'DD={base_2025["max_dd"]}%, PUT: {base_2025["put_trades"]}t WR={base_2025["put_wr"]}% '
          f'PnL={D}{base_2025["put_pnl"]:+,.0f} PF={base_2025["put_pf"]}')
    print(f'    2026: {base_2026["trades"]}t, WR={base_2026["wr"]}%, Ret={base_2026["ret"]}%, '
          f'DD={base_2026["max_dd"]}%, PUT: {base_2026["put_trades"]}t WR={base_2026["put_wr"]}% '
          f'PnL={D}{base_2026["put_pnl"]:+,.0f} PF={base_2026["put_pf"]}')

    # Optimize
    print(f'\n  Running {n_trials} optimization trials...')
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=30),
    )

    # Seed with baseline
    study.enqueue_trial({
        'put_pt': 0.50, 'put_sl': 0.35, 'put_hold': 16,
        'rsi_put': 30, 'put_min_rsi': 25.0,
        'put_skip_days': 'mon', 'put_min_entry_min': 0,
        'put_adaptive': True, 'put_streak_thr': 2, 'put_cooldown': 3,
        'require_uptrend': True,
    })

    # Also seed some variations
    study.enqueue_trial({  # tighter SL, shorter hold
        'put_pt': 0.40, 'put_sl': 0.25, 'put_hold': 8,
        'rsi_put': 30, 'put_min_rsi': 20.0,
        'put_skip_days': 'mon', 'put_min_entry_min': 600,
        'put_adaptive': True, 'put_streak_thr': 1, 'put_cooldown': 5,
        'require_uptrend': False,
    })
    study.enqueue_trial({  # aggressive adaptive filter
        'put_pt': 0.50, 'put_sl': 0.35, 'put_hold': 12,
        'rsi_put': 25, 'put_min_rsi': 15.0,
        'put_skip_days': 'mon_fri', 'put_min_entry_min': 615,
        'put_adaptive': True, 'put_streak_thr': 1, 'put_cooldown': 8,
        'require_uptrend': True,
    })
    study.enqueue_trial({  # no filters (see raw PUT performance)
        'put_pt': 0.50, 'put_sl': 0.35, 'put_hold': 16,
        'rsi_put': 30, 'put_min_rsi': 0.0,
        'put_skip_days': 'none', 'put_min_entry_min': 0,
        'put_adaptive': False,
        'require_uptrend': False,
    })

    objective = create_objective('2025')
    t2 = time.time()
    best_score = [float('-inf')]
    completed = [0]

    def progress_callback(study, trial):
        completed[0] += 1
        n = completed[0]
        if n % 25 == 0 or (trial.value is not None and trial.value > best_score[0]):
            if trial.value is not None and trial.value > best_score[0]:
                best_score[0] = trial.value
            best = study.best_trial
            m = {k: best.user_attrs[k] for k in ['ret', 'put_wr', 'put_pf', 'put_pnl', 'max_dd'] if k in best.user_attrs}
            elapsed = time.time() - t2
            rate = n / elapsed
            eta = (n_trials - n) / rate
            print(f'    [{n:>4}/{n_trials}] best: Ret={m.get("ret",0)}% PUT_WR={m.get("put_wr",0)}% '
                  f'PUT_PF={m.get("put_pf",0)} PUT_PnL={D}{m.get("put_pnl",0):+,.0f} DD={m.get("max_dd",0)}% '
                  f'({rate:.1f} trials/s, ETA {eta:.0f}s)')

    print(f'\n  Running {n_trials} trials with {n_workers} parallel workers...')
    study.optimize(objective, n_trials=n_trials, n_jobs=n_workers,
                   show_progress_bar=False, callbacks=[progress_callback])

    print(f'\n  Optimization complete: {time.time()-t2:.0f}s')

    # ============================================================
    # TOP 10 RESULTS
    # ============================================================
    print('\n' + '=' * 70)
    print('  TOP 10 CONFIGS (by composite score)')
    print('=' * 70)

    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)
    top_configs = []

    for rank, trial in enumerate(trials_sorted[:10], 1):
        attrs = trial.user_attrs
        print(f'\n  #{rank} (score={trial.value:.4f}):')
        print(f'    Total: {attrs.get("trades",0)}t WR={attrs.get("wr",0)}% Ret={attrs.get("ret",0)}% '
              f'DD={attrs.get("max_dd",0)}% PF={attrs.get("pf",0)}')
        print(f'    PUT:   {attrs.get("put_trades",0)}t WR={attrs.get("put_wr",0)}% '
              f'PnL={D}{attrs.get("put_pnl",0):+,.0f} PF={attrs.get("put_pf",0)} '
              f'STOP={attrs.get("put_stop_rate",0)}%')
        print(f'    CALL:  {attrs.get("call_trades",0)}t WR={attrs.get("call_wr",0)}% '
              f'PnL={D}{attrs.get("call_pnl",0):+,.0f}')

        # Extract params
        p = {}
        for k, v in attrs.items():
            if k.startswith('p_'):
                p[k[2:]] = v
        print(f'    Params: PT={p.get("put_pt","")} SL={p.get("put_sl","")} Hold={p.get("put_hold","")} '
              f'RSI<={p.get("rsi_put","")} MinRSI={p.get("put_min_rsi","")}')
        print(f'    Filter: skip={p.get("put_skip_days","")} minTime={p.get("put_min_entry_minutes","")} '
              f'uptrend={p.get("put_filter_require_uptrend","")}')
        print(f'    Adaptive: on={p.get("put_adaptive_filter","")} streak={p.get("put_loss_streak_threshold","")} '
              f'cooldown={p.get("put_adaptive_cooldown","")}')

        if rank <= 5:
            top_configs.append((rank, trial.params, p, attrs))

    # ============================================================
    # OOS VALIDATION of TOP 5
    # ============================================================
    print('\n' + '=' * 70)
    print('  OOS VALIDATION — 2026 (Jan-Feb)')
    print('=' * 70)

    best_oos = None
    best_oos_rank = None

    for rank, trial_params, p_attrs, train_attrs in top_configs:
        # Reconstruct params dict
        skip_key = trial_params.get('put_skip_days', 'none')
        put_adaptive = trial_params.get('put_adaptive', True)

        params = {
            'put_pt': trial_params['put_pt'],
            'put_sl': trial_params['put_sl'],
            'put_hold': trial_params['put_hold'],
            'rsi_put': trial_params['rsi_put'],
            'put_min_rsi': trial_params['put_min_rsi'],
            'put_skip_days': SKIP_DAY_OPTIONS.get(skip_key),
            'put_min_entry_minutes': trial_params['put_min_entry_min'],
            'put_filter_require_uptrend': trial_params['require_uptrend'],
            'put_adaptive_filter': put_adaptive,
            'put_loss_streak_threshold': trial_params.get('put_streak_thr', 2),
            'put_adaptive_cooldown': trial_params.get('put_cooldown', 3),
        }

        oos = run_single_config(params, '2026')

        print(f'\n  #{rank} OOS:')
        if oos is None:
            print(f'    FAILED (too few trades)')
            continue

        print(f'    Total: {oos["trades"]}t WR={oos["wr"]}% Ret={oos["ret"]}% DD={oos["max_dd"]}%')
        print(f'    PUT:   {oos["put_trades"]}t WR={oos["put_wr"]}% PnL={D}{oos["put_pnl"]:+,.0f} '
              f'PF={oos["put_pf"]} STOP={oos["put_stop_rate"]}%')
        print(f'    CALL:  {oos["call_trades"]}t WR={oos["call_wr"]}% PnL={D}{oos["call_pnl"]:+,.0f}')
        print(f'    vs baseline 2026: PUT PnL {D}{oos["put_pnl"]:+,.0f} vs {D}{base_2026["put_pnl"]:+,.0f}')

        # Best OOS = best total return with PUT not losing catastrophically
        if best_oos is None or oos['ret'] > best_oos['ret']:
            best_oos = oos
            best_oos_rank = rank
            best_oos_params = params

    # ============================================================
    # SUMMARY
    # ============================================================
    print('\n' + '=' * 70)
    print('  SUMMARY — BEST OOS CONFIG')
    print('=' * 70)

    if best_oos:
        print(f'\n  Best OOS was #{best_oos_rank}:')
        print(f'    2026: Ret={best_oos["ret"]}% PUT WR={best_oos["put_wr"]}% '
              f'PUT PnL={D}{best_oos["put_pnl"]:+,.0f} PUT PF={best_oos["put_pf"]}')
        print(f'\n  Params to apply:')
        for k, v in best_oos_params.items():
            print(f'    {k}: {v}')

        print(f'\n  vs CURRENT Phase 8 baseline:')
        print(f'    2025 PUT: WR {base_2025["put_wr"]}% -> check top config')
        print(f'    2026 PUT: WR {base_2026["put_wr"]}% PnL {D}{base_2026["put_pnl"]:+,.0f} '
              f'-> WR {best_oos["put_wr"]}% PnL {D}{best_oos["put_pnl"]:+,.0f}')

    print(f'\n  Total runtime: {time.time()-t0:.0f}s')
    print('=' * 70)


if __name__ == '__main__':
    main()
