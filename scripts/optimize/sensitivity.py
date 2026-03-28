"""
Parameter Sensitivity Analysis & Focused Parallel Optimization

Phase 1: One-at-a-time (OAT) sensitivity sweep
  - Sweeps each parameter across its range while holding others at baseline
  - Measures impact on composite score (same objective as full.py)
  - Ranks parameters by sensitivity (score range / mean)

Phase 2: Focused optimization on top-N sensitive parameters
  - Freezes insensitive parameters at baseline values
  - Concentrates all trial budget on high-impact parameters
  - Runs Optuna TPE with narrower search space → faster convergence

Phase 3: OOS validation of best focused config

Usage:
  python scripts/optimize/sensitivity.py [--sweep-points N] [--trials N] [--top N] [--workers N]
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
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from backtest.engine import Backtest0DTE, TradeConfig
from core.risk_manager import RiskConfig
from config import defaults as cfg
from config.config_manager import save_optimization_run

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# REUSE: Data loading, config runner, metrics from full.py
# ============================================================
import importlib.util
_spec = importlib.util.spec_from_file_location('full', 'scripts/optimize/full.py')
_full_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_full_mod)

load_data_cached = _full_mod.load_data_cached
run_single_config = _full_mod.run_single_config
BASELINE_PARAMS = _full_mod.BASELINE_PARAMS
_load_strategy_json = _full_mod._load_strategy_json


# ============================================================
# PARAMETER DEFINITIONS: name → (type, range, step)
# ============================================================
PARAM_DEFS = {
    # Core trading params
    'call_pt':           ('float', 0.25, 0.55, 0.05),
    'put_pt':            ('float', 0.30, 0.60, 0.05),
    'call_sl':           ('float', 0.20, 0.45, 0.05),
    'put_sl':            ('float', 0.25, 0.45, 0.05),
    'rsi_call':          ('int',   65,   80,   5),
    'rsi_put':           ('int',   25,   40,   5),
    'call_hold':         ('int',   10,   20,   2),
    'put_hold':          ('int',    8,   18,   2),

    # Position sizing
    'kelly':             ('float', 0.04, 0.08, 0.01),
    'max_risk':          ('float', 0.02, 0.05, 0.01),
    'min_contracts':     ('int',   1,    2,    1),

    # Risk controls
    'max_daily_losses':  ('cat',   [1, 2, 3, 999]),
    'max_consec_losses': ('int',   2,    5,    1),
    'max_daily_loss_pct':('float', 0.005, 0.03, 0.005),

    # Post-loss
    'post_loss':         ('cat',   ['none', 'momentum_confirm', 'multi_confirm']),
    'momentum_threshold':('float', 0.02, 0.20, 0.02),

    # Regime tuning
    'regime_rsi_buffer':    ('int',   0,  15,  5),
    'regime_size_reduction':('float', 0.0, 0.30, 0.05),

    # PUT adaptive filter
    'put_loss_streak':   ('int',   1,    3,    1),
    'put_cooldown':      ('int',   2,    8,    2),
    'put_min_rsi':       ('int',   10,   25,   5),

    # Trading window
    'trade_end_hour':    ('int',   13,   15,   1),
}


def compute_composite(m):
    """Same composite score as full.py objective."""
    if m is None or m['trades'] < 100 or m['max_dd'] > 20:
        return float('-inf')

    ret_score = m['ret'] / 2100
    sharpe_score = m['sharpe'] / 8
    pf_score = min(m['pf'] / 4, 1)
    wr_score = (m['wr'] - 50) / 25
    trade_score = min(m['trades'] / 1200, 1.0)
    dd_penalty = max(0, (m['max_dd'] - 12) / 10)

    return (
        0.40 * ret_score +
        0.20 * sharpe_score +
        0.15 * trade_score +
        0.10 * pf_score +
        0.10 * wr_score +
        0.05 * (1 - dd_penalty)
    )


def generate_sweep_values(param_name, n_points=10):
    """Generate sweep values for a parameter."""
    defn = PARAM_DEFS[param_name]

    if defn[0] == 'cat':
        return defn[1]  # all categories
    elif defn[0] == 'int':
        lo, hi, step = defn[1], defn[2], defn[3]
        vals = list(range(lo, hi + 1, step))
        if len(vals) > n_points:
            indices = np.linspace(0, len(vals) - 1, n_points, dtype=int)
            vals = [vals[i] for i in indices]
        return vals
    else:  # float
        lo, hi, step = defn[1], defn[2], defn[3]
        vals = np.arange(lo, hi + step / 2, step).tolist()
        vals = [round(v, 4) for v in vals]
        if len(vals) > n_points:
            indices = np.linspace(0, len(vals) - 1, n_points, dtype=int)
            vals = [vals[i] for i in indices]
        return vals


# ============================================================
# PHASE 1: ONE-AT-A-TIME SENSITIVITY SWEEP
# ============================================================

def sweep_single_param(param_name, period='2025', n_points=10):
    """Sweep one parameter across its range, return (value, score, metrics) tuples."""
    sweep_vals = generate_sweep_values(param_name, n_points)
    results = []

    for val in sweep_vals:
        params = dict(BASELINE_PARAMS)
        params[param_name] = val
        m = run_single_config(params, period=period)
        score = compute_composite(m)
        results.append({
            'param': param_name,
            'value': val,
            'score': score,
            'ret': m['ret'] if m else None,
            'max_dd': m['max_dd'] if m else None,
            'sharpe': m['sharpe'] if m else None,
            'trades': m['trades'] if m else None,
            'wr': m['wr'] if m else None,
            'pf': m['pf'] if m else None,
        })

    return results


def run_sensitivity_analysis(period='2025', n_points=10, max_workers=6):
    """Run OAT sensitivity sweep across all parameters in parallel."""
    print('=' * 70)
    print('  PHASE 1: PARAMETER SENSITIVITY ANALYSIS')
    print(f'  {len(PARAM_DEFS)} parameters × ~{n_points} sweep points')
    print(f'  {max_workers} parallel workers')
    print('=' * 70)

    # Load data once upfront
    print('\n  Loading data...')
    t0 = time.time()
    load_data_cached(period)
    print(f'  Data loaded in {time.time() - t0:.1f}s')

    # Run baseline
    baseline_m = run_single_config(BASELINE_PARAMS, period=period)
    baseline_score = compute_composite(baseline_m)
    print(f'  BASELINE: score={baseline_score:.4f} | ret={baseline_m["ret"]:+.1f}% '
          f'dd={baseline_m["max_dd"]:.1f}% sharpe={baseline_m["sharpe"]:.2f} '
          f'trades={baseline_m["trades"]}')

    # Parallel sweep of all parameters
    all_results = []
    sensitivity = {}
    t_start = time.time()
    completed = 0

    print(f'\n  Sweeping parameters...')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for param_name in PARAM_DEFS:
            fut = executor.submit(sweep_single_param, param_name, period, n_points)
            futures[fut] = param_name

        for fut in as_completed(futures):
            param_name = futures[fut]
            completed += 1
            try:
                results = fut.result()
                all_results.extend(results)

                # Compute sensitivity metrics
                valid_scores = [r['score'] for r in results if r['score'] > float('-inf')]
                if len(valid_scores) >= 2:
                    score_range = max(valid_scores) - min(valid_scores)
                    score_std = np.std(valid_scores)
                    score_mean = np.mean(valid_scores)
                    # Coefficient of variation: how much the score varies relative to mean
                    cv = score_std / abs(score_mean) if score_mean != 0 else 0
                    best_val = results[np.argmax([r['score'] for r in results])]['value']
                    worst_val = results[np.argmin(
                        [r['score'] if r['score'] > float('-inf') else float('inf')
                         for r in results])]['value']

                    sensitivity[param_name] = {
                        'range': round(score_range, 6),
                        'std': round(score_std, 6),
                        'cv': round(cv, 6),
                        'mean': round(score_mean, 6),
                        'best_val': best_val,
                        'worst_val': worst_val,
                        'n_valid': len(valid_scores),
                        'n_total': len(results),
                    }

                    elapsed = time.time() - t_start
                    print(f'  [{completed:>2}/{len(PARAM_DEFS)}] {param_name:<22} '
                          f'range={score_range:.4f}  std={score_std:.4f}  '
                          f'best={best_val}  ({elapsed:.0f}s)', flush=True)
                else:
                    sensitivity[param_name] = {
                        'range': 0, 'std': 0, 'cv': 0, 'mean': 0,
                        'best_val': BASELINE_PARAMS.get(param_name),
                        'worst_val': None, 'n_valid': len(valid_scores),
                        'n_total': len(results),
                    }
                    print(f'  [{completed:>2}/{len(PARAM_DEFS)}] {param_name:<22} '
                          f'INSUFFICIENT DATA ({len(valid_scores)} valid)', flush=True)

            except Exception as e:
                print(f'  [{completed:>2}/{len(PARAM_DEFS)}] {param_name:<22} ERROR: {e}',
                      flush=True)
                sensitivity[param_name] = {
                    'range': 0, 'std': 0, 'cv': 0, 'mean': 0,
                    'best_val': BASELINE_PARAMS.get(param_name),
                    'worst_val': None, 'n_valid': 0, 'n_total': 0,
                }

    elapsed = time.time() - t_start
    print(f'\n  Sensitivity sweep completed in {elapsed:.0f}s ({elapsed/60:.1f}min)')

    return sensitivity, all_results, baseline_score


def rank_parameters(sensitivity):
    """Rank parameters by sensitivity (score range). Return sorted list."""
    ranked = sorted(sensitivity.items(), key=lambda x: x[1]['range'], reverse=True)
    return ranked


def print_sensitivity_report(ranked, baseline_score):
    """Print detailed sensitivity ranking report."""
    print(f'\n  {"="*70}')
    print(f'  PARAMETER SENSITIVITY RANKING')
    print(f'  Baseline score: {baseline_score:.4f}')
    print(f'  {"="*70}')
    print(f'  {"Rank":>4} {"Parameter":<24} {"ScoreRange":>10} {"StdDev":>10} '
          f'{"CV":>8} {"BestVal":>10} {"Tier":>8}')
    print(f'  {"─"*80}')

    # Classify into tiers based on score range distribution
    ranges = [s['range'] for _, s in ranked if s['range'] > 0]
    if ranges:
        p75 = np.percentile(ranges, 75)
        p50 = np.percentile(ranges, 50)
    else:
        p75 = p50 = 0

    tiers = {}
    for i, (name, s) in enumerate(ranked):
        if s['range'] >= p75:
            tier = 'HIGH'
        elif s['range'] >= p50:
            tier = 'MEDIUM'
        else:
            tier = 'LOW'
        tiers[name] = tier

        print(f'  {i+1:>4} {name:<24} {s["range"]:>10.4f} {s["std"]:>10.4f} '
              f'{s["cv"]:>8.4f} {str(s["best_val"]):>10} {tier:>8}')

    # Summary
    high = [n for n, t in tiers.items() if t == 'HIGH']
    medium = [n for n, t in tiers.items() if t == 'MEDIUM']
    low = [n for n, t in tiers.items() if t == 'LOW']

    print(f'\n  HIGH sensitivity ({len(high)}):   {", ".join(high)}')
    print(f'  MEDIUM sensitivity ({len(medium)}): {", ".join(medium)}')
    print(f'  LOW sensitivity ({len(low)}):    {", ".join(low)}')

    return tiers


# ============================================================
# PHASE 2: FOCUSED OPTIMIZATION ON SENSITIVE PARAMETERS
# ============================================================

def create_focused_objective(sensitive_params, period='2025'):
    """Create Optuna objective that only optimizes sensitive parameters."""

    def objective(trial):
        params = dict(BASELINE_PARAMS)

        for pname in sensitive_params:
            defn = PARAM_DEFS[pname]
            if defn[0] == 'cat':
                params[pname] = trial.suggest_categorical(pname, defn[1])
            elif defn[0] == 'int':
                params[pname] = trial.suggest_int(pname, defn[1], defn[2], step=defn[3])
            else:
                params[pname] = trial.suggest_float(pname, defn[1], defn[2], step=defn[3])

        m = run_single_config(params, period=period)
        score = compute_composite(m)

        if score > float('-inf') and m is not None:
            for k, v in m.items():
                trial.set_user_attr(k, v)
            for k, v in params.items():
                trial.set_user_attr(f'p_{k}', v)

        return score

    return objective


def run_focused_optimization(sensitive_params, sensitivity, n_trials=300,
                             n_workers=6, period='2025'):
    """Optimize only the most sensitive parameters."""
    print(f'\n{"="*70}')
    print(f'  PHASE 2: FOCUSED OPTIMIZATION')
    print(f'  Optimizing {len(sensitive_params)} sensitive parameters:')
    for p in sensitive_params:
        s = sensitivity[p]
        defn = PARAM_DEFS[p]
        if defn[0] == 'cat':
            rng = str(defn[1])
        else:
            rng = f'{defn[1]}–{defn[2]} (step {defn[3]})'
        print(f'    • {p:<24} range: {rng:<30} sensitivity: {s["range"]:.4f}')

    frozen = [p for p in PARAM_DEFS if p not in sensitive_params]
    print(f'\n  Frozen at baseline ({len(frozen)} params):')
    for p in frozen:
        print(f'    • {p:<24} = {BASELINE_PARAMS.get(p)}')

    print(f'\n  {n_trials} trials | {n_workers} workers | Optuna TPE')
    print(f'{"="*70}')

    # Create study with focused search space
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=min(20, n_trials // 5),
        ),
    )

    # Enqueue baseline (only sensitive params)
    baseline_trial = {p: BASELINE_PARAMS[p] for p in sensitive_params
                      if p in BASELINE_PARAMS}
    study.enqueue_trial(baseline_trial)

    # Enqueue "best from sensitivity" trial
    best_trial = {}
    for p in sensitive_params:
        best_val = sensitivity[p]['best_val']
        if best_val is not None:
            best_trial[p] = best_val
    if best_trial and best_trial != baseline_trial:
        study.enqueue_trial(best_trial)

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
                  f'| {elapsed:.0f}s ~{remaining:.0f}s left', flush=True)
        elif n % 50 == 0:
            print(f'  [{n:>3}/{n_trials}] best={best_score:.4f} | '
                  f'{elapsed:.0f}s ~{remaining:.0f}s left', flush=True)

    # Run optimization
    objective = create_focused_objective(sensitive_params, period=period)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_workers,
                   show_progress_bar=False, callbacks=[progress_callback])

    elapsed = time.time() - t_start
    print(f'\n  Focused optimization completed in {elapsed:.0f}s ({elapsed/60:.1f}min)')
    print(f'  Rate: {n_trials / elapsed:.1f} trials/sec')

    return study


def analyze_focused_results(study, sensitive_params, baseline_score, top_n=15):
    """Analyze focused optimization results."""
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
              and t.value is not None and t.value > float('-inf')]

    if not trials:
        print('  No completed trials!')
        return None

    trials.sort(key=lambda t: t.value, reverse=True)

    print(f'\n  FOCUSED OPTIMIZATION RESULTS')
    print(f'  Total completed trials: {len(trials)}')
    print(f'  Best score: {trials[0].value:.4f} (baseline: {baseline_score:.4f}, '
          f'delta: {trials[0].value - baseline_score:+.4f})')

    # Print top results
    print(f'\n  TOP {min(top_n, len(trials))} CONFIGS:')
    header_params = ' '.join(f'{p[:6]:>7}' for p in sensitive_params[:8])
    print(f'  {"Rk":>3} {"Score":>7} {"Ret%":>7} {"DD%":>5} {"Shrp":>5} '
          f'{"PF":>5} {"WR%":>5} {"N":>4} | {header_params}')
    print(f'  {"─"*100}')

    for rank, t in enumerate(trials[:top_n], 1):
        ua = t.user_attrs
        param_vals = ' '.join(
            f'{str(ua.get("p_"+p, "?"))[:7]:>7}'
            for p in sensitive_params[:8]
        )
        print(f'  {rank:>3} {t.value:>7.4f} {ua.get("ret",0):>+6.1f}% '
              f'{ua.get("max_dd",0):>4.1f}% {ua.get("sharpe",0):>5.2f} '
              f'{ua.get("pf",0):>5.2f} {ua.get("wr",0):>4.1f}% '
              f'{ua.get("trades",0):>4} | {param_vals}')

    # Best config detail
    best = trials[0]
    ua = best.user_attrs
    print(f'\n  BEST FOCUSED CONFIG:')
    for p in sensitive_params:
        baseline_val = BASELINE_PARAMS.get(p)
        best_val = ua.get(f'p_{p}')
        changed = '  ← CHANGED' if best_val != baseline_val else ''
        print(f'    {p:<24} = {best_val:<12} (baseline: {baseline_val}){changed}')

    print(f'\n    Return: {ua.get("ret",0):+.1f}%  DD: {ua.get("max_dd",0):.1f}%  '
          f'Sharpe: {ua.get("sharpe",0):.2f}')
    print(f'    Trades: {ua.get("trades",0)}  WR: {ua.get("wr",0):.1f}%  '
          f'PF: {ua.get("pf",0):.2f}')
    print(f'    CALL: {ua.get("call_trades",0)} trades, WR={ua.get("call_wr",0):.1f}%, '
          f'P&L=${ua.get("call_pnl",0):+,.0f}')
    print(f'    PUT:  {ua.get("put_trades",0)} trades, WR={ua.get("put_wr",0):.1f}%, '
          f'P&L=${ua.get("put_pnl",0):+,.0f}')

    # Save full results
    all_rows = []
    for t in trials:
        ua = t.user_attrs
        row = {'score': t.value}
        row.update({k: v for k, v in ua.items()})
        all_rows.append(row)
    pd.DataFrame(all_rows).to_csv('output/sensitivity_optimization_results.csv', index=False)
    print(f'\n  Results saved to output/sensitivity_optimization_results.csv')

    return trials[0]


def validate_focused_on_oos(best_trial, period='2026'):
    """Validate focused best config on OOS data."""
    ua = best_trial.user_attrs
    params = dict(BASELINE_PARAMS)
    # Override with optimized values
    for k, v in ua.items():
        if k.startswith('p_'):
            params[k[2:]] = v

    print(f'\n{"="*70}')
    print(f'  PHASE 3: OOS VALIDATION ({period})')
    print(f'{"="*70}')

    load_data_cached(period)
    m = run_single_config(params, period=period)

    if m is None:
        print('  FAILED: No trades on OOS data!')
        return None

    print(f'\n  OOS Results:')
    print(f'    Trades: {m["trades"]}  WR: {m["wr"]:.1f}%  Return: {m["ret"]:+.1f}%')
    print(f'    Max DD: {m["max_dd"]:.1f}%  Sharpe: {m["sharpe"]:.2f}  PF: {m["pf"]:.2f}')
    print(f'    CALL: {m["call_trades"]} trades ${m["call_pnl"]:+,.0f}  '
          f'PUT: {m["put_trades"]} trades ${m["put_pnl"]:+,.0f}')

    # IS vs OOS comparison
    is_ret = ua.get('ret', 0)
    print(f'\n  IS vs OOS:')
    print(f'    {"Metric":<15} {"IS":>10} {"OOS":>10} {"Decay":>10}')
    print(f'    {"─"*45}')
    for name, is_v, oos_v in [
        ('Return %', is_ret, m['ret']),
        ('Max DD %', ua.get('max_dd', 0), m['max_dd']),
        ('Sharpe', ua.get('sharpe', 0), m['sharpe']),
        ('Win Rate %', ua.get('wr', 0), m['wr']),
        ('Profit Factor', ua.get('pf', 0), m['pf']),
    ]:
        if is_v != 0:
            decay = f'{(oos_v / is_v - 1) * 100:+.0f}%'
        else:
            decay = 'N/A'
        print(f'    {name:<15} {is_v:>10.1f} {oos_v:>10.1f} {decay:>10}')

    return m


# ============================================================
# VISUALIZATION
# ============================================================

def generate_sensitivity_charts(sensitivity, all_results, sensitive_params):
    """Generate sensitivity analysis charts."""
    ranked = rank_parameters(sensitivity)
    top_params = [r[0] for r in ranked[:5]]

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')

    # 1. Sensitivity ranking (horizontal bar chart)
    ax = axes[0, 0]
    names = [r[0] for r in ranked]
    ranges = [r[1]['range'] for r in ranked]
    colors = ['#d32f2f' if n in sensitive_params else '#78909c' for n in names]
    ax.barh(range(len(names)), ranges, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Score Range (higher = more sensitive)')
    ax.set_title('Parameter Sensitivity Ranking (red = selected for optimization)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # 2-6. Sweep curves for top 5 sensitive params
    subplot_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for idx, pname in enumerate(top_params):
        r, c = subplot_positions[idx]
        ax = axes[r, c]

        param_results = [r for r in all_results if r['param'] == pname]
        vals = [r['value'] for r in param_results]
        scores = [r['score'] if r['score'] > float('-inf') else None for r in param_results]

        valid = [(v, s) for v, s in zip(vals, scores) if s is not None]
        if valid:
            x, y = zip(*valid)
            ax.plot(x, y, 'o-', color='steelblue', label='Composite Score', lw=2, ms=6)

        baseline_val = BASELINE_PARAMS.get(pname)
        if baseline_val is not None:
            ax.axvline(baseline_val, color='red', ls='--', alpha=0.7,
                       label=f'Baseline={baseline_val}')

        ax.set_xlabel(pname)
        ax.set_ylabel('Composite Score')
        ax.set_title(f'Sensitivity: {pname}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = 'output/sensitivity_analysis.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'\n  Charts saved to {out}')


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Parameter Sensitivity & Focused Optimization')
    parser.add_argument('--sweep-points', type=int, default=10,
                        help='Points per parameter sweep (default: 10)')
    parser.add_argument('--trials', type=int, default=300,
                        help='Optimization trials for Phase 2 (default: 300)')
    parser.add_argument('--top', type=int, default=0,
                        help='Number of top sensitive params to optimize (0 = auto)')
    parser.add_argument('--workers', type=int, default=6,
                        help='Parallel workers (default: 6)')
    parser.add_argument('--period', type=str, default='2025',
                        help='Training period (default: 2025)')
    parser.add_argument('--oos', type=str, default='2026',
                        help='OOS period (default: 2026)')
    parser.add_argument('--skip-sensitivity', action='store_true',
                        help='Skip Phase 1, load cached sensitivity results')
    parser.add_argument('--sensitivity-only', action='store_true',
                        help='Run Phase 1 only (no optimization)')
    args = parser.parse_args()

    t_total = time.time()

    # ── Phase 1: Sensitivity Analysis ──
    cache_path = 'output/sensitivity_cache.json'

    if args.skip_sensitivity and os.path.exists(cache_path):
        print('  Loading cached sensitivity results...')
        with open(cache_path, 'r') as f:
            cached = json.load(f)
        sensitivity = cached['sensitivity']
        baseline_score = cached['baseline_score']
        all_results = cached.get('sweep_results', [])
        ranked = sorted(sensitivity.items(), key=lambda x: x[1]['range'], reverse=True)
    else:
        sensitivity, all_results, baseline_score = run_sensitivity_analysis(
            period=args.period,
            n_points=args.sweep_points,
            max_workers=args.workers,
        )
        ranked = rank_parameters(sensitivity)

        # Cache results for re-runs
        with open(cache_path, 'w') as f:
            json.dump({
                'sensitivity': sensitivity,
                'baseline_score': baseline_score,
                'sweep_results': all_results,
            }, f, indent=2, default=str)
        print(f'  Sensitivity results cached to {cache_path}')

    # Print sensitivity report
    tiers = print_sensitivity_report(ranked, baseline_score)

    # Select params for focused optimization
    if args.top > 0:
        n_sensitive = args.top
    else:
        # Auto: select HIGH + MEDIUM tier params
        n_sensitive = sum(1 for t in tiers.values() if t in ('HIGH', 'MEDIUM'))
        n_sensitive = max(n_sensitive, 3)  # at least 3

    sensitive_params = [name for name, _ in ranked[:n_sensitive]]
    print(f'\n  Selected {n_sensitive} parameters for focused optimization: '
          f'{", ".join(sensitive_params)}')

    # Generate sensitivity charts
    print('\n  Generating sensitivity charts...')
    generate_sensitivity_charts(sensitivity, all_results, sensitive_params)

    if args.sensitivity_only:
        print(f'\n  Total time: {time.time() - t_total:.0f}s')
        return

    # ── Phase 2: Focused Optimization ──
    study = run_focused_optimization(
        sensitive_params=sensitive_params,
        sensitivity=sensitivity,
        n_trials=args.trials,
        n_workers=args.workers,
        period=args.period,
    )

    best_trial = analyze_focused_results(
        study, sensitive_params, baseline_score, top_n=15
    )

    if best_trial is None:
        print('  Focused optimization failed!')
        return

    # ── Phase 3: OOS Validation ──
    oos_m = validate_focused_on_oos(best_trial, period=args.oos)

    # Save best config
    ua = best_trial.user_attrs
    best_config = {
        'optimization_date': time.strftime('%Y-%m-%d'),
        'source': 'sensitivity_focused',
        'n_sensitive_params': n_sensitive,
        'sensitive_params': sensitive_params,
        'sensitivity_ranking': {name: s for name, s in ranked},
        'trade_config': {},
        'risk_config': {},
        'is_results': {},
        'oos_results': oos_m,
    }

    # Populate trade/risk config from user_attrs
    for k, v in ua.items():
        if k.startswith('p_'):
            pname = k[2:]
            if pname in ('kelly', 'max_risk', 'max_daily_losses', 'max_consec_losses',
                         'max_daily_loss_pct'):
                best_config['risk_config'][pname] = v
            else:
                best_config['trade_config'][pname] = v
        elif not k.startswith('p_'):
            best_config['is_results'][k] = v

    with open('output/sensitivity_optimized_config.json', 'w') as f:
        json.dump(best_config, f, indent=2, default=str)
    print(f'\n  Best config saved to output/sensitivity_optimized_config.json')

    # Save to centralized history
    save_optimization_run(
        source='sensitivity_focused',
        trade_config=best_config['trade_config'],
        risk_config=best_config['risk_config'],
        results={
            'is_results': best_config['is_results'],
            'oos_results': oos_m,
        },
        metadata={
            'trials': len(study.trials),
            'sensitive_params': sensitive_params,
            'n_total_params': len(PARAM_DEFS),
        },
    )

    elapsed = time.time() - t_total
    print(f'\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)')
    print('  Done.')


if __name__ == '__main__':
    main()
