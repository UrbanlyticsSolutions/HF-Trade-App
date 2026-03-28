"""
Post-Loss Momentum Confirm — Performance Analysis & Threshold Optimization

Analyzes and optimizes the post_loss_strategy='momentum_confirm' system:
  1. Compare none / momentum_confirm / multi_confirm at current baseline
  2. Sweep momentum_threshold (0.02 → 0.30) to find optimal sensitivity
  3. Track post-loss specific metrics (skipped signals, post-loss WR, flip accuracy)
  4. OOS validation of best threshold on 2026 data

Usage:
  python scripts/optimize_post_loss.py
"""
import sys
sys.path.insert(0, '.')

import io
import time
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from backtest.engine import Backtest0DTE, TradeConfig, Trade0DTE
from core.risk_manager import RiskConfig
from config import defaults as cfg

# ============================================================
# DATA LOADING (reuse from optimize_full)
# ============================================================
_DATA_CACHE = {}

def load_data(period='2025'):
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


# ============================================================
# BASELINE CONFIG (from optimization run #2)
# ============================================================
OPTIMIZED_PARAMS = {
    'call_pt': 0.55, 'put_pt': 0.55,
    'call_sl': 0.40, 'put_sl': 0.40,
    'rsi_call': 65, 'rsi_put': 35,
    'call_hold': 18, 'put_hold': 20,
    'kelly': 0.06, 'max_risk': 0.05,
    'min_contracts': 1,
    'max_daily_losses': 2, 'max_consec_losses': 3,
    'max_daily_loss_pct': 0.025,
}

BASELINE_PARAMS = {
    'call_pt': 0.50, 'put_pt': 0.50,
    'call_sl': 0.35, 'put_sl': 0.35,
    'rsi_call': 70, 'rsi_put': 30,
    'call_hold': 16, 'put_hold': 16,
    'kelly': 0.05, 'max_risk': 0.03,
    'min_contracts': 1,
    'max_daily_losses': 2, 'max_consec_losses': 3,
    'max_daily_loss_pct': 0.008,
}


def run_config(params, post_loss='none', momentum_threshold=0.10, period='2025'):
    """Run backtest with given params and post-loss strategy."""
    u, o, f, v, dt_idx, td_idx = load_data(period)
    cap = cfg.initial_capital()

    tc = TradeConfig(
        strategy='momentum',
        trade_start_hour=10, trade_start_minute=0,
        trade_end_hour=11, trade_end_minute=0,
        rsi_call_threshold=params['rsi_call'],
        rsi_put_threshold=params['rsi_put'],
        profit_target_pct=params['call_pt'],
        stop_loss_pct=params['call_sl'],
        call_profit_target_pct=params['call_pt'],
        put_profit_target_pct=params['put_pt'],
        call_stop_loss_pct=params['call_sl'],
        put_stop_loss_pct=params['put_sl'],
        max_hold_bars=params['call_hold'],
        call_max_hold_bars=params['call_hold'],
        put_max_hold_bars=params['put_hold'],
        min_option_price=0.50, max_option_price=2.00,
        use_adaptive_exits=False, use_trailing_stop=False,
        use_time_decay_exit=False, use_quick_exit=False,
        use_ml_filter=False, skip_day_filter=True,
        min_contracts_per_trade=params['min_contracts'],
        post_loss_strategy=post_loss,
        post_loss_momentum_threshold=momentum_threshold,
    )

    rc = RiskConfig(
        max_risk_per_trade_pct=params['max_risk'],
        max_position_pct=0.10,
        max_daily_losses=params['max_daily_losses'],
        max_consecutive_losses=params['max_consec_losses'],
        max_daily_loss_pct=params['max_daily_loss_pct'],
        max_trades_per_day=999,
        reduce_size_at_dd_pct=0.99,
    )

    bt = Backtest0DTE(tc, rc, initial_capital=cap)
    bt._opt_by_date_time = dt_idx
    bt._opt_by_ticker_date = td_idx
    bt.risk_manager.set_kelly(params['kelly'])

    old_out = sys.stdout
    sys.stdout = io.StringIO()
    try:
        trades = bt.run_no_ml(u, o, f, v, verbose=False)
    finally:
        sys.stdout = old_out

    return compute_metrics(trades, cap)


def compute_metrics(trades, cap):
    """Compute metrics from trade list."""
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
        if c > peak: peak = c
        dd = (peak - c) / peak
        if dd > max_dd: max_dd = dd

    # Sharpe
    rets = [t.pnl / max(t.capital - t.pnl, 1) for t in trades]
    mu = np.mean(rets)
    sigma = np.std(rets) if len(rets) > 1 else 1
    sharpe = (mu * 252) / (sigma * np.sqrt(252)) if sigma > 0 else 0

    # Profit factor
    gp = sum(t.pnl for t in trades if t.pnl > 0)
    gl = abs(sum(t.pnl for t in trades if t.pnl <= 0)) or 0.01
    pf = gp / gl

    calmar = ret / (max_dd * 100) if max_dd > 0.001 else 0

    # Breakdown
    call_trades = [t for t in trades if t.direction == 'CALL']
    put_trades = [t for t in trades if t.direction == 'PUT']
    call_pnl = sum(t.pnl for t in call_trades)
    put_pnl = sum(t.pnl for t in put_trades)
    call_wr = sum(1 for t in call_trades if t.pnl > 0) / max(len(call_trades), 1) * 100
    put_wr = sum(1 for t in put_trades if t.pnl > 0) / max(len(put_trades), 1) * 100

    # Exit distribution
    profit_exits = sum(1 for t in trades if t.exit_reason == 'PROFIT')
    stop_exits = sum(1 for t in trades if t.exit_reason == 'STOP')
    time_exits = sum(1 for t in trades if t.exit_reason == 'TIME')

    # Daily stats
    daily_pnl = {}
    for t in trades:
        daily_pnl[t.date] = daily_pnl.get(t.date, 0) + t.pnl
    profitable_days = sum(1 for p in daily_pnl.values() if p > 0)
    total_days = len(daily_pnl)
    day_win_pct = profitable_days / total_days * 100 if total_days > 0 else 0

    # Post-loss specific: trades that occur after first daily loss
    # We can identify these by looking at sequential trades per day
    post_loss_trades = []
    pre_loss_trades = []
    daily_trades = {}
    for t in trades:
        daily_trades.setdefault(t.date, []).append(t)

    for date, day_trades in daily_trades.items():
        first_loss_idx = None
        for i, t in enumerate(day_trades):
            if t.pnl < 0 and first_loss_idx is None:
                first_loss_idx = i
                pre_loss_trades.append(t)
            elif first_loss_idx is not None:
                post_loss_trades.append(t)
            else:
                pre_loss_trades.append(t)

    pl_n = len(post_loss_trades)
    pl_wins = sum(1 for t in post_loss_trades if t.pnl > 0)
    pl_wr = pl_wins / pl_n * 100 if pl_n > 0 else 0
    pl_pnl = sum(t.pnl for t in post_loss_trades)
    pl_avg = pl_pnl / pl_n if pl_n > 0 else 0

    pre_n = len(pre_loss_trades)
    pre_wins = sum(1 for t in pre_loss_trades if t.pnl > 0)
    pre_wr = pre_wins / pre_n * 100 if pre_n > 0 else 0
    pre_pnl = sum(t.pnl for t in pre_loss_trades)
    pre_avg = pre_pnl / pre_n if pre_n > 0 else 0

    # Loss days count
    loss_days = sum(1 for day_trades in daily_trades.values()
                    if any(t.pnl < 0 for t in day_trades))

    return {
        'trades': n, 'wins': wins, 'wr': round(wr, 1),
        'pnl': round(total_pnl, 0), 'ret': round(ret, 1),
        'max_dd': round(max_dd * 100, 1),
        'sharpe': round(sharpe, 2), 'pf': round(pf, 2),
        'calmar': round(calmar, 1),
        'call_trades': len(call_trades), 'put_trades': len(put_trades),
        'call_pnl': round(call_pnl, 0), 'put_pnl': round(put_pnl, 0),
        'call_wr': round(call_wr, 1), 'put_wr': round(put_wr, 1),
        'profit_exits': profit_exits, 'stop_exits': stop_exits, 'time_exits': time_exits,
        'day_wr': round(day_win_pct, 1),
        'loss_days': loss_days,
        'post_loss_n': pl_n, 'post_loss_wins': pl_wins,
        'post_loss_wr': round(pl_wr, 1), 'post_loss_pnl': round(pl_pnl, 0),
        'post_loss_avg': round(pl_avg, 2),
        'pre_loss_n': pre_n, 'pre_loss_wr': round(pre_wr, 1),
        'pre_loss_pnl': round(pre_pnl, 0), 'pre_loss_avg': round(pre_avg, 2),
    }


def main():
    t_start = time.time()
    cap = cfg.initial_capital()

    print('=' * 70)
    print('  POST-LOSS STRATEGY — PERFORMANCE ANALYSIS & OPTIMIZATION')
    print('=' * 70)

    # ============================================================
    # PHASE 1: Compare none / momentum_confirm / multi_confirm
    # ============================================================
    print('\n  Loading 2025 data...')
    load_data('2025')

    strategies = ['none', 'momentum_confirm', 'multi_confirm']
    configs = {'Baseline': BASELINE_PARAMS, 'Optimized': OPTIMIZED_PARAMS}

    print(f'\n  ── PHASE 1: Strategy Comparison (threshold=0.10) ──')

    all_results = {}
    for config_name, params in configs.items():
        print(f'\n  [{config_name} config]')
        for strat in strategies:
            m = run_config(params, post_loss=strat, period='2025')
            key = f'{config_name}_{strat}'
            all_results[key] = m
            if m:
                print(f'    {strat:20s} | {m["trades"]:>3} trades | ret={m["ret"]:>+7.1f}% | '
                      f'dd={m["max_dd"]:>4.1f}% | sharpe={m["sharpe"]:>5.2f} | '
                      f'wr={m["wr"]:>4.1f}% | pf={m["pf"]:>4.2f}')
                print(f'      Post-loss trades: {m["post_loss_n"]:>3} | '
                      f'WR={m["post_loss_wr"]:>4.1f}% | P&L=${m["post_loss_pnl"]:>+7.0f} | '
                      f'AvgP&L=${m["post_loss_avg"]:>+6.2f}')
                print(f'      Pre-loss trades:  {m["pre_loss_n"]:>3} | '
                      f'WR={m["pre_loss_wr"]:>4.1f}% | P&L=${m["pre_loss_pnl"]:>+7.0f} | '
                      f'AvgP&L=${m["pre_loss_avg"]:>+6.2f}')
                print(f'      Loss days: {m["loss_days"]} | Day WR: {m["day_wr"]:.1f}%')

    # ============================================================
    # PHASE 2: Momentum threshold sweep
    # ============================================================
    print(f'\n  ── PHASE 2: Momentum Threshold Sweep (IS 2025) ──')
    thresholds = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

    sweep_results_baseline = []
    sweep_results_optimized = []

    print(f'\n  {"Thr":>5} | {"Config":>10} | {"N":>3} | {"Ret%":>7} | {"DD%":>5} | '
          f'{"Shrp":>5} | {"WR%":>5} | {"PF":>5} | {"PLn":>3} | {"PLWR":>5} | {"PLP&L":>7}')
    print(f'  {"─" * 90}')

    for thr in thresholds:
        for config_name, params in configs.items():
            m = run_config(params, post_loss='momentum_confirm',
                          momentum_threshold=thr, period='2025')
            if m:
                row = {'threshold': thr, **m}
                if config_name == 'Baseline':
                    sweep_results_baseline.append(row)
                else:
                    sweep_results_optimized.append(row)

                print(f'  {thr:>5.2f} | {config_name:>10} | {m["trades"]:>3} | '
                      f'{m["ret"]:>+6.1f}% | {m["max_dd"]:>4.1f}% | {m["sharpe"]:>5.2f} | '
                      f'{m["wr"]:>4.1f}% | {m["pf"]:>4.2f} | {m["post_loss_n"]:>3} | '
                      f'{m["post_loss_wr"]:>4.1f}% | ${m["post_loss_pnl"]:>+6.0f}')

    # Find best thresholds
    print(f'\n  ── BEST THRESHOLDS ──')
    for name, results in [('Baseline', sweep_results_baseline),
                           ('Optimized', sweep_results_optimized)]:
        if not results:
            continue
        # Score by composite: ret + sharpe + post_loss_wr
        for r in results:
            r['score'] = (r['ret'] / 300) + (r['sharpe'] / 8) + (r['post_loss_wr'] / 100)
        best = max(results, key=lambda r: r['score'])
        best_ret = max(results, key=lambda r: r['ret'])
        best_sharpe = max(results, key=lambda r: r['sharpe'])
        best_plwr = max(results, key=lambda r: r['post_loss_wr'])

        print(f'\n  [{name}]')
        print(f'    Best composite score:   thr={best["threshold"]:.2f} → '
              f'ret={best["ret"]:+.1f}% sharpe={best["sharpe"]:.2f} PL_WR={best["post_loss_wr"]:.1f}%')
        print(f'    Best return:            thr={best_ret["threshold"]:.2f} → '
              f'ret={best_ret["ret"]:+.1f}%')
        print(f'    Best risk-adjusted:     thr={best_sharpe["threshold"]:.2f} → '
              f'sharpe={best_sharpe["sharpe"]:.2f}')
        print(f'    Best post-loss WR:      thr={best_plwr["threshold"]:.2f} → '
              f'PL_WR={best_plwr["post_loss_wr"]:.1f}% ({best_plwr["post_loss_n"]} trades)')

    # ============================================================
    # PHASE 3: Compare best threshold against none (delta analysis)
    # ============================================================
    print(f'\n  ── PHASE 3: Impact Analysis (Best vs None) ──')

    for config_name, params, results in [
        ('Baseline', BASELINE_PARAMS, sweep_results_baseline),
        ('Optimized', OPTIMIZED_PARAMS, sweep_results_optimized)
    ]:
        if not results:
            continue
        best = max(results, key=lambda r: r['score'])
        none_key = f'{config_name}_none'
        none_m = all_results.get(none_key)
        if not none_m:
            continue

        print(f'\n  [{config_name}] Best threshold={best["threshold"]:.2f}')
        print(f'    {"Metric":<20} {"No Post-Loss":>14} {"Momentum Confirm":>16} {"Delta":>10}')
        print(f'    {"─" * 62}')
        for metric in ['ret', 'max_dd', 'sharpe', 'wr', 'pf', 'trades']:
            nv = none_m[metric]
            bv = best[metric]
            if metric in ('max_dd',):
                delta = f'{bv - nv:+.1f}pp'
            elif metric == 'trades':
                delta = f'{bv - nv:+d}'
            elif nv != 0:
                delta = f'{(bv/nv - 1)*100:+.1f}%'
            else:
                delta = 'N/A'
            if metric == 'trades':
                print(f'    {metric:<20} {nv:>14d} {bv:>16d} {delta:>10}')
            elif metric in ('sharpe', 'pf'):
                print(f'    {metric:<20} {nv:>14.2f} {bv:>16.2f} {delta:>10}')
            else:
                print(f'    {metric:<20} {nv:>14.1f} {bv:>16.1f} {delta:>10}')
        print(f'    {"post_loss_n":<20} {none_m["post_loss_n"]:>14d} {best["post_loss_n"]:>16d}')
        print(f'    {"post_loss_wr":<20} {none_m["post_loss_wr"]:>14.1f} {best["post_loss_wr"]:>16.1f}')
        print(f'    {"post_loss_pnl":<20} ${none_m["post_loss_pnl"]:>+13.0f} ${best["post_loss_pnl"]:>+15.0f}')

    # ============================================================
    # PHASE 4: OOS Validation (2026)
    # ============================================================
    print(f'\n  ── PHASE 4: Out-of-Sample Validation (2026) ──')
    print(f'  Loading 2026 data...')
    load_data('2026')

    oos_results = {}
    for config_name, params, results in [
        ('Baseline', BASELINE_PARAMS, sweep_results_baseline),
        ('Optimized', OPTIMIZED_PARAMS, sweep_results_optimized)
    ]:
        if not results:
            continue
        best = max(results, key=lambda r: r['score'])
        best_thr = best['threshold']

        # Run none, best momentum_confirm, and multi_confirm on OOS
        print(f'\n  [{config_name}] OOS with best threshold={best_thr:.2f}')
        for strat, thr in [('none', 0.10), ('momentum_confirm', best_thr), ('multi_confirm', best_thr)]:
            m = run_config(params, post_loss=strat, momentum_threshold=thr, period='2026')
            oos_key = f'{config_name}_{strat}'
            oos_results[oos_key] = m
            if m:
                print(f'    {strat:20s} | {m["trades"]:>3} trades | ret={m["ret"]:>+7.1f}% | '
                      f'dd={m["max_dd"]:>4.1f}% | sharpe={m["sharpe"]:>5.2f} | '
                      f'wr={m["wr"]:>4.1f}% | pf={m["pf"]:>4.2f}')
                print(f'      Post-loss: {m["post_loss_n"]:>2} trades | '
                      f'WR={m["post_loss_wr"]:>4.1f}% | P&L=${m["post_loss_pnl"]:>+7.0f}')

    # ============================================================
    # PHASE 5: Charts
    # ============================================================
    print(f'\n  Generating charts...')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Post-Loss Momentum Confirm — Threshold Optimization', fontsize=14, fontweight='bold')

    # Chart 1: Return vs Threshold
    ax = axes[0, 0]
    for name, results, color in [('Baseline', sweep_results_baseline, 'blue'),
                                   ('Optimized', sweep_results_optimized, 'red')]:
        if results:
            thrs = [r['threshold'] for r in results]
            rets = [r['ret'] for r in results]
            ax.plot(thrs, rets, 'o-', color=color, label=name, lw=2, markersize=6)
    ax.set_xlabel('Momentum Threshold')
    ax.set_ylabel('Return %')
    ax.set_title('Return vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Chart 2: Sharpe vs Threshold
    ax = axes[0, 1]
    for name, results, color in [('Baseline', sweep_results_baseline, 'blue'),
                                   ('Optimized', sweep_results_optimized, 'red')]:
        if results:
            thrs = [r['threshold'] for r in results]
            sharpes = [r['sharpe'] for r in results]
            ax.plot(thrs, sharpes, 'o-', color=color, label=name, lw=2, markersize=6)
    ax.set_xlabel('Momentum Threshold')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Chart 3: Post-loss WR vs Threshold
    ax = axes[1, 0]
    for name, results, color in [('Baseline', sweep_results_baseline, 'blue'),
                                   ('Optimized', sweep_results_optimized, 'red')]:
        if results:
            thrs = [r['threshold'] for r in results]
            plwr = [r['post_loss_wr'] for r in results]
            pln = [r['post_loss_n'] for r in results]
            ax.plot(thrs, plwr, 'o-', color=color, label=f'{name} WR', lw=2, markersize=6)
            ax2 = ax.twinx()
            ax2.bar([t + (0.005 if name == 'Optimized' else -0.005) for t in thrs],
                    pln, width=0.008, alpha=0.3, color=color, label=f'{name} count')
            ax2.set_ylabel('Post-loss trade count')
    ax.set_xlabel('Momentum Threshold')
    ax.set_ylabel('Post-loss Win Rate %')
    ax.set_title('Post-loss Quality vs Threshold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Chart 4: Strategy comparison (bar chart)
    ax = axes[1, 1]
    strats = strategies
    for i, (config_name, _) in enumerate(configs.items()):
        ret_vals = []
        for s in strats:
            key = f'{config_name}_{s}'
            m = all_results.get(key)
            ret_vals.append(m['ret'] if m else 0)
        x = np.arange(len(strats))
        offset = (i - 0.5) * 0.35
        bars = ax.bar(x + offset, ret_vals, 0.3, label=config_name,
                      alpha=0.8, color=['blue', 'red'][i])
        for bar, val in zip(bars, ret_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{val:+.0f}%', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(np.arange(len(strats)))
    ax.set_xticklabels(strats, fontsize=9)
    ax.set_ylabel('Return %')
    ax.set_title('Post-Loss Strategy Comparison (IS 2025)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart_path = 'output/post_loss_optimization.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'  Charts saved to {chart_path}')

    # ============================================================
    # SUMMARY
    # ============================================================
    elapsed = time.time() - t_start
    print(f'\n{"=" * 70}')
    print(f'  POST-LOSS OPTIMIZATION COMPLETE | {elapsed:.0f}s ({elapsed/60:.1f} min)')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
