"""
Advanced Profit-Taking Strategy Comparison
===========================================
Tests 6 exit strategies against Phase 8 baseline on Jan-Feb 2026 OOS data.

Strategies:
  0. Baseline     - Fixed PT=50% / SL=35% / TIME=16
  A. Ratchet Lock - Discrete profit-lock floors at 25%/35%/45% peaks
  B. Partial Exit - Sell half at +25%, rest runs to +50%/SL/TIME
  C. Trailing Stop- Optuna-optimized trailing stop (trained on 2025)
  D. Dynamic PT   - PT decreases as trade ages (50%->35%->20%)
  E. Hybrid       - Ratchet floors + Dynamic PT combined
"""
import sys
sys.path.insert(0, '.')

import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from backtest.engine import Backtest0DTE, TradeConfig, Trade0DTE
from core.risk_manager import RiskConfig, RiskManager
from core.signals import get_basic_signal

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS
# ============================================================
SLIPPAGE = 0.005
COMMISSION = 0.65
INITIAL_CAPITAL = 10000


# ============================================================
# EXIT STRATEGY FUNCTIONS
# Each returns list of (bar_idx, exit_reason, fraction)
# fraction=1.0 for full exits, 0.5 for partial
# ============================================================

def exit_baseline(entry_price, closes, max_hold, pt, sl, **kwargs):
    """Strategy 0: Fixed PT / SL / TIME"""
    n = min(len(closes), max_hold)
    for i in range(n):
        pct = (closes[i] - entry_price) / entry_price
        if pct >= pt:
            return [(i, 'PROFIT', 1.0)]
        if pct <= -sl:
            return [(i, 'STOP', 1.0)]
        if i + 1 >= max_hold:
            return [(i, 'TIME', 1.0)]
    return [(max(0, n - 1), 'TIME', 1.0)]


def exit_ratchet(entry_price, closes, max_hold, pt, sl, **kwargs):
    """Strategy A: Ratchet lock-in with discrete profit floors.

    Once peak gain crosses a threshold, a floor is locked in.
    If price drops below the floor, exit immediately.
    Levels: [(threshold, floor), ...]
    """
    levels = kwargs.get('levels', [(0.25, 0.10), (0.35, 0.20), (0.45, 0.30)])
    max_pct = 0.0
    floor = None
    n = min(len(closes), max_hold)

    for i in range(n):
        pct = (closes[i] - entry_price) / entry_price
        if pct > max_pct:
            max_pct = pct

        # Update floor to highest earned level
        for threshold, lock in levels:
            if max_pct >= threshold:
                floor = lock  # ascending order -> keeps highest

        # Check exits in priority order
        if pct >= pt:
            return [(i, 'PROFIT', 1.0)]
        if floor is not None and pct < floor:
            return [(i, 'RATCHET', 1.0)]
        if pct <= -sl:
            return [(i, 'STOP', 1.0)]
        if i + 1 >= max_hold:
            return [(i, 'TIME', 1.0)]

    return [(max(0, n - 1), 'TIME', 1.0)]


def exit_partial(entry_price, closes, max_hold, pt, sl, **kwargs):
    """Strategy B: Sell half at partial_pt, rest runs to full PT/SL/TIME.

    If num_contracts < 2, falls back to baseline (can't split 1 contract).
    """
    partial_pt = kwargs.get('partial_pt', 0.25)
    results = []
    first_done = False
    n = min(len(closes), max_hold)

    for i in range(n):
        pct = (closes[i] - entry_price) / entry_price

        if not first_done:
            # Full position still open
            if pct >= pt:
                return [(i, 'PROFIT', 1.0)]  # Hit full PT before partial
            if pct >= partial_pt:
                results.append((i, 'PARTIAL', 0.5))
                first_done = True
                continue
            if pct <= -sl:
                return [(i, 'STOP', 1.0)]
            if i + 1 >= max_hold:
                return [(i, 'TIME', 1.0)]
        else:
            # Only second half remains
            if pct >= pt:
                results.append((i, 'PROFIT', 0.5))
                return results
            if pct <= -sl:
                results.append((i, 'STOP', 0.5))
                return results
            if i + 1 >= max_hold:
                results.append((i, 'TIME', 0.5))
                return results

    # Edge case
    if first_done:
        results.append((max(0, n - 1), 'TIME', 0.5))
    else:
        results = [(max(0, n - 1), 'TIME', 1.0)]
    return results


def exit_trailing(entry_price, closes, max_hold, pt, sl, **kwargs):
    """Strategy C: Trailing stop with breakeven lock.

    - After be_activation gain -> stop moves to breakeven
    - After activation gain -> trailing stop tracks at (1-trail_dist) of peak
    - Still respects PT and SL
    """
    activation = kwargs.get('activation', 0.25)
    trail_dist = kwargs.get('trail_dist', 0.35)
    be_activation = kwargs.get('be_activation', 0.12)

    max_pct = 0.0
    trail_price = None
    n = min(len(closes), max_hold)

    for i in range(n):
        pct = (closes[i] - entry_price) / entry_price
        if pct > max_pct:
            max_pct = pct

        # Breakeven activation
        if trail_price is None and max_pct >= be_activation:
            trail_price = entry_price * 1.001  # tiny profit above entry

        # Trailing stop activation (overrides breakeven level)
        if max_pct >= activation:
            new_trail = entry_price * (1 + max_pct * (1 - trail_dist))
            if trail_price is None or new_trail > trail_price:
                trail_price = new_trail

        # Check exits
        if pct >= pt:
            return [(i, 'PROFIT', 1.0)]
        if trail_price is not None and closes[i] <= trail_price:
            return [(i, 'TRAIL', 1.0)]
        if pct <= -sl:
            return [(i, 'STOP', 1.0)]
        if i + 1 >= max_hold:
            return [(i, 'TIME', 1.0)]

    return [(max(0, n - 1), 'TIME', 1.0)]


def exit_dynamic_pt(entry_price, closes, max_hold, pt, sl, **kwargs):
    """Strategy D: Profit target decreases as trade ages.

    Bars 1-8:  PT = 50% (original)
    Bars 9-12: PT = 35% (lowered)
    Bars 13-16: PT = 20% (take what you can)
    """
    tiers = kwargs.get('tiers', [(8, 0.50), (12, 0.35), (16, 0.20)])
    n = min(len(closes), max_hold)

    for i in range(n):
        pct = (closes[i] - entry_price) / entry_price
        bars = i + 1

        # Determine current PT based on bar count
        current_pt = pt
        for max_bar, tier_pt in tiers:
            if bars <= max_bar:
                current_pt = tier_pt
                break

        if pct >= current_pt:
            return [(i, 'PROFIT', 1.0)]
        if pct <= -sl:
            return [(i, 'STOP', 1.0)]
        if bars >= max_hold:
            return [(i, 'TIME', 1.0)]

    return [(max(0, n - 1), 'TIME', 1.0)]


def exit_hybrid(entry_price, closes, max_hold, pt, sl, **kwargs):
    """Strategy E: Ratchet floors + Dynamic PT combined.

    Lock-in floors protect against giveback.
    Declining PT ensures we take profits before TIME exit.
    """
    levels = kwargs.get('levels', [(0.25, 0.10), (0.35, 0.20), (0.45, 0.30)])
    tiers = kwargs.get('tiers', [(8, 0.50), (12, 0.35), (16, 0.20)])
    max_pct = 0.0
    floor = None
    n = min(len(closes), max_hold)

    for i in range(n):
        pct = (closes[i] - entry_price) / entry_price
        bars = i + 1
        if pct > max_pct:
            max_pct = pct

        # Update ratchet floor
        for threshold, lock in levels:
            if max_pct >= threshold:
                floor = lock

        # Dynamic PT
        current_pt = pt
        for max_bar, tier_pt in tiers:
            if bars <= max_bar:
                current_pt = tier_pt
                break

        # Check exits
        if pct >= current_pt:
            return [(i, 'PROFIT', 1.0)]
        if floor is not None and pct < floor:
            return [(i, 'RATCHET', 1.0)]
        if pct <= -sl:
            return [(i, 'STOP', 1.0)]
        if bars >= max_hold:
            return [(i, 'TIME', 1.0)]

    return [(max(0, n - 1), 'TIME', 1.0)]


# ============================================================
# BACKTEST RUNNER
# ============================================================

def run_backtest(exit_fn, exit_kwargs, bt, underlying_df, features_df,
                 trade_cfg, risk_cfg, initial_capital=INITIAL_CAPITAL,
                 verbose=False):
    """
    Run full backtest with a custom exit strategy function.
    Mirrors engine.run_no_ml() exactly for signal generation, option finding,
    and position sizing. Only the exit simulation is pluggable.

    Returns: list of trade dicts
    """
    cfg = trade_cfg
    risk_mgr = RiskManager(initial_capital, risk_cfg)
    # Kelly defaults to config.default_position_pct = 0.07 (0 training samples)

    trades = []

    for idx in range(len(underlying_df)):
        row = underlying_df.iloc[idx]
        date = row['date']
        current_time = row['time']
        hour = row['hour']
        minute = row.get('minute', 0)
        underlying_price = row['close']

        # Time filter (replicates engine exactly - hour only, no minute extraction)
        if hour < cfg.trade_start_hour:
            continue
        if hour > cfg.trade_end_hour:
            continue
        if hour == cfg.trade_end_hour and minute > cfg.trade_end_minute:
            continue

        # Risk check
        can_trade, reason = risk_mgr.can_trade(date)
        if not can_trade:
            continue

        # Signal
        feat = features_df.iloc[idx]
        direction = get_basic_signal(
            feat,
            rsi_call_threshold=cfg.rsi_call_threshold,
            rsi_put_threshold=cfg.rsi_put_threshold,
            strategy=cfg.strategy,
            bb_buffer_pct=cfg.bb_buffer_pct,
            vwap_dev_threshold=cfg.vwap_dev_threshold,
            orb_buffer_pct=cfg.orb_buffer_pct,
        )
        if direction is None:
            continue

        option_type = 'call' if direction == 'CALL' else 'put'

        # Find option (uses engine's internal indexes)
        option = bt._find_option(None, underlying_price, option_type, date, current_time)
        if option is None:
            continue

        entry_price = option['close'] * (1 + SLIPPAGE)
        strike = option['strike']
        ticker = option['option_ticker']

        # Future bars
        future_bars = bt._get_future_bars(None, ticker, date, current_time)
        if len(future_bars) == 0:
            continue
        closes = future_bars['close'].values

        # Position sizing
        num_contracts, _ = risk_mgr.get_position_size(
            entry_price, ml_confidence=None, stop_loss_pct=cfg.stop_loss_pct
        )
        if num_contracts == 0:
            continue

        # === Exit strategy ===
        # For partial exits with < 2 contracts, fall back to baseline
        effective_exit_fn = exit_fn
        effective_kwargs = exit_kwargs
        if exit_fn == exit_partial and num_contracts < 2:
            effective_exit_fn = exit_baseline
            effective_kwargs = {}

        exits = effective_exit_fn(
            entry_price, closes, cfg.max_hold_bars,
            cfg.profit_target_pct, cfg.stop_loss_pct,
            **effective_kwargs
        )

        # === Calculate PnL ===
        total_net_pnl = 0.0

        for part_idx, (bar_idx, reason, fraction) in enumerate(exits):
            if len(exits) == 1:
                n = num_contracts
            elif part_idx == 0:
                n = (num_contracts + 1) // 2   # ceil
            else:
                n = num_contracts - (num_contracts + 1) // 2  # floor
            if n <= 0:
                continue

            raw_exit = closes[bar_idx]
            slipped_exit = raw_exit * (1 - SLIPPAGE)
            gross = n * 100 * (slipped_exit - entry_price)
            comm = COMMISSION * n * 2
            part_pnl = gross - comm
            total_net_pnl += part_pnl

        # Record with risk manager
        risk_mgr.record_trade(date, total_net_pnl)

        last_bar_idx = exits[-1][0]
        exit_reasons = '+'.join(r for _, r, _ in exits)
        last_raw_exit = closes[last_bar_idx] * (1 - SLIPPAGE)

        trades.append({
            'date': date,
            'time': current_time,
            'direction': direction,
            'strike': strike,
            'ticker': ticker,
            'rsi': feat['rsi'],
            'entry': entry_price,
            'exit': last_raw_exit,
            'exit_reason': exit_reasons,
            'bars_held': last_bar_idx + 1,
            'num_contracts': num_contracts,
            'pnl': total_net_pnl,
            'capital': risk_mgr.capital,
        })

        if verbose:
            emoji = "+" if total_net_pnl > 0 else "x"
            print(f"  {emoji} {date} {current_time} | {direction} {strike:.0f} "
                  f"| {exit_reasons} | ${total_net_pnl:+.2f} | Cap=${risk_mgr.capital:,.0f}")

    return trades


# ============================================================
# OPTUNA TRAILING STOP OPTIMIZER
# ============================================================

def optimize_trailing_stop(bt, underlying_df, features_df, trade_cfg, risk_cfg,
                           n_trials=50, initial_capital=INITIAL_CAPITAL):
    """
    Search for optimal trailing stop params on train data using Optuna.

    Search space:
      activation:    [0.15, 0.45]  - min gain before trail activates
      trail_dist:    [0.20, 0.50]  - trail distance as fraction of peak
      be_activation: [0.05, 0.25]  - min gain before breakeven lock

    Objective: composite score (40% return + 30% 1/DD + 30% Sharpe)
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [WARN] Optuna not installed, using default trailing stop params")
        return {'activation': 0.25, 'trail_dist': 0.35, 'be_activation': 0.12}

    best_score = [-999]
    best_params = [{'activation': 0.25, 'trail_dist': 0.35, 'be_activation': 0.12}]
    trial_count = [0]

    def objective(trial):
        activation = trial.suggest_float('activation', 0.15, 0.45, step=0.05)
        trail_dist = trial.suggest_float('trail_dist', 0.20, 0.50, step=0.05)
        be_activation = trial.suggest_float('be_activation', 0.05, 0.25, step=0.05)

        kwargs = {
            'activation': activation,
            'trail_dist': trail_dist,
            'be_activation': be_activation,
        }

        trades = run_backtest(
            exit_trailing, kwargs, bt, underlying_df, features_df,
            trade_cfg, risk_cfg, initial_capital, verbose=False
        )

        if len(trades) < 5:
            return -999

        # Compute metrics
        final_cap = trades[-1]['capital']
        ret_pct = (final_cap / initial_capital - 1) * 100

        peak = initial_capital
        max_dd = 0
        for t in trades:
            if t['capital'] > peak:
                peak = t['capital']
            dd = (peak - t['capital']) / peak
            if dd > max_dd:
                max_dd = dd
        dd_pct = max(max_dd * 100, 0.5)

        returns = [t['pnl'] / max(t['capital'] - t['pnl'], 100) for t in trades]
        sharpe = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) \
            if np.std(returns) > 0 else 0

        # Composite score
        score = (0.4 * min(ret_pct, 300) / 300 +
                 0.3 * min(100 / dd_pct, 20) / 20 +
                 0.3 * min(max(sharpe, 0), 15) / 15)

        trial_count[0] += 1
        if score > best_score[0]:
            best_score[0] = score
            best_params[0] = kwargs.copy()
            print(f"    Trial {trial_count[0]:3d} | act={activation:.2f} trail={trail_dist:.2f} "
                  f"be={be_activation:.2f} | Ret={ret_pct:+.1f}% DD={dd_pct:.1f}% "
                  f"Sh={sharpe:.2f} | Score={score:.4f} *")
        elif trial_count[0] % 10 == 0:
            print(f"    Trial {trial_count[0]:3d} | Score={score:.4f} (best={best_score[0]:.4f})")

        return score

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    print(f"\n  Best trailing stop params (from {n_trials} trials):")
    print(f"    activation:    {study.best_params['activation']:.2f}")
    print(f"    trail_dist:    {study.best_params['trail_dist']:.2f}")
    print(f"    be_activation: {study.best_params['be_activation']:.2f}")
    print(f"    Score:         {study.best_value:.4f}")

    return study.best_params


# ============================================================
# METRICS COMPUTATION
# ============================================================

def calc_metrics(trades, initial_capital=INITIAL_CAPITAL):
    """Compute full suite of performance metrics from trade list."""
    if not trades:
        return {
            'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
            'final_cap': initial_capital, 'return_pct': 0, 'max_dd_pct': 0,
            'profit_factor': 0, 'sharpe': 0, 'sortino': 0,
            'avg_win': 0, 'avg_loss': 0, 'exit_reasons': {},
        }

    n = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = n - wins
    final_cap = trades[-1]['capital']
    ret_pct = (final_cap / initial_capital - 1) * 100

    # Drawdown
    peak = initial_capital
    max_dd = 0
    for t in trades:
        if t['capital'] > peak:
            peak = t['capital']
        dd = (peak - t['capital']) / peak
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_win = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe / Sortino
    returns = [t['pnl'] / max(t['capital'] - t['pnl'], 100) for t in trades]
    avg_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = (avg_ret * 252) / (std_ret * np.sqrt(252)) if std_ret > 0 else 0

    down_rets = [r for r in returns if r < 0]
    down_std = np.std(down_rets) if down_rets else 0
    sortino = (avg_ret * 252) / (down_std * np.sqrt(252)) if down_std > 0 else 0

    # Avg win/loss
    win_pnls = [t['pnl'] for t in trades if t['pnl'] > 0]
    loss_pnls = [t['pnl'] for t in trades if t['pnl'] <= 0]
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0

    # Exit reason distribution
    reasons = {}
    for t in trades:
        for r in t['exit_reason'].split('+'):
            reasons[r] = reasons.get(r, 0) + 1

    return {
        'trades': n,
        'wins': wins,
        'losses': losses,
        'win_rate': wins / n * 100,
        'final_cap': final_cap,
        'return_pct': ret_pct,
        'max_dd_pct': max_dd * 100,
        'profit_factor': pf,
        'sharpe': sharpe,
        'sortino': sortino,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'exit_reasons': reasons,
    }


# ============================================================
# COMPARISON TABLE
# ============================================================

def print_comparison(results: Dict[str, dict]):
    """Print side-by-side comparison table."""
    names = list(results.keys())
    metrics = [results[n] for n in names]

    # Header
    print('\n' + '=' * 120)
    print('  ADVANCED PROFIT-TAKING STRATEGY COMPARISON (OOS: Jan-Feb 2026)')
    print('=' * 120)

    # Table header
    hdr = (f"{'Strategy':<22} {'Trades':>6} {'WR':>6} {'Return':>8} "
           f"{'DD':>6} {'Sharpe':>7} {'PF':>6} {'AvgW':>7} {'AvgL':>7} {'Sort':>6}")
    print(hdr)
    print('-' * len(hdr))

    for name, m in zip(names, metrics):
        print(f"{name:<22} {m['trades']:>6d} {m['win_rate']:>5.1f}% "
              f"{m['return_pct']:>+7.1f}% {m['max_dd_pct']:>5.1f}% "
              f"{m['sharpe']:>7.2f} {m['profit_factor']:>6.2f} "
              f"${m['avg_win']:>6.0f} ${m['avg_loss']:>6.0f} "
              f"{m['sortino']:>6.1f}")

    # Exit reason breakdown
    all_reasons = set()
    for m in metrics:
        all_reasons.update(m['exit_reasons'].keys())
    all_reasons = sorted(all_reasons)

    print('\n  Exit Reason Breakdown:')
    hdr2 = f"  {'Strategy':<22}" + ''.join(f" {r:>8}" for r in all_reasons)
    print(hdr2)
    print('  ' + '-' * (len(hdr2) - 2))
    for name, m in zip(names, metrics):
        row = f"  {name:<22}"
        for r in all_reasons:
            count = m['exit_reasons'].get(r, 0)
            row += f" {count:>8d}"
        print(row)

    # Highlight winners
    print('\n' + '-' * 60)
    best_ret = max(names, key=lambda n: results[n]['return_pct'])
    best_dd = min(names, key=lambda n: results[n]['max_dd_pct'])
    best_sh = max(names, key=lambda n: results[n]['sharpe'])
    best_pf = max(names, key=lambda n: results[n]['profit_factor'])
    best_sort = max(names, key=lambda n: results[n]['sortino'])

    print(f"  Best Return:        {best_ret} ({results[best_ret]['return_pct']:+.1f}%)")
    print(f"  Lowest Drawdown:    {best_dd} ({results[best_dd]['max_dd_pct']:.1f}%)")
    print(f"  Best Sharpe:        {best_sh} ({results[best_sh]['sharpe']:.2f})")
    print(f"  Best Profit Factor: {best_pf} ({results[best_pf]['profit_factor']:.2f})")
    print(f"  Best Sortino:       {best_sort} ({results[best_sort]['sortino']:.1f})")

    # Composite ranking
    print('\n  Composite Ranking (40% Ret + 30% 1/DD + 30% Sharpe):')
    scores = {}
    max_ret = max(m['return_pct'] for m in metrics)
    min_dd = min(m['max_dd_pct'] for m in metrics)
    max_sh = max(m['sharpe'] for m in metrics)
    for name, m in zip(names, metrics):
        s = (0.4 * m['return_pct'] / max(max_ret, 1) +
             0.3 * min_dd / max(m['max_dd_pct'], 0.1) +
             0.3 * m['sharpe'] / max(max_sh, 0.1))
        scores[name] = s
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    for rank, (name, score) in enumerate(ranked, 1):
        marker = ' <-- WINNER' if rank == 1 else ''
        print(f"    #{rank} {name:<22} score={score:.4f}{marker}")


# ============================================================
# EQUITY CURVE CHART
# ============================================================

def plot_comparison(all_results: Dict[str, Tuple[list, dict]], output_path: str):
    """Plot overlaid equity curves for all strategies."""
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={'height_ratios': [2, 1]})

    ax1 = axes[0]
    ax2 = axes[1]

    for i, (name, (trades, metrics)) in enumerate(all_results.items()):
        color = colors[i % len(colors)]
        capitals = [INITIAL_CAPITAL] + [t['capital'] for t in trades]
        x = list(range(len(capitals)))
        label = f"{name} ({metrics['return_pct']:+.0f}%, DD={metrics['max_dd_pct']:.1f}%)"
        ax1.plot(x, capitals, color=color, linewidth=2, label=label, alpha=0.85)

        # Drawdown
        peak = INITIAL_CAPITAL
        dds = [0]
        for t in trades:
            if t['capital'] > peak:
                peak = t['capital']
            dds.append((peak - t['capital']) / peak * 100)
        ax2.plot(x, dds, color=color, linewidth=1.5, alpha=0.7)

    ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.4)
    ax1.set_title('Equity Curve Comparison: Advanced Profit-Taking (OOS Jan-Feb 2026)',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Capital ($)')
    ax1.set_xlabel('Trade #')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Drawdown Comparison', fontsize=12)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Trade #')
    ax2.set_ylim(bottom=0)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved to {output_path}")


# ============================================================
# TRADE-LEVEL DETAIL FOR TIME EXITS
# ============================================================

def analyze_time_exits(all_results: Dict[str, Tuple[list, dict]]):
    """Show how each strategy handles what were TIME exits in baseline."""
    baseline_trades = all_results.get('0. Baseline', ([], {}))[0]
    if not baseline_trades:
        return

    time_trades = [t for t in baseline_trades if t['exit_reason'] == 'TIME']
    if not time_trades:
        return

    print(f'\n  TIME-exit trade analysis ({len(time_trades)} baseline TIME exits):')

    # Build header
    other_names = [n for n in all_results if n != '0. Baseline']
    hdr = f"  {'Date':<12} {'Dir':<5} {'BasePnL':>8} "
    for name in other_names:
        short = name.split('.')[1].strip()[:8] if '.' in name else name[:8]
        hdr += f"| {short:>8} {'Rsn':<8}"
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    for bt_trade in time_trades[:15]:  # Show first 15
        line = (f"  {bt_trade['date']:<12} {bt_trade['direction']:<5} "
                f"${bt_trade['pnl']:>+7.0f} ")
        for name in other_names:
            trades_list = all_results[name][0]
            match = None
            for t in trades_list:
                if t['date'] == bt_trade['date'] and t['time'] == bt_trade['time']:
                    match = t
                    break
            if match:
                line += f"| ${match['pnl']:>+6.0f} {match['exit_reason']:<8}"
            else:
                line += f"|  {'skip':>6} {'---':<8}"
        print(line)


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()

    # Load config
    with open('config/strategy.json') as f:
        cfg_data = json.load(f)

    tc_data = cfg_data['trade_config']
    rc_data = cfg_data['risk_config']

    trade_cfg = TradeConfig(**{k: v for k, v in tc_data.items()
                               if k in TradeConfig.__dataclass_fields__})
    risk_cfg = RiskConfig(**{k: v for k, v in rc_data.items()
                             if k in RiskConfig.__dataclass_fields__})

    print('=' * 70)
    print('  ADVANCED PROFIT-TAKING STRATEGY COMPARISON')
    print('=' * 70)
    print(f'  Config: PT={trade_cfg.profit_target_pct:.0%} '
          f'SL={trade_cfg.stop_loss_pct:.0%} '
          f'H={trade_cfg.max_hold_bars} '
          f'RSI={trade_cfg.rsi_call_threshold}/{trade_cfg.rsi_put_threshold}')
    print(f'  Risk: SFL={risk_cfg.stop_after_first_loss} '
          f'CL={risk_cfg.max_consecutive_losses} '
          f'DLL={risk_cfg.max_daily_loss_pct}')
    print(f'  Window: {trade_cfg.trade_start_hour}:{trade_cfg.trade_start_minute:02d}'
          f'-{trade_cfg.trade_end_hour}:{trade_cfg.trade_end_minute:02d}')
    print()

    # Create engine for data loading and index building
    bt = Backtest0DTE(trade_cfg, risk_cfg, initial_capital=INITIAL_CAPITAL)

    # ================================================================
    # PHASE 1: Load train data + Optuna trailing stop optimization
    # ================================================================
    train_start = '2025-01-02'
    train_end = '2025-12-31'
    test_start = '2026-01-02'
    test_end = '2026-02-14'

    print(f'  [1/4] Loading train data ({train_start} to {train_end})...')
    train_underlying, train_options, train_features = bt.load_data(
        train_start, train_end)
    print(f'        {len(train_underlying)} underlying bars, '
          f'{train_underlying["date"].nunique()} days')

    # Kelly calibration (will default to 0.07)
    train_vol = bt.compute_historical_volatility(train_underlying)
    training_data = bt.generate_training_samples(
        train_underlying, train_options, train_features, train_vol)
    bt.calculate_kelly_only(training_data)
    kelly_pct = bt.risk_manager.position_sizer.kelly_pct
    print(f'        Kelly = {kelly_pct:.2%}')

    print(f'\n  [2/4] Optimizing trailing stop on train data (50 trials)...')
    trail_params = optimize_trailing_stop(
        bt, train_underlying, train_features, trade_cfg, risk_cfg,
        n_trials=50, initial_capital=INITIAL_CAPITAL
    )

    # ================================================================
    # PHASE 2: Load test data
    # ================================================================
    print(f'\n  [3/4] Loading test data ({test_start} to {test_end})...')
    test_underlying, test_options, test_features = bt.load_data(
        test_start, test_end)
    test_vol = bt.compute_historical_volatility(test_underlying)
    print(f'        {len(test_underlying)} underlying bars, '
          f'{test_underlying["date"].nunique()} days')

    # ================================================================
    # PHASE 3: Run all strategies on test data
    # ================================================================
    print(f'\n  [4/4] Running 6 strategies on OOS test data...')

    strategies = [
        ('0. Baseline', exit_baseline, {}),
        ('A. Ratchet Lock', exit_ratchet, {
            'levels': [(0.25, 0.10), (0.35, 0.20), (0.45, 0.30)]
        }),
        ('B. Partial Exit', exit_partial, {
            'partial_pt': 0.25
        }),
        ('C. Trailing Stop', exit_trailing, trail_params),
        ('D. Dynamic PT', exit_dynamic_pt, {
            'tiers': [(8, 0.50), (12, 0.35), (16, 0.20)]
        }),
        ('E. Hybrid (R+D)', exit_hybrid, {
            'levels': [(0.25, 0.10), (0.35, 0.20), (0.45, 0.30)],
            'tiers': [(8, 0.50), (12, 0.35), (16, 0.20)],
        }),
    ]

    all_results = {}  # name -> (trades, metrics)

    for name, exit_fn, exit_kwargs in strategies:
        t1 = time.time()
        trades = run_backtest(
            exit_fn, exit_kwargs, bt,
            test_underlying, test_features,
            trade_cfg, risk_cfg, INITIAL_CAPITAL,
            verbose=False
        )
        metrics = calc_metrics(trades, INITIAL_CAPITAL)
        elapsed = time.time() - t1
        all_results[name] = (trades, metrics)
        print(f"    {name:<22} | {metrics['trades']:>3d} trades | "
              f"{metrics['return_pct']:>+7.1f}% | DD={metrics['max_dd_pct']:.1f}% | "
              f"Sh={metrics['sharpe']:.2f} | {elapsed:.1f}s")

    # ================================================================
    # PHASE 4: Compare results
    # ================================================================
    results_only = {n: m for n, (_, m) in all_results.items()}
    print_comparison(results_only)
    analyze_time_exits(all_results)

    # Save chart
    plot_comparison(all_results, 'output/profit_taking_comparison.png')

    # Save detailed results to CSV
    rows = []
    for name, (trades, metrics) in all_results.items():
        for t in trades:
            t_copy = t.copy()
            t_copy['strategy'] = name
            rows.append(t_copy)
    pd.DataFrame(rows).to_csv('output/profit_taking_trades.csv', index=False)
    print(f"  Trades saved to output/profit_taking_trades.csv")

    elapsed_total = time.time() - t0
    print(f'\n  Total runtime: {elapsed_total:.0f}s ({elapsed_total / 60:.1f} min)')
    print('=' * 70)


if __name__ == '__main__':
    main()
