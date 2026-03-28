"""
Strategy Optimization: Parallel Config Comparison

Tests multiple configurations on 2025 + 2026 data side-by-side.
Loads data once, runs each config variant, produces comparison table.
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from dataclasses import replace
from backtest.engine import Backtest0DTE, TradeConfig
from config import defaults as cfg
from core.risk_manager import RiskConfig


def load_base_config():
    with open('config/strategy.json') as f:
        data = json.load(f)
    tc = data['trade_config']
    rc = data['risk_config']
    trade_cfg = TradeConfig(**{k: v for k, v in tc.items() if k in TradeConfig.__dataclass_fields__})
    risk_cfg = RiskConfig(**{k: v for k, v in rc.items() if k in RiskConfig.__dataclass_fields__})
    return trade_cfg, risk_cfg


def compute_metrics(trades, initial_capital=10000):
    if not trades:
        return {
            'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
            'total_pnl': 0, 'final_cap': initial_capital, 'return_pct': 0,
            'max_dd': 0, 'profit_factor': 0, 'sharpe': 0,
            'avg_win': 0, 'avg_loss': 0, 'trades_per_day': 0,
        }
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = len(trades) - wins
    total_pnl = sum(t.pnl for t in trades)
    final_cap = trades[-1].capital
    ret_pct = (final_cap / initial_capital - 1) * 100

    # Drawdown
    peak = initial_capital
    max_dd = 0
    for t in trades:
        if t.capital > peak:
            peak = t.capital
        dd = (peak - t.capital) / peak
        max_dd = max(max_dd, dd)

    # Profit factor
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe
    returns = [t.pnl / max(t.capital - t.pnl, 1) for t in trades]
    avg_ret = np.mean(returns) if returns else 0
    std_ret = np.std(returns) if returns else 1
    sharpe = (avg_ret * 252) / (std_ret * np.sqrt(252)) if std_ret > 0 else 0

    avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if wins > 0 else 0
    avg_loss = np.mean([t.pnl for t in trades if t.pnl <= 0]) if losses > 0 else 0

    # Trades per day
    unique_days = len(set(t.date for t in trades))
    trades_per_day = len(trades) / unique_days if unique_days > 0 else 0

    return {
        'trades': len(trades), 'wins': wins, 'losses': losses,
        'win_rate': wins / len(trades) * 100 if trades else 0,
        'total_pnl': total_pnl, 'final_cap': final_cap, 'return_pct': ret_pct,
        'max_dd': max_dd * 100, 'profit_factor': pf, 'sharpe': sharpe,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'trades_per_day': trades_per_day,
    }


def run_config(name, trade_cfg, risk_cfg, train_data, test_underlying, test_options,
               test_features, test_vol, initial_capital=10000):
    """Run a single config and return metrics."""
    bt = Backtest0DTE(trade_cfg, risk_cfg, initial_capital=initial_capital)
    bt._build_option_index(test_options)
    bt.calculate_kelly_only(train_data)
    trades = bt.run_no_ml(test_underlying, test_options, test_features, test_vol, verbose=False)
    metrics = compute_metrics(trades, initial_capital)
    return trades, metrics


def main():
    base_tc, base_rc = load_base_config()
    initial_capital = cfg.initial_capital()

    # ============================================================
    # DEFINE CONFIGS TO TEST
    # ============================================================
    configs = []

    # A: Baseline (current config: max_daily_losses=2, no DLL, no SFL)
    configs.append(("A: Baseline",
                    replace(base_tc),
                    replace(base_rc)))

    # B: SFL-1 (Stop First Loss - max 1 daily loss)
    configs.append(("B: SFL-1",
                    replace(base_tc),
                    replace(base_rc, max_daily_losses=1)))

    # C: SFL-1 + DLL 0.8% (daily loss limit)
    configs.append(("C: SFL-1+DLL",
                    replace(base_tc),
                    replace(base_rc, max_daily_losses=1, max_daily_loss_pct=0.008)))

    # D: SFL-1 + wider window 10:00-12:00
    configs.append(("D: SFL-1+Wide",
                    replace(base_tc, trade_end_hour=12, trade_end_minute=0),
                    replace(base_rc, max_daily_losses=1)))

    # E: Post-loss flip (keep trading after loss, but flip direction)
    configs.append(("E: Flip",
                    replace(base_tc, post_loss_strategy="flip"),
                    replace(base_rc, max_daily_losses=999)))

    # F: Post-loss momentum_confirm (needs momentum_3 feature)
    configs.append(("F: MomConfirm",
                    replace(base_tc, post_loss_strategy="momentum_confirm"),
                    replace(base_rc, max_daily_losses=999)))

    # G: Post-loss multi_confirm (needs momentum_3 + vwap_distance + trend_strength)
    configs.append(("G: MultiConfirm",
                    replace(base_tc, post_loss_strategy="multi_confirm"),
                    replace(base_rc, max_daily_losses=999)))

    # H: Tighter RSI 75/25 (fewer, higher quality signals)
    configs.append(("H: RSI 75/25",
                    replace(base_tc, rsi_call_threshold=75, rsi_put_threshold=25),
                    replace(base_rc)))

    # I: SFL-1 + Tighter RSI 75/25
    configs.append(("I: SFL+RSI75",
                    replace(base_tc, rsi_call_threshold=75, rsi_put_threshold=25),
                    replace(base_rc, max_daily_losses=1)))

    # J: SFL-1 + wider window + DLL (full Phase 8)
    configs.append(("J: SFL+Wide+DLL",
                    replace(base_tc, trade_end_hour=12, trade_end_minute=0),
                    replace(base_rc, max_daily_losses=1, max_daily_loss_pct=0.008)))

    # K: SFL-1 + skip first 10 min (avoid opening noise)
    configs.append(("K: SFL+Skip10m",
                    replace(base_tc, trade_start_minute=10),
                    replace(base_rc, max_daily_losses=1)))

    # L: SFL-1 + tighter stop 25%
    configs.append(("L: SFL+SL25",
                    replace(base_tc, stop_loss_pct=0.25),
                    replace(base_rc, max_daily_losses=1)))

    # M: SFL-1 + wider stop 45% + higher PT 60%
    configs.append(("M: SFL+PT60/SL45",
                    replace(base_tc, profit_target_pct=0.60, stop_loss_pct=0.45),
                    replace(base_rc, max_daily_losses=1)))

    # N: SFL-1 + DLL + CL=2 (tighter consecutive loss)
    configs.append(("N: SFL+DLL+CL2",
                    replace(base_tc),
                    replace(base_rc, max_daily_losses=1, max_daily_loss_pct=0.008,
                            max_consecutive_losses=2)))

    # ============================================================
    # LOAD DATA ONCE
    # ============================================================
    print("=" * 100)
    print("  STRATEGY COMPARISON: PARALLEL CONFIG TESTING")
    print("=" * 100)

    bt0 = Backtest0DTE(base_tc, base_rc, initial_capital=initial_capital)

    print("\n--- Loading training data (Kelly calibration) ---")
    train_u, train_o, train_f = bt0.load_data('2024-07-01', '2024-12-31')
    train_v = bt0.compute_historical_volatility(train_u)
    train_d = bt0.generate_training_samples(train_u, train_o, train_f, train_v)
    print(f"  Training samples: {len(train_d)}")

    # Test periods
    periods = [
        ("2025", '2025-01-01', '2025-12-31'),
        ("2026", '2026-01-01', '2026-02-28'),
    ]

    all_results = {}

    for period_name, test_start, test_end in periods:
        print(f"\n{'='*100}")
        print(f"  TEST PERIOD: {period_name} ({test_start} to {test_end})")
        print(f"{'='*100}")

        bt_loader = Backtest0DTE(base_tc, base_rc, initial_capital=initial_capital)
        test_u, test_o, test_f = bt_loader.load_data(test_start, test_end)
        test_v = bt_loader.compute_historical_volatility(test_u)

        period_results = {}

        for name, tc, rc in configs:
            print(f"  Running {name}...", end=" ", flush=True)
            trades, metrics = run_config(
                name, tc, rc, train_d,
                test_u, test_o, test_f, test_v, initial_capital
            )
            period_results[name] = metrics
            print(f"{metrics['trades']} trades, WR={metrics['win_rate']:.1f}%, "
                  f"P&L=${metrics['total_pnl']:+,.0f}, DD={metrics['max_dd']:.1f}%, "
                  f"PF={metrics['profit_factor']:.2f}")

        all_results[period_name] = period_results

    # ============================================================
    # PRINT COMPARISON TABLES
    # ============================================================
    for period_name in all_results:
        results = all_results[period_name]

        print(f"\n{'='*130}")
        print(f"  COMPARISON TABLE: {period_name}")
        print(f"{'='*130}")
        header = (f"  {'Config':<18} {'Trades':>6} {'Wins':>5} {'WR%':>6} "
                  f"{'P&L':>10} {'Ret%':>8} {'MaxDD':>7} {'PF':>6} "
                  f"{'Sharpe':>7} {'AvgWin':>8} {'AvgLoss':>8} {'T/Day':>5}")
        print(header)
        sep = (f"  {'─'*18} {'─'*6} {'─'*5} {'─'*6} "
               f"{'─'*10} {'─'*8} {'─'*7} {'─'*6} "
               f"{'─'*7} {'─'*8} {'─'*8} {'─'*5}")
        print(sep)

        for name in results:
            m = results[name]
            pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 100 else "inf"
            print(f"  {name:<18} {m['trades']:>6} {m['wins']:>5} {m['win_rate']:>5.1f}% "
                  f"${m['total_pnl']:>+9,.0f} {m['return_pct']:>+7.1f}% {m['max_dd']:>6.1f}% "
                  f"{pf_str:>6} {m['sharpe']:>7.2f} ${m['avg_win']:>7,.0f} ${m['avg_loss']:>7,.0f} "
                  f"{m['trades_per_day']:>5.1f}")

        # Rank by P&L
        print(f"\n  RANKING by P&L:")
        ranked = sorted(results.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
        for i, (name, m) in enumerate(ranked):
            marker = " *** BEST ***" if i == 0 else ""
            print(f"    #{i+1}: {name:<18} P&L=${m['total_pnl']:>+9,.0f}  "
                  f"WR={m['win_rate']:.1f}%  DD={m['max_dd']:.1f}%  PF={m['profit_factor']:.2f}"
                  f"{marker}")

        # Rank by Return/DD
        print(f"\n  RANKING by Return/DD ratio:")
        ranked_rd = sorted(results.items(),
                           key=lambda x: x[1]['return_pct'] / max(x[1]['max_dd'], 0.1),
                           reverse=True)
        for i, (name, m) in enumerate(ranked_rd):
            ret_dd = m['return_pct'] / max(m['max_dd'], 0.1)
            marker = " *** BEST ***" if i == 0 else ""
            print(f"    #{i+1}: {name:<18} Ret/DD={ret_dd:>6.1f}x  "
                  f"(Ret={m['return_pct']:>+.1f}%, DD={m['max_dd']:.1f}%){marker}")

    # ============================================================
    # CROSS-PERIOD CONSISTENCY
    # ============================================================
    if len(all_results) > 1:
        print(f"\n{'='*130}")
        print(f"  CROSS-PERIOD CONSISTENCY (both 2025 + 2026 profitable = robust)")
        print(f"{'='*130}")
        period_names = list(all_results.keys())
        config_names = [c[0] for c in configs]

        header_parts = [f"  {'Config':<18}"]
        for p in period_names:
            header_parts.append(f"  {p+' Ret%':>10} {p+' DD%':>7} {p+' PF':>6}")
        header_parts.append(f"  {'Score':>7} {'Verdict':>14}")
        print("".join(header_parts))

        sep_parts = [f"  {'─'*18}"]
        for p in period_names:
            sep_parts.append(f"  {'─'*10} {'─'*7} {'─'*6}")
        sep_parts.append(f"  {'─'*7} {'─'*14}")
        print("".join(sep_parts))

        scored = []
        for name in config_names:
            parts = [f"  {name:<18}"]
            rets = []
            dds = []
            pfs = []
            for p in period_names:
                m = all_results[p][name]
                pf_s = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 100 else "inf"
                parts.append(f"  {m['return_pct']:>+9.1f}% {m['max_dd']:>6.1f}% {pf_s:>6}")
                rets.append(m['return_pct'])
                dds.append(m['max_dd'])
                pfs.append(min(m['profit_factor'], 10))

            both_pos = all(r > 0 for r in rets)
            # Composite score: avg return - 2*avg_dd + avg_pf*10
            score = np.mean(rets) - 2 * np.mean(dds) + np.mean(pfs) * 5
            verdict = "ROBUST" if both_pos else "MIXED"
            parts.append(f"  {score:>7.1f} {verdict:>14}")
            scored.append((name, score, both_pos))
            print("".join(parts))

        # Final recommendation
        robust = [(n, s) for n, s, b in scored if b]
        if robust:
            best_name, best_score = max(robust, key=lambda x: x[1])
            print(f"\n  {'='*60}")
            print(f"  RECOMMENDATION: {best_name}")
            print(f"  Composite Score: {best_score:.1f}")
            print(f"  {'='*60}")

    print(f"\n{'='*100}")
    print("  ALL CONFIGS TESTED")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
