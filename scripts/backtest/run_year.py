"""
Full 2025 Backtest — Complete Optimized Strategy

Runs the best-performing config across the entire 2025 dataset
with verbose trade-by-trade output and comprehensive metrics.

Loads all parameters from config/strategy.json (single source of truth).

Usage:
  python scripts/run_2025_full.py
"""
import sys
sys.path.insert(0, '.')

import time
import numpy as np
import pandas as pd

from backtest.engine import Backtest0DTE, TradeConfig
from core.risk_manager import RiskConfig
from config import defaults as cfg


def main():
    cap = cfg.initial_capital()
    print('=' * 80)
    print('  FULL 2025 BACKTEST — COMPLETE OPTIMIZED STRATEGY')
    print('=' * 80)
    print()

    # Load config from strategy.json (single source of truth)
    import json
    config = json.load(open('config/strategy.json'))
    tc_dict = config['trade_config']
    rc_dict = config['risk_config']
    tc = TradeConfig(**{k: v for k, v in tc_dict.items() if k in TradeConfig.__dataclass_fields__})
    rc = RiskConfig(**{k: v for k, v in rc_dict.items() if k in RiskConfig.__dataclass_fields__})

    print('  CONFIG (loaded from strategy.json):')
    print(f'    Strategy:     {tc.strategy} (RSI>{tc.rsi_call_threshold} CALL, RSI<{tc.rsi_put_threshold} PUT)')
    pt = tc.call_profit_target_pct or tc.profit_target_pct
    sl = tc.call_stop_loss_pct or tc.stop_loss_pct
    hold = tc.call_max_hold_bars or tc.max_hold_bars
    print(f'    PT={pt*100:.0f}%  SL={sl*100:.0f}%  Hold={hold} bars')
    print(f'    Post-Loss:    {tc.post_loss_strategy}')
    print(f'    Kelly: {rc.kelly_fraction*100:.0f}%  MaxRisk: {rc.max_risk_per_trade_pct*100:.0f}%  DailyLosses: {rc.max_daily_losses}  ConsecLosses: {rc.max_consecutive_losses}  DLL: {rc.max_daily_loss_pct*100:.1f}%')
    print(f'    Capital:      ${cap:,.0f}')
    print()

    bt = Backtest0DTE(tc, rc, initial_capital=cap)
    kelly_pct = rc_dict.get('kelly_pct', 0.06)
    bt.risk_manager.set_kelly(kelly_pct)

    print('  Loading 2025 data...')
    t0 = time.time()
    u, o, f = bt.load_data('2025-01-01', '2025-12-31')
    v = bt.compute_historical_volatility(u)
    print(f'  Data loaded in {time.time()-t0:.1f}s')
    print()

    print('  Running backtest (verbose)...')
    print('-' * 80)
    trades = bt.run_no_ml(u, o, f, v, verbose=True)
    print('-' * 80)

    if not trades:
        print('No trades!')
        return

    # Compute full metrics
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    total_pnl = sum(t.pnl for t in trades)
    wr = wins / n * 100
    ret = total_pnl / cap * 100

    peak = cap
    max_dd = 0
    max_dd_date = ''
    for t in trades:
        c = t.capital
        if c > peak:
            peak = c
        dd = (peak - c) / peak
        if dd > max_dd:
            max_dd = dd
            max_dd_date = t.date

    rets_list = [t.pnl / max(t.capital - t.pnl, 1) for t in trades]
    mu = np.mean(rets_list)
    sigma = np.std(rets_list)
    sharpe = (mu * 252) / (sigma * np.sqrt(252)) if sigma > 0 else 0
    down = [r for r in rets_list if r < 0]
    ds = np.std(down) if len(down) > 1 else 1
    sortino = (mu * 252) / (ds * np.sqrt(252)) if ds > 0 else 0

    gp = sum(t.pnl for t in trades if t.pnl > 0)
    gl = abs(sum(t.pnl for t in trades if t.pnl <= 0)) or 0.01
    pf = gp / gl

    call_t = [t for t in trades if t.direction == 'CALL']
    put_t = [t for t in trades if t.direction == 'PUT']

    pe = sum(1 for t in trades if t.exit_reason == 'PROFIT')
    se = sum(1 for t in trades if t.exit_reason == 'STOP')
    te = sum(1 for t in trades if t.exit_reason == 'TIME')

    daily_pnl = {}
    for t in trades:
        daily_pnl[t.date] = daily_pnl.get(t.date, 0) + t.pnl
    green_days = sum(1 for p in daily_pnl.values() if p > 0)
    red_days = sum(1 for p in daily_pnl.values() if p <= 0)
    total_days = len(daily_pnl)

    # Monthly breakdown
    monthly = {}
    for t in trades:
        m = t.date[:7]
        if m not in monthly:
            monthly[m] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
        monthly[m]['trades'] += 1
        if t.pnl > 0:
            monthly[m]['wins'] += 1
        monthly[m]['pnl'] += t.pnl

    print()
    print('=' * 80)
    print('  RESULTS SUMMARY')
    print('=' * 80)
    print(f'    Total Trades:     {n}')
    print(f'    Win Rate:         {wr:.1f}% ({wins}W / {n-wins}L)')
    print(f'    Total PnL:        ${total_pnl:+,.0f}')
    print(f'    Return:           {ret:+.1f}%')
    print(f'    Final Capital:    ${trades[-1].capital:,.0f}')
    print(f'    Max Drawdown:     {max_dd*100:.1f}% (at {max_dd_date})')
    print(f'    Sharpe Ratio:     {sharpe:.2f}')
    print(f'    Sortino Ratio:    {sortino:.2f}')
    print(f'    Profit Factor:    {pf:.2f}')
    calmar = ret / (max_dd * 100) if max_dd > 0.001 else 0
    print(f'    Calmar Ratio:     {calmar:.1f}')
    print()
    print('  DIRECTION BREAKDOWN:')
    call_wins = sum(1 for t in call_t if t.pnl > 0)
    call_wr = call_wins / max(len(call_t), 1) * 100
    call_pnl = sum(t.pnl for t in call_t)
    put_wins = sum(1 for t in put_t if t.pnl > 0)
    put_wr = put_wins / max(len(put_t), 1) * 100
    put_pnl = sum(t.pnl for t in put_t)
    print(f'    CALL: {len(call_t)} trades, WR={call_wr:.1f}%, PnL=${call_pnl:+,.0f}')
    print(f'    PUT:  {len(put_t)} trades, WR={put_wr:.1f}%, PnL=${put_pnl:+,.0f}')
    print()
    print('  EXIT DISTRIBUTION:')
    print(f'    PROFIT: {pe} ({pe/n*100:.1f}%)')
    print(f'    STOP:   {se} ({se/n*100:.1f}%)')
    print(f'    TIME:   {te} ({te/n*100:.1f}%)')
    print()
    print('  DAILY STATS:')
    print(f'    Trading Days:     {total_days}')
    print(f'    Green Days:       {green_days} ({green_days/total_days*100:.1f}%)')
    print(f'    Red Days:         {red_days} ({red_days/total_days*100:.1f}%)')
    daily_vals = list(daily_pnl.values())
    print(f'    Avg Daily PnL:    ${np.mean(daily_vals):+,.0f}')
    print(f'    Best Day:         ${max(daily_vals):+,.0f}')
    print(f'    Worst Day:        ${min(daily_vals):+,.0f}')
    print()
    print('  MONTHLY BREAKDOWN:')
    print(f'    {"Month":<8} {"Trades":>6} {"WR":>6} {"PnL":>10} {"Cumulative":>12}')
    print(f'    {"-"*44}')
    cum = 0
    for m in sorted(monthly.keys()):
        d = monthly[m]
        cum += d['pnl']
        mwr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
        print(f'    {m:<8} {d["trades"]:>6} {mwr:>5.1f}% ${d["pnl"]:>+9,.0f} ${cum:>+11,.0f}')
    print()

    # Post-loss specific analysis
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

    if post_loss_trades:
        pl_n = len(post_loss_trades)
        pl_wins = sum(1 for t in post_loss_trades if t.pnl > 0)
        pl_wr = pl_wins / pl_n * 100
        pl_pnl = sum(t.pnl for t in post_loss_trades)
        print('  POST-LOSS TRADE ANALYSIS:')
        print(f'    Post-loss trades: {pl_n}')
        print(f'    Post-loss WR:     {pl_wr:.1f}%')
        print(f'    Post-loss PnL:    ${pl_pnl:+,.0f}')
        print(f'    Avg Post-loss:    ${pl_pnl/pl_n:+,.0f}')
        print(f'    Pre-loss trades:  {len(pre_loss_trades)}')
        pre_pnl = sum(t.pnl for t in pre_loss_trades)
        print(f'    Pre-loss PnL:     ${pre_pnl:+,.0f}')
        print()

    # Avg trade stats
    avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if wins > 0 else 0
    avg_loss = np.mean([t.pnl for t in trades if t.pnl <= 0]) if (n - wins) > 0 else 0
    avg_bars = np.mean([t.bars_held for t in trades])
    print('  TRADE STATISTICS:')
    print(f'    Avg Winner:       ${avg_win:+,.0f}')
    print(f'    Avg Loser:        ${avg_loss:+,.0f}')
    print(f'    Win/Loss Ratio:   {abs(avg_win/avg_loss):.2f}x' if avg_loss != 0 else '    Win/Loss Ratio:   inf')
    print(f'    Avg Bars Held:    {avg_bars:.1f}')
    print(f'    Avg PnL/Trade:    ${total_pnl/n:+,.0f}')
    print()

    # Save trades CSV
    df = pd.DataFrame([{
        'date': t.date, 'time': t.time, 'direction': t.direction,
        'strike': t.strike, 'ticker': t.option_ticker,
        'rsi': round(t.rsi, 1), 'entry': round(t.entry, 4),
        'exit': round(t.exit, 4), 'exit_reason': t.exit_reason,
        'bars_held': t.bars_held, 'contracts': t.num_contracts,
        'pnl': round(t.pnl, 2), 'capital': round(t.capital, 2),
    } for t in trades])
    df.to_csv('output/backtest_2025_full.csv', index=False)
    print(f'  Trades saved to output/backtest_2025_full.csv ({n} trades)')

    # Now run OOS 2026 with same config
    print()
    print('=' * 80)
    print('  OOS VALIDATION — 2026')
    print('=' * 80)

    bt2 = Backtest0DTE(tc, rc, initial_capital=cap)
    bt2.risk_manager.set_kelly(kelly_pct)
    u2, o2, f2 = bt2.load_data('2026-01-02', '2026-02-25')
    v2 = bt2.compute_historical_volatility(u2)
    trades2 = bt2.run_no_ml(u2, o2, f2, v2, verbose=True)

    if trades2:
        n2 = len(trades2)
        w2 = sum(1 for t in trades2 if t.pnl > 0)
        pnl2 = sum(t.pnl for t in trades2)
        ret2 = pnl2 / cap * 100

        peak2 = cap; dd2 = 0
        for t in trades2:
            c = t.capital
            if c > peak2: peak2 = c
            dd = (peak2 - c) / peak2
            if dd > dd2: dd2 = dd

        rets2 = [t.pnl / max(t.capital - t.pnl, 1) for t in trades2]
        mu2 = np.mean(rets2); sig2 = np.std(rets2)
        sh2 = (mu2 * 252) / (sig2 * np.sqrt(252)) if sig2 > 0 else 0
        gp2 = sum(t.pnl for t in trades2 if t.pnl > 0)
        gl2 = abs(sum(t.pnl for t in trades2 if t.pnl <= 0)) or 0.01
        pf2 = gp2 / gl2

        call2 = [t for t in trades2 if t.direction == 'CALL']
        put2 = [t for t in trades2 if t.direction == 'PUT']

        print()
        print(f'  OOS 2026 RESULTS:')
        print(f'    Trades: {n2}  WR: {w2/n2*100:.1f}%  PnL: ${pnl2:+,.0f}  Return: {ret2:+.1f}%')
        print(f'    Max DD: {dd2*100:.1f}%  Sharpe: {sh2:.2f}  PF: {pf2:.2f}')
        print(f'    CALL: {len(call2)}t WR={sum(1 for t in call2 if t.pnl>0)/max(len(call2),1)*100:.1f}% ${sum(t.pnl for t in call2):+,.0f}')
        print(f'    PUT:  {len(put2)}t WR={sum(1 for t in put2 if t.pnl>0)/max(len(put2),1)*100:.1f}% ${sum(t.pnl for t in put2):+,.0f}')

        # Save OOS trades
        df2 = pd.DataFrame([{
            'date': t.date, 'time': t.time, 'direction': t.direction,
            'strike': t.strike, 'ticker': t.option_ticker,
            'rsi': round(t.rsi, 1), 'entry': round(t.entry, 4),
            'exit': round(t.exit, 4), 'exit_reason': t.exit_reason,
            'bars_held': t.bars_held, 'contracts': t.num_contracts,
            'pnl': round(t.pnl, 2), 'capital': round(t.capital, 2),
        } for t in trades2])
        df2.to_csv('output/backtest_2026_oos.csv', index=False)
        print(f'    Trades saved to output/backtest_2026_oos.csv')

    # IS vs OOS comparison
    print()
    print('=' * 80)
    print('  IS vs OOS COMPARISON')
    print('=' * 80)
    if trades2:
        print(f'    {"Metric":<15} {"2025 IS":>12} {"2026 OOS":>12}')
        print(f'    {"-"*40}')
        print(f'    {"Return %":<15} {ret:>+11.1f}% {ret2:>+11.1f}%')
        print(f'    {"Trades":<15} {n:>12} {n2:>12}')
        print(f'    {"Win Rate %":<15} {wr:>11.1f}% {w2/n2*100:>11.1f}%')
        print(f'    {"Sharpe":<15} {sharpe:>12.2f} {sh2:>12.2f}')
        print(f'    {"Max DD %":<15} {max_dd*100:>11.1f}% {dd2*100:>11.1f}%')
        print(f'    {"Profit Factor":<15} {pf:>12.2f} {pf2:>12.2f}')

    print()
    print('=' * 80)
    print(f'  COMPLETE | Total runtime: {time.time()-t0:.0f}s')
    print('=' * 80)


if __name__ == '__main__':
    main()
