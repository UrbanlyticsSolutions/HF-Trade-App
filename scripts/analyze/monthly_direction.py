"""
Monthly & Direction Performance Analysis + Consecutive Loss Deep-Dive

Runs current strategy.json config on 2025 + 2026 data and produces:
  1. Monthly P&L breakdown by CALL vs PUT
  2. Red month root cause (which direction, which exit reasons)
  3. Consecutive loss streak analysis (length, direction, time-of-day, RSI)
  4. Post-loss recovery patterns
  5. Actionable recommendations

Usage:
  python scripts/analyze/monthly_direction.py
"""
import sys
sys.path.insert(0, '.')

import io
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import asdict

from backtest.engine import Backtest0DTE, TradeConfig
from core.risk_manager import RiskConfig
from config import defaults as cfg


def load_and_run(start, end):
    """Run backtest with current strategy.json config, return trade list."""
    tc_dict = cfg.get_trade_config()
    tc = TradeConfig(**{k: v for k, v in tc_dict.items() if k in TradeConfig.__dataclass_fields__})
    rc_dict = cfg.get_risk_config()
    rc = RiskConfig(**{k: v for k, v in rc_dict.items() if k in RiskConfig.__dataclass_fields__})
    cap = cfg.initial_capital()

    bt = Backtest0DTE(tc, rc, initial_capital=cap)
    u, o, f = bt.load_data(start, end)
    v = bt.compute_historical_volatility(u)

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        trades = bt.run_no_ml(u, o, f, v, verbose=False)
    finally:
        sys.stdout = old_stdout

    return trades, cap


def trades_to_df(trades):
    """Convert trade objects to DataFrame."""
    rows = []
    for t in trades:
        rows.append({
            'date': t.date,
            'time': t.time,
            'direction': t.direction,
            'strike': t.strike,
            'rsi': t.rsi,
            'entry': t.entry,
            'exit': t.exit,
            'exit_reason': t.exit_reason,
            'bars_held': t.bars_held,
            'num_contracts': t.num_contracts,
            'pnl': t.pnl,
            'capital': t.capital,
        })
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    df['weekday'] = df['date'].dt.day_name()
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    df['win'] = df['pnl'] > 0
    df['return_pct'] = df['pnl'] / (df['capital'] - df['pnl']) * 100
    return df


def section(title):
    print(f'\n{"="*80}')
    print(f'  {title}')
    print(f'{"="*80}')


def monthly_analysis(df, cap):
    """Monthly breakdown by direction."""
    section('MONTHLY PERFORMANCE BY DIRECTION')

    months = sorted(df['month'].unique())

    # Header
    print(f'\n  {"Month":<10} | {"TOTAL":>8} {"#":>4} {"WR":>5} | '
          f'{"CALL $":>9} {"#":>4} {"WR":>5} {"PT%":>4} {"SL%":>4} {"TM%":>4} | '
          f'{"PUT $":>9} {"#":>4} {"WR":>5} {"PT%":>4} {"SL%":>4} {"TM%":>4} | '
          f'{"DD%":>5} {"Status":>6}')
    print(f'  {"─"*120}')

    monthly_stats = []
    running_capital = cap

    for m in months:
        mdf = df[df['month'] == m]
        total_pnl = mdf['pnl'].sum()
        total_n = len(mdf)
        total_wr = mdf['win'].mean() * 100

        # Direction splits
        calls = mdf[mdf['direction'] == 'CALL']
        puts = mdf[mdf['direction'] == 'PUT']

        call_pnl = calls['pnl'].sum() if len(calls) > 0 else 0
        call_n = len(calls)
        call_wr = calls['win'].mean() * 100 if call_n > 0 else 0

        put_pnl = puts['pnl'].sum() if len(puts) > 0 else 0
        put_n = len(puts)
        put_wr = puts['win'].mean() * 100 if put_n > 0 else 0

        # Exit reasons by direction
        def exit_pcts(sub):
            n = len(sub)
            if n == 0:
                return 0, 0, 0
            pt = (sub['exit_reason'] == 'PROFIT').sum() / n * 100
            sl = (sub['exit_reason'] == 'STOP').sum() / n * 100
            tm = (sub['exit_reason'] == 'TIME').sum() / n * 100
            return pt, sl, tm

        c_pt, c_sl, c_tm = exit_pcts(calls)
        p_pt, p_sl, p_tm = exit_pcts(puts)

        # Drawdown within month
        peak = running_capital
        max_dd = 0
        for _, row in mdf.iterrows():
            c = row['capital']
            if c > peak:
                peak = c
            dd = (peak - c) / peak * 100
            if dd > max_dd:
                max_dd = dd
        running_capital = mdf['capital'].iloc[-1] if len(mdf) > 0 else running_capital

        status = 'GREEN' if total_pnl > 0 else 'RED'

        print(f'  {str(m):<10} | {total_pnl:>+8,.0f} {total_n:>4} {total_wr:>4.0f}% | '
              f'{call_pnl:>+9,.0f} {call_n:>4} {call_wr:>4.0f}% {c_pt:>3.0f}% {c_sl:>3.0f}% {c_tm:>3.0f}% | '
              f'{put_pnl:>+9,.0f} {put_n:>4} {put_wr:>4.0f}% {p_pt:>3.0f}% {p_sl:>3.0f}% {p_tm:>3.0f}% | '
              f'{max_dd:>4.1f}% {status:>6}')

        monthly_stats.append({
            'month': str(m), 'total_pnl': total_pnl, 'status': status,
            'call_pnl': call_pnl, 'call_n': call_n, 'call_wr': call_wr,
            'put_pnl': put_pnl, 'put_n': put_n, 'put_wr': put_wr,
            'c_pt': c_pt, 'c_sl': c_sl, 'c_tm': c_tm,
            'p_pt': p_pt, 'p_sl': p_sl, 'p_tm': p_tm,
            'max_dd': max_dd,
        })

    # Summary
    red_months = [s for s in monthly_stats if s['status'] == 'RED']
    green_months = [s for s in monthly_stats if s['status'] == 'GREEN']
    print(f'\n  Summary: {len(green_months)} green months, {len(red_months)} red months')

    if red_months:
        print(f'\n  RED MONTH DRILL-DOWN:')
        for rm in red_months:
            print(f'\n    {rm["month"]}:  total={rm["total_pnl"]:+,.0f}')
            print(f'      CALL: {rm["call_n"]} trades, WR={rm["call_wr"]:.0f}%, '
                  f'P&L={rm["call_pnl"]:+,.0f}  (PT={rm["c_pt"]:.0f}% SL={rm["c_sl"]:.0f}% TIME={rm["c_tm"]:.0f}%)')
            print(f'      PUT:  {rm["put_n"]} trades, WR={rm["put_wr"]:.0f}%, '
                  f'P&L={rm["put_pnl"]:+,.0f}  (PT={rm["p_pt"]:.0f}% SL={rm["p_sl"]:.0f}% TIME={rm["p_tm"]:.0f}%)')
            # Which direction caused the loss?
            if rm['call_pnl'] < 0 and rm['put_pnl'] < 0:
                print(f'      → Both CALL and PUT losing')
            elif rm['call_pnl'] < rm['put_pnl']:
                print(f'      → CALL is primary drag ({rm["call_pnl"]:+,.0f})')
            else:
                print(f'      → PUT is primary drag ({rm["put_pnl"]:+,.0f})')

    return monthly_stats


def daily_direction_analysis(df):
    """Daily P&L patterns by direction."""
    section('DAILY P&L PATTERNS')

    daily = df.groupby(['date', 'direction']).agg(
        pnl=('pnl', 'sum'),
        trades=('pnl', 'count'),
        wins=('win', 'sum'),
    ).reset_index()

    # Days where CALL lost vs PUT lost
    daily_call = daily[daily['direction'] == 'CALL'].set_index('date')
    daily_put = daily[daily['direction'] == 'PUT'].set_index('date')

    all_dates = sorted(df['date'].unique())
    both_lose = 0
    call_only_lose = 0
    put_only_lose = 0
    both_win = 0

    for d in all_dates:
        c_pnl = daily_call.loc[d, 'pnl'] if d in daily_call.index else 0
        p_pnl = daily_put.loc[d, 'pnl'] if d in daily_put.index else 0
        if isinstance(c_pnl, pd.Series):
            c_pnl = c_pnl.sum()
        if isinstance(p_pnl, pd.Series):
            p_pnl = p_pnl.sum()

        if c_pnl < 0 and p_pnl < 0:
            both_lose += 1
        elif c_pnl < 0 and p_pnl >= 0:
            call_only_lose += 1
        elif c_pnl >= 0 and p_pnl < 0:
            put_only_lose += 1
        else:
            both_win += 1

    total_days = len(all_dates)
    print(f'\n  Daily direction patterns ({total_days} days):')
    print(f'    Both directions win:   {both_win:>4} ({both_win/total_days*100:>5.1f}%)')
    print(f'    CALL loses, PUT wins:  {call_only_lose:>4} ({call_only_lose/total_days*100:>5.1f}%)')
    print(f'    PUT loses, CALL wins:  {put_only_lose:>4} ({put_only_lose/total_days*100:>5.1f}%)')
    print(f'    Both directions lose:  {both_lose:>4} ({both_lose/total_days*100:>5.1f}%)')

    # Weekday analysis
    print(f'\n  P&L by Weekday:')
    for wd in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        wdf = df[df['weekday'] == wd]
        if len(wdf) == 0:
            continue
        calls = wdf[wdf['direction'] == 'CALL']
        puts = wdf[wdf['direction'] == 'PUT']
        print(f'    {wd:<10}  total={wdf["pnl"].sum():>+9,.0f}  '
              f'CALL={calls["pnl"].sum():>+8,.0f} ({len(calls):>3} tr, WR={calls["win"].mean()*100 if len(calls)>0 else 0:>4.0f}%)  '
              f'PUT={puts["pnl"].sum():>+8,.0f} ({len(puts):>3} tr, WR={puts["win"].mean()*100 if len(puts)>0 else 0:>4.0f}%)')

    # Hour-of-day analysis
    print(f'\n  P&L by Entry Hour:')
    for h in sorted(df['hour'].unique()):
        hdf = df[df['hour'] == h]
        calls = hdf[hdf['direction'] == 'CALL']
        puts = hdf[hdf['direction'] == 'PUT']
        print(f'    {h:>2}:00  total={hdf["pnl"].sum():>+9,.0f}  '
              f'CALL={calls["pnl"].sum():>+8,.0f} ({len(calls):>3} tr, WR={calls["win"].mean()*100 if len(calls)>0 else 0:>4.0f}%)  '
              f'PUT={puts["pnl"].sum():>+8,.0f} ({len(puts):>3} tr, WR={puts["win"].mean()*100 if len(puts)>0 else 0:>4.0f}%)')


def consecutive_loss_analysis(df):
    """Deep analysis of consecutive losing streaks."""
    section('CONSECUTIVE LOSS STREAK ANALYSIS')

    # Identify streaks
    df = df.copy()
    df['is_loss'] = df['pnl'] <= 0

    streaks = []
    current_streak = []

    for _, row in df.iterrows():
        if row['is_loss']:
            current_streak.append(row)
        else:
            if len(current_streak) >= 2:
                streaks.append(current_streak[:])
            current_streak = []
    if len(current_streak) >= 2:
        streaks.append(current_streak[:])

    if not streaks:
        print('  No consecutive loss streaks of 2+ found.')
        return

    # Streak length distribution
    lengths = [len(s) for s in streaks]
    print(f'\n  Total streaks (2+ consecutive losses): {len(streaks)}')
    print(f'  Max streak length: {max(lengths)}')
    print(f'  Mean streak length: {np.mean(lengths):.1f}')

    print(f'\n  Streak Length Distribution:')
    for l in sorted(set(lengths)):
        count = lengths.count(l)
        total_loss = sum(sum(r['pnl'] for r in s) for s in streaks if len(s) == l)
        print(f'    {l} consecutive losses: {count} occurrences, '
              f'total damage: ${total_loss:+,.0f}')

    # Detailed streak analysis
    print(f'\n  {"─"*100}')
    print(f'  DETAILED STREAKS (3+ losses):')
    print(f'  {"─"*100}')

    big_streaks = [s for s in streaks if len(s) >= 3]
    for i, streak in enumerate(big_streaks):
        streak_pnl = sum(r['pnl'] for r in streak)
        print(f'\n  Streak #{i+1}: {len(streak)} losses, damage=${streak_pnl:+,.0f}')
        print(f'    {"Date":<12} {"Time":<9} {"Dir":<5} {"RSI":>6} '
              f'{"Exit":>7} {"Bars":>4} {"P&L":>10} {"Capital":>10}')
        for r in streak:
            print(f'    {str(r["date"])[:10]:<12} {r["time"]:<9} {r["direction"]:<5} '
                  f'{r["rsi"]:>6.1f} {r["exit_reason"]:>7} {r["bars_held"]:>4} '
                  f'{r["pnl"]:>+10,.0f} {r["capital"]:>10,.0f}')

    # Direction composition of streaks
    print(f'\n  Streak Direction Composition:')
    all_streak_trades = [r for s in streaks for r in s]
    streak_df = pd.DataFrame(all_streak_trades)
    call_in_streaks = len(streak_df[streak_df['direction'] == 'CALL'])
    put_in_streaks = len(streak_df[streak_df['direction'] == 'PUT'])
    total_in_streaks = len(streak_df)
    print(f'    CALL in streaks: {call_in_streaks} ({call_in_streaks/total_in_streaks*100:.0f}%)')
    print(f'    PUT in streaks: {put_in_streaks} ({put_in_streaks/total_in_streaks*100:.0f}%)')

    # Exit reasons within streaks
    print(f'\n  Exit Reasons INSIDE Streaks vs OUTSIDE:')
    non_streak_idx = set(range(len(df))) - set()
    streak_indices = set()
    for s in streaks:
        for r in s:
            idx = df[(df['date'] == r['date']) & (df['time'] == r['time'])].index
            streak_indices.update(idx.tolist())

    in_streak = df.loc[list(streak_indices)]
    out_streak = df.drop(list(streak_indices))

    for reason in ['PROFIT', 'STOP', 'TIME']:
        in_pct = (in_streak['exit_reason'] == reason).mean() * 100 if len(in_streak) > 0 else 0
        out_pct = (out_streak['exit_reason'] == reason).mean() * 100 if len(out_streak) > 0 else 0
        print(f'    {reason:<7}: in-streak={in_pct:>5.1f}%  outside={out_pct:>5.1f}%  '
              f'delta={in_pct - out_pct:>+5.1f}pp')

    # RSI at entry during streaks vs normal
    print(f'\n  RSI at Entry:')
    for d in ['CALL', 'PUT']:
        in_rsi = in_streak[in_streak['direction'] == d]['rsi'].mean() if len(in_streak[in_streak['direction'] == d]) > 0 else 0
        out_rsi = out_streak[out_streak['direction'] == d]['rsi'].mean() if len(out_streak[out_streak['direction'] == d]) > 0 else 0
        print(f'    {d}: in-streak={in_rsi:.1f}  outside={out_rsi:.1f}  delta={in_rsi - out_rsi:+.1f}')

    return streaks


def post_loss_recovery(df):
    """Analyze what happens after a loss or consecutive losses."""
    section('POST-LOSS RECOVERY PATTERNS')

    df = df.copy().reset_index(drop=True)

    # After 1 loss, 2 consecutive, 3 consecutive
    for streak_len in [1, 2, 3]:
        print(f'\n  After {streak_len} consecutive loss(es):')

        recovery_trades = []
        for i in range(streak_len, len(df)):
            all_losses = all(df.loc[i - streak_len + j, 'pnl'] <= 0 for j in range(streak_len))
            if all_losses:
                # Next trade
                recovery_trades.append(df.loc[i])

        if not recovery_trades:
            print(f'    No data')
            continue

        rdf = pd.DataFrame(recovery_trades)

        for d in ['ALL', 'CALL', 'PUT']:
            sub = rdf if d == 'ALL' else rdf[rdf['direction'] == d]
            if len(sub) == 0:
                continue
            wr = sub['win'].mean() * 100
            avg_pnl = sub['pnl'].mean()
            n = len(sub)
            pf_num = sub[sub['pnl'] > 0]['pnl'].sum()
            pf_den = abs(sub[sub['pnl'] <= 0]['pnl'].sum()) or 0.01
            pf = pf_num / pf_den
            print(f'    {d:<5}: {n:>4} trades, WR={wr:>5.1f}%, '
                  f'avg P&L={avg_pnl:>+7.0f}, PF={pf:.2f}')


def exit_reason_deep_dive(df):
    """Analyze exit reasons and their impact on P&L by direction."""
    section('EXIT REASON ANALYSIS BY DIRECTION')

    for d in ['CALL', 'PUT']:
        sub = df[df['direction'] == d]
        if len(sub) == 0:
            continue
        print(f'\n  {d} ({len(sub)} trades, total P&L={sub["pnl"].sum():+,.0f}):')
        for reason in ['PROFIT', 'STOP', 'TIME']:
            r = sub[sub['exit_reason'] == reason]
            if len(r) == 0:
                continue
            avg_pnl = r['pnl'].mean()
            total_pnl = r['pnl'].sum()
            wr = (r['pnl'] > 0).mean() * 100
            avg_bars = r['bars_held'].mean()
            avg_rsi = r['rsi'].mean()
            print(f'    {reason:<7}: {len(r):>4} trades ({len(r)/len(sub)*100:>4.0f}%), '
                  f'total={total_pnl:>+10,.0f}, avg={avg_pnl:>+7.0f}, '
                  f'WR={wr:>4.0f}%, bars={avg_bars:.1f}, RSI={avg_rsi:.1f}')

    # Losing TIME exits analysis
    print(f'\n  TIME EXIT LOSERS (held to max, lost money):')
    time_losers = df[(df['exit_reason'] == 'TIME') & (df['pnl'] <= 0)]
    if len(time_losers) > 0:
        for d in ['CALL', 'PUT']:
            tl = time_losers[time_losers['direction'] == d]
            if len(tl) == 0:
                continue
            avg_loss = tl['pnl'].mean()
            total_loss = tl['pnl'].sum()
            avg_rsi = tl['rsi'].mean()
            print(f'    {d}: {len(tl)} trades, avg loss={avg_loss:+,.0f}, '
                  f'total loss={total_loss:+,.0f}, avg RSI={avg_rsi:.1f}')

    # Losing STOP exits - are they concentrated?
    print(f'\n  STOP LOSS CLUSTERING:')
    stop_losses = df[df['exit_reason'] == 'STOP'].copy()
    if len(stop_losses) > 0:
        stop_losses['date_str'] = stop_losses['date'].dt.strftime('%Y-%m-%d')
        daily_stops = stop_losses.groupby('date_str').size()
        multi_stop_days = daily_stops[daily_stops >= 2]
        print(f'    Days with 2+ stop losses: {len(multi_stop_days)}')
        if len(multi_stop_days) > 0:
            total_multi_damage = 0
            for date, count in multi_stop_days.items():
                day_stops = stop_losses[stop_losses['date_str'] == date]
                damage = day_stops['pnl'].sum()
                total_multi_damage += damage
                dirs = day_stops['direction'].value_counts().to_dict()
                print(f'      {date}: {count} stops, damage=${damage:+,.0f}, '
                      f'directions={dirs}')
            print(f'    Total multi-stop-day damage: ${total_multi_damage:+,.0f}')


def recommendations(df, monthly_stats):
    """Generate actionable recommendations based on analysis."""
    section('ACTIONABLE RECOMMENDATIONS')

    # 1. Red month patterns
    red = [m for m in monthly_stats if m['status'] == 'RED']
    if red:
        print(f'\n  RED MONTH MITIGATION ({len(red)} red months):')
        for rm in red:
            if rm['call_pnl'] < 0 and rm['put_pnl'] < 0:
                print(f'    {rm["month"]}: Both directions down → consider tighter SL or reduced sizing')
            elif rm['put_pnl'] < rm['call_pnl']:
                print(f'    {rm["month"]}: PUT drag (${rm["put_pnl"]:+,.0f}) → '
                      f'PUT WR={rm["put_wr"]:.0f}%, STOP={rm["p_sl"]:.0f}%, TIME={rm["p_tm"]:.0f}%')
                if rm['p_sl'] > 40:
                    print(f'      → High PUT stop rate suggests tighter PUT SL or lower PUT RSI threshold')
                if rm['p_tm'] > 50:
                    print(f'      → High PUT time exit suggests shorter PUT hold or wider PUT PT')
            else:
                print(f'    {rm["month"]}: CALL drag (${rm["call_pnl"]:+,.0f}) → '
                      f'CALL WR={rm["call_wr"]:.0f}%, STOP={rm["c_sl"]:.0f}%, TIME={rm["c_tm"]:.0f}%')

    # 2. Direction imbalance
    calls = df[df['direction'] == 'CALL']
    puts = df[df['direction'] == 'PUT']
    call_per_trade = calls['pnl'].mean() if len(calls) > 0 else 0
    put_per_trade = puts['pnl'].mean() if len(puts) > 0 else 0
    print(f'\n  DIRECTION EFFICIENCY:')
    print(f'    CALL: avg P&L/trade = ${call_per_trade:+,.0f}  '
          f'({len(calls)} trades, WR={calls["win"].mean()*100:.1f}%)')
    print(f'    PUT:  avg P&L/trade = ${put_per_trade:+,.0f}  '
          f'({len(puts)} trades, WR={puts["win"].mean()*100:.1f}%)')

    if call_per_trade < 0 and put_per_trade > 0:
        print(f'    → CALLs are net negative. Consider tighter CALL filters or skip in certain regimes.')
    elif put_per_trade < 0 and call_per_trade > 0:
        print(f'    → PUTs are net negative. Consider tighter PUT filters.')

    # 3. Consecutive loss prevention
    print(f'\n  CONSECUTIVE LOSS PREVENTION:')
    df_copy = df.copy().reset_index(drop=True)
    df_copy['is_loss'] = df_copy['pnl'] <= 0
    max_streak = 0
    cur = 0
    for _, row in df_copy.iterrows():
        if row['is_loss']:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    print(f'    Max consecutive losses: {max_streak}')

    # Check if same-direction clusters dominate
    same_dir_streaks = 0
    mixed_dir_streaks = 0
    current_streak = []
    for _, row in df_copy.iterrows():
        if row['is_loss']:
            current_streak.append(row['direction'])
        else:
            if len(current_streak) >= 3:
                if len(set(current_streak)) == 1:
                    same_dir_streaks += 1
                else:
                    mixed_dir_streaks += 1
            current_streak = []
    print(f'    3+ loss streaks: {same_dir_streaks} same-direction, {mixed_dir_streaks} mixed-direction')
    if same_dir_streaks > mixed_dir_streaks:
        print(f'    → Same-direction streaks dominate → post-loss direction cooldown may help')
    else:
        print(f'    → Mixed-direction streaks → market-wide risk (regime detection should help)')


def main():
    print('='*80)
    print('  MONTHLY & DIRECTION PERFORMANCE ANALYSIS')
    print('  Current strategy.json config')
    print('='*80)

    # Run on both periods
    for label, start, end in [('2025 (IS)', '2025-01-01', '2025-12-31'),
                               ('2026 (OOS)', '2026-01-02', '2026-02-25')]:
        print(f'\n\n{"#"*80}')
        print(f'  PERIOD: {label}')
        print(f'{"#"*80}')

        trades, cap = load_and_run(start, end)
        if not trades:
            print(f'  No trades for {label}')
            continue

        df = trades_to_df(trades)
        total_pnl = df['pnl'].sum()
        total_ret = total_pnl / cap * 100
        print(f'\n  {len(trades)} trades, P&L=${total_pnl:+,.0f} ({total_ret:+.1f}%), '
              f'WR={df["win"].mean()*100:.1f}%')

        monthly_stats = monthly_analysis(df, cap)
        daily_direction_analysis(df)
        exit_reason_deep_dive(df)
        streaks = consecutive_loss_analysis(df)
        post_loss_recovery(df)
        recommendations(df, monthly_stats)

    print(f'\n\n  Analysis complete.')


if __name__ == '__main__':
    main()
