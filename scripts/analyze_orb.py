"""
ORB Strategy Analysis Script
"""
import pandas as pd
import numpy as np

df = pd.read_csv('output/backtest_trades.csv')

print('='*60)
print('ORB STRATEGY - DETAILED ANALYSIS (2025)')
print('='*60)

total = len(df)
wins = df[df['pnl'] > 0]
losses = df[df['pnl'] <= 0]

print(f"\nTotal Trades: {total}")
print(f"Win Rate: {len(wins)/total*100:.1f}%")
print(f"Total P&L: ${df['pnl'].sum():,.0f}")

# By direction
print('\n' + '-'*40)
print('BY DIRECTION')
print('-'*40)
for d in ['CALL', 'PUT']:
    s = df[df['direction'] == d]
    if len(s) > 0:
        wr = (s['pnl'] > 0).mean()*100
        print(f"{d}: {len(s)} trades ({len(s)/total*100:.1f}%), {wr:.1f}% WR, ${s['pnl'].sum():,.0f}")

# By exit reason
print('\n' + '-'*40)
print('BY EXIT REASON')
print('-'*40)
for reason in df['exit_reason'].unique():
    s = df[df['exit_reason'] == reason]
    wr = (s['pnl'] > 0).mean()*100
    avg = s['pnl'].mean()
    print(f"{reason:8s}: {len(s):4d} ({len(s)/total*100:5.1f}%), {wr:5.1f}% WR, Avg ${avg:+,.0f}")

# By hour
print('\n' + '-'*40)
print('BY HOUR')
print('-'*40)
df['hour'] = pd.to_datetime(df['time']).dt.hour
for h in sorted(df['hour'].unique()):
    s = df[df['hour'] == h]
    wr = (s['pnl'] > 0).mean()*100
    print(f"{h}:00: {len(s):4d} trades, {wr:5.1f}% WR, ${s['pnl'].sum():>12,.0f}")

# Bars held distribution
print('\n' + '-'*40)
print('BARS HELD DISTRIBUTION')
print('-'*40)
for bars in sorted(df['bars_held'].unique()):
    s = df[df['bars_held'] == bars]
    wr = (s['pnl'] > 0).mean()*100
    print(f"{bars} bars: {len(s):4d} trades ({len(s)/total*100:5.1f}%), {wr:5.1f}% WR")

# Trade stats
print('\n' + '-'*40)
print('TRADE STATISTICS')
print('-'*40)
print(f"Avg Winner:   ${wins['pnl'].mean():,.0f}")
print(f"Avg Loser:    ${losses['pnl'].mean():,.0f}")
print(f"Largest Win:  ${wins['pnl'].max():,.0f}")
print(f"Largest Loss: ${losses['pnl'].min():,.0f}")
print(f"Avg Bars Held: {df['bars_held'].mean():.2f}")
print(f"Win/Loss Ratio: {wins['pnl'].mean()/abs(losses['pnl'].mean()):.2f}")

# Monthly breakdown
print('\n' + '-'*40)
print('MONTHLY BREAKDOWN')
print('-'*40)
df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
monthly = df.groupby('month').agg({
    'pnl': ['sum', 'count'],
    'direction': lambda x: (x == 'CALL').sum()
})
monthly.columns = ['pnl', 'trades', 'calls']
monthly['puts'] = monthly['trades'] - monthly['calls']
monthly['wr'] = df.groupby('month').apply(lambda x: (x['pnl'] > 0).mean() * 100)

for m, row in monthly.iterrows():
    print(f"{m}: {row['trades']:3.0f} trades ({row['calls']:.0f}C/{row['puts']:.0f}P), "
          f"{row['wr']:.0f}% WR, ${row['pnl']:>10,.0f}")

# Best/worst days
print('\n' + '-'*40)
print('TOP 5 BEST DAYS')
print('-'*40)
daily = df.groupby('date')['pnl'].sum().sort_values(ascending=False)
for d, pnl in daily.head(5).items():
    day_df = df[df['date'] == d]
    print(f"{d}: ${pnl:>10,.0f} ({len(day_df)} trades)")

print('\n' + '-'*40)
print('TOP 5 WORST DAYS')
print('-'*40)
for d, pnl in daily.tail(5).items():
    day_df = df[df['date'] == d]
    print(f"{d}: ${pnl:>10,.0f} ({len(day_df)} trades)")

# Capital curve stats
print('\n' + '-'*40)
print('CAPITAL CURVE')
print('-'*40)
capitals = [10000] + df['capital'].tolist()
peak = 10000
max_dd = 0
max_dd_date = df['date'].iloc[0]
for i, c in enumerate(capitals[1:], 1):
    if c > peak:
        peak = c
    dd = (peak - c) / peak * 100
    if dd > max_dd:
        max_dd = dd
        max_dd_date = df['date'].iloc[i-1]

print(f"Starting Capital: $10,000")
print(f"Final Capital: ${capitals[-1]:,.0f}")
print(f"Return: {(capitals[-1]/10000 - 1)*100:.0f}%")
print(f"Max Drawdown: {max_dd:.1f}% (on {max_dd_date})")
print(f"Peak Capital: ${max(capitals):,.0f}")
