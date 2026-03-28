"""Analyze backtest trades: separate CALL vs PUT PnL + loss limit metrics."""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

df = pd.read_csv('backtest_trades.csv')
INITIAL_CAPITAL = 10000

print("=" * 70)
print("CALL vs PUT P&L BREAKDOWN")
print("=" * 70)

for direction in ['CALL', 'PUT']:
    d = df[df['direction'] == direction]
    if len(d) == 0:
        print(f"\n  {direction}: No trades")
        continue
    wins = d[d['pnl'] > 0]
    losses = d[d['pnl'] <= 0]
    gross_win = wins['pnl'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0

    dollar = "$"
    print(f"\n  {direction}:")
    print(f"    Trades:        {len(d)}")
    print(f"    Wins:          {len(wins)} ({len(wins)/len(d)*100:.1f}%)")
    print(f"    Losses:        {len(losses)} ({len(losses)/len(d)*100:.1f}%)")
    print(f"    Total P&L:     {dollar}{d['pnl'].sum():+,.2f}")
    print(f"    Gross Profit:  {dollar}{gross_win:,.2f}")
    print(f"    Gross Loss:    -{dollar}{gross_loss:,.2f}")
    print(f"    Profit Factor: {pf:.2f}")
    print(f"    Avg Winner:    {dollar}{avg_win:+,.2f}")
    print(f"    Avg Loser:     {dollar}{avg_loss:+,.2f}")
    print(f"    Largest Win:   {dollar}{d['pnl'].max():+,.2f}")
    print(f"    Largest Loss:  {dollar}{d['pnl'].min():+,.2f}")

    # Exit reason breakdown
    print(f"    Exit Reasons:")
    for reason in d['exit_reason'].value_counts().index:
        s = d[d['exit_reason'] == reason]
        print(f"      {reason}: {len(s)} trades, {dollar}{s['pnl'].sum():+,.2f}")

# ====================================================================
# LOSS LIMIT / RISK METRICS
# ====================================================================
print()
print("=" * 70)
print("LOSS LIMIT / RISK METRICS")
print("=" * 70)

dollar = "$"
losses = df[df['pnl'] < 0].copy()
print(f"\n  Total Losing Trades: {len(losses)} / {len(df)} ({len(losses)/len(df)*100:.1f}%)")
print(f"  Total Loss Amount:  {dollar}{losses['pnl'].sum():,.2f}")
print(f"  Avg Loss Size:      {dollar}{losses['pnl'].mean():,.2f}")
print(f"  Median Loss:        {dollar}{losses['pnl'].median():,.2f}")
print(f"  Worst Loss:         {dollar}{losses['pnl'].min():,.2f}")

# Loss as % of capital at time
losses['pre_capital'] = losses['capital'] - losses['pnl']
losses['loss_pct'] = losses['pnl'] / losses['pre_capital'] * 100
print(f"  Avg Loss % Capital: {losses['loss_pct'].mean():.2f}%")
print(f"  Worst Loss % Cap:   {losses['loss_pct'].min():.2f}%")

# Consecutive losses
streak = 0
max_streak = 0
current_streak_pnl = 0
max_streak_pnl = 0
for _, row in df.iterrows():
    if row['pnl'] < 0:
        streak += 1
        current_streak_pnl += row['pnl']
    else:
        if streak > max_streak:
            max_streak = streak
            max_streak_pnl = current_streak_pnl
        streak = 0
        current_streak_pnl = 0
if streak > max_streak:
    max_streak = streak
    max_streak_pnl = current_streak_pnl

print(f"\n  Max Consecutive Losses: {max_streak}")
print(f"  Max Consec Loss P&L:   {dollar}{max_streak_pnl:,.2f}")

# Drawdown analysis
capitals = [INITIAL_CAPITAL] + df['capital'].tolist()
peak = capitals[0]
max_dd_pct = 0
max_dd_dollars = 0
drawdowns = []
current_dd_start = None
for i, c in enumerate(capitals):
    if c >= peak:
        peak = c
        if current_dd_start is not None:
            drawdowns.append((current_dd_start, i - 1, dd_pct_val, dd_dlr_val))
            current_dd_start = None
    dd_pct_val = (peak - c) / peak * 100
    dd_dlr_val = peak - c
    if dd_pct_val > max_dd_pct:
        max_dd_pct = dd_pct_val
        max_dd_dollars = dd_dlr_val
    if dd_pct_val > 0 and current_dd_start is None:
        current_dd_start = i
if current_dd_start is not None:
    drawdowns.append((current_dd_start, len(capitals) - 1, dd_pct_val, dd_dlr_val))

print(f"\n  Max Drawdown:       {max_dd_pct:.1f}% ({dollar}{max_dd_dollars:,.0f})")
print(f"  Drawdown Episodes:  {len(drawdowns)}")
dd_sorted = sorted(drawdowns, key=lambda x: x[2], reverse=True)[:5]
for j, (s, e, pct, dlr) in enumerate(dd_sorted):
    print(f"    #{j+1}: {pct:.1f}% ({dollar}{dlr:,.0f})")

# Daily P&L
df['date_only'] = pd.to_datetime(df['date']).dt.date
daily = df.groupby('date_only')['pnl'].sum()
losing_days = daily[daily < 0]
print(f"\n  Trading Days:      {len(daily)}")
print(f"  Losing Days:       {len(losing_days)} ({len(losing_days)/len(daily)*100:.1f}%)")
print(f"  Worst Day P&L:     {dollar}{daily.min():,.2f}")
print(f"  Best Day P&L:      {dollar}{daily.max():,.2f}")
print(f"  Avg Day P&L:       {dollar}{daily.mean():,.2f}")

# ====================================================================
# STOP-AFTER-FIRST-LOSS ANALYSIS
# ====================================================================
print()
print("=" * 70)
print("STOP-AFTER-FIRST-LOSS (SFL) ANALYSIS")
print("=" * 70)

daily_stats = df.groupby('date_only').agg(
    trades=('pnl', 'count'),
    wins=('pnl', lambda x: (x > 0).sum()),
    losses_count=('pnl', lambda x: (x < 0).sum()),
    day_pnl=('pnl', 'sum'),
    first_trade_pnl=('pnl', 'first')
).reset_index()

sfl_days = daily_stats[daily_stats['first_trade_pnl'] < 0]
print(f"  Days where 1st trade was a loss: {len(sfl_days)}")
if len(sfl_days) > 0:
    print(f"    Avg trades on those days:   {sfl_days['trades'].mean():.1f}")
    print(f"    Total P&L on 1st-loss days: {dollar}{sfl_days['day_pnl'].sum():,.2f}")

non_sfl_days = daily_stats[daily_stats['first_trade_pnl'] >= 0]
print(f"  Days where 1st trade was a win:  {len(non_sfl_days)}")
if len(non_sfl_days) > 0:
    print(f"    Avg trades on those days:   {non_sfl_days['trades'].mean():.1f}")
    print(f"    Total P&L on 1st-win days:  {dollar}{non_sfl_days['day_pnl'].sum():,.2f}")

# How many trades would be saved if SFL was disabled?
multi_loss_days = daily_stats[daily_stats['losses_count'] > 1]
print(f"\n  Days with >1 loss: {len(multi_loss_days)}")
if len(multi_loss_days) > 0:
    print(f"    Total losses on those days: {multi_loss_days['losses_count'].sum()}")
    print(f"    Total P&L those days:       {dollar}{multi_loss_days['day_pnl'].sum():,.2f}")

# ====================================================================
# LOSS SIZE DISTRIBUTION
# ====================================================================
print()
print("=" * 70)
print("LOSS SIZE DISTRIBUTION")
print("=" * 70)

losses_abs = losses['pnl'].abs()
bins = [0, 50, 100, 200, 500, 1000, float('inf')]
labels = ['$0-50', '$50-100', '$100-200', '$200-500', '$500-1K', '>$1K']
for i in range(len(bins) - 1):
    bucket = losses_abs[(losses_abs >= bins[i]) & (losses_abs < bins[i + 1])]
    if len(bucket) > 0:
        print(f"  {labels[i]:>10}: {len(bucket)} trades, total {dollar}{bucket.sum():,.2f}")

# ====================================================================
# CALL vs PUT LOSS COMPARISON
# ====================================================================
print()
print("=" * 70)
print("CALL vs PUT LOSS COMPARISON")
print("=" * 70)

for direction in ['CALL', 'PUT']:
    d_losses = df[(df['direction'] == direction) & (df['pnl'] < 0)]
    if len(d_losses) == 0:
        print(f"\n  {direction}: No losses")
        continue
    d_losses = d_losses.copy()
    d_losses['pre_capital'] = d_losses['capital'] - d_losses['pnl']
    d_losses['loss_pct_cap'] = d_losses['pnl'] / d_losses['pre_capital'] * 100
    print(f"\n  {direction} Losses:")
    print(f"    Count:           {len(d_losses)}")
    print(f"    Total Loss:      {dollar}{d_losses['pnl'].sum():,.2f}")
    print(f"    Avg Loss:        {dollar}{d_losses['pnl'].mean():,.2f}")
    print(f"    Worst Loss:      {dollar}{d_losses['pnl'].min():,.2f}")
    print(f"    Avg Loss % Cap:  {d_losses['loss_pct_cap'].mean():.2f}%")
    print(f"    Exit Reasons:")
    for reason in d_losses['exit_reason'].value_counts().index:
        s = d_losses[d_losses['exit_reason'] == reason]
        print(f"      {reason}: {len(s)}, {dollar}{s['pnl'].sum():,.2f}")

# ====================================================================
# DAILY LOSS LIMIT SENSITIVITY
# ====================================================================
print()
print("=" * 70)
print("DAILY LOSS LIMIT SENSITIVITY (current: 0.8% of capital)")
print("=" * 70)

# For each daily loss limit, count how many additional trades would be blocked
for dll_pct in [0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050]:
    # Simulate: on each day, once cumulative loss exceeds dll_pct of start-of-day capital, block further trades
    blocked = 0
    blocked_pnl = 0
    allowed_pnl = 0
    for date, group in df.groupby('date_only'):
        day_capital = group.iloc[0]['capital'] - group.iloc[0]['pnl']  # approx start-of-day capital
        limit = day_capital * dll_pct
        cum_loss = 0
        hit_limit = False
        for _, row in group.iterrows():
            if hit_limit:
                blocked += 1
                blocked_pnl += row['pnl']
            else:
                allowed_pnl += row['pnl']
                if row['pnl'] < 0:
                    cum_loss += abs(row['pnl'])
                    if cum_loss >= limit:
                        hit_limit = True
    total_pnl = allowed_pnl + blocked_pnl
    print(f"  DLL={dll_pct*100:.1f}%: blocked {blocked} trades, "
          f"blocked P&L={dollar}{blocked_pnl:+,.0f}, "
          f"allowed P&L={dollar}{allowed_pnl:+,.0f}, "
          f"net={dollar}{total_pnl:+,.0f}")

# ====================================================================
# STOP LOSS PCT SENSITIVITY
# ====================================================================
print()
print("=" * 70)
print("STOP LOSS % SENSITIVITY (current: 35%)")
print("=" * 70)

# Can only analyze from trade data exit reasons
stops = df[df['exit_reason'] == 'STOP']
if len(stops) > 0:
    print(f"\n  Trades stopped out: {len(stops)} ({len(stops)/len(df)*100:.1f}%)")
    print(f"  Total stop loss P&L: {dollar}{stops['pnl'].sum():,.2f}")
    print(f"  Avg stop loss:       {dollar}{stops['pnl'].mean():,.2f}")
    
    # Loss per contract for stops
    stops = stops.copy()
    stops['pnl_per_contract'] = stops['pnl'] / stops['num_contracts']
    stops['loss_pct'] = (stops['exit'] - stops['entry']) / stops['entry'] * 100
    print(f"  Avg loss %:          {stops['loss_pct'].mean():.1f}%")
    print(f"  Avg P&L/contract:    {dollar}{stops['pnl_per_contract'].mean():,.2f}")

time_exits = df[df['exit_reason'] == 'TIME']
if len(time_exits) > 0:
    print(f"\n  Time exits: {len(time_exits)} ({len(time_exits)/len(df)*100:.1f}%)")
    print(f"  Time exit P&L: {dollar}{time_exits['pnl'].sum():,.2f}")
    print(f"  Win rate on time exits: {(time_exits['pnl'] > 0).mean()*100:.1f}%")

profit_exits = df[df['exit_reason'] == 'PROFIT']
if len(profit_exits) > 0:
    print(f"\n  Profit target hits: {len(profit_exits)} ({len(profit_exits)/len(df)*100:.1f}%)")
    print(f"  Profit exit P&L: {dollar}{profit_exits['pnl'].sum():,.2f}")
