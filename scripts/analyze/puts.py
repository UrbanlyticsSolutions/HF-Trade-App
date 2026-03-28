"""Deep analysis of PUT option trades from backtest."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

# Accept command line arg for year
year = sys.argv[1] if len(sys.argv) > 1 else '2025'
if year == '2026':
    csv_path = 'output/backtest_2026_oos.csv'
else:
    csv_path = 'output/backtest_2025_full.csv'

df = pd.read_csv(csv_path)
puts = df[df['direction'] == 'PUT'].copy()
calls = df[df['direction'] == 'CALL'].copy()
D = "$"

n = len(puts)
wins = (puts['pnl'] > 0).sum()
losses = (puts['pnl'] <= 0).sum()
total_pnl = puts['pnl'].sum()
wr = wins / n * 100
gp = puts[puts['pnl'] > 0]['pnl'].sum()
gl = abs(puts[puts['pnl'] <= 0]['pnl'].sum()) or 0.01
pf = gp / gl

print('=' * 70)
print(f'  {year} BACKTEST — PUT OPTION DEEP ANALYSIS  ({csv_path})')
print('=' * 70)

print(f'\n  OVERALL PUT STATISTICS:')
print(f'    Trades:          {n} (vs {len(calls)} CALL)')
print(f'    Win Rate:        {wr:.1f}% ({wins}W / {losses}L)')
print(f'    Total PnL:       {D}{total_pnl:+,.0f}')
print(f'    Gross Profit:    {D}{gp:,.0f}')
print(f'    Gross Loss:      -{D}{gl:,.0f}')
print(f'    Profit Factor:   {pf:.2f}')
pw = puts[puts['pnl'] > 0]
pl = puts[puts['pnl'] <= 0]
print(f'    Avg Winner:      {D}{pw["pnl"].mean():+,.0f}')
print(f'    Avg Loser:       {D}{pl["pnl"].mean():+,.0f}')
print(f'    Largest Win:     {D}{puts["pnl"].max():+,.0f}')
print(f'    Largest Loss:    {D}{puts["pnl"].min():+,.0f}')
print(f'    Avg PnL/Trade:   {D}{total_pnl/n:+,.0f}')
print(f'    Avg Bars Held:   {puts["bars_held"].mean():.1f}')

# ---- EXIT REASON BREAKDOWN ----
print(f'\n  PUT EXIT REASONS:')
for reason in ['PROFIT', 'STOP', 'TIME']:
    s = puts[puts['exit_reason'] == reason]
    if len(s) == 0:
        continue
    sw = (s['pnl'] > 0).sum()
    print(f'    {reason:8s}: {len(s):>4} trades ({len(s)/n*100:5.1f}%), '
          f'WR={sw/len(s)*100:.1f}%, PnL={D}{s["pnl"].sum():+,.0f}, Avg={D}{s["pnl"].mean():+,.0f}')

# ---- RSI DISTRIBUTION ----
print(f'\n  PUT RSI AT ENTRY:')
for lo, hi, label in [(0, 10, '0-10'), (10, 15, '10-15'), (15, 20, '15-20'),
                       (20, 25, '20-25'), (25, 30, '25-30')]:
    mask = (puts['rsi'] >= lo) & (puts['rsi'] < hi)
    s = puts[mask]
    if len(s) == 0:
        continue
    sw = (s['pnl'] > 0).sum()
    print(f'    RSI {label:>5}: {len(s):>4} trades, WR={sw/max(len(s),1)*100:.1f}%, '
          f'PnL={D}{s["pnl"].sum():+,.0f}, Avg={D}{s["pnl"].mean():+,.0f}')

# ---- MONTHLY BREAKDOWN ----
puts['month'] = puts['date'].str[:7]
print(f'\n  PUT MONTHLY BREAKDOWN:')
print(f'    {"Month":<8} {"Trades":>6} {"WR":>6} {"PnL":>10} {"AvgPnL":>8}')
print(f'    {"-" * 42}')
for m in sorted(puts['month'].unique()):
    s = puts[puts['month'] == m]
    sw = (s['pnl'] > 0).sum()
    mwr = sw / len(s) * 100
    mpnl = s['pnl'].sum()
    print(f'    {m:<8} {len(s):>6} {mwr:>5.1f}% {D}{mpnl:>+9,.0f} {D}{mpnl/len(s):>+7,.0f}')

# ---- TIME OF DAY ----
puts['hour'] = pd.to_datetime(puts['time']).dt.hour
print(f'\n  PUT BY TIME OF DAY:')
for h in sorted(puts['hour'].unique()):
    s = puts[puts['hour'] == h]
    sw = (s['pnl'] > 0).sum()
    print(f'    {h:02d}:xx  {len(s):>4} trades, WR={sw/max(len(s),1)*100:.1f}%, '
          f'PnL={D}{s["pnl"].sum():+,.0f}, Avg={D}{s["pnl"].mean():+,.0f}')

# ---- WINNING vs LOSING PUT CHARACTERISTICS ----
print(f'\n  WINNING PUT CHARACTERISTICS:')
print(f'    Count:         {len(pw)}')
print(f'    Avg RSI:       {pw["rsi"].mean():.1f}')
print(f'    Avg Bars Held: {pw["bars_held"].mean():.1f}')
print(f'    Avg Contracts: {pw["contracts"].mean():.1f}')
print(f'\n  LOSING PUT CHARACTERISTICS:')
print(f'    Count:         {len(pl)}')
print(f'    Avg RSI:       {pl["rsi"].mean():.1f}')
print(f'    Avg Bars Held: {pl["bars_held"].mean():.1f}')
print(f'    Avg Contracts: {pl["contracts"].mean():.1f}')

# ---- PUT vs CALL COMPARISON TABLE ----
cw = (calls['pnl'] > 0).sum()
cn = len(calls)
cgp = calls[calls['pnl'] > 0]['pnl'].sum()
cgl = abs(calls[calls['pnl'] <= 0]['pnl'].sum()) or 0.01
cpf = cgp / cgl
call_pnl = calls['pnl'].sum()

print(f'\n  PUT vs CALL COMPARISON:')
print(f'    {"":18s} {"PUT":>12} {"CALL":>12}')
print(f'    {"-" * 44}')
print(f'    {"Trades":18s} {n:>12} {cn:>12}')
print(f'    {"Win Rate":18s} {wr:>11.1f}% {cw/cn*100:>11.1f}%')
print(f'    {"Total PnL":18s} {D}{total_pnl:>+11,.0f} {D}{call_pnl:>+11,.0f}')
print(f'    {"Avg PnL/Trade":18s} {D}{total_pnl/n:>+11,.0f} {D}{call_pnl/cn:>+11,.0f}')
print(f'    {"Profit Factor":18s} {pf:>12.2f} {cpf:>12.2f}')
print(f'    {"Avg Bars Held":18s} {puts["bars_held"].mean():>12.1f} {calls["bars_held"].mean():>12.1f}')
print(f'    {"Largest Win":18s} {D}{puts["pnl"].max():>+11,.0f} {D}{calls["pnl"].max():>+11,.0f}')
print(f'    {"Largest Loss":18s} {D}{puts["pnl"].min():>+11,.0f} {D}{calls["pnl"].min():>+11,.0f}')

# ---- CONSECUTIVE LOSS STREAKS (PUT ONLY) ----
streak = 0
max_streak = 0
streak_pnl = 0
max_streak_pnl = 0
for _, row in puts.iterrows():
    if row['pnl'] < 0:
        streak += 1
        streak_pnl += row['pnl']
    else:
        if streak > max_streak:
            max_streak = streak
            max_streak_pnl = streak_pnl
        streak = 0
        streak_pnl = 0
if streak > max_streak:
    max_streak = streak
    max_streak_pnl = streak_pnl

print(f'\n  PUT CONSECUTIVE LOSS STREAKS:')
print(f'    Max Streak:     {max_streak}')
print(f'    Streak PnL:     {D}{max_streak_pnl:,.0f}')

# ---- DRAWDOWN (PUT ONLY EQUITY CURVE) ----
cum = 0
peak = 0
max_dd = 0
max_dd_date = ''
for _, row in puts.iterrows():
    cum += row['pnl']
    if cum > peak:
        peak = cum
    dd = (peak - cum) / max(peak, 1)
    if dd > max_dd:
        max_dd = dd
        max_dd_date = row['date']

print(f'\n  PUT EQUITY CURVE DRAWDOWN:')
print(f'    Max DD:         {max_dd * 100:.1f}% (at {max_dd_date})')
print(f'    Final Cum PnL:  {D}{cum:+,.0f}')

# ---- LOSS ANALYSIS: STOP vs TIME LOSSES ----
stop_l = puts[(puts['exit_reason'] == 'STOP') & (puts['pnl'] < 0)]
time_l = puts[(puts['exit_reason'] == 'TIME') & (puts['pnl'] < 0)]
print(f'\n  PUT LOSS BREAKDOWN:')
if len(stop_l) > 0:
    print(f'    STOP Losses:  {len(stop_l)} trades, Total={D}{stop_l["pnl"].sum():,.0f}, Avg={D}{stop_l["pnl"].mean():,.0f}')
if len(time_l) > 0:
    print(f'    TIME Losses:  {len(time_l)} trades, Total={D}{time_l["pnl"].sum():,.0f}, Avg={D}{time_l["pnl"].mean():,.0f}')

# ---- PUT DAILY PNL ----
puts['date_only'] = pd.to_datetime(puts['date']).dt.date
daily = puts.groupby('date_only')['pnl'].agg(['sum', 'count'])
green = (daily['sum'] > 0).sum()
red = (daily['sum'] <= 0).sum()
print(f'\n  PUT DAILY STATS:')
print(f'    Trading Days:    {len(daily)}')
print(f'    Green Days:      {green} ({green/len(daily)*100:.1f}%)')
print(f'    Red Days:        {red} ({red/len(daily)*100:.1f}%)')
print(f'    Avg Daily PnL:   {D}{daily["sum"].mean():+,.0f}')
print(f'    Best Day:        {D}{daily["sum"].max():+,.0f}')
print(f'    Worst Day:       {D}{daily["sum"].min():+,.0f}')
print(f'    Avg Trades/Day:  {daily["count"].mean():.1f}')

# ---- PUT FILTER IMPACT ----
print(f'\n  PUT FILTER STATS (from strategy.json):')
print(f'    put_min_rsi=25, skip_days=[Mon], min_entry_min=610')
print(f'    put_filter_require_uptrend=True')
print(f'    put_adaptive_filter=True (streak>=2, cooldown=3)')


# ---- QUARTERLY BREAKDOWN ----
puts['quarter'] = pd.to_datetime(puts['date']).dt.quarter
print(f'\n  PUT BY QUARTER:')
for q in sorted(puts['quarter'].unique()):
    s = puts[puts['quarter'] == q]
    sw = (s['pnl'] > 0).sum()
    print(f'    Q{q}: {len(s):>4} trades, WR={sw/len(s)*100:.1f}%, PnL={D}{s["pnl"].sum():+,.0f}')

print()
