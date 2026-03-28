"""Generate backtest chart insights from output/backtest_trades.csv"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

df = pd.read_csv('output/backtest_trades.csv')
df['date'] = pd.to_datetime(df['date'])
df['win'] = df['pnl'] > 0
initial_capital = 10000

fig = plt.figure(figsize=(20, 24))
gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

# === 1. EQUITY CURVE (top, full width) ===
ax1 = fig.add_subplot(gs[0, :])
capitals = [initial_capital] + df['capital'].tolist()
trade_nums = list(range(len(capitals)))
ax1.plot(trade_nums, capitals, 'b-', linewidth=2)
ax1.fill_between(trade_nums, initial_capital, capitals,
                 where=[c >= initial_capital for c in capitals],
                 color='green', alpha=0.2, interpolate=True)
ax1.fill_between(trade_nums, initial_capital, capitals,
                 where=[c < initial_capital for c in capitals],
                 color='red', alpha=0.2, interpolate=True)
ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
peak = initial_capital
max_dd = 0
max_dd_idx = 0
for i, c in enumerate(capitals):
    peak = max(peak, c)
    dd = (peak - c) / peak
    if dd > max_dd:
        max_dd = dd
        max_dd_idx = i
ax1.scatter([max_dd_idx], [capitals[max_dd_idx]], color='red', s=120, zorder=5, marker='o')
ax1.annotate(f'Max DD: {max_dd*100:.1f}%', xy=(max_dd_idx, capitals[max_dd_idx]),
             xytext=(max_dd_idx+5, capitals[max_dd_idx]*0.95), fontsize=11, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red'))
final_ret = (capitals[-1] / initial_capital - 1) * 100
total_pnl = df['pnl'].sum()
wr = df['win'].mean() * 100
gross_profit = df[df['win']]['pnl'].sum()
gross_loss = abs(df[~df['win']]['pnl'].sum())
pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
ax1.set_title(f'Equity Curve | ${initial_capital:,} -> ${capitals[-1]:,.0f} (+{final_ret:.0f}%) | '
              f'{len(df)} trades, {wr:.0f}% WR | PF {pf:.2f}',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Trade #')
ax1.set_ylabel('Capital ($)')
ax1.grid(True, alpha=0.3)

# === 2. DAILY P&L BAR CHART ===
ax2 = fig.add_subplot(gs[1, 0])
daily_pnl = df.groupby('date')['pnl'].sum()
colors = ['green' if p > 0 else 'red' for p in daily_pnl.values]
ax2.bar(range(len(daily_pnl)), daily_pnl.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.3)
ax2.axhline(y=0, color='black', linewidth=0.8)
win_days = (daily_pnl > 0).sum()
loss_days = (daily_pnl <= 0).sum()
ax2.set_title(f'Daily P&L | {win_days} green days, {loss_days} red days', fontsize=12, fontweight='bold')
ax2.set_xlabel('Trading Day')
ax2.set_ylabel('P&L ($)')
ax2.grid(True, alpha=0.3, axis='y')
dates_short = [d.strftime('%m/%d') for d in daily_pnl.index]
ax2.set_xticks(range(0, len(dates_short), 2))
ax2.set_xticklabels(dates_short[::2], rotation=45, fontsize=7)

# === 3. CUMULATIVE P&L BY DIRECTION ===
ax3 = fig.add_subplot(gs[1, 1])
for dir_name, color in [('CALL', 'royalblue'), ('PUT', 'darkorange')]:
    mask = df['direction'] == dir_name
    sub = df[mask].copy()
    sub['cum_pnl'] = sub['pnl'].cumsum()
    d_wr = sub['win'].mean() * 100
    d_total = sub['pnl'].sum()
    ax3.plot(range(len(sub)), sub['cum_pnl'].values, color=color, linewidth=2,
             label=f'{dir_name}: {len(sub)} trades, {d_wr:.0f}% WR, ${d_total:,.0f}')
ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax3.set_title('Cumulative P&L by Direction', fontsize=12, fontweight='bold')
ax3.set_xlabel('Trade #')
ax3.set_ylabel('Cum P&L ($)')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# === 4. WIN RATE BY EXIT REASON ===
ax4 = fig.add_subplot(gs[2, 0])
exit_stats = df.groupby('exit_reason').agg(
    count=('pnl', 'size'),
    win_rate=('win', 'mean'),
    avg_pnl=('pnl', 'mean'),
    total_pnl=('pnl', 'sum')
).sort_values('count', ascending=True)
ax4.barh(exit_stats.index, exit_stats['win_rate'] * 100,
         color=['green' if wr > 50 else 'red' for wr in exit_stats['win_rate']], alpha=0.8)
for i, (idx, row) in enumerate(exit_stats.iterrows()):
    ax4.text(row['win_rate'] * 100 + 1, i,
             f"n={int(row['count'])}, avg ${row['avg_pnl']:+,.0f}",
             va='center', fontsize=10)
ax4.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
ax4.set_title('Win Rate by Exit Reason', fontsize=12, fontweight='bold')
ax4.set_xlabel('Win Rate (%)')
ax4.set_xlim(0, 110)
ax4.grid(True, alpha=0.3, axis='x')

# === 5. P&L DISTRIBUTION HISTOGRAM ===
ax5 = fig.add_subplot(gs[2, 1])
wins_pnl = df[df['win']]['pnl']
losses_pnl = df[~df['win']]['pnl']
bins = np.linspace(df['pnl'].min(), df['pnl'].max(), 30)
ax5.hist(wins_pnl, bins=bins, color='green', alpha=0.7,
         label=f'Winners (avg ${wins_pnl.mean():,.0f})', edgecolor='black', linewidth=0.3)
ax5.hist(losses_pnl, bins=bins, color='red', alpha=0.7,
         label=f'Losers (avg ${losses_pnl.mean():,.0f})', edgecolor='black', linewidth=0.3)
ax5.axvline(x=0, color='black', linewidth=1)
ax5.axvline(x=df['pnl'].mean(), color='blue', linestyle='--', linewidth=1.5,
            label=f'Avg ${df["pnl"].mean():,.0f}')
ax5.set_title('P&L Distribution', fontsize=12, fontweight='bold')
ax5.set_xlabel('P&L ($)')
ax5.set_ylabel('Count')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# === 6. DRAWDOWN CHART ===
ax6 = fig.add_subplot(gs[3, 0])
peak = initial_capital
drawdowns = []
for c in capitals:
    peak = max(peak, c)
    drawdowns.append((peak - c) / peak * 100)
ax6.fill_between(trade_nums, 0, drawdowns, color='red', alpha=0.5)
ax6.plot(trade_nums, drawdowns, 'darkred', linewidth=1)
ax6.axhline(y=max(drawdowns), color='darkred', linestyle='--', alpha=0.7)
ax6.text(len(drawdowns) * 0.02, max(drawdowns) + 0.3,
         f'Max: {max(drawdowns):.1f}%', color='darkred', fontweight='bold')
ax6.set_title('Drawdown %', fontsize=12, fontweight='bold')
ax6.set_xlabel('Trade #')
ax6.set_ylabel('Drawdown (%)')
ax6.set_ylim(0, max(drawdowns) * 1.4)
ax6.invert_yaxis()
ax6.grid(True, alpha=0.3)

# === 7. BARS HELD vs P&L ===
ax7 = fig.add_subplot(gs[3, 1])
colors_scatter = ['green' if p > 0 else 'red' for p in df['pnl']]
ax7.scatter(df['bars_held'], df['pnl'], c=colors_scatter, alpha=0.6, s=40,
            edgecolors='black', linewidth=0.3)
ax7.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
bars_group = df.groupby('bars_held').agg(avg_pnl=('pnl', 'mean'), count=('pnl', 'size'))
best_bars = bars_group[bars_group['count'] >= 3].sort_values('avg_pnl', ascending=False)
if len(best_bars) > 0:
    ax7.set_title(f"Bars Held vs P&L | Best hold: {best_bars.index[0]} bars "
                  f"(avg ${best_bars.iloc[0]['avg_pnl']:,.0f})",
                  fontsize=12, fontweight='bold')
else:
    ax7.set_title('Bars Held vs P&L', fontsize=12, fontweight='bold')
ax7.set_xlabel('Bars Held')
ax7.set_ylabel('P&L ($)')
ax7.grid(True, alpha=0.3)

out_path = 'output/backtest_chart_insights.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Chart saved to {out_path}')
