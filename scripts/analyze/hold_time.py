"""
Hold Time Analysis for ORB Strategy
"""
import pandas as pd

df = pd.read_csv('output/backtest_trades.csv')

# Each bar = 5 minutes
df['hold_time_minutes'] = df['bars_held'] * 5

# Save updated CSV
df.to_csv('output/backtest_trades.csv', index=False)

print('='*60)
print('HOLD TIME ANALYSIS - ORB STRATEGY (2025)')
print('='*60)
print(f'\nTotal Trades: {len(df)}')

print('\n' + '-'*40)
print('HOLD TIME DISTRIBUTION')
print('-'*40)
for mins in sorted(df['hold_time_minutes'].unique()):
    subset = df[df['hold_time_minutes'] == mins]
    count = len(subset)
    pct = count / len(df) * 100
    wr = (subset['pnl'] > 0).mean() * 100
    avg_pnl = subset['pnl'].mean()
    total_pnl = subset['pnl'].sum()
    print(f"{mins:2d} min: {count:4d} trades ({pct:5.1f}%), WR: {wr:5.1f}%, Avg: ${avg_pnl:+,.0f}, Total: ${total_pnl:+,.0f}")

print('\n' + '-'*40)
print('SUMMARY STATISTICS')
print('-'*40)
print(f"Average Hold Time: {df['hold_time_minutes'].mean():.1f} min")
print(f"Median Hold Time: {df['hold_time_minutes'].median():.0f} min")
print(f"Min Hold Time: {df['hold_time_minutes'].min():.0f} min")
print(f"Max Hold Time: {df['hold_time_minutes'].max():.0f} min")
print(f"Total Time in Market: {df['hold_time_minutes'].sum():,} min ({df['hold_time_minutes'].sum()/60:.1f} hrs)")

# Quick trades (<=5 min) vs Long trades (>5 min)
quick = df[df['hold_time_minutes'] <= 5]
long_trades = df[df['hold_time_minutes'] > 5]

print('\n' + '-'*40)
print('QUICK vs LONG TRADES')
print('-'*40)
print(f"Quick (<=5 min): {len(quick)} trades ({len(quick)/len(df)*100:.1f}%), WR: {(quick['pnl']>0).mean()*100:.1f}%, PnL: ${quick['pnl'].sum():,.0f}")
print(f"Long (>5 min):   {len(long_trades)} trades ({len(long_trades)/len(df)*100:.1f}%), WR: {(long_trades['pnl']>0).mean()*100:.1f}%, PnL: ${long_trades['pnl'].sum():,.0f}")

print('\n' + '-'*40)
print('CSV SAMPLE (first 15 trades)')
print('-'*40)
print(df[['date','time','direction','strike','bars_held','hold_time_minutes','exit_reason','pnl']].head(15).to_string(index=False))

print(f"\n\nUpdated CSV saved to: output/backtest_trades.csv")
print(f"Columns: {df.columns.tolist()}")
