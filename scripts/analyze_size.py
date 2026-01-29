import pandas as pd

df = pd.read_csv('output/backtest_trades.csv')
df['win'] = df['pnl'] > 0

print('=== Trade Size Analysis ===\n')

# Basic stats
print('Contract Distribution:')
print(f"  Min: {df['num_contracts'].min()}")
print(f"  Max: {df['num_contracts'].max()}")
print(f"  Mean: {df['num_contracts'].mean():.1f}")
print(f"  Median: {df['num_contracts'].median():.0f}")

# By size bucket
print('\n=== Performance by Trade Size ===\n')

small = df[df['num_contracts'] < 10]
medium = df[(df['num_contracts'] >= 10) & (df['num_contracts'] < 50)]
large = df[df['num_contracts'] >= 50]

print(f"Small (2-9 contracts):  {len(small):3d} trades | WR: {100*small['win'].mean():5.1f}% | Avg P&L: ${small['pnl'].mean():8,.0f} | Total: ${small['pnl'].sum():12,.0f}")
print(f"Medium (10-49):         {len(medium):3d} trades | WR: {100*medium['win'].mean():5.1f}% | Avg P&L: ${medium['pnl'].mean():8,.0f} | Total: ${medium['pnl'].sum():12,.0f}")
print(f"Large (50):             {len(large):3d} trades | WR: {100*large['win'].mean():5.1f}% | Avg P&L: ${large['pnl'].mean():8,.0f} | Total: ${large['pnl'].sum():12,.0f}")

# Why small trades?
print('\n=== Why Small Trades? (first 20 small trades) ===\n')
small_sorted = small.sort_values('date').head(20)
print(small_sorted[['date', 'time', 'num_contracts', 'capital', 'pnl', 'exit_reason']].to_string(index=False))

# Capital at small trade times
print('\n=== Small Trade Details ===')
print(f"Capital range during small trades: ${small['capital'].min():,.0f} - ${small['capital'].max():,.0f}")
print(f"These occur early when capital is low (Kelly sizing)")
