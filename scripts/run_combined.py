"""Run combined 2025-2026 backtest."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import Backtest0DTE, TradeConfig
from core.risk_manager import RiskConfig

config_path = Path(__file__).parent.parent / 'config' / 'strategy.json'
with open(config_path) as f:
    config = json.load(f)

tc = config['trade_config']
rc = config['risk_config']
trade_cfg = TradeConfig(**{k: v for k, v in tc.items() if k in TradeConfig.__dataclass_fields__})
risk_cfg = RiskConfig(**{k: v for k, v in rc.items() if k in RiskConfig.__dataclass_fields__})

bt = Backtest0DTE(trade_cfg, risk_cfg, initial_capital=config['backtest']['initial_capital'])
kelly_pct = rc.get('kelly_pct', 0.06)
bt.risk_manager.set_kelly(kelly_pct)

start, end = '2025-01-01', '2026-12-31'
print(f'Loading data {start} to {end}...')
underlying, options, features = bt.load_data(start, end)
vol = bt.compute_historical_volatility(underlying)
trades = bt.run_no_ml(underlying, options, features, vol, verbose=False)

if not trades:
    print('No trades')
    sys.exit(1)

import pandas as pd
wins = sum(1 for t in trades if t.pnl > 0)
total_pnl = sum(t.pnl for t in trades)
capitals = [config['backtest']['initial_capital']] + [t.capital for t in trades]
peak = capitals[0]
max_dd = 0
for c in capitals:
    peak = max(peak, c)
    dd = (peak - c) / peak * 100
    max_dd = max(max_dd, dd)
gp = sum(t.pnl for t in trades if t.pnl > 0)
gl = abs(sum(t.pnl for t in trades if t.pnl < 0))
pf = gp / gl if gl > 0 else 0

print()
print('=' * 60)
print('COMBINED 2025-2026 RESULTS')
print('=' * 60)
print(f'  Days Traded:     {len(set(t.date for t in trades))}')
print(f'  Total Trades:    {len(trades)}')
print(f'  Winning Trades:  {wins} ({wins/len(trades)*100:.1f}%)')
print(f'  Total P&L:       ${total_pnl:,.0f}')
print(f'  Final Capital:   ${trades[-1].capital:,.0f}')
print(f'  Max Drawdown:    {max_dd:.1f}%')
print(f'  Profit Factor:   {pf:.2f}')

# Save
trades_df = pd.DataFrame([t.to_dict() for t in trades])
trades_df.to_csv('output/backtest_trades.csv', index=False)
print(f'  Trades saved to: output/backtest_trades.csv')

# Direction breakdown
print()
df = trades_df
df['win'] = df['pnl'] > 0
ic = config['backtest']['initial_capital']
for d in ['CALL', 'PUT', 'ALL']:
    s = df if d == 'ALL' else df[df['direction'] == d]
    w = s[s['win']]
    l = s[~s['win']]
    cap = ic
    peak_c = ic
    mdd = 0
    for p in s['pnl']:
        cap += p
        peak_c = max(peak_c, cap)
        dd = (peak_c - cap) / peak_c * 100
        if dd > mdd:
            mdd = dd
    gp_d = w['pnl'].sum()
    gl_d = abs(l['pnl'].sum())
    pf_d = gp_d / gl_d if gl_d > 0 else 0
    print(f'{d:4s}: {len(s):3d} trades | WR {len(w)}/{len(s)} ({len(w)/len(s)*100:.1f}%) | '
          f'PnL ${s.pnl.sum():+,.0f} | AvgW ${w.pnl.mean():+,.0f} | AvgL ${l.pnl.mean():+,.0f} | '
          f'PF {pf_d:.2f} | MaxDD {mdd:.1f}%')

# Monthly
print()
print('Monthly breakdown:')
df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
for m, g in df.groupby('month'):
    wc = g['win'].sum()
    t = len(g)
    calls = g[g['direction'] == 'CALL']
    puts = g[g['direction'] == 'PUT']
    print(f'  {m}: {t:3d} trades (C:{len(calls):2d} P:{len(puts):2d}) | '
          f'WR {wc/t*100:4.0f}% | PnL ${g.pnl.sum():+8,.0f} | '
          f'C ${calls.pnl.sum():+8,.0f} | P ${puts.pnl.sum():+8,.0f}')
