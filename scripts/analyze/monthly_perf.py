"""Monthly performance breakdown for current config."""
import sys, io, json
import pandas as pd
sys.path.insert(0, '.')
from backtest.engine import Backtest0DTE, TradeConfig
from core.risk_manager import RiskConfig
from config import defaults as cfg

cap = cfg.initial_capital()
config = json.load(open('config/strategy.json'))
tc = TradeConfig(**{k: v for k, v in config['trade_config'].items() if k in TradeConfig.__dataclass_fields__})
rc = RiskConfig(**{k: v for k, v in config['risk_config'].items() if k in RiskConfig.__dataclass_fields__})
bt = Backtest0DTE(tc, rc, initial_capital=cap)
bt.risk_manager.set_kelly(config['risk_config'].get('kelly_pct', 0.06))
u, o, f = bt.load_data('2025-01-01', '2025-12-31')
v = bt.compute_historical_volatility(u)
old = sys.stdout
sys.stdout = io.StringIO()
trades = bt.run_no_ml(u, o, f, v, verbose=False)
sys.stdout = old

rows = [{'date': t.date, 'dir': t.direction, 'pnl': t.pnl, 'capital': t.capital,
         'exit': t.exit_reason, 'bars': t.bars_held} for t in trades]
df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.strftime('%Y-%m')
df['win'] = df['pnl'] > 0

hdr = f"{'Month':>8} | {'Trd':>5} | {'Win':>4} | {'WR%':>6} | {'PnL ($)':>12} | {'Avg ($)':>9} | {'C':>4} | {'P':>4} | {'C_WR':>5} | {'P_WR':>5} | {'PF':>6} | {'Cumul ($)':>12}"
sep = '-' * len(hdr)

print()
print('MONTHLY PERFORMANCE — Current Config (2025)')
print('=' * len(hdr))
print(hdr)
print(sep)

cum = 0
for m in sorted(df['month'].unique()):
    s = df[df['month'] == m]
    w = int(s['win'].sum())
    wr = w / len(s) * 100
    pnl = s['pnl'].sum()
    cum += pnl
    c = s[s['dir'] == 'CALL']
    p = s[s['dir'] == 'PUT']
    cwr = c['win'].mean() * 100 if len(c) > 0 else 0
    pwr = p['win'].mean() * 100 if len(p) > 0 else 0
    gp = s[s['pnl'] > 0]['pnl'].sum()
    gl = abs(s[s['pnl'] < 0]['pnl'].sum())
    pf = gp / gl if gl > 0 else 999.0
    avg = pnl / len(s)
    print(f'{m:>8} | {len(s):>5} | {w:>4} | {wr:>5.1f}% | {pnl:>12,.0f} | {avg:>9,.0f} | {len(c):>4} | {len(p):>4} | {cwr:>4.0f}% | {pwr:>4.0f}% | {pf:>6.2f} | {cum:>12,.0f}')

print(sep)
tot_pnl = df['pnl'].sum()
wr = df['win'].mean() * 100
gp = df[df['pnl'] > 0]['pnl'].sum()
gl = abs(df[df['pnl'] < 0]['pnl'].sum())
pf = gp / gl if gl > 0 else 999
c = df[df['dir'] == 'CALL']
p = df[df['dir'] == 'PUT']
print(f'{"TOTAL":>8} | {len(df):>5} | {int(df["win"].sum()):>4} | {wr:>5.1f}% | {tot_pnl:>12,.0f} | {tot_pnl/len(df):>9,.0f} | {len(c):>4} | {len(p):>4} | {c["win"].mean()*100:>4.0f}% | {p["win"].mean()*100:>4.0f}% | {pf:>6.2f} | {cum:>12,.0f}')

print()
print(f'Return: {tot_pnl/cap*100:.1f}%  |  Start: ${cap:,.0f}  |  Final: ${cap+tot_pnl:,.0f}')
month_pnl = df.groupby('month')['pnl'].sum()
win_months = (month_pnl > 0).sum()
lose_months = (month_pnl < 0).sum()
print(f'Winning months: {win_months}/12  |  Losing months: {lose_months}/12')
print(f'Best:  {month_pnl.idxmax()} (${month_pnl.max():,.0f})')
print(f'Worst: {month_pnl.idxmin()} (${month_pnl.min():,.0f})')
