"""Drawdown event analysis — identifies and dissects every DD episode."""
import sys, io, json
sys.path.insert(0, '.')

from backtest.engine import Backtest0DTE, TradeConfig
from core.risk_manager import RiskConfig
from config import defaults as cfg

# Load config
config = json.load(open('config/strategy.json'))
tc_dict = config['trade_config']
tc = TradeConfig(**{k: v for k, v in tc_dict.items() if k in TradeConfig.__dataclass_fields__})
rc_dict = config['risk_config']
rc = RiskConfig(**{k: v for k, v in rc_dict.items() if k in RiskConfig.__dataclass_fields__})

bt = Backtest0DTE(tc, rc, initial_capital=10000)
u, o, f = bt.load_data('2025-01-01', '2025-12-31')
v = bt.compute_historical_volatility(u)
bt.risk_manager.set_kelly(0.05)

old = sys.stdout; sys.stdout = io.StringIO()
trades = bt.run_no_ml(u, o, f, v, verbose=False)
sys.stdout = old

D = "$"

# === Build equity curve ===
cap = 10000
equity = [cap]
for t in trades:
    equity.append(t.capital)

# === Find all DD events (>1%) ===
peak = cap
dd_events = []
in_dd = False
dd_start_idx = 0
dd_start_peak = cap
eq_low = cap
trough_idx = 0

for i, eq in enumerate(equity):
    if eq >= peak:
        if in_dd and (dd_start_peak - eq_low) / dd_start_peak > 0.01:
            dd_events.append({
                'start_idx': dd_start_idx,
                'end_idx': i,
                'peak': dd_start_peak,
                'trough': eq_low,
                'trough_idx': trough_idx,
                'dd_pct': (dd_start_peak - eq_low) / dd_start_peak * 100,
                'trades_in_dd': i - dd_start_idx,
            })
        peak = eq
        in_dd = False
    else:
        if not in_dd:
            in_dd = True
            dd_start_idx = i
            dd_start_peak = peak
            eq_low = eq
            trough_idx = i
        if eq < eq_low:
            eq_low = eq
            trough_idx = i

if in_dd and (dd_start_peak - eq_low) / dd_start_peak > 0.01:
    dd_events.append({
        'start_idx': dd_start_idx,
        'end_idx': len(equity)-1,
        'peak': dd_start_peak,
        'trough': eq_low,
        'trough_idx': trough_idx,
        'dd_pct': (dd_start_peak - eq_low) / dd_start_peak * 100,
        'trades_in_dd': len(equity)-1 - dd_start_idx,
    })

dd_events.sort(key=lambda x: -x['dd_pct'])

print(f"Total DD events (>1%): {len(dd_events)}")
print()
print("TOP 10 DRAWDOWN EVENTS:")
hdr = f"{'#':>3} {'DD%':>6} {'Peak'+D:>10} {'Trough'+D:>10} {'Loss'+D:>10} {'Trades':>7} {'Start':>12} {'Trough':>12}"
print(hdr)
print("-" * len(hdr))

for rank, dd in enumerate(dd_events[:10], 1):
    si = dd['start_idx']
    ti = dd['trough_idx']
    t_start = trades[min(si, len(trades)-1)]
    t_trough = trades[min(ti, len(trades)-1)]
    loss = dd['peak'] - dd['trough']
    print(f"{rank:>3} {dd['dd_pct']:>5.1f}% {dd['peak']:>10,.0f} {dd['trough']:>10,.0f} "
          f"{loss:>10,.0f} {dd['trades_in_dd']:>7} {t_start.date:>12} {t_trough.date:>12}")

# === Detailed breakdown of top 5 ===
print()
print("=" * 90)
print("DETAILED TRADE-BY-TRADE FOR TOP 5 DD EVENTS")
print("=" * 90)

for rank, dd in enumerate(dd_events[:5], 1):
    si = max(dd['start_idx'] - 1, 0)
    ei = min(dd['end_idx'], len(trades)-1)
    print(f"\n--- DD #{rank}: {dd['dd_pct']:.1f}% "
          f"(peak={D}{dd['peak']:,.0f} trough={D}{dd['trough']:,.0f}) ---")

    losers_in_dd = []
    for idx in range(si, ei + 1):
        if idx < len(trades):
            t = trades[idx]
            marker = "***" if t.pnl < 0 else "   "
            if t.pnl < 0 or abs(idx - dd['trough_idx']) <= 2:
                regime = getattr(t, 'regime', 'N/A')
                print(f"  {marker} [{idx:>4}] {t.date} {t.time} {t.direction:>4} "
                      f"pnl={t.pnl:>+8,.0f} cap={t.capital:>10,.0f} exit={t.exit_reason:<6} "
                      f"ct={t.num_contracts} regime={regime}")
                if t.pnl < 0:
                    losers_in_dd.append(t)

    all_in_dd = [trades[i] for i in range(si, min(ei+1, len(trades)))]
    calls = [t for t in losers_in_dd if t.direction == 'CALL']
    puts = [t for t in losers_in_dd if t.direction == 'PUT']
    stop_exits = sum(1 for t in losers_in_dd if t.exit_reason == 'STOP')
    time_exits = sum(1 for t in losers_in_dd if t.exit_reason == 'TIME')
    total_loss = sum(t.pnl for t in losers_in_dd)

    # Date clustering
    dates = sorted(set(t.date for t in losers_in_dd))
    losses_per_date = {}
    for t in losers_in_dd:
        losses_per_date[t.date] = losses_per_date.get(t.date, 0) + t.pnl
    worst_date = min(losses_per_date, key=losses_per_date.get) if losses_per_date else 'N/A'

    print(f"  Summary: {len(losers_in_dd)} losers ({len(calls)}C/{len(puts)}P), "
          f"stop={stop_exits} time={time_exits}, total_loss={D}{total_loss:+,.0f}")
    print(f"  Dates: {', '.join(dates[:8])}")
    print(f"  Worst date: {worst_date} ({D}{losses_per_date.get(worst_date, 0):+,.0f})")

# === Daily P&L analysis around DD events ===
print()
print("=" * 90)
print("DAILY P&L AROUND TOP 3 DD EVENTS")
print("=" * 90)

for rank, dd in enumerate(dd_events[:3], 1):
    si = max(dd['start_idx'] - 5, 0)
    ei = min(dd['end_idx'] + 5, len(trades)-1)
    all_trades = [trades[i] for i in range(si, ei + 1)]

    daily = {}
    for t in all_trades:
        if t.date not in daily:
            daily[t.date] = {'pnl': 0, 'n': 0, 'wins': 0, 'calls': 0, 'puts': 0,
                             'stops': 0, 'cap_start': t.capital - t.pnl}
        daily[t.date]['pnl'] += t.pnl
        daily[t.date]['n'] += 1
        if t.pnl > 0: daily[t.date]['wins'] += 1
        if t.direction == 'CALL': daily[t.date]['calls'] += 1
        else: daily[t.date]['puts'] += 1
        if t.exit_reason == 'STOP': daily[t.date]['stops'] += 1

    print(f"\n--- DD #{rank}: {dd['dd_pct']:.1f}% ---")
    print(f"  {'Date':>12} {'Trades':>7} {'C/P':>6} {'WR%':>5} {'Stops':>6} {'PnL':>10} {'Capital':>10}")
    print(f"  " + "-" * 68)

    for date in sorted(daily.keys()):
        d = daily[date]
        wr = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
        marker = " <<<" if d['pnl'] < -500 else ""
        print(f"  {date:>12} {d['n']:>7} {d['calls']:>2}/{d['puts']:<2} {wr:>4.0f}% "
              f"{d['stops']:>6} {D}{d['pnl']:>+9,.0f} {D}{d['cap_start']+d['pnl']:>9,.0f}{marker}")

# === Risk metrics by exit reason during DD ===
print()
print("=" * 90)
print("LOSS ANALYSIS BY EXIT REASON AND DIRECTION")
print("=" * 90)

all_losers = [t for t in trades if t.pnl < 0]
print(f"\nAll losers: {len(all_losers)} trades, total loss {D}{sum(t.pnl for t in all_losers):+,.0f}")

for reason in ['STOP', 'TIME', 'PROFIT']:
    subset = [t for t in all_losers if t.exit_reason == reason]
    if subset:
        avg_loss = sum(t.pnl for t in subset) / len(subset)
        max_loss = min(t.pnl for t in subset)
        calls = [t for t in subset if t.direction == 'CALL']
        puts = [t for t in subset if t.direction == 'PUT']
        print(f"  {reason:>6}: {len(subset)} trades, avg={D}{avg_loss:+,.0f}, "
              f"worst={D}{max_loss:+,.0f}, {len(calls)}C/{len(puts)}P")

# Call vs Put loss profile
print()
for d in ['CALL', 'PUT']:
    subset = [t for t in all_losers if t.direction == d]
    if subset:
        avg_loss = sum(t.pnl for t in subset) / len(subset)
        total_loss = sum(t.pnl for t in subset)
        max_loss = min(t.pnl for t in subset)
        avg_ct = sum(t.num_contracts for t in subset) / len(subset)
        print(f"  {d}: {len(subset)} losers, total={D}{total_loss:+,.0f}, "
              f"avg={D}{avg_loss:+,.0f}, worst={D}{max_loss:+,.0f}, avg_ct={avg_ct:.1f}")

# === Consecutive loss streaks ===
print()
print("=" * 90)
print("CONSECUTIVE LOSS STREAKS (>=3)")
print("=" * 90)

streak = 0
streak_start = 0
streaks = []
for i, t in enumerate(trades):
    if t.pnl < 0:
        if streak == 0:
            streak_start = i
        streak += 1
    else:
        if streak >= 3:
            streak_trades = trades[streak_start:streak_start + streak]
            total_loss = sum(tt.pnl for tt in streak_trades)
            streaks.append({
                'start': streak_start,
                'length': streak,
                'loss': total_loss,
                'date_start': streak_trades[0].date,
                'date_end': streak_trades[-1].date,
                'directions': '/'.join(tt.direction[0] for tt in streak_trades),
            })
        streak = 0
if streak >= 3:
    streak_trades = trades[streak_start:streak_start + streak]
    total_loss = sum(tt.pnl for tt in streak_trades)
    streaks.append({
        'start': streak_start, 'length': streak, 'loss': total_loss,
        'date_start': streak_trades[0].date, 'date_end': streak_trades[-1].date,
        'directions': '/'.join(tt.direction[0] for tt in streak_trades),
    })

streaks.sort(key=lambda x: x['loss'])
print(f"Total streaks >=3: {len(streaks)}")
for s in streaks[:15]:
    print(f"  len={s['length']} {D}{s['loss']:>+8,.0f}  {s['date_start']} - {s['date_end']}  dirs={s['directions']}")

# === Contract size at loss moments ===
print()
print("=" * 90)
print("CONTRACT SIZE DISTRIBUTION AT LOSSES vs WINS")
print("=" * 90)

win_cts = [t.num_contracts for t in trades if t.pnl > 0]
loss_cts = [t.num_contracts for t in all_losers]
import numpy as np
print(f"  Winners: avg={np.mean(win_cts):.1f} med={np.median(win_cts):.0f} "
      f"max={max(win_cts)} contracts")
print(f"  Losers:  avg={np.mean(loss_cts):.1f} med={np.median(loss_cts):.0f} "
      f"max={max(loss_cts)} contracts")

# Big losses (>$1000)
big_losses = [t for t in all_losers if t.pnl < -1000]
print(f"\n  Big losses (>{D}1000): {len(big_losses)}")
for t in sorted(big_losses, key=lambda x: x.pnl)[:10]:
    regime = getattr(t, 'regime', 'N/A')
    print(f"    {t.date} {t.time} {t.direction} ct={t.num_contracts} "
          f"pnl={D}{t.pnl:+,.0f} exit={t.exit_reason} cap={D}{t.capital:,.0f} regime={regime}")
