"""
Backtest Comparison: 2025 vs 2026
Generates side-by-side equity curves, drawdown, monthly breakdown charts.
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from backtest.engine import Backtest0DTE, TradeConfig
from config import defaults as cfg
from core.risk_manager import RiskConfig


def load_config():
    with open('config/strategy.json') as f:
        data = json.load(f)
    tc = data['trade_config']
    rc = data['risk_config']
    trade_cfg = TradeConfig(**{k: v for k, v in tc.items() if k in TradeConfig.__dataclass_fields__})
    risk_cfg = RiskConfig(**{k: v for k, v in rc.items() if k in RiskConfig.__dataclass_fields__})
    return trade_cfg, risk_cfg


def run_backtest(trade_cfg, risk_cfg, train_start, train_end, test_start, test_end, label, initial_capital=None):
    if initial_capital is None:
        initial_capital = cfg.initial_capital()
    """Run a single backtest and return trades + metrics."""
    print(f'\n{"="*60}')
    print(f'  BACKTEST: {label}')
    print(f'{"="*60}')
    print(f'  Train: {train_start} to {train_end}')
    print(f'  Test:  {test_start} to {test_end}')

    bt = Backtest0DTE(trade_cfg, risk_cfg, initial_capital=initial_capital)

    # Kelly calibration from training data
    print('  Loading training data (Kelly)...')
    train_u, train_o, train_f = bt.load_data(train_start, train_end)
    train_v = bt.compute_historical_volatility(train_u)
    train_d = bt.generate_training_samples(train_u, train_o, train_f, train_v)
    bt.calculate_kelly_only(train_d)

    # Test period
    print('  Loading test data...')
    test_u, test_o, test_f = bt.load_data(test_start, test_end)
    test_v = bt.compute_historical_volatility(test_u)

    print('  Running...')
    trades = bt.run_no_ml(test_u, test_o, test_f, test_v, verbose=False)
    print(f'  → {len(trades)} trades')
    return trades


def compute_metrics(trades, initial_capital=None):
    if initial_capital is None:
        initial_capital = cfg.initial_capital()
    """Compute key metrics from trade list."""
    if not trades:
        return {}
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = len(trades) - wins
    total_pnl = sum(t.pnl for t in trades)
    final_cap = trades[-1].capital
    ret_pct = (final_cap / initial_capital - 1) * 100

    # Drawdown
    peak = initial_capital
    max_dd = 0
    for t in trades:
        if t.capital > peak:
            peak = t.capital
        dd = (peak - t.capital) / peak
        max_dd = max(max_dd, dd)

    # Profit factor
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe / Sortino
    returns = [t.pnl / (t.capital - t.pnl) for t in trades]
    avg_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = (avg_ret * 252) / (std_ret * np.sqrt(252)) if std_ret > 0 else 0
    down_rets = [r for r in returns if r < 0]
    down_std = np.std(down_rets) if down_rets else 0
    sortino = (avg_ret * 252) / (down_std * np.sqrt(252)) if down_std > 0 else 0

    avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if wins else 0
    avg_loss = np.mean([t.pnl for t in trades if t.pnl <= 0]) if losses else 0

    # Streaks
    streak, max_win_streak, max_loss_streak = 0, 0, 0
    for t in trades:
        if t.pnl > 0:
            streak = streak + 1 if streak > 0 else 1
            max_win_streak = max(max_win_streak, streak)
        else:
            streak = streak - 1 if streak < 0 else -1
            max_loss_streak = max(max_loss_streak, abs(streak))

    return {
        'trades': len(trades), 'wins': wins, 'losses': losses,
        'win_rate': wins / len(trades) * 100,
        'total_pnl': total_pnl, 'final_cap': final_cap, 'return_pct': ret_pct,
        'max_dd': max_dd * 100, 'profit_factor': pf,
        'sharpe': sharpe, 'sortino': sortino,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'max_win_streak': max_win_streak, 'max_loss_streak': max_loss_streak,
        'avg_ret_per_trade': avg_ret * 100,
    }


def print_metrics(label, m):
    """Print metrics summary."""
    print(f'\n  {label}')
    print(f'  {"─"*50}')
    print(f'  Trades:       {m["trades"]}  ({m["wins"]}W / {m["losses"]}L)')
    print(f'  Win Rate:     {m["win_rate"]:.1f}%')
    print(f'  Return:       +{m["return_pct"]:.1f}%  (${m["total_pnl"]:+,.0f})')
    print(f'  Max DD:       {m["max_dd"]:.1f}%')
    print(f'  PF:           {m["profit_factor"]:.2f}')
    print(f'  Sharpe:       {m["sharpe"]:.2f}')
    print(f'  Sortino:      {m["sortino"]:.2f}')
    print(f'  Avg Win:      ${m["avg_win"]:,.2f}')
    print(f'  Avg Loss:     ${m["avg_loss"]:,.2f}')


def generate_plots(trades_2025, trades_2026, m25, m26, initial_capital=None):
    if initial_capital is None:
        initial_capital = cfg.initial_capital()
    """Generate comparison charts for 2025 vs 2026."""
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Backtest Comparison: 2025 vs 2026 (Jan-Feb)\nPhase 8 Strategy  |  $10,000 Starting Capital',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    c25 = '#3498db'  # blue
    c26 = '#2ecc71'  # green

    # ---- 1. Equity Curves (% Return) ----
    ax1 = fig.add_subplot(gs[0, 0])
    for trades, color, label, m in [
        (trades_2025, c25, '2025', m25),
        (trades_2026, c26, '2026', m26),
    ]:
        caps = [initial_capital] + [t.capital for t in trades]
        rets = [(c / initial_capital - 1) * 100 for c in caps]
        ax1.plot(range(len(rets)), rets, color=color, linewidth=2,
                 label=f'{label} ({m["trades"]} trades, +{m["return_pct"]:.0f}%)', alpha=0.9)
        ax1.fill_between(range(len(rets)), 0, rets, color=color, alpha=0.08)

    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Equity Curve (% Return)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Return %')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}%'))
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ---- 2. Drawdown Comparison ----
    ax2 = fig.add_subplot(gs[0, 1])
    for trades, color, label, m in [
        (trades_2025, c25, '2025', m25),
        (trades_2026, c26, '2026', m26),
    ]:
        caps = [initial_capital] + [t.capital for t in trades]
        peak = initial_capital
        dds = []
        for c in caps:
            if c > peak:
                peak = c
            dds.append((peak - c) / peak * 100)
        ax2.fill_between(range(len(dds)), 0, dds, color=color, alpha=0.4,
                         label=f'{label} (max {m["max_dd"]:.1f}%)')
        ax2.plot(range(len(dds)), dds, color=color, linewidth=1)

    ax2.set_title('Drawdown %', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Trade #')
    ax2.set_ylabel('Drawdown %')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    # ---- 3. Daily P&L Bar Charts ----
    ax3 = fig.add_subplot(gs[1, 0])
    for trades, color, label in [
        (trades_2025, c25, '2025'),
        (trades_2026, c26, '2026'),
    ]:
        tdf = pd.DataFrame([{'date': t.date, 'pnl': t.pnl} for t in trades])
        tdf['date'] = pd.to_datetime(tdf['date'])
        daily = tdf.groupby(tdf['date'].dt.date)['pnl'].sum()
        # Normalize to % of starting capital for comparability
        daily_pct = daily / initial_capital * 100
        ax3.bar([f'{label}\n{d.strftime("%m/%d")}' for d in daily.index],
                daily_pct.values,
                color=[color if p > 0 else '#e74c3c' for p in daily_pct.values],
                alpha=0.7, width=0.8, edgecolor='white', linewidth=0.3)

    ax3.axhline(0, color='gray', linewidth=0.5)
    ax3.set_title('Daily P&L (% of Capital)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Daily Return %')
    ax3.tick_params(axis='x', rotation=90, labelsize=6)
    ax3.grid(True, alpha=0.3, axis='y')

    # ---- 4. Monthly Breakdown Bars ----
    ax4 = fig.add_subplot(gs[1, 1])
    months_all = []
    pnl_by_month = {}
    wr_by_month = {}
    for trades, label in [(trades_2025, '2025'), (trades_2026, '2026')]:
        tdf = pd.DataFrame([{'date': t.date, 'pnl': t.pnl} for t in trades])
        tdf['date'] = pd.to_datetime(tdf['date'])
        tdf['month'] = tdf['date'].dt.to_period('M').astype(str)
        for month_str in tdf['month'].unique():
            key = f'{label}\n{month_str}'
            months_all.append(key)
            sub = tdf[tdf['month'] == month_str]
            pnl_by_month[key] = sub['pnl'].sum()
            wr_by_month[key] = (sub['pnl'] > 0).mean() * 100

    x_pos = range(len(months_all))
    colors_bar = [c25 if '2025' in m else c26 for m in months_all]
    bars = ax4.bar(x_pos, [pnl_by_month[m] for m in months_all],
                   color=colors_bar, alpha=0.8, edgecolor='white')
    # Add win rate labels
    for i, m in enumerate(months_all):
        pnl = pnl_by_month[m]
        wr = wr_by_month[m]
        ax4.text(i, pnl + (200 if pnl >= 0 else -200),
                 f'WR {wr:.0f}%', ha='center', va='bottom' if pnl >= 0 else 'top',
                 fontsize=8, fontweight='bold')

    ax4.set_xticks(list(x_pos))
    ax4.set_xticklabels(months_all, fontsize=8)
    ax4.set_title('Monthly P&L ($)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('P&L ($)')
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='y')

    # ---- 5. Win Rate by Hour ----
    ax5 = fig.add_subplot(gs[2, 0])
    hours = range(9, 16)
    width = 0.35
    for i, (trades, color, label) in enumerate([
        (trades_2025, c25, '2025'),
        (trades_2026, c26, '2026'),
    ]):
        tdf = pd.DataFrame([{'time': t.time, 'pnl': t.pnl} for t in trades])
        tdf['hour'] = tdf['time'].apply(lambda x: int(x.split(':')[0]))
        wr_list = []
        count_list = []
        for h in hours:
            sub = tdf[tdf.hour == h]
            wr_list.append((sub.pnl > 0).mean() * 100 if len(sub) > 0 else 0)
            count_list.append(len(sub))
        x = np.arange(len(hours))
        offset = -width / 2 + i * width
        b = ax5.bar(x + offset, wr_list, width, color=color, alpha=0.8,
                    label=label, edgecolor='white')
        for bar, cnt in zip(b, count_list):
            if cnt > 0:
                ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         str(cnt), ha='center', va='bottom', fontsize=7, color=color)

    ax5.set_xticks(np.arange(len(hours)))
    ax5.set_xticklabels([f'{h}:00' for h in hours])
    ax5.set_title('Win Rate by Hour (numbers = count)', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Win Rate %')
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 110)

    # ---- 6. Summary Table ----
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    table_data = [
        ['Metric', 'Jan-Feb 2025', 'Jan-Feb 2026', 'Delta'],
        ['Total Trades', f'{m25["trades"]}', f'{m26["trades"]}', f'{m26["trades"]-m25["trades"]:+d}'],
        ['Win Rate', f'{m25["win_rate"]:.1f}%', f'{m26["win_rate"]:.1f}%', f'{m26["win_rate"]-m25["win_rate"]:+.1f}pp'],
        ['Total Return', f'+{m25["return_pct"]:.0f}%', f'+{m26["return_pct"]:.0f}%', f'{m26["return_pct"]-m25["return_pct"]:+.0f}%'],
        ['Total P&L', f'${m25["total_pnl"]:+,.0f}', f'${m26["total_pnl"]:+,.0f}', f'${m26["total_pnl"]-m25["total_pnl"]:+,.0f}'],
        ['Max Drawdown', f'{m25["max_dd"]:.1f}%', f'{m26["max_dd"]:.1f}%', f'{m26["max_dd"]-m25["max_dd"]:+.1f}pp'],
        ['Profit Factor', f'{m25["profit_factor"]:.2f}', f'{m26["profit_factor"]:.2f}', f'{m26["profit_factor"]-m25["profit_factor"]:+.2f}'],
        ['Sharpe', f'{m25["sharpe"]:.2f}', f'{m26["sharpe"]:.2f}', f'{m26["sharpe"]-m25["sharpe"]:+.2f}'],
        ['Sortino', f'{m25["sortino"]:.2f}', f'{m26["sortino"]:.2f}', f'{m26["sortino"]-m25["sortino"]:+.2f}'],
        ['Avg Win', f'${m25["avg_win"]:,.2f}', f'${m26["avg_win"]:,.2f}', ''],
        ['Avg Loss', f'${m25["avg_loss"]:,.2f}', f'${m26["avg_loss"]:,.2f}', ''],
        ['Return/DD', f'{m25["return_pct"]/m25["max_dd"]:.1f}x', f'{m26["return_pct"]/m26["max_dd"]:.1f}x', ''],
    ]

    table = ax6.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for j in range(4):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(table_data)):
        for j in range(4):
            table[i, j].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
            table[i, j].set_edgecolor('#bdc3c7')

    ax6.set_title('Summary Comparison', fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = 'output/backtest_2025_vs_2026.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'\nChart saved to {out_path}')
    plt.close()


def generate_individual_plot(trades, metrics, year_label, test_start, test_end, initial_capital=None):
    if initial_capital is None:
        initial_capital = cfg.initial_capital()
    """Generate a standalone equity curve + drawdown chart for one period."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1.5, 1]})

    tdf = pd.DataFrame([{'date': t.date, 'pnl': t.pnl, 'capital': t.capital} for t in trades])
    tdf['date'] = pd.to_datetime(tdf['date'])

    daily = tdf.groupby(tdf['date'].dt.date).agg(
        pnl=('pnl', 'sum'), capital=('capital', 'last'), trades_count=('pnl', 'count'),
        wins=('pnl', lambda x: (x > 0).sum())
    )

    # ---- 1. Daily P&L bars + equity overlay ----
    ax1 = axes[0]
    colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in daily['pnl'].values]
    ax1.bar(daily.index, daily['pnl'].values, color=colors, alpha=0.7, width=0.8)
    ax1.axhline(0, color='gray', linewidth=0.5)

    ax1b = ax1.twinx()
    ax1b.plot(daily.index, daily['capital'].values, 'b-', linewidth=2, label='Equity')
    ax1b.set_ylabel('Equity ($)', color='blue', fontsize=11)
    ax1b.tick_params(axis='y', labelcolor='blue')

    m = metrics
    ax1.set_title(
        f'{year_label} Backtest | {m["trades"]} trades | WR {m["win_rate"]:.1f}% | '
        f'Return +{m["return_pct"]:.0f}% | DD {m["max_dd"]:.1f}% | PF {m["profit_factor"]:.2f}',
        fontsize=13, fontweight='bold')
    ax1.set_ylabel('Daily P&L ($)', fontsize=11)
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.grid(True, alpha=0.3)

    # ---- 2. Equity curve by trade ----
    ax2 = axes[1]
    capitals = [initial_capital] + [t.capital for t in trades]
    trade_nums = list(range(len(capitals)))
    ax2.plot(trade_nums, capitals, 'b-', linewidth=2)
    ax2.fill_between(trade_nums, initial_capital, capitals,
                     where=[c >= initial_capital for c in capitals], color='green', alpha=0.3, interpolate=True)
    ax2.fill_between(trade_nums, initial_capital, capitals,
                     where=[c < initial_capital for c in capitals], color='red', alpha=0.3, interpolate=True)
    ax2.axhline(initial_capital, color='gray', linestyle='--', alpha=0.5, label=f'Initial ${initial_capital:,}')
    ax2.scatter([len(capitals) - 1], [capitals[-1]], color='blue', s=100, zorder=5)
    ax2.annotate(f'${capitals[-1]:,.0f}\n(+{m["return_pct"]:.0f}%)',
                 xy=(len(capitals) - 1, capitals[-1]),
                 xytext=(len(capitals) - 1 - len(capitals) * 0.15, capitals[-1]),
                 fontsize=11, fontweight='bold', color='blue')
    ax2.set_title(f'Equity Curve | ${initial_capital:,} → ${capitals[-1]:,.0f}', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Trade #', fontsize=11)
    ax2.set_ylabel('Capital ($)', fontsize=11)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # ---- 3. Drawdown ----
    ax3 = axes[2]
    peak = initial_capital
    dds = []
    for c in capitals:
        if c > peak:
            peak = c
        dds.append((peak - c) / peak * 100)
    ax3.fill_between(trade_nums, 0, dds, color='red', alpha=0.5)
    ax3.plot(trade_nums, dds, 'darkred', linewidth=1)
    if max(dds) > 0:
        ax3.axhline(max(dds), color='darkred', linestyle='--', alpha=0.5)
        ax3.text(len(dds) * 0.02, max(dds) + 0.3, f'Max: {max(dds):.1f}%', color='darkred', fontweight='bold')
    ax3.set_title('Drawdown %', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Trade #', fontsize=11)
    ax3.set_ylabel('DD (%)', fontsize=11)
    ax3.set_ylim(0, max(dds) * 1.4 if max(dds) > 0 else 5)
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_label = year_label.lower().replace(' ', '_').replace('-', '_')
    out_path = f'output/backtest_{safe_label}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Chart saved to {out_path}')
    plt.close()

    # Save trades CSV
    csv_path = f'output/backtest_{safe_label}.csv'
    trades_df = pd.DataFrame([t.to_dict() for t in trades])
    trades_df.to_csv(csv_path, index=False)
    print(f'Trades saved to {csv_path}')


def analyze_monthly_restart(trades_all, initial_capital=None):
    if initial_capital is None:
        initial_capital = cfg.initial_capital()
    """Analyze trades month-by-month, restarting capital to $10K each month."""
    if not trades_all:
        print('No trades to analyze.')
        return

    tdf = pd.DataFrame([t.to_dict() for t in trades_all])
    tdf['date'] = pd.to_datetime(tdf['date'])
    tdf['month'] = tdf['date'].dt.to_period('M').astype(str)

    months = sorted(tdf['month'].unique())
    monthly_results = []

    print('\n' + '=' * 80)
    print('  MONTHLY ANALYSIS (Capital Restarted to $10,000 Each Month)')
    print('=' * 80)
    print(f'  {"Month":<10} {"Trades":>6} {"W/L":>8} {"WR%":>6} {"P&L":>10} {"Ret%":>8} {"PF":>6} {"MaxDD":>7} {"AvgWin":>8} {"AvgLoss":>8}')
    print(f'  {"─"*10} {"─"*6} {"─"*8} {"─"*6} {"─"*10} {"─"*8} {"─"*6} {"─"*7} {"─"*8} {"─"*8}')

    for month in months:
        mt = tdf[tdf['month'] == month].copy()
        n = len(mt)
        wins = (mt['pnl'] > 0).sum()
        losses = n - wins
        total_pnl = mt['pnl'].sum()

        # Recompute capital with restart
        cap = initial_capital
        peak = cap
        max_dd = 0
        caps = [cap]
        for _, row in mt.iterrows():
            cap += row['pnl']
            caps.append(cap)
            if cap > peak:
                peak = cap
            dd = (peak - cap) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        ret_pct = (cap / initial_capital - 1) * 100
        gp = mt.loc[mt['pnl'] > 0, 'pnl'].sum()
        gl = abs(mt.loc[mt['pnl'] <= 0, 'pnl'].sum())
        pf = gp / gl if gl > 0 else float('inf')
        avg_win = mt.loc[mt['pnl'] > 0, 'pnl'].mean() if wins > 0 else 0
        avg_loss = mt.loc[mt['pnl'] <= 0, 'pnl'].mean() if losses > 0 else 0

        monthly_results.append({
            'month': month, 'trades': n, 'wins': wins, 'losses': losses,
            'wr': wins / n * 100 if n else 0, 'pnl': total_pnl,
            'return_pct': ret_pct, 'max_dd': max_dd * 100, 'pf': pf,
            'avg_win': avg_win, 'avg_loss': avg_loss, 'final_cap': cap,
        })

        pf_str = f'{pf:.1f}' if pf < 100 else 'inf'
        print(f'  {month:<10} {n:>6} {wins:>3}/{losses:<4} {wins/n*100:>5.1f}% ${total_pnl:>+8,.0f} {ret_pct:>+7.1f}% {pf_str:>6} {max_dd*100:>6.1f}% ${avg_win:>7,.0f} ${avg_loss:>7,.0f}')

    # Totals
    total_trades = sum(r['trades'] for r in monthly_results)
    total_wins = sum(r['wins'] for r in monthly_results)
    total_losses = sum(r['losses'] for r in monthly_results)
    total_pnl = sum(r['pnl'] for r in monthly_results)
    profitable_months = sum(1 for r in monthly_results if r['pnl'] > 0)
    losing_months = sum(1 for r in monthly_results if r['pnl'] <= 0)
    avg_monthly_ret = np.mean([r['return_pct'] for r in monthly_results])
    avg_monthly_pnl = np.mean([r['pnl'] for r in monthly_results])

    print(f'  {"─"*10} {"─"*6} {"─"*8} {"─"*6} {"─"*10} {"─"*8} {"─"*6} {"─"*7} {"─"*8} {"─"*8}')
    print(f'  {"TOTAL":<10} {total_trades:>6} {total_wins:>3}/{total_losses:<4} {total_wins/total_trades*100:>5.1f}% ${total_pnl:>+8,.0f} {"":>8} {"":>6} {"":>7} {"":>8} {"":>8}')

    print(f'\n  Profitable Months:  {profitable_months}/{len(monthly_results)} ({profitable_months/len(monthly_results)*100:.0f}%)')
    print(f'  Losing Months:      {losing_months}/{len(monthly_results)}')
    print(f'  Avg Monthly Return: {avg_monthly_ret:+.1f}%  (${avg_monthly_pnl:+,.0f})')
    print(f'  Best Month:         {max(monthly_results, key=lambda x: x["pnl"])["month"]} (${max(monthly_results, key=lambda x: x["pnl"])["pnl"]:+,.0f})')
    print(f'  Worst Month:        {min(monthly_results, key=lambda x: x["pnl"])["month"]} (${min(monthly_results, key=lambda x: x["pnl"])["pnl"]:+,.0f})')

    return monthly_results, tdf


def generate_analysis_chart(monthly_results, tdf, trades_all, initial_capital=None):
    if initial_capital is None:
        initial_capital = cfg.initial_capital()
    """Generate comprehensive monthly restart analysis chart."""
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle('Backtest Analysis: Jan 2025 → Feb 2026\nCapital Restarted $10,000 Each Month',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

    months = [r['month'] for r in monthly_results]
    pnls = [r['pnl'] for r in monthly_results]
    rets = [r['return_pct'] for r in monthly_results]
    wrs = [r['wr'] for r in monthly_results]
    dds = [r['max_dd'] for r in monthly_results]
    trade_counts = [r['trades'] for r in monthly_results]

    # ---- 1. Monthly P&L bars ----
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in pnls]
    bars = ax1.bar(range(len(months)), pnls, color=colors, alpha=0.8, edgecolor='white')
    for i, (bar, pnl) in enumerate(zip(bars, pnls)):
        ax1.text(bar.get_x() + bar.get_width() / 2, pnl + (50 if pnl >= 0 else -50),
                 f'${pnl:+,.0f}', ha='center', va='bottom' if pnl >= 0 else 'top',
                 fontsize=7, fontweight='bold')
    ax1.set_xticks(range(len(months)))
    ax1.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax1.set_title('Monthly P&L ($10K Restart)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('P&L ($)')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # ---- 2. Monthly Return % ----
    ax2 = fig.add_subplot(gs[0, 1])
    colors_ret = ['#2ecc71' if r > 0 else '#e74c3c' for r in rets]
    bars2 = ax2.bar(range(len(months)), rets, color=colors_ret, alpha=0.8, edgecolor='white')
    for i, (bar, ret) in enumerate(zip(bars2, rets)):
        ax2.text(bar.get_x() + bar.get_width() / 2, ret + (0.5 if ret >= 0 else -0.5),
                 f'{ret:+.1f}%', ha='center', va='bottom' if ret >= 0 else 'top',
                 fontsize=7, fontweight='bold')
    ax2.set_xticks(range(len(months)))
    ax2.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax2.set_title('Monthly Return % ($10K Restart)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Return %')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # ---- 3. Win Rate + Trade Count by month ----
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(range(len(months)), wrs, color='#3498db', alpha=0.8, edgecolor='white')
    for i, (wr, n) in enumerate(zip(wrs, trade_counts)):
        ax3.text(i, wr + 1, f'{wr:.0f}%\n({n})', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax3.set_xticks(range(len(months)))
    ax3.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax3.set_title('Win Rate % by Month (count in parens)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Win Rate %')
    ax3.axhline(np.mean(wrs), color='red', linestyle='--', alpha=0.7, label=f'Avg {np.mean(wrs):.1f}%')
    ax3.set_ylim(0, 100)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # ---- 4. Max Drawdown by month ----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(range(len(months)), dds, color='#e74c3c', alpha=0.6, edgecolor='white')
    for i, dd in enumerate(dds):
        ax4.text(i, dd + 0.2, f'{dd:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax4.set_xticks(range(len(months)))
    ax4.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax4.set_title('Max Drawdown % by Month ($10K Restart)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Max DD %')
    ax4.axhline(np.mean(dds), color='darkred', linestyle='--', alpha=0.7, label=f'Avg {np.mean(dds):.1f}%')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # ---- 5. Continuous equity curve ----
    ax5 = fig.add_subplot(gs[2, :])
    capitals = [initial_capital] + [t.capital for t in trades_all]
    ax5.plot(range(len(capitals)), capitals, 'b-', linewidth=2)
    ax5.fill_between(range(len(capitals)), initial_capital, capitals,
                     where=[c >= initial_capital for c in capitals], color='green', alpha=0.2, interpolate=True)
    ax5.fill_between(range(len(capitals)), initial_capital, capitals,
                     where=[c < initial_capital for c in capitals], color='red', alpha=0.2, interpolate=True)
    ax5.axhline(initial_capital, color='gray', linestyle='--', alpha=0.5)

    # Mark month boundaries
    trade_dates = [t.date for t in trades_all]
    prev_month = None
    for i, d in enumerate(trade_dates):
        m = d[:7]
        if m != prev_month and prev_month is not None:
            ax5.axvline(i, color='gray', linestyle=':', alpha=0.4)
            ax5.text(i, max(capitals) * 0.98, m, fontsize=7, rotation=90, va='top', alpha=0.6)
        prev_month = m

    ax5.scatter([len(capitals) - 1], [capitals[-1]], color='blue', s=80, zorder=5)
    final_ret = (capitals[-1] / initial_capital - 1) * 100
    ax5.annotate(f'${capitals[-1]:,.0f}\n(+{final_ret:.0f}%)',
                 xy=(len(capitals) - 1, capitals[-1]),
                 xytext=(len(capitals) * 0.85, capitals[-1] * 0.9),
                 fontsize=11, fontweight='bold', color='blue',
                 arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))

    ax5.set_title(f'Continuous Equity Curve | ${initial_capital:,} → ${capitals[-1]:,.0f} (+{final_ret:.0f}%)',
                  fontsize=13, fontweight='bold')
    ax5.set_xlabel('Trade #', fontsize=11)
    ax5.set_ylabel('Capital ($)', fontsize=11)
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax5.grid(True, alpha=0.3)

    # ---- 6. Daily P&L heatmap-style ----
    ax6 = fig.add_subplot(gs[3, 0])
    daily = tdf.groupby(tdf['date'].dt.date)['pnl'].sum()
    daily_pct = daily / initial_capital * 100
    colors_d = ['#2ecc71' if p > 0 else '#e74c3c' for p in daily_pct.values]
    ax6.bar(range(len(daily)), daily_pct.values, color=colors_d, alpha=0.7, width=1.0)
    ax6.axhline(0, color='gray', linewidth=0.5)
    ax6.set_title('Daily P&L % (Every Trading Day)', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Trading Day #')
    ax6.set_ylabel('Daily Return %')
    winning_days = (daily_pct > 0).sum()
    total_days = len(daily_pct)
    ax6.text(0.02, 0.95, f'Win Days: {winning_days}/{total_days} ({winning_days/total_days*100:.0f}%)',
             transform=ax6.transAxes, fontsize=9, fontweight='bold', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax6.grid(True, alpha=0.3, axis='y')

    # ---- 7. Summary table ----
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.axis('off')

    profitable = sum(1 for r in monthly_results if r['pnl'] > 0)
    total_m = len(monthly_results)
    total_trades = sum(r['trades'] for r in monthly_results)
    total_wins = sum(r['wins'] for r in monthly_results)
    avg_ret = np.mean(rets)
    avg_pnl = np.mean(pnls)
    best = max(monthly_results, key=lambda x: x['pnl'])
    worst = min(monthly_results, key=lambda x: x['pnl'])
    final_cap = capitals[-1]

    table_data = [
        ['Metric', 'Value'],
        ['Total Trades', f'{total_trades}'],
        ['Overall Win Rate', f'{total_wins/total_trades*100:.1f}%'],
        ['Profitable Months', f'{profitable}/{total_m} ({profitable/total_m*100:.0f}%)'],
        ['Avg Monthly Return', f'{avg_ret:+.1f}%'],
        ['Avg Monthly P&L', f'${avg_pnl:+,.0f}'],
        ['Best Month', f'{best["month"]} (${best["pnl"]:+,.0f})'],
        ['Worst Month', f'{worst["month"]} (${worst["pnl"]:+,.0f})'],
        ['Avg Max DD/Month', f'{np.mean(dds):.1f}%'],
        ['Compounded Return', f'+{final_ret:.0f}%'],
        ['Final Capital', f'${final_cap:,.0f}'],
        ['Win Day Rate', f'{winning_days}/{total_days} ({winning_days/total_days*100:.0f}%)'],
    ]

    table = ax7.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for j in range(2):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(table_data)):
        for j in range(2):
            table[i, j].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
            table[i, j].set_edgecolor('#bdc3c7')
    ax7.set_title('Summary', fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = 'output/backtest_2025_2026_analysis.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'\nAnalysis chart saved to {out_path}')
    plt.close()


def main():
    trade_cfg, risk_cfg = load_config()
    initial_capital = cfg.initial_capital()
    trades_all = run_backtest(
        trade_cfg, risk_cfg,
        train_start='2024-07-01', train_end='2025-12-31',
        test_start='2026-01-02', test_end='2026-02-14',
        label='Jan-Feb 2026',
        initial_capital=initial_capital,
    )

    m = compute_metrics(trades_all, initial_capital)

    # Print summary
    print('\n' + '=' * 60)
    print('  CONTINUOUS BACKTEST SUMMARY: Jan 2025 → Feb 2026')
    print('=' * 60)
    print_metrics('Jan 2025 – Feb 2026', m)

    # Monthly restart analysis
    monthly_results, tdf = analyze_monthly_restart(trades_all, initial_capital)

    # Generate charts
    print('\n' + '=' * 60)
    print('  GENERATING CHARTS')
    print('=' * 60)
    generate_individual_plot(trades_all, m, '2025_2026_continuous', '2025-01-02', '2026-02-14', initial_capital)
    generate_analysis_chart(monthly_results, tdf, trades_all, initial_capital)

    # Save trades CSV
    trades_df = pd.DataFrame([t.to_dict() for t in trades_all])
    csv_path = 'output/backtest_2025_2026_continuous.csv'
    trades_df.to_csv(csv_path, index=False)
    print(f'Trades saved to {csv_path}')

    # Save monthly summary CSV
    mdf = pd.DataFrame(monthly_results)
    mdf.to_csv('output/backtest_monthly_restart_summary.csv', index=False)
    print(f'Monthly summary saved to output/backtest_monthly_restart_summary.csv')

    print('\n' + '=' * 60)
    print('  ALL DONE')
    print('=' * 60)
    print('  Output files:')
    print('    output/backtest_2025_2026_continuous.png')
    print('    output/backtest_2025_2026_continuous.csv')
    print('    output/backtest_2025_2026_analysis.png')
    print('    output/backtest_monthly_restart_summary.csv')


if __name__ == '__main__':
    main()
