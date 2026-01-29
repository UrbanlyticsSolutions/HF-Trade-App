"""
0DTE Backtest Runner V3 - NO ML
- RSI calibration filter only (RSI > 65 for CALL, RSI < 35 for PUT)
- No ML model needed
- Simpler, faster, better results
- Supports JSON config loading
"""
import sys
sys.path.insert(0, '.')

import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from backtest.engine import Backtest0DTE, TradeConfig
from core.risk_manager import RiskConfig


def plot_results(trades, underlying_df):
    """Plot daily SPY price chart with trades, equity curve, and drawdown"""
    import matplotlib.dates as mdates
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1.5, 1]})
    
    # Aggregate intraday to DAILY OHLC for ALL days
    daily_price = underlying_df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    daily_price['date'] = pd.to_datetime(daily_price['date'])
    daily_price = daily_price.sort_values('date').reset_index(drop=True)
    
    print(f"\nPlotting {len(daily_price)} trading days of SPY data")
    print(f"Date range: {daily_price['date'].min().strftime('%Y-%m-%d')} to {daily_price['date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${daily_price['low'].min():.2f} - ${daily_price['high'].max():.2f}")
    
    # ============================================
    # 1. DAILY SPY PRICE CHART WITH WIN/LOSS MARKERS
    # ============================================
    ax1 = axes[0]
    
    # Plot daily price line for ALL days
    ax1.plot(daily_price['date'], daily_price['close'], 'b-', linewidth=1.5, label='SPY Daily Close')
    ax1.fill_between(daily_price['date'], daily_price['low'], daily_price['high'], 
                     alpha=0.15, color='blue', label='Daily Range')
    
    # Plot win/loss markers at daily close price
    win_dates, win_prices = [], []
    loss_dates, loss_prices = [], []
    
    for t in trades:
        trade_date = pd.to_datetime(t.date)
        price_row = daily_price[daily_price['date'] == trade_date]
        if len(price_row) > 0:
            price = price_row['close'].values[0]
            if t.pnl > 0:
                win_dates.append(trade_date)
                win_prices.append(price + 1)  # Offset above price
            else:
                loss_dates.append(trade_date)
                loss_prices.append(price - 1)  # Offset below price
    
    # Plot markers
    ax1.scatter(win_dates, win_prices, marker='^', color='green', s=50, alpha=0.9, 
                label=f'Win ({len(win_dates)})', zorder=5, edgecolors='darkgreen', linewidths=0.5)
    ax1.scatter(loss_dates, loss_prices, marker='v', color='red', s=50, alpha=0.9, 
                label=f'Loss ({len(loss_dates)})', zorder=5, edgecolors='darkred', linewidths=0.5)
    
    # Formatting
    win_rate = len(win_dates) / len(trades) * 100
    ax1.set_title(f'SPY Daily Price Chart 2024-2025 | {len(trades)} trades | {win_rate:.1f}% Win Rate', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('SPY Price ($)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.set_xlim(daily_price['date'].min(), daily_price['date'].max())
    
    # ============================================
    # 2. EQUITY CURVE
    # ============================================
    ax2 = axes[1]
    
    capitals = [10000] + [t.capital for t in trades]
    trade_indices = list(range(len(capitals)))
    
    # Color equity curve based on trend
    ax2.plot(trade_indices, capitals, 'b-', linewidth=2)
    ax2.fill_between(trade_indices, 10000, capitals, where=[c >= 10000 for c in capitals], 
                     color='green', alpha=0.3, interpolate=True)
    ax2.fill_between(trade_indices, 10000, capitals, where=[c < 10000 for c in capitals], 
                     color='red', alpha=0.3, interpolate=True)
    
    # Mark key points
    ax2.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial $10K')
    
    # Find max drawdown point
    peak = 10000
    max_dd = 0
    max_dd_idx = 0
    for i, cap in enumerate(capitals):
        if cap > peak:
            peak = cap
        dd = (peak - cap) / peak
        if dd > max_dd:
            max_dd = dd
            max_dd_idx = i
    
    ax2.scatter([max_dd_idx], [capitals[max_dd_idx]], color='red', s=100, zorder=5, marker='o')
    ax2.annotate(f'Max DD: {max_dd*100:.1f}%', xy=(max_dd_idx, capitals[max_dd_idx]), 
                 xytext=(max_dd_idx + 10, capitals[max_dd_idx] * 0.9),
                 fontsize=10, color='red', fontweight='bold')
    
    final_return = (capitals[-1] / 10000 - 1) * 100
    ax2.set_title(f'Equity Curve | Final: ${capitals[-1]:,.0f} (+{final_return:.0f}%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Trade #')
    ax2.set_ylabel('Capital ($)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # ============================================
    # 3. DRAWDOWN CHART
    # ============================================
    ax3 = axes[2]
    
    peak = 10000
    drawdowns = []
    for cap in capitals:
        if cap > peak:
            peak = cap
        dd = (peak - cap) / peak * 100
        drawdowns.append(dd)
    
    ax3.fill_between(trade_indices, 0, drawdowns, color='red', alpha=0.6)
    ax3.plot(trade_indices, drawdowns, 'darkred', linewidth=1)
    
    # Mark max drawdown
    ax3.axhline(y=max(drawdowns), color='darkred', linestyle='--', alpha=0.7)
    ax3.text(len(drawdowns) * 0.02, max(drawdowns) + 0.5, f'Max: {max(drawdowns):.1f}%', 
             color='darkred', fontweight='bold')
    
    ax3.set_title('Drawdown %', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Trade #')
    ax3.set_ylabel('Drawdown (%)')
    ax3.set_ylim(0, max(drawdowns) * 1.3 if max(drawdowns) > 0 else 10)
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('\nPlot saved to backtest_results.png')


def load_config_from_json(json_path: str = "config/strategy.json"):
    """Load configuration from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    trade_cfg_data = data.get('trade_config', {})
    risk_cfg_data = data.get('risk_config', {})
    backtest_data = data.get('backtest', {})
    
    # Build TradeConfig
    trade_cfg = TradeConfig(**{k: v for k, v in trade_cfg_data.items() 
                               if k in TradeConfig.__dataclass_fields__})
    
    # Build RiskConfig
    risk_cfg = RiskConfig(**{k: v for k, v in risk_cfg_data.items() 
                             if k in RiskConfig.__dataclass_fields__})
    
    return trade_cfg, risk_cfg, backtest_data


def run_backtest(trade_cfg: TradeConfig, risk_cfg: RiskConfig, backtest_cfg: dict, plot: bool = True):
    """Run backtest with given configuration"""
    initial_capital = backtest_cfg.get('initial_capital', 10000)
    
    bt = Backtest0DTE(trade_cfg, risk_cfg, initial_capital=initial_capital)

    # === LOAD TRAINING DATA (for Kelly calculation only) ===
    print('=' * 50)
    print('LOADING TRAINING DATA (for Kelly)')
    print('=' * 50)
    
    train_start = backtest_cfg.get('train_start', '2023-01-01')
    train_end = backtest_cfg.get('train_end', '2023-12-31')
    
    train_underlying, train_options, train_features = bt.load_data(train_start, train_end)
    train_vol = bt.compute_historical_volatility(train_underlying)
    
    # Generate training samples for Kelly calculation
    training_data = bt.generate_training_samples(train_underlying, train_options, train_features, train_vol)
    
    # Calculate Kelly from training data (no ML training)
    kelly = bt.calculate_kelly_only(training_data)

    # === TEST ===
    print('\n' + '=' * 50)
    print('TESTING')
    print('=' * 50)
    
    test_start = backtest_cfg.get('test_start', '2024-01-01')
    test_end = backtest_cfg.get('test_end', '2025-06-30')
    
    test_underlying, test_options, test_features = bt.load_data(test_start, test_end)
    test_vol = bt.compute_historical_volatility(test_underlying)

    trades = bt.run_no_ml(test_underlying, test_options, test_features, test_vol, verbose=False)

    # === RESULTS ===
    if not trades:
        print('\nNo trades executed!')
        return None

    wins = sum(1 for t in trades if t.pnl > 0)
    losses = len(trades) - wins
    
    # Calculate max drawdown
    peak = initial_capital
    max_dd = 0
    for t in trades:
        if t.capital > peak:
            peak = t.capital
        dd = (peak - t.capital) / peak
        if dd > max_dd:
            max_dd = dd

    # Calculate profit factor
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # === CONSISTENCY METRICS ===
    import numpy as np
    
    # Daily returns
    returns = [t.pnl / (t.capital - t.pnl) for t in trades]
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Sharpe Ratio
    sharpe = (avg_return * 250) / (std_return * np.sqrt(250)) if std_return > 0 else 0
    
    # Sortino Ratio
    downside_returns = [r for r in returns if r < 0]
    downside_std = np.std(downside_returns) if downside_returns else 0
    sortino = (avg_return * 250) / (downside_std * np.sqrt(250)) if downside_std > 0 else 0
    
    # Monthly returns
    trades_df_temp = pd.DataFrame([{'date': t.date, 'pnl': t.pnl, 'capital': t.capital} for t in trades])
    trades_df_temp['date'] = pd.to_datetime(trades_df_temp['date'])
    trades_df_temp['month'] = trades_df_temp['date'].dt.to_period('M')
    
    monthly_pnl = trades_df_temp.groupby('month')['pnl'].sum()
    profitable_months = sum(1 for pnl in monthly_pnl if pnl > 0)
    total_months = len(monthly_pnl)
    monthly_consistency = profitable_months / total_months * 100 if total_months > 0 else 0
    
    worst_month_pnl = monthly_pnl.min()
    worst_month = monthly_pnl.idxmin()
    best_month_pnl = monthly_pnl.max()
    best_month = monthly_pnl.idxmax()

    print('\n' + '=' * 50)
    print('RESULTS')
    print('=' * 50)
    print(f'  Trading Hours: {trade_cfg.trade_start_hour}:00 - {trade_cfg.trade_end_hour}:{trade_cfg.trade_end_minute:02d}')
    print(f'  Trades: {len(trades)}')
    print(f'  Wins: {wins} | Losses: {losses}')
    print(f'  Win Rate: {wins/len(trades)*100:.1f}%')
    print(f'  Profit Factor: {profit_factor:.2f}')
    print(f'  Final Capital: ${trades[-1].capital:,.0f}')
    print(f'  Return: +{(trades[-1].capital/initial_capital-1)*100:.0f}%')
    print(f'  Max Drawdown: {max_dd*100:.1f}%')
    
    print('\n' + '=' * 50)
    print('CONSISTENCY METRICS')
    print('=' * 50)
    print(f'  Sharpe Ratio: {sharpe:.2f}')
    print(f'  Sortino Ratio: {sortino:.2f}')
    print(f'  Risk-Adjusted (Return/DD): {(trades[-1].capital/initial_capital-1)*100/max_dd/100:.0f}')
    print(f'  Avg Return/Trade: {avg_return*100:.1f}%')
    print(f'  Return Std Dev: {std_return*100:.1f}%')
    print(f'  Monthly Win Rate: {monthly_consistency:.0f}% ({profitable_months}/{total_months} months)')
    print(f'  Best Month: {best_month} (+${best_month_pnl:,.0f})')
    print(f'  Worst Month: {worst_month} (${worst_month_pnl:,.0f})')

    # Save trades to CSV
    trades_df = pd.DataFrame([t.to_dict() for t in trades])
    trades_df.to_csv('backtest_trades.csv', index=False)
    print(f'\nTrades saved to backtest_trades.csv')

    if plot:
        # Load FULL underlying data for plotting
        print('\nLoading full SPY data for plotting...')
        full_underlying = pd.DataFrame(bt.db.get_intraday_5min('SPY'))
        full_underlying = full_underlying.rename(columns={'date': 'timestamp'})
        full_underlying['date'] = pd.to_datetime(full_underlying['timestamp']).dt.strftime('%Y-%m-%d')
        full_underlying = full_underlying[
            (full_underlying['date'] >= test_start) & 
            (full_underlying['date'] <= test_end)
        ]
        plot_results(trades, full_underlying)
    
    return trades


def main():
    parser = argparse.ArgumentParser(description='0DTE Backtest Runner')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file')
    parser.add_argument('--no-time-restriction', action='store_true', help='Remove trading hour restrictions (9:30-15:30)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    if args.config:
        # Load from JSON config
        trade_cfg, risk_cfg, backtest_cfg = load_config_from_json(args.config)
    else:
        # Default configuration
        trade_cfg = TradeConfig(
            profit_target_pct=0.25,
            stop_loss_pct=0.28,
            use_ml_filter=False,
            skip_day_filter=True,
            use_adaptive_exits=True,
            profit_low_vol=0.20,
            profit_mid_vol=0.25,
            profit_high_vol=0.35,
            stop_low_vol=0.18,
            stop_mid_vol=0.28,
            stop_high_vol=0.40,
            use_trailing_stop=False,
        )
        
        risk_cfg = RiskConfig(
            kelly_fraction=0.20,
            max_risk_per_trade_pct=0.02,
            max_position_pct=0.07,
            max_position_value=5000,
            stop_after_first_loss=True,
            max_consecutive_losses=2,
            reduce_size_at_dd_pct=0.05,
        )
        
        backtest_cfg = {
            'initial_capital': 10000,
            'train_start': '2023-01-01',
            'train_end': '2023-12-31',
            'test_start': '2024-01-01',
            'test_end': '2025-06-30',
        }
    
    # Apply no-time-restriction if requested
    if args.no_time_restriction:
        trade_cfg.trade_start_hour = 9
        trade_cfg.trade_end_hour = 15
        trade_cfg.trade_end_minute = 30
        print("*** RUNNING WITHOUT TIME RESTRICTIONS (9:30 - 15:30) ***\n")
    
    run_backtest(trade_cfg, risk_cfg, backtest_cfg, plot=not args.no_plot)


if __name__ == '__main__':
    main()
