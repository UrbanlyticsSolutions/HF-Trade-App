"""
0DTE SPY Options Backtest

Usage:
    python -m backtest.run backtest              # Run backtest on configured dates
    python -m backtest.run backtest --year 2025  # Backtest specific year
    python -m backtest.run analyze               # Analyze saved results
"""
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config():
    """Load strategy configuration from JSON file."""
    config_path = Path(__file__).parent.parent / 'config' / 'strategy.json'
    with open(config_path) as f:
        return json.load(f)


def run_backtest(year: int = None, verbose: bool = True):
    """Run backtest with configured parameters."""
    from backtest.engine import Backtest0DTE, TradeConfig
    from core.risk_manager import RiskConfig
    
    config = load_config()
    tc = config['trade_config']
    rc = config['risk_config']
    strategy = tc.get('strategy', config.get('strategy', 'momentum'))
    
    trade_cfg = TradeConfig(**{k: v for k, v in tc.items() if k in TradeConfig.__dataclass_fields__})
    
    risk_cfg = RiskConfig(**{k: v for k, v in rc.items() if k in RiskConfig.__dataclass_fields__})
    
    # Determine date range
    if year:
        start = f"{year}-01-01"
        end = f"{year}-12-31"
    else:
        start = config['backtest']['train_start']
        end = config['backtest']['test_end']
    
    print("="*60)
    print(f"0DTE SPY OPTIONS - {strategy.upper()} STRATEGY")
    print("="*60)
    print(f"Period: {start} to {end}")
    print(f"Window: {tc['trade_start_hour']}:{tc['trade_start_minute']:02d} - {tc['trade_end_hour']}:{tc.get('trade_end_minute', 0):02d}")
    print(f"RSI Thresholds: CALL>{tc.get('rsi_call_threshold', 70)}, PUT<{tc.get('rsi_put_threshold', 30)}")
    print(f"Options: ${tc['min_option_price']:.2f} - ${tc['max_option_price']:.2f}")
    print(f"Targets: +{tc['profit_target_pct']*100:.0f}% / -{tc['stop_loss_pct']*100:.0f}%")
    print()
    
    bt = Backtest0DTE(trade_cfg, risk_cfg, initial_capital=config['backtest']['initial_capital'])
    
    # Use configured Kelly % (from optimization) instead of calculating from training data
    kelly_pct = rc.get('kelly_pct', 0.06)
    bt.risk_manager.set_kelly(kelly_pct)
    print(f"Kelly: {kelly_pct*100:.0f}%")
    
    # Load test data
    underlying, options, features = bt.load_data(start, end)
    vol = bt.compute_historical_volatility(underlying)
    trades = bt.run_no_ml(underlying, options, features, vol, verbose=verbose)
    
    if not trades:
        print("No trades executed")
        return []
    
    # Results
    wins = sum(1 for t in trades if t.pnl > 0)
    total_pnl = sum(t.pnl for t in trades)
    final_cap = trades[-1].capital
    
    # Drawdown
    capitals = [config['backtest']['initial_capital']] + [t.capital for t in trades]
    peak = capitals[0]
    max_dd = 0
    for c in capitals:
        peak = max(peak, c)
        dd = (peak - c) / peak * 100
        max_dd = max(max_dd, dd)
    
    # Profit factor
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  Total Trades:    {len(trades)}")
    print(f"  Winning Trades:  {wins} ({wins/len(trades)*100:.1f}%)")
    print(f"  Total P&L:       ${total_pnl:,.0f}")
    print(f"  Final Capital:   ${final_cap:,.0f}")
    print(f"  Max Drawdown:    {max_dd:.1f}%")
    print(f"  Profit Factor:   {pf:.2f}")
    
    # Save trades to CSV
    import pandas as pd
    trades_df = pd.DataFrame([t.to_dict() for t in trades])
    trades_df.to_csv('output/backtest_trades.csv', index=False)
    print(f"\n  Trades saved to: output/backtest_trades.csv")
    
    return trades


def run_analyze():
    """Analyze saved backtest results."""
    import pandas as pd
    
    csv_path = Path(__file__).parent.parent / 'output' / 'all_backtest_trades.csv'
    if not csv_path.exists():
        print("No backtest results found. Run 'python run.py backtest' first.")
        return
    
    df = pd.read_csv(csv_path)
    df['win'] = df['pnl'] > 0
    
    print("="*60)
    print("0DTE SPY OPTIONS - ANALYSIS")
    print("="*60)
    print(f"\nTotal Trades: {len(df)}")
    print(f"Win Rate: {df['win'].mean()*100:.1f}%")
    print(f"Total P&L: ${df['pnl'].sum():,.0f}")
    print(f"Avg Winner: ${df[df['win']]['pnl'].mean():,.0f}")
    print(f"Avg Loser: ${df[~df['win']]['pnl'].mean():,.0f}")
    
    print("\nBy Period:")
    for period in df['period'].unique():
        s = df[df['period'] == period]
        print(f"  {period}: {len(s)} trades, {s['win'].mean()*100:.1f}% WR, ${s['pnl'].sum():,.0f}")
    
    print("\nBy Direction:")
    for direction in ['CALL', 'PUT']:
        s = df[df['direction'] == direction]
        print(f"  {direction}: {len(s)} trades, {s['win'].mean()*100:.1f}% WR, ${s['pnl'].sum():,.0f}")
    
    print("\nBy Exit Reason:")
    for reason in df['exit_reason'].unique():
        s = df[df['exit_reason'] == reason]
        print(f"  {reason}: {len(s)} trades ({len(s)/len(df)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='0DTE SPY Options Backtest')
    parser.add_argument('command', choices=['backtest', 'analyze'], 
                        help='Command to run')
    parser.add_argument('--year', type=int, help='Year for backtest (default: use config dates)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    if args.command == 'backtest':
        run_backtest(year=args.year, verbose=not args.quiet)
    elif args.command == 'analyze':
        run_analyze()


if __name__ == '__main__':
    main()
