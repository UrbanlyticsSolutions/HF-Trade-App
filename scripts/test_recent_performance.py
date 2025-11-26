"""
Test Strategy Performance on Recent Data
Shows win rate, returns, and detailed metrics
"""
import sys
import os
import pandas as pd
import sqlite3
import logging
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy.ml_trend_following_strategy import MLTrendFollowingStrategy
from strategy.risk_manager import RiskLimits

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger('RecentTest')
logger.setLevel(logging.INFO)

def load_recent_data(symbol="QQQ", days=30):
    """Load most recent N days of data"""
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'market_data.db')
    conn = sqlite3.connect(db_path)
    
    # Get data from last N days
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    query = """
        SELECT * FROM intraday_ticker_data 
        WHERE ticker = ? AND timestamp >= ?
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn, params=(symbol, cutoff_date))
    conn.close()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def calculate_metrics(trades):
    """Calculate detailed performance metrics"""
    if not trades:
        return None
    
    df = pd.DataFrame(trades)
    
    # Basic stats
    total_trades = len(df)
    wins = df[df['profitable'] == 1]
    losses = df[df['profitable'] == 0]
    
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    
    # PnL stats
    total_pnl = df['pnl'].sum()
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
    
    # Profit factor
    gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Return calculation (assuming $10k starting capital, $100/share position)
    capital = 10000
    position_size = 100  # shares
    total_return_pct = (total_pnl / capital) * 100
    
    # Win/Loss ratio
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    return {
        'total_trades': total_trades,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_return_pct': total_return_pct,
        'win_loss_ratio': win_loss_ratio
    }

def run_recent_test(days=30):
    print("="*80)
    print(f"TESTING STRATEGY ON LAST {days} DAYS")
    print("="*80)
    
    # Load data
    print(f"\nLoading last {days} days of data...")
    df = load_recent_data("QQQ", days)
    
    if len(df) == 0:
        print("No recent data found!")
        return
    
    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Initialize Strategy
    risk_limits = RiskLimits(
        max_position_size=100,
        max_portfolio_value=100000,
        max_position_pct=0.2,
        max_drawdown_pct=0.1,
        stop_loss_pct=0.01,
        profit_target_pct=0.03,
        trailing_stop_pct=0.01,
        max_daily_loss=5000
    )
    
    strategy = MLTrendFollowingStrategy(
        symbol="QQQ",
        risk_limits=risk_limits,
        ml_entry_threshold=0.60,
        ml_exit_threshold=0.50
    )
    
    print("\nRunning backtest...")
    
    # Simulate ticks
    for i, row in df.iterrows():
        tick = {
            'timestamp': row['timestamp'],
            'price': row['close'],
            'volume': row['volume'],
            'high': row['high'],
            'low': row['low']
        }
        strategy.on_tick(tick)
    
    # Get metrics
    metrics = calculate_metrics(strategy.trades)
    
    print("\n" + "="*80)
    print("PERFORMANCE RESULTS")
    print("="*80)
    
    if metrics:
        print(f"\n📊 TRADING STATISTICS:")
        print(f"  Total Trades:     {metrics['total_trades']}")
        print(f"  Wins:             {metrics['wins']}")
        print(f"  Losses:           {metrics['losses']}")
        print(f"  Win Rate:         {metrics['win_rate']:.2%} ⭐")
        
        print(f"\n💰 PROFIT & LOSS:")
        print(f"  Total P&L:        ${metrics['total_pnl']:.2f}")
        print(f"  Avg Win:          ${metrics['avg_win']:.2f}")
        print(f"  Avg Loss:         ${metrics['avg_loss']:.2f}")
        print(f"  Win/Loss Ratio:   {metrics['win_loss_ratio']:.2f}")
        print(f"  Profit Factor:    {metrics['profit_factor']:.2f}")
        
        print(f"\n📈 RETURNS (on $10k capital):")
        print(f"  Total Return:     {metrics['total_return_pct']:.2f}%")
        print(f"  Per Trade Avg:    {metrics['total_pnl']/metrics['total_trades']:.2f}")
        
    else:
        print("No trades executed in this period.")
    
    print("="*80)
    
    return metrics

if __name__ == "__main__":
    # Test on last 30 days
    run_recent_test(days=30)
