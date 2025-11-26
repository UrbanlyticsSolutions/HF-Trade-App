"""
Backtest Intraday ML Trend-Following Strategy
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import json

from strategy.ml_trend_following_strategy import MLTrendFollowingStrategy
from strategy.risk_manager import RiskLimits

def load_intraday_data(symbol="QQQ", limit=None):
    """Load intraday data from database"""
    conn = sqlite3.connect('market_data.db')
    query = f"""
        SELECT * FROM intraday_ticker_data 
        WHERE ticker = '{symbol}' 
        ORDER BY timestamp ASC
    """
    if limit:
        query += f" LIMIT {limit}"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def backtest_trend_following(symbol="QQQ", ml_entry_threshold=0.65):
    """Run backtest on historical data"""
    print("="*80)
    print("INTRADAY ML TREND-FOLLOWING BACKTEST")
    print("="*80)
    print(f"Symbol: {symbol}")
    print(f"ML Entry Threshold: {ml_entry_threshold}")
    print()
    
    # Load data
    print("Loading historical data...")
    df = load_intraday_data(symbol)
    print(f"Loaded {len(df)} bars")
    print()
    
    # Initialize strategy
    risk_limits = RiskLimits(
        max_position_size=100,
        max_portfolio_value=100000,
        stop_loss_pct=0.015,
        profit_target_pct=0.035
    )
    
    strategy = MLTrendFollowingStrategy(
        symbol=symbol,
        risk_limits=risk_limits,
        ml_entry_threshold=ml_entry_threshold,
        ml_exit_threshold=0.60
    )
    
    # Run backtest
    print("Running backtest...")
    for idx, row in df.iterrows():
        tick = {
            'timestamp': row['timestamp'],
            'price': row['close'],
            'volume': row['volume']
        }
        strategy.on_tick(tick)
        
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(df)} bars...")
    
    # Close any open position
    if strategy.position is not None:
        strategy.close_position(df.iloc[-1]['close'])
    
    # Get results
    stats = strategy.get_performance_stats()
    
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"Total Trades: {stats.get('total_trades', 0)}")
    print(f"Wins: {stats.get('wins', 0)}")
    print(f"Losses: {stats.get('losses', 0)}")
    print(f"Win Rate: {stats.get('win_rate', 0):.1%}")
    print(f"Total P/L: ${stats.get('total_pnl', 0):.2f}")
    print(f"Avg Win: ${stats.get('avg_win', 0):.2f}")
    print(f"Avg Loss: ${stats.get('avg_loss', 0):.2f}")
    print(f"Profit Factor: {stats.get('profit_factor', 0):.2f}")
    print("="*80)
    
    # Save results
    import os
    os.makedirs('output', exist_ok=True)
    
    with open('output/trend_following_backtest.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save trades
    if strategy.trades:
        trades_df = pd.DataFrame(strategy.trades)
        trades_df.to_csv('output/trend_following_trades.csv', index=False)
        print(f"\n✓ Trades saved to output/trend_following_trades.csv")
    
    return stats

if __name__ == "__main__":
    backtest_trend_following(symbol="QQQ", ml_entry_threshold=0.65)
