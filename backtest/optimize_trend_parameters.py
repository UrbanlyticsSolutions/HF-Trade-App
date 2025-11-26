"""
Optimize Trend-Following Strategy Parameters
Tests different ML thresholds and MA periods
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import json

from strategy.ml_trend_following_strategy import MLTrendFollowingStrategy
from strategy.risk_manager import RiskLimits

def load_intraday_data(symbol="QQQ"):
    """Load intraday data from database"""
    conn = sqlite3.connect('market_data.db')
    query = f"""
        SELECT * FROM intraday_ticker_data 
        WHERE ticker = '{symbol}' 
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def backtest_with_params(df, ml_threshold, fast_ma, slow_ma):
    """Run backtest with specific parameters"""
    risk_limits = RiskLimits(
        max_position_size=100,
        max_portfolio_value=100000,
        stop_loss_pct=0.015,
        profit_target_pct=0.035
    )
    
    strategy = MLTrendFollowingStrategy(
        symbol="QQQ",
        risk_limits=risk_limits,
        ml_entry_threshold=ml_threshold,
        ml_exit_threshold=0.60
    )
    
    # Override MA periods
    strategy.fast_ma_period = fast_ma
    strategy.slow_ma_period = slow_ma
    
    # Run backtest
    for idx, row in df.iterrows():
        tick = {
            'timestamp': row['timestamp'],
            'price': row['close'],
            'volume': row['volume']
        }
        strategy.on_tick(tick)
    
    # Close any open position
    if strategy.position is not None:
        strategy.close_position(df.iloc[-1]['close'])
    
    return strategy.get_performance_stats()

def optimize_parameters():
    """Grid search for best parameters"""
    print("="*80)
    print("TREND-FOLLOWING PARAMETER OPTIMIZATION")
    print("="*80)
    print()
    
    # Load data
    print("Loading data...")
    df = load_intraday_data("QQQ")
    print(f"Loaded {len(df)} bars\n")
    
    # Parameter grid
    ml_thresholds = [0.55, 0.60, 0.65, 0.70]
    ma_configs = [(10, 30), (15, 40), (20, 50)]
    
    results = []
    
    print("Testing parameter combinations...")
    for ml_thresh in ml_thresholds:
        for fast_ma, slow_ma in ma_configs:
            print(f"Testing: ML={ml_thresh}, MA=({fast_ma},{slow_ma})...", end=" ")
            
            stats = backtest_with_params(df, ml_thresh, fast_ma, slow_ma)
            
            result = {
                'ml_threshold': ml_thresh,
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'total_trades': stats.get('total_trades', 0),
                'win_rate': stats.get('win_rate', 0),
                'profit_factor': stats.get('profit_factor', 0),
                'total_pnl': stats.get('total_pnl', 0)
            }
            
            results.append(result)
            print(f"Trades={result['total_trades']}, WinRate={result['win_rate']:.1%}, PF={result['profit_factor']:.2f}")
    
    # Find best configuration
    results_df = pd.DataFrame(results)
    
    # Sort by profit factor (primary) and win rate (secondary)
    results_df['score'] = results_df['profit_factor'] * 0.6 + results_df['win_rate'] * 0.4
    results_df = results_df.sort_values('score', ascending=False)
    
    print("\n" + "="*80)
    print("TOP 5 CONFIGURATIONS")
    print("="*80)
    print(results_df.head(5).to_string(index=False))
    print("="*80)
    
    # Save results
    import os
    os.makedirs('output', exist_ok=True)
    results_df.to_csv('output/trend_optimization_results.csv', index=False)
    
    # Save best config
    best = results_df.iloc[0]
    best_config = {
        'ml_threshold': float(best['ml_threshold']),
        'fast_ma': int(best['fast_ma']),
        'slow_ma': int(best['slow_ma']),
        'win_rate': float(best['win_rate']),
        'profit_factor': float(best['profit_factor']),
        'total_pnl': float(best['total_pnl'])
    }
    
    with open('output/best_trend_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\n✓ Best configuration saved to output/best_trend_config.json")
    print(f"✓ Full results saved to output/trend_optimization_results.csv")
    
    return best_config

if __name__ == "__main__":
    best_config = optimize_parameters()
