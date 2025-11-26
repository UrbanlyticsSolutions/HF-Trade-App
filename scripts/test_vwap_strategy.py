"""
VWAP Strategy Test
Trades around Volume-Weighted Average Price
"""
import sys
import os
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_vwap(prices, volumes):
    """Calculate VWAP"""
    if len(prices) == 0:
        return None
    
    cumulative_pv = np.cumsum(np.array(prices) * np.array(volumes))
    cumulative_v = np.cumsum(volumes)
    
    return cumulative_pv[-1] / cumulative_v[-1] if cumulative_v[-1] > 0 else None

def test_vwap_strategy(days=30):
    print("="*80)
    print("VWAP STRATEGY TEST")
    print("="*80)
    
    # Load data
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'market_data.db')
    conn = sqlite3.connect(db_path)
    
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    query = """
        SELECT * FROM intraday_ticker_data 
        WHERE ticker = 'QQQ' AND timestamp >= ?
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn, params=(cutoff_date,))
    conn.close()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Strategy parameters
    profit_target = 0.008  # 0.8%
    stop_loss = 0.004      # 0.4%
    vwap_threshold = 0.002 # 0.2% deviation from VWAP
    
    # Track state
    position = None
    entry_price = None
    trades = []
    
    # Group by day for daily VWAP
    df['date'] = df['timestamp'].dt.date
    
    for date, day_df in df.groupby('date'):
        day_prices = []
        day_volumes = []
        
        for i, row in day_df.iterrows():
            price = row['close']
            volume = row['volume']
            
            day_prices.append(price)
            day_volumes.append(volume)
            
            if len(day_prices) < 10:  # Need minimum bars
                continue
            
            vwap = calculate_vwap(day_prices, day_volumes)
            if vwap is None:
                continue
            
            deviation = (price - vwap) / vwap
            
            # Exit logic
            if position:
                pnl_pct = (price - entry_price) / entry_price
                
                if position == 'LONG':
                    if pnl_pct >= profit_target or pnl_pct <= -stop_loss:
                        trades.append({
                            'entry': entry_price,
                            'exit': price,
                            'pnl': price - entry_price,
                            'profitable': 1 if pnl_pct > 0 else 0
                        })
                        position = None
                elif position == 'SHORT':
                    pnl_pct = (entry_price - price) / entry_price
                    if pnl_pct >= profit_target or pnl_pct <= -stop_loss:
                        trades.append({
                            'entry': entry_price,
                            'exit': price,
                            'pnl': entry_price - price,
                            'profitable': 1 if pnl_pct > 0 else 0
                        })
                        position = None
            
            # Entry logic
            if not position:
                # LONG: Price below VWAP with volume
                if deviation < -vwap_threshold and volume > np.mean(day_volumes):
                    position = 'LONG'
                    entry_price = price
                
                # SHORT: Price above VWAP with volume
                elif deviation > vwap_threshold and volume > np.mean(day_volumes):
                    position = 'SHORT'
                    entry_price = price
        
        # Close position at end of day
        if position:
            last_price = day_df.iloc[-1]['close']
            if position == 'LONG':
                pnl = last_price - entry_price
            else:
                pnl = entry_price - last_price
            
            trades.append({
                'entry': entry_price,
                'exit': last_price,
                'pnl': pnl,
                'profitable': 1 if pnl > 0 else 0
            })
            position = None
    
    # Results
    if trades:
        df_trades = pd.DataFrame(trades)
        win_rate = df_trades['profitable'].mean()
        total_pnl = df_trades['pnl'].sum()
        avg_pnl = df_trades['pnl'].mean()
        
        print(f"\n📊 RESULTS:")
        print(f"  Total Trades: {len(trades)}")
        print(f"  Trades/Day: {len(trades)/days:.1f}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Total P&L: ${total_pnl:.2f}")
        print(f"  Avg P&L: ${avg_pnl:.2f}")
        print(f"  Return: {(total_pnl/10000)*100:.2f}%")
    else:
        print("No trades executed")
    
    print("="*80)
    return trades

if __name__ == "__main__":
    test_vwap_strategy(days=30)
