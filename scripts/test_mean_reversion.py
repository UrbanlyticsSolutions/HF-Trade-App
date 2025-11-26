"""
Mean Reversion Strategy Test
Trades when price deviates from Bollinger Bands
"""
import sys
import os
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return None, None, None
    
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper, sma, lower

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    if len(prices) < period + 1:
        return 50.0
    
    deltas = np.diff(prices[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def test_mean_reversion(days=30):
    print("="*80)
    print("MEAN REVERSION STRATEGY TEST")
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
    bb_period = 20
    rsi_period = 14
    profit_target = 0.006  # 0.6%
    stop_loss = 0.003      # 0.3%
    
    # Track state
    prices = []
    position = None
    entry_price = None
    trades = []
    
    for i, row in df.iterrows():
        price = row['close']
        prices.append(price)
        
        if len(prices) < bb_period:
            continue
        
        # Calculate indicators
        upper, middle, lower = calculate_bollinger_bands(prices, bb_period)
        rsi = calculate_rsi(prices, rsi_period)
        
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
            # LONG: Price below lower band + RSI oversold
            if price < lower and rsi < 30:
                position = 'LONG'
                entry_price = price
            
            # SHORT: Price above upper band + RSI overbought
            elif price > upper and rsi > 70:
                position = 'SHORT'
                entry_price = price
    
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
    test_mean_reversion(days=30)
