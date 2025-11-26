"""
Generate Training Data for Price Prediction
Creates labeled data: UP/DOWN/NEUTRAL for next bar
"""
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_indicators(prices, volumes, highs, lows):
    """Calculate technical indicators"""
    if len(prices) < 20:
        return None
    
    price = prices[-1]
    
    # Moving averages
    ma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else price
    ma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else price
    ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else price
    
    # Price changes
    price_change_1 = (price - prices[-2]) / prices[-2] if len(prices) >= 2 else 0
    price_change_3 = (price - prices[-4]) / prices[-4] if len(prices) >= 4 else 0
    price_change_5 = (price - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
    
    # Volatility
    if len(prices) >= 10:
        returns = np.diff(prices[-10:]) / prices[-10:-1]
        price_volatility = np.std(returns)
    else:
        price_volatility = 0
    
    # RSI
    if len(prices) >= 15:
        deltas = np.diff(prices[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100 if avg_gain > 0 else 50
    else:
        rsi = 50
    
    # MACD (simple version)
    if len(prices) >= 26:
        ema_12 = pd.Series(prices).ewm(span=12).mean().iloc[-1]
        ema_26 = pd.Series(prices).ewm(span=26).mean().iloc[-1]
        macd = ema_12 - ema_26
        macd_signal = pd.Series([macd]).ewm(span=9).mean().iloc[-1]
        macd_hist = macd - macd_signal
    else:
        macd = 0
        macd_hist = 0
    
    # Volume
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 and np.mean(volumes[-20:]) > 0 else 1.0
    if len(volumes) >= 2 and volumes[-2] > 0:
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2]
    else:
        volume_change = 0
    
    # ADX (simplified)
    adx = 20  # Placeholder
    trend_strength = (ma_5 - ma_20) / ma_20 if ma_20 > 0 else 0
    
    return {
        'price_change_1': price_change_1,
        'price_change_3': price_change_3,
        'price_change_5': price_change_5,
        'price_volatility': price_volatility,
        'ma_5': ma_5,
        'ma_10': ma_10,
        'ma_20': ma_20,
        'price_to_ma5': price / ma_5 if ma_5 > 0 else 1.0,
        'price_to_ma20': price / ma_20 if ma_20 > 0 else 1.0,
        'rsi': rsi,
        'macd': macd,
        'macd_hist': macd_hist,
        'volume_ratio': volume_ratio,
        'volume_change': volume_change,
        'adx': adx,
        'trend_strength': trend_strength
    }

def create_labeled_data(df):
    """
    Create labeled training data from raw OHLCV dataframe
    Returns DataFrame with features and 'target' column
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    training_samples = []
    prices = []
    volumes = []
    highs = []
    lows = []
    
    for i, row in df.iterrows():
        prices.append(row['close'])
        volumes.append(row['volume'])
        highs.append(row['high'])
        lows.append(row['low'])
        
        if len(prices) < 30:
            continue
        
        # Calculate features
        features = calculate_indicators(prices, volumes, highs, lows)
        if features is None:
            continue
        
        # Add time features
        features['hour'] = row['timestamp'].hour
        features['minute'] = row['timestamp'].minute
        
        # Look ahead to create label
        if i + 5 < len(df):  # Predict 5 bars ahead
            current_price = row['close']
            future_price = df.iloc[i + 5]['close']
            price_change_pct = (future_price - current_price) / current_price
            
            # Label: 0=DOWN, 1=NEUTRAL, 2=UP
            threshold = 0.001  # 0.1% threshold
            if price_change_pct > threshold:
                target = 2  # UP
            elif price_change_pct < -threshold:
                target = 0  # DOWN
            else:
                target = 1  # NEUTRAL
            
            features['target'] = target
            training_samples.append(features)
            
    return pd.DataFrame(training_samples)

def generate_prediction_data():
    print("="*80)
    print("GENERATING PRICE PREDICTION TRAINING DATA")
    print("="*80)
    
    # Load data
    conn = sqlite3.connect('market_data.db')
    query = "SELECT * FROM intraday_ticker_data WHERE ticker = 'QQQ' ORDER BY timestamp ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} rows")
    
    train_df = create_labeled_data(df)
    
    # Save to CSV
    if not train_df.empty:
        train_df.to_csv('prediction_training_data.csv', index=False)
        
        print(f"\n✓ Saved {len(train_df)} samples to prediction_training_data.csv")
        print(f"\nTarget Distribution:")
        print(f"  DOWN: {(train_df['target'] == 0).sum()} ({(train_df['target'] == 0).mean():.1%})")
        print(f"  NEUTRAL: {(train_df['target'] == 1).sum()} ({(train_df['target'] == 1).mean():.1%})")
        print(f"  UP: {(train_df['target'] == 2).sum()} ({(train_df['target'] == 2).mean():.1%})")
    else:
        print("No training samples generated")
    
    print("="*80)

if __name__ == "__main__":
    generate_prediction_data()
