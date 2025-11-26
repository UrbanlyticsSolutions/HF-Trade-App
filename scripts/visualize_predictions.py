
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from price_prediction_model import PricePredictionModel

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

def visualize():
    print("="*80)
    print("GENERATING PREDICTION VISUALIZATION")
    print("="*80)
    
    # Load model
    model = PricePredictionModel()
    try:
        model.load_model()
        print("Model loaded successfully.")
    except:
        print("Error: Could not load model. Please train it first.")
        return

    # Load data
    conn = sqlite3.connect('market_data.db')
    query = "SELECT * FROM intraday_ticker_data WHERE ticker = 'QQQ' ORDER BY timestamp DESC LIMIT 200"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) < 50:
        print("Not enough data to visualize.")
        return
        
    # Sort chronological
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Loaded {len(df)} rows. Visualizing last 100...")
    
    # Prepare data for plotting
    plot_data = df.iloc[-100:].copy().reset_index(drop=True)
    
    predictions = []
    valid_indices = []
    
    # We need history for indicators, so we iterate through the full df but only keep predictions for the plot part
    # Actually, let's just calculate indicators for the whole df and then slice
    
    prices = df['close'].values
    volumes = df['volume'].values
    highs = df['high'].values
    lows = df['low'].values
    
    # Generate predictions
    pred_signals = {} # index -> prediction
    
    start_idx = len(df) - 100
    
    for i in range(start_idx, len(df)):
        # Get history up to i
        # We need at least 30 bars history for indicators
        if i < 30: 
            continue
            
        p_slice = prices[:i+1]
        v_slice = volumes[:i+1]
        h_slice = highs[:i+1]
        l_slice = lows[:i+1]
        
        features = calculate_indicators(p_slice, v_slice, h_slice, l_slice)
        if features is None:
            continue
            
        # Add time features
        ts = df.iloc[i]['timestamp']
        features['hour'] = ts.hour
        features['minute'] = ts.minute
        
        # Format for model
        feat_vector = [features.get(k, 0) for k in model.feature_names]
        
        # Predict
        pred = model.predict(feat_vector)
        pred_signals[i - start_idx] = pred # Map to plot index
        
    # Plotting
    plt.figure(figsize=(15, 8))
    plt.plot(plot_data.index, plot_data['close'], label='Close Price', color='black', alpha=0.6)
    
    # Add markers
    # 0: DOWN (Red v), 1: NEUTRAL (Gray .), 2: UP (Green ^)
    
    up_idx = [i for i, p in pred_signals.items() if p == 2]
    down_idx = [i for i, p in pred_signals.items() if p == 0]
    neutral_idx = [i for i, p in pred_signals.items() if p == 1]
    
    plt.scatter(up_idx, plot_data.iloc[up_idx]['close'], marker='^', color='green', s=100, label='Predict UP', zorder=5)
    plt.scatter(down_idx, plot_data.iloc[down_idx]['close'], marker='v', color='red', s=100, label='Predict DOWN', zorder=5)
    # plt.scatter(neutral_idx, plot_data.iloc[neutral_idx]['close'], marker='.', color='gray', s=30, label='Predict NEUTRAL')
    
    plt.title(f"Price Prediction Model - Forecast Horizon: 5 Minutes (QQQ)")
    plt.xlabel("Bar Index (1 min)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = 'prediction_visualization.png'
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    visualize()
