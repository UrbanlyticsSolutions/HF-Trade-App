
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def generate_chart():
    print("="*80)
    print("GENERATING INTERACTIVE PREDICTION CHART")
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
    # Load ALL data for QQQ
    query = "SELECT * FROM intraday_ticker_data WHERE ticker = 'QQQ' ORDER BY timestamp ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) < 50:
        print("Not enough data to visualize.")
        return
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded {len(df)} rows.")
    
    # Calculate MAs for plotting
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # Generate predictions
    predictions = []
    
    prices = df['close'].values
    volumes = df['volume'].values
    highs = df['high'].values
    lows = df['low'].values
    
    pred_signals = {} # index -> prediction
    
    print("Generating predictions...")
    # We can skip the first 30 rows as we need history
    for i in range(30, len(df)):
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
        pred_signals[i] = pred
        
    print(f"Generated {len(pred_signals)} predictions.")
    
    # Create Plotly Figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('Price & Predictions', 'Volume'),
                        row_heights=[0.7, 0.3])

    # Candlestick
    fig.add_trace(go.Candlestick(x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC'), row=1, col=1)

    # Moving Averages
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MA5'], line=dict(color='orange', width=1), name='MA 5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MA20'], line=dict(color='blue', width=1), name='MA 20'), row=1, col=1)

    # Predictions
    # 0: DOWN, 1: NEUTRAL, 2: UP
    up_indices = [i for i, p in pred_signals.items() if p == 2]
    down_indices = [i for i, p in pred_signals.items() if p == 0]
    
    if up_indices:
        fig.add_trace(go.Scatter(
            x=df.iloc[up_indices]['timestamp'],
            y=df.iloc[up_indices]['low'] * 0.9995, # Slightly below candle
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='green'),
            name='Predict UP'
        ), row=1, col=1)
        
    if down_indices:
        fig.add_trace(go.Scatter(
            x=df.iloc[down_indices]['timestamp'],
            y=df.iloc[down_indices]['high'] * 1.0005, # Slightly above candle
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color='red'),
            name='Predict DOWN'
        ), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color='gray'), row=2, col=1)

    # Layout
    fig.update_layout(
        title='Price Prediction Model Analysis (QQQ)',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark'
    )

    output_file = 'interactive_prediction_chart.html'
    fig.write_html(output_file)
    print(f"Interactive chart saved to {output_file}")

if __name__ == "__main__":
    generate_chart()
