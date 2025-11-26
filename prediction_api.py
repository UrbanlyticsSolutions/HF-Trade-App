"""
Real-Time Prediction API Server
Serves live ML predictions from the trained model
"""
import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from price_prediction_model import PricePredictionModel
from clients.fmp_stable_client import FMPStableClient
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Initialize model
model = PricePredictionModel(model_path="models/price_predictor.pkl")
last_model_load_time = 0

def check_and_reload_model():
    """Check if model file has changed and reload if necessary"""
    global last_model_load_time
    try:
        if os.path.exists(model.model_path):
            mtime = os.path.getmtime(model.model_path)
            if last_model_load_time == 0:
                last_model_load_time = mtime
            
            if mtime > last_model_load_time:
                print(f"🔄 Model file changed, reloading... (New mtime: {mtime})")
                model.load_model()
                last_model_load_time = mtime
                print("✅ Model reloaded successfully")
    except Exception as e:
        print(f"⚠️ Error checking model reload: {e}")

try:
    model.load_model()
    if os.path.exists(model.model_path):
        last_model_load_time = os.path.getmtime(model.model_path)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️  Model not found, will need training: {e}")

# Initialize FMP client for real-time data
api_key = os.getenv("FMP_API_KEY")
fmp_client = FMPStableClient(api_key=api_key) if api_key else None

SYMBOL = "QQQ"  # Default symbol


def safe_float(value, fallback):
    """Convert value to float, falling back when NaN/None/inf."""
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return float(fallback)
        return result
    except (TypeError, ValueError):
        return float(fallback)


def calculate_indicators(df):
    """Calculate technical indicators for the dataframe"""
    # Price changes
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_3'] = df['close'].pct_change(3)
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_volatility'] = df['close'].rolling(window=20).std()
    
    # Moving averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['price_to_ma5'] = df['close'] / df['ma_5']
    df['price_to_ma20'] = df['close'] / df['ma_20']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Volume
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    df['volume_change'] = df['volume'].pct_change(1)
    
    # ADX (simplified)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['adx'] = tr.rolling(window=14).mean() / df['close'] * 100
    df['trend_strength'] = df['adx']
    
    # Time features
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['minute'] = pd.to_datetime(df['timestamp']).dt.minute
    
    return df


def get_latest_data_from_db(limit=100):
    """Get latest intraday data from the local database"""
    try:
        conn = sqlite3.connect('market_data.db')
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM intraday_ticker_data
            WHERE ticker = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(SYMBOL, limit))
        conn.close()

        if df.empty:
            return None

        df = df.sort_values('timestamp').reset_index(drop=True)
        # Note: Indicators are calculated later in get_prediction after merging live quote
        return df
    except Exception as e:
        print(f"Error reading from database: {e}")
        return None


@app.route('/api/prediction', methods=['GET'])
def get_prediction():
    """Get current prediction"""
    try:
        # Check for model updates
        check_and_reload_model()
        
        # Get latest data from DB
        df = get_latest_data_from_db(limit=100)
        
        # Fetch real-time quote
        real_time_quote = None
        if fmp_client:
            try:
                quotes = fmp_client.quote(SYMBOL)
                if quotes:
                    q = quotes[0]
                    # Convert timestamp to string
                    ts = datetime.fromtimestamp(q['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    
                    real_time_quote = {
                        'timestamp': ts,
                        'open': float(q.get('open', 0)),
                        'high': float(q.get('dayHigh', 0)),
                        'low': float(q.get('dayLow', 0)),
                        'close': float(q.get('price', 0)),
                        'volume': int(q.get('volume', 0))
                    }
            except Exception as e:
                print(f"Error fetching real-time quote: {e}")

        if df is None and real_time_quote is None:
             return jsonify({'error': 'No data available'}), 500

        # Append real-time quote if available
        if real_time_quote:
            new_row = pd.DataFrame([real_time_quote])
            if df is not None:
                # Check if the last row has the same minute timestamp
                last_ts = pd.to_datetime(df.iloc[-1]['timestamp'])
                new_ts = pd.to_datetime(real_time_quote['timestamp'])
                
                if last_ts.minute == new_ts.minute and last_ts.hour == new_ts.hour and last_ts.day == new_ts.day:
                    # Update last row
                    df.iloc[-1] = new_row.iloc[0]
                else:
                    # Append new row
                    df = pd.concat([df, new_row], ignore_index=True)
            else:
                df = new_row

        if len(df) < 50:
            return jsonify({'error': 'Insufficient data'}), 500
            
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Get latest row
        latest = df.iloc[-1]
        
        # Extract features
        features = model.extract_features(latest.to_dict())
        
        # Make prediction
        prediction_class = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # Map prediction to direction
        direction_map = {0: 'DOWN', 1: 'NEUTRAL', 2: 'UP'}
        direction = direction_map.get(prediction_class, 'NEUTRAL')
        confidence = float(max(probabilities) * 100)
        
        # Calculate trend
        ma20 = safe_float(latest.get('ma_20', latest['close']), latest['close'])
        ma50_raw = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else latest['close']
        ma50 = safe_float(ma50_raw, latest['close'])
        trend = 'BULLISH' if ma20 > ma50 else 'BEARISH'
        adx_value = safe_float(latest.get('adx', 0), 0)
        
        # Price target (simple: current + predicted direction)
        current_price = float(latest['close'])
        if direction == 'UP':
            target_price = current_price * 1.002  # +0.2%
        elif direction == 'DOWN':
            target_price = current_price * 0.998  # -0.2%
        else:
            target_price = current_price
        
        return jsonify({
            'timestamp': latest['timestamp'],
            'current_price': current_price,
            'prediction': {
                'direction': direction,
                'confidence': confidence,
                'probabilities': {
                    'down': float(probabilities[0] * 100),
                    'neutral': float(probabilities[1] * 100),
                    'up': float(probabilities[2] * 100)
                }
            },
            'trend': {
                'direction': trend,
                'adx': adx_value,
                'strength': 'Strong' if adx_value > 25 else 'Moderate' if adx_value > 15 else 'Weak'
            },
            'target_price': target_price,
            'indicators': {
                'rsi': safe_float(latest.get('rsi', 50), 50),
                'macd': safe_float(latest.get('macd', 0), 0),
                'ma_20': ma20,
                'ma_50': ma50
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chart_data', methods=['GET'])
def get_chart_data():
    """Get chart data for visualization"""
    try:
        minutes = request.args.get('minutes', default=None, type=int)
        limit = 390  # default to full trading day
        if minutes is not None:
            limit = max(1, min(minutes, 1000))  # guardrail to keep response small

        df = get_latest_data_from_db(limit=limit)
        
        if df is None:
            return jsonify({'error': 'No data available'}), 500
        
        # Prepare chart data
        chart_data = []
        for idx, row in df.iterrows():
            # Make prediction for each point
            try:
                features = model.extract_features(row.to_dict())
                pred_class = model.predict(features)
                
                # Estimate predicted price
                if pred_class == 2:  # UP
                    pred_price = row['close'] * 1.002
                elif pred_class == 0:  # DOWN
                    pred_price = row['close'] * 0.998
                else:  # NEUTRAL
                    pred_price = row['close']
            except:
                pred_price = row['close']
            
            ma20 = safe_float(row.get('ma_20', row['close']), row['close'])
            rsi = safe_float(row.get('rsi', 50), 50)
            adx = safe_float(row.get('adx', 0), 0)

            chart_data.append({
                'timestamp': row['timestamp'],
                'price': safe_float(row['close'], row['close']),
                'ma20': ma20,
                'prediction': safe_float(pred_price, row['close']),
                'rsi': rsi,
                'adx': adx
            })
        
        return jsonify({'data': chart_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/historical_data', methods=['GET'])
def get_historical_data():
    """Get historical data with multiple days/weeks of data"""
    try:
        # Get timeframe parameter (default to 5 days)
        days = int(request.args.get('days', 5))
        limit = min(days * 390, 10000)  # Cap at 10k bars to prevent overload
        
        df = get_latest_data_from_db(limit=limit)
        
        if df is None or df.empty:
            return jsonify({'error': 'No data available'}), 500
        
        # For longer timeframes, aggregate to avoid too many points
        if len(df) > 1000:
            # Resample to 5-min bars
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df_resampled = df.resample('5T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            df_resampled = df_resampled.reset_index()
            df_resampled['timestamp'] = df_resampled['timestamp'].astype(str)
            df = calculate_indicators(df_resampled)
        
        ma50_series = df['close'].rolling(window=50).mean().fillna(df['close'])

        # Prepare historical data
        historical_data = []
        for idx, row in df.iterrows():
            # Make prediction for each point
            try:
                features = model.extract_features(row.to_dict())
                pred_class = model.predict(features)
                
                # Estimate predicted price
                if pred_class == 2:  # UP
                    pred_price = row['close'] * 1.002
                elif pred_class == 0:  # DOWN
                    pred_price = row['close'] * 0.998
                else:  # NEUTRAL
                    pred_price = row['close']
            except:
                pred_price = row['close']
            
            ma20 = safe_float(row.get('ma_20', row['close']), row['close'])
            ma50 = safe_float(ma50_series.iloc[idx], row['close'])
            rsi = safe_float(row.get('rsi', 50), 50)
            macd = safe_float(row.get('macd', 0), 0)
            adx = safe_float(row.get('adx', 0), 0)

            historical_data.append({
                'timestamp': row['timestamp'],
                'open': safe_float(row['open'], row['close']),
                'high': safe_float(row['high'], row['close']),
                'low': safe_float(row['low'], row['close']),
                'close': safe_float(row['close'], row['close']),
                'volume': safe_float(row['volume'], 0),
                'ma20': ma20,
                'ma50': ma50,
                'prediction': safe_float(pred_price, row['close']),
                'rsi': rsi,
                'macd': macd,
                'adx': adx
            })
        
        return jsonify({
            'data': historical_data,
            'timeframe': f'{days} days',
            'bars': len(historical_data)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model.model is not None,
        'symbol': SYMBOL
    })


if __name__ == '__main__':
    print(f"🚀 Starting Real-Time Prediction API for {SYMBOL}")
    print(f"📊 Model: {model.model_path}")
    print(f"🌐 API will be available at http://localhost:5000")
    print(f"\nEndpoints:")
    print(f"  - GET /api/prediction - Get current prediction")
    print(f"  - GET /api/chart_data - Get chart data")
    print(f"  - GET /api/health - Health check")
    
    app.run(debug=False, port=5000, host='0.0.0.0')
