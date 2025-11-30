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

from core.market_analyzer import MarketAnalyzer
from core.features import build_feature_frame
from clients.fmp_stable_client import FMPStableClient
from clients.database import MarketDatabase
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Initialize Analyzer
analyzer = MarketAnalyzer(model_path="models/price_predictor.pkl")

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


def format_timestamp(value):
    """Return a consistent string timestamp for JSON payloads."""
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    return str(value)


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
        return df
    except Exception as e:
        print(f"Error reading from database: {e}")
        return None


@app.route('/api/prediction', methods=['GET'])
def get_prediction():
    """Get current prediction"""
    try:
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
            
        # Use MarketAnalyzer
        result = analyzer.analyze(df)
        
        if result:
            # Format timestamp for JSON
            result['timestamp'] = format_timestamp(result['timestamp'])
            return jsonify(result)
        else:
            return jsonify({'error': 'Analysis failed'}), 500

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
        
        df = build_feature_frame(df, dropna=False)

        # Prepare chart data
        
        # 1. Fetch stored predictions from DB
        db = MarketDatabase()
        stored_preds = db.get_predictions(SYMBOL, limit=len(df) + 10) # Fetch a bit more to be safe
        db.close()
        
        # Create map: timestamp -> prediction
        pred_map = {format_timestamp(p['timestamp']): p['prediction'] for p in stored_preds}

        # 2. Calculate fallback predictions (for old data not in DB)
        calculated_predictions = []
        for idx, row in df.iterrows():
            try:
                features = analyzer.model.extract_features(row.to_dict())
                pred_class = analyzer.model.predict(features)
                if pred_class == 2: pred = row['close'] * 1.002
                elif pred_class == 0: pred = row['close'] * 0.998
                else: pred = row['close']
                calculated_predictions.append(pred)
            except:
                calculated_predictions.append(row['close'])

        # 3. Build final chart list
        chart_data = []
        for i in range(len(df)):
            row = df.iloc[i]
            ts = pd.to_datetime(row['timestamp'])
            
            item = {
                'timestamp': format_timestamp(row['timestamp']),
                'price': safe_float(row['close'], row['close']),
                'ma20': safe_float(row.get('ma_20', row['close']), row['close']),
                'rsi': safe_float(row.get('rsi', 50), 50),
                'adx': safe_float(row.get('adx', 0), 0),
                'prediction': None 
            }
            
            # If we have a prediction from the PREVIOUS bar (i-1), that applies to THIS bar (i)
            if i > 0:
                prev_ts = df.iloc[i-1]['timestamp']
                # Check DB first
                prev_ts_key = format_timestamp(prev_ts)
                if prev_ts_key in pred_map:
                    item['prediction'] = pred_map[prev_ts_key]
                else:
                    # Fallback to calculated
                    item['prediction'] = calculated_predictions[i-1]
            
            chart_data.append(item)

        # 3. Add the FINAL prediction (for T+1)
        if len(df) > 0:
            last_row = df.iloc[-1]
            last_ts = pd.to_datetime(last_row['timestamp'])
            future_ts = last_ts + timedelta(minutes=1)
            future_pred = calculated_predictions[-1]
            
            chart_data.append({
                'timestamp': future_ts.strftime('%Y-%m-%d %H:%M:%S'),
                'price': None, # Future, no price yet
                'ma20': None,
                'rsi': None,
                'adx': None,
                'prediction': future_pred
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
            df = build_feature_frame(df_resampled, dropna=True)
        
        ma50_series = df['close'].rolling(window=50).mean().fillna(df['close'])

        # Prepare historical data
        historical_data = []
        for idx, row in df.iterrows():
            # Make prediction for each point
            try:
                features = analyzer.model.extract_features(row.to_dict())
                pred_class = analyzer.model.predict(features)
                
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
                'timestamp': format_timestamp(row['timestamp']),
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
        'model_loaded': analyzer.model.model is not None,
        'symbol': SYMBOL
    })


if __name__ == '__main__':
    print(f"🚀 Starting Real-Time Prediction API for {SYMBOL}")
    print(f"📊 Model: {analyzer.model.model_path}")
    print(f"🌐 API will be available at http://localhost:5000")
    print(f"\nEndpoints:")
    print(f"  - GET /api/prediction - Get current prediction")
    print(f"  - GET /api/chart_data - Get chart data")
    print(f"  - GET /api/health - Health check")
    
    app.run(debug=False, port=5001, host='0.0.0.0')
