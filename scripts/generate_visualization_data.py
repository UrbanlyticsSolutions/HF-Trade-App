"""
Generate Visualization Data
Generates live_data.json for the static visualization page.
"""
import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.market_analyzer import MarketAnalyzer
from core.features import build_feature_frame

SYMBOL = "QQQ"
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualization", "live_data.json")

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

def get_latest_data_from_db(limit=390):
    """Get latest intraday data from the local database"""
    try:
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'market_data.db')
        conn = sqlite3.connect(db_path)
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

def generate_data():
    print(f"Generating visualization data for {SYMBOL}...")
    
    # 1. Get Data
    df = get_latest_data_from_db(limit=390) # 1 day of data
    if df is None or df.empty:
        print("Error: No data found in database.")
        return False

    # 2. Analyze
    analyzer = MarketAnalyzer(model_path="models/price_predictor.pkl")
    
    # We need to build features to get indicators like RSI, ADX, etc.
    df_features = build_feature_frame(df, dropna=False)
    
    # Get the latest analysis (prediction, trend, etc.)
    # Note: analyzer.analyze() expects a dataframe with raw OHLCV data, it calls build_feature_frame internally
    # but we also need the full feature frame for the chart history.
    # So we'll use the analyzer for the *latest* prediction and our own df_features for history.
    
    analysis_result = analyzer.analyze(df)
    
    if not analysis_result:
        print("Error: Analysis failed.")
        return False

    # 3. Build Chart Data
    chart_data = []
    
    # Simple prediction simulation for history (since we don't store all historical predictions in this script)
    # In a real system, you'd fetch these from a predictions table.
    calculated_predictions = []
    for idx, row in df_features.iterrows():
        try:
            # This is a simplified replay of the model logic for visualization purposes
            # Ideally this should match prediction_api.py logic exactly or fetch from DB
            features = analyzer.model.extract_features(row.to_dict())
            pred_class = analyzer.model.predict(features)
            if pred_class == 2: pred = row['close'] * 1.002
            elif pred_class == 0: pred = row['close'] * 0.998
            else: pred = row['close']
            calculated_predictions.append(pred)
        except:
            calculated_predictions.append(row['close'])

    for i in range(len(df_features)):
        row = df_features.iloc[i]
        
        item = {
            'timestamp': format_timestamp(row['timestamp']),
            'price': safe_float(row['close'], row['close']),
            'ma20': safe_float(row.get('ma_20', row['close']), row['close']),
            'ma50': safe_float(row.get('ma_50', row['close']), row['close']),
            'rsi': safe_float(row.get('rsi', 50), 50),
            'adx': safe_float(row.get('adx', 0), 0),
            'prediction': None
        }
        
        # Lagged prediction (prediction made at T-1 applies to T)
        if i > 0:
            item['prediction'] = calculated_predictions[i-1]
            
        chart_data.append(item)

    # Add future point (T+1)
    if len(df) > 0:
        last_row = df.iloc[-1]
        last_ts = pd.to_datetime(last_row['timestamp'])
        future_ts = last_ts + timedelta(minutes=5) # Assuming 5m bars
        future_pred = calculated_predictions[-1]
        
        chart_data.append({
            'timestamp': format_timestamp(future_ts),
            'price': None,
            'ma20': None,
            'ma50': None,
            'rsi': None,
            'adx': None,
            'prediction': future_pred
        })

    # 4. Construct Final JSON
    output_data = {
        "prediction": analysis_result['prediction'],
        "trend": analysis_result['trend'],
        "current_price": analysis_result['current_price'],
        "target_price": analysis_result['target_price'],
        "timestamp": format_timestamp(analysis_result['timestamp']),
        "chart_data": {
            "data": chart_data
        }
    }

    # 5. Write to File
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Successfully wrote data to {OUTPUT_FILE}")
        return True
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False

if __name__ == "__main__":
    generate_data()
