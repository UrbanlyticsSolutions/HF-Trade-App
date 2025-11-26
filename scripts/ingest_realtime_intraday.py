"""Continuous intraday updater that keeps market_data.db current in near real time."""
import argparse
import os
import sys
import time
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

from dotenv import load_dotenv

# Ensure repository root is importable when running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Ensure scripts dir is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clients.fmp_stable_client import FMPStableClient
from clients.database import MarketDatabase
from price_prediction_model import PricePredictionModel
from generate_prediction_data import create_labeled_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously refresh intraday bars for a ticker")
    parser.add_argument("--symbol", default="QQQ", help="Ticker symbol to ingest")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between refresh calls")
    parser.add_argument("--lookback-minutes", type=int, default=60,
                        help="Only reinsert rows newer than this many minutes")
    parser.add_argument("--iterations", type=int, default=0,
                        help="Number of refresh loops (0 = run forever)")
    return parser.parse_args()


def filter_recent_rows(rows, lookback_minutes, last_timestamp):
    if not rows:
        return []
    cutoff = datetime.now() - timedelta(minutes=lookback_minutes)
    filtered = []
    for row in rows:
        ts = row.get("date")
        if not ts:
            continue
        if last_timestamp and ts <= last_timestamp:
            continue
        try:
            ts_dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if ts_dt >= cutoff:
            filtered.append(row)
    return filtered


def get_latest_timestamp(db, symbol):
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT MAX(timestamp) FROM intraday_ticker_data WHERE ticker = ?",
        (symbol.upper(),)
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None


def refresh_once(client, db, symbol, lookback_minutes):
    today = datetime.now().strftime("%Y-%m-%d")
    last_timestamp = get_latest_timestamp(db, symbol)
    try:
        data = client.historical_chart_1min(symbol, today, today)
    except Exception as exc:
        print(f"[ERROR] Failed to fetch intraday data: {exc}")
        return 0

    recent_rows = filter_recent_rows(data, lookback_minutes, last_timestamp)
    if not recent_rows:
        # print("[WARN] No recent rows returned from FMP") # Reduce noise
        return 0

    recent_rows = sorted(recent_rows, key=lambda row: row.get("date"))
    db.insert_intraday_data(symbol, recent_rows)
    print(f"[INFO] Upserted {len(recent_rows)} rows ending at {recent_rows[-1]['date']}")
    return len(recent_rows)


def retrain_model(symbol, db_path="market_data.db"):
    """Retrain the price prediction model"""
    print(f"\n[TRAIN] Starting model retraining for {symbol}...")
    try:
        # Load data
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM intraday_ticker_data WHERE ticker = ? ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()
        
        if len(df) < 500:
            print("[TRAIN] Not enough data to retrain (need 500+ rows)")
            return

        # Create labeled data
        train_df = create_labeled_data(df)
        
        if train_df.empty:
            print("[TRAIN] No labeled data created")
            return

        # Prepare X and y
        model = PricePredictionModel(model_path="models/price_predictor.pkl")
        
        # We need to convert train_df features to X, y
        feature_names = model.feature_names
        X = train_df[feature_names].values
        y = train_df['target'].values
        
        # Train
        print(f"[TRAIN] Training on {len(X)} samples...")
        model.train(X, y)
        
        # Save
        model.save_model()
        print("[TRAIN] Model retrained and saved successfully\n")
        
    except Exception as e:
        print(f"[TRAIN] Error retraining model: {e}")


def main():
    args = parse_args()
    load_dotenv()
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise SystemExit("FMP_API_KEY missing; set it in .env")

    client = FMPStableClient(api_key)
    db = MarketDatabase()
    loop = 0
    new_bars_count = 0
    RETRAIN_INTERVAL = 100

    print(f"Starting intraday refresher for {args.symbol} (interval={args.interval}s)")
    print(f"Retraining model every {RETRAIN_INTERVAL} new bars")
    
    try:
        while args.iterations == 0 or loop < args.iterations:
            loop += 1
            inserted = refresh_once(client, db, args.symbol, args.lookback_minutes)
            
            if inserted > 0:
                new_bars_count += inserted
                print(f"[INFO] New bars count: {new_bars_count}/{RETRAIN_INTERVAL}")
                
                if new_bars_count >= RETRAIN_INTERVAL:
                    retrain_model(args.symbol, db.db_path)
                    new_bars_count = 0
            
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopping intraday refresher...")
    finally:
        db.close()


if __name__ == "__main__":
    main()
