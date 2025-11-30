"""
Fetch Intraday Data
Fetches 5-minute intraday data from FMP and stores it in the local SQLite database.
"""
import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients.fmp_stable_client import FMPStableClient

def init_db():
    """Initialize the database table if it doesn't exist"""
    conn = sqlite3.connect('market_data.db')
    cursor = conn.cursor()
    
    # Create table matching prediction_api.py expectations
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS intraday_ticker_data (
        timestamp TEXT,
        ticker TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        PRIMARY KEY (timestamp, ticker)
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized.")

def fetch_and_store_data(symbol="QQQ", days=5):
    """Fetch data and store in DB"""
    load_dotenv()
    api_key = os.getenv("FMP_API_KEY")
    
    if not api_key:
        print("Error: FMP_API_KEY not found.")
        return

    client = FMPStableClient(api_key=api_key)
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    print(f"Fetching 5min data for {symbol} from {start_date} to {end_date}...")
    
    try:
        # Fetch 5min data
        data = client.historical_chart_5min(symbol, from_date=start_date, to_date=end_date)
        
        if not data:
            print("No data returned from API.")
            return

        print(f"Retrieved {len(data)} records.")
        
        # Prepare for insertion
        records = []
        for row in data:
            records.append((
                row['date'], # timestamp
                symbol,      # ticker
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume']
            ))
            
        # Insert into DB
        conn = sqlite3.connect('market_data.db')
        cursor = conn.cursor()
        
        cursor.executemany('''
        INSERT OR REPLACE INTO intraday_ticker_data 
        (timestamp, ticker, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', records)
        
        conn.commit()
        conn.close()
        print(f"Successfully stored {len(records)} records in market_data.db")
        
    except Exception as e:
        print(f"Error fetching/storing data: {e}")

if __name__ == "__main__":
    init_db()
    fetch_and_store_data()
