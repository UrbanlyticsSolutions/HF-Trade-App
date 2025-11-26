import sqlite3
import pandas as pd

conn = sqlite3.connect('market_data.db')

# Get all tables
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
print("Available tables:")
print(tables)
print()

# Check intraday data
print("Intraday data sample:")
intraday = pd.read_sql_query("SELECT * FROM intraday_ticker_data WHERE ticker='QQQ' ORDER BY timestamp DESC LIMIT 5", conn)
print(intraday)
print(f"\nTotal intraday records: {pd.read_sql_query('SELECT COUNT(*) FROM intraday_ticker_data WHERE ticker=\"QQQ\"', conn).iloc[0,0]}")
print()

# Check for daily/historical data
try:
    daily = pd.read_sql_query("SELECT * FROM daily_ticker_data WHERE ticker='QQQ' ORDER BY date DESC LIMIT 5", conn)
    print("Daily data sample:")
    print(daily)
    print(f"\nTotal daily records: {pd.read_sql_query('SELECT COUNT(*) FROM daily_ticker_data WHERE ticker=\"QQQ\"', conn).iloc[0,0]}")
except:
    print("No daily_ticker_data table found")

conn.close()
