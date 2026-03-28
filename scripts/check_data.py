"""Check what data we have and what's missing for 2026 backtest."""
import sqlite3
conn = sqlite3.connect('data/market_data.db')

# Time resolution sample (use indexed date column)
r2 = conn.execute("SELECT DISTINCT time FROM options_intraday WHERE underlying='SPY' AND date='2026-01-02' ORDER BY time LIMIT 20").fetchall()
print('Sample times on 2026-01-02:', [x[0] for x in r2])

# Count per date in 2026
r3 = conn.execute("SELECT date, COUNT(*) FROM options_intraday WHERE underlying='SPY' AND date>='2026-01-01' GROUP BY date ORDER BY date").fetchall()
print('\n2026 option records per date:')
for date, cnt in r3:
    print(f'  {date}: {cnt} records')

# Same for underlying
r4 = conn.execute("SELECT date, COUNT(*) FROM intraday_5min_data WHERE ticker='SPY' AND date>='2026-01-01' GROUP BY date ORDER BY date").fetchall()
print('\n2026 underlying 5min records per date:')
for date, cnt in r4:
    print(f'  {date}: {cnt} records')

conn.close()
