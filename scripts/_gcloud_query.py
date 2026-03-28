#!/usr/bin/env python3
"""Query trades DB directly (runs inside container)."""
import sqlite3, csv, sys

DB_PATHS = ["/app/data/live_0dte_trades.db", "/app/data/live_trades.db", "/app/data/trades.db"]
for DB_PATH in DB_PATHS:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        if tables:
            print(f"\n=== DB: {DB_PATH} | Tables: {tables} ===")
            break
    except Exception:
        continue

# Find trades table
tbl = "trades" if "trades" in tables else tables[0] if tables else None
if not tbl:
    print("No trade table found!")
    sys.exit(1)

# Get columns
cur.execute(f"PRAGMA table_info({tbl})")
cols = [r[1] for r in cur.fetchall()]
print(f"Columns: {cols}")

# Last 30 trades
cur.execute(f"SELECT * FROM {tbl} ORDER BY rowid DESC LIMIT 30")
rows = cur.fetchall()
if rows:
    w = csv.writer(sys.stdout)
    w.writerow(rows[0].keys())
    for r in rows:
        w.writerow(list(r))

# Summary
try:
    cur.execute(f"SELECT COUNT(*) as total, SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as wins, SUM(pnl) as total_pnl FROM {tbl} WHERE status='closed'")
except Exception:
    cur.execute(f"SELECT COUNT(*) as total, 0 as wins, 0 as total_pnl FROM {tbl}")
s = cur.fetchone()
print(f"\n--- SUMMARY: {s['total']} trades, {s['wins']} wins, PnL=${s['total_pnl'] or 0:.2f} ---")

# Open positions
try:
    cur.execute(f"SELECT * FROM {tbl} WHERE status='open' OR status='OPEN'")
    opens = cur.fetchall()
    if opens:
        print("\n--- OPEN POSITIONS ---")
        for o in opens:
            print(f"  {dict(o)}")
except Exception as e:
    print(f"Open positions check: {e}")

conn.close()
