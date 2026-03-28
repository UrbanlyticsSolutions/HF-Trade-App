"""Delete phantom STK (stock) trades from the live_0dte_trades.db.
These were imported by trade_sync.py before the OPT-only filter was added.
"""
import sqlite3
import os

DB_PATHS = [
    "/app/data/live_0dte_trades.db",
    "/app/data/live_trades.db",
    "/app/data/trades.db",
]

for db_path in DB_PATHS:
    if not os.path.exists(db_path):
        print(f"[SKIP] {db_path} not found")
        continue

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Find tables
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if "trades" not in tables:
        print(f"[SKIP] {db_path} has no 'trades' table")
        conn.close()
        continue

    # Count phantom STK trades (trade_type='stock' OR notes contain 'imported:ibkr_flex' AND symbol is plain 'SPY')
    cur.execute("""
        SELECT COUNT(*) FROM trades 
        WHERE (trade_type = 'stock') 
           OR (option_type IS NULL OR option_type = '') 
              AND (notes LIKE '%imported:ibkr_flex%')
              AND (symbol = 'SPY')
    """)
    count = cur.fetchone()[0]
    print(f"\n[DB] {db_path}: Found {count} phantom STK trades to delete")

    if count > 0:
        # Show what we're deleting
        cur.execute("""
            SELECT id, symbol, trade_type, option_type, action, quantity, pnl, entry_time, notes
            FROM trades 
            WHERE (trade_type = 'stock') 
               OR (option_type IS NULL OR option_type = '') 
                  AND (notes LIKE '%imported:ibkr_flex%')
                  AND (symbol = 'SPY')
            ORDER BY id DESC
            LIMIT 5
        """)
        print("Sample rows being deleted:")
        for row in cur.fetchall():
            print(f"  id={row[0]}, sym={row[1]}, type={row[2]}, opt_type={row[3]}, qty={row[5]}, pnl={row[6]}")

        # Delete them
        cur.execute("""
            DELETE FROM trades 
            WHERE (trade_type = 'stock') 
               OR (option_type IS NULL OR option_type = '') 
                  AND (notes LIKE '%imported:ibkr_flex%')
                  AND (symbol = 'SPY')
        """)
        deleted = cur.rowcount
        conn.commit()
        print(f"  => Deleted {deleted} phantom STK trades")

        # Show remaining
        cur.execute("SELECT COUNT(*) FROM trades")
        remaining = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM trades WHERE status='open'")
        open_count = cur.fetchone()[0]
        print(f"  => Remaining: {remaining} trades total, {open_count} open")

        # Show remaining trade summary
        cur.execute("""
            SELECT id, symbol, option_type, action, quantity, status, pnl, pnl_percent, entry_time
            FROM trades ORDER BY id DESC
        """)
        print("\n  Remaining trades:")
        for row in cur.fetchall():
            print(f"    id={row[0]} {row[1]} {row[2] or ''} {row[3]} qty={row[4]} status={row[5]} pnl={row[6]} pnl%={row[7]}")

    conn.close()

print("\nDone.")
