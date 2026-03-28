"""Query DB for today's trades and orders."""
import sqlite3

conn = sqlite3.connect('/app/data/live_0dte_trades.db')
conn.row_factory = sqlite3.Row

print("=== TRADES TABLE SCHEMA ===")
for r in conn.execute("PRAGMA table_info(trades)"):
    print(dict(r))

print()
print("=== TRADES (March 10) ===")
for r in conn.execute("SELECT * FROM trades WHERE date(entry_time) >= '2026-03-10' ORDER BY id"):
    print(dict(r))

print()
print("=== CURRENT_POSITIONS ===")
for r in conn.execute("SELECT * FROM current_positions"):
    print(dict(r))

print()
print("=== ORDERS (March 10) ===")
for r in conn.execute("""
    SELECT id, trade_id, order_id, action, symbol, quantity, order_type,
           limit_price, status, filled_quantity, avg_fill_price, created_at
    FROM orders WHERE date(created_at) >= '2026-03-10' ORDER BY id
"""):
    print(dict(r))

conn.close()
