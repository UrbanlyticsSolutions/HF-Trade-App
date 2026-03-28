#!/usr/bin/env python3
"""Quick query of live trading DB state."""
import sqlite3
import sys
from datetime import date

db_path = sys.argv[1] if len(sys.argv) > 1 else "/app/data/live_0dte_trades.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

today = date.today().isoformat()

print("=== ALL TRADES TODAY ===")
cur.execute(
    "SELECT id, symbol, quantity, entry_price, entry_time, exit_price, exit_time, status, pnl, notes "
    "FROM trades WHERE date(entry_time) >= ? ORDER BY entry_time",
    (today,)
)
for r in cur.fetchall():
    d = dict(r)
    print(f"  #{d['id']} {d['symbol']} qty={d['quantity']} entry=${d['entry_price']} "
          f"exit=${d['exit_price']} status={d['status']} pnl=${d['pnl']} "
          f"entry_t={d['entry_time'][:19]} exit_t={str(d['exit_time'] or '')[:19]}")
    if d['notes']:
        print(f"       notes: {d['notes'][:120]}")

print("\n=== OPEN TRADES ===")
cur.execute(
    "SELECT id, symbol, quantity, entry_price, entry_time, status, entry_order_id, notes "
    "FROM trades WHERE status='open' ORDER BY entry_time"
)
rows = cur.fetchall()
if not rows:
    print("  (none)")
for r in rows:
    d = dict(r)
    print(f"  #{d['id']} {d['symbol']} qty={d['quantity']} entry=${d['entry_price']} "
          f"order={d['entry_order_id']} entry_t={d['entry_time'][:19]}")
    if d['notes']:
        print(f"       notes: {d['notes'][:120]}")

print("\n=== CURRENT POSITIONS TABLE ===")
cur.execute("SELECT * FROM current_positions")
rows = cur.fetchall()
if not rows:
    print("  (none)")
for r in rows:
    d = dict(r)
    print(f"  {d['symbol']} qty={d['quantity']} avg_cost=${d['avg_cost']} "
          f"price=${d['current_price']} mkt_val=${d['market_value']} "
          f"unrealized=${d['unrealized_pnl']}")

conn.close()
