#!/usr/bin/env python3
"""Query the live trading DB on GCloud for actual trade records."""
import sqlite3
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "/opt/trading-engine/data/live_0dte_trades.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=== TABLES ===")
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
for r in cur.fetchall():
    cur2 = conn.cursor()
    cur2.execute(f"SELECT count(*) FROM [{r[0]}]")
    cnt = cur2.fetchone()[0]
    print(f"  {r[0]}: {cnt} rows")

print("\n=== TRADE COUNT BY STATUS ===")
cur.execute("SELECT status, count(*) as cnt FROM trades GROUP BY status")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]}")

print("\n=== TRADES TODAY (2026-03-24) ===")
cur.execute("""
    SELECT id, symbol, quantity, entry_price, entry_time, exit_price, exit_time,
           status, pnl, entry_order_id, exit_order_id, notes
    FROM trades WHERE date(entry_time) >= '2026-03-24' ORDER BY entry_time
""")
rows = cur.fetchall()
if not rows:
    print("  (none)")
for r in rows:
    d = dict(r)
    print(f"  #{d['id']} {d['symbol']} qty={d['quantity']} "
          f"entry=${d['entry_price']} exit=${d['exit_price']} "
          f"st={d['status']} pnl=${d['pnl']} "
          f"entry_oid={d['entry_order_id']} exit_oid={d['exit_order_id']} "
          f"entry_t={str(d['entry_time'])[:19]} exit_t={str(d['exit_time'] or '')[:19]}")
    if d['notes']:
        print(f"       notes: {str(d['notes'])[:150]}")

print("\n=== LAST 10 TRADES ===")
cur.execute("""
    SELECT id, symbol, quantity, entry_price, entry_time, exit_price, exit_time,
           status, pnl, entry_order_id, exit_order_id, notes
    FROM trades ORDER BY entry_time DESC LIMIT 10
""")
for r in cur.fetchall():
    d = dict(r)
    print(f"  #{d['id']} {d['symbol']} qty={d['quantity']} "
          f"entry=${d['entry_price']} exit=${d['exit_price']} "
          f"st={d['status']} pnl=${d['pnl']} "
          f"entry_oid={d['entry_order_id']} exit_oid={d['exit_order_id']} "
          f"entry_t={str(d['entry_time'])[:19]} exit_t={str(d['exit_time'] or '')[:19]}")
    if d['notes']:
        print(f"       notes: {str(d['notes'])[:150]}")

print("\n=== OPEN TRADES ===")
cur.execute("""
    SELECT id, symbol, quantity, entry_price, entry_time, status, entry_order_id, notes
    FROM trades WHERE status='open' ORDER BY entry_time
""")
rows = cur.fetchall()
if not rows:
    print("  (none)")
for r in rows:
    d = dict(r)
    print(f"  #{d['id']} {d['symbol']} qty={d['quantity']} entry=${d['entry_price']} "
          f"order={d['entry_order_id']} entry_t={str(d['entry_time'])[:19]}")

print("\n=== CURRENT POSITIONS TABLE ===")
try:
    cur.execute("SELECT * FROM current_positions")
    rows = cur.fetchall()
    if not rows:
        print("  (none)")
    for r in rows:
        d = dict(r)
        print(f"  {d}")
except Exception as e:
    print(f"  (error: {e})")

conn.close()
