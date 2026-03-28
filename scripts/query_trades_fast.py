#!/usr/bin/env python3
"""Quick targeted query of trade records only (skips large tables)."""
import sqlite3
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "/opt/trading-engine/data/live_0dte_trades.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

cur = conn.cursor()

print("=== TRADE COUNT BY STATUS ===")
cur.execute("SELECT status, count(*) FROM trades GROUP BY status")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]}")

print("\n=== TODAY TRADES (2026-03-18) ===")
cur.execute("""
    SELECT id, symbol, quantity, entry_price, entry_time, exit_price, exit_time, 
           status, pnl, entry_order_id, exit_order_id, notes 
    FROM trades WHERE date(entry_time) >= '2026-03-18' ORDER BY entry_time
""")
rows = cur.fetchall()
if not rows:
    print("  (none)")
for r in rows:
    d = dict(r)
    notes_short = str(d['notes'] or '')[:120]
    print(f"  #{d['id']} {d['symbol']} qty={d['quantity']} "
          f"entry=${d['entry_price']} exit=${d['exit_price']} "
          f"st={d['status']} pnl=${d['pnl']} "
          f"eoid={d['entry_order_id']} xoid={d['exit_order_id']} "
          f"in={str(d['entry_time'])[:19]} out={str(d['exit_time'] or '')[:19]}")
    if notes_short:
        print(f"       {notes_short}")

print("\n=== LAST 15 TRADES ===")
cur.execute("""
    SELECT id, symbol, quantity, entry_price, entry_time, exit_price, exit_time,
           status, pnl, entry_order_id, exit_order_id, notes
    FROM trades ORDER BY id DESC LIMIT 15
""")
for r in cur.fetchall():
    d = dict(r)
    notes_short = str(d['notes'] or '')[:120]
    print(f"  #{d['id']} {d['symbol']} qty={d['quantity']} "
          f"entry=${d['entry_price']} exit=${d['exit_price']} "
          f"st={d['status']} pnl=${d['pnl']} "
          f"eoid={d['entry_order_id']} xoid={d['exit_order_id']} "
          f"in={str(d['entry_time'])[:19]} out={str(d['exit_time'] or '')[:19]}")
    if notes_short:
        print(f"       {notes_short}")

print("\n=== OPEN TRADES ===")
cur.execute("SELECT id, symbol, quantity, entry_price, entry_time, entry_order_id FROM trades WHERE status='open'")
rows = cur.fetchall()
if not rows:
    print("  (none)")
for r in rows:
    d = dict(r)
    print(f"  #{d['id']} {d['symbol']} qty={d['quantity']} entry=${d['entry_price']} oid={d['entry_order_id']} {str(d['entry_time'])[:19]}")

conn.close()
