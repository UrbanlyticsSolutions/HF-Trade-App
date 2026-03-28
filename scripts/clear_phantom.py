"""Clear phantom open trades from DB"""
import sqlite3
import sys
import os

db_path = os.environ.get("DB_PATH", "/app/data/live_0dte_trades.db")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

rows = conn.execute("SELECT * FROM trades WHERE status='open'").fetchall()
print(f"Open trades: {len(rows)}")
for r in rows:
    d = dict(r)
    print(f"  id={d['id']} symbol={d.get('symbol')} qty={d.get('quantity')} "
          f"entry=${d.get('entry_price')} time={d.get('entry_time')} "
          f"order_id={d.get('entry_order_id')}")

if rows and "--fix" in sys.argv:
    for r in rows:
        conn.execute("UPDATE trades SET status='cancelled', notes='PHANTOM: no actual IBKR fill' WHERE id=?", (r['id'],))
    conn.commit()
    print(f"Closed {len(rows)} phantom trade(s)")

conn.close()
