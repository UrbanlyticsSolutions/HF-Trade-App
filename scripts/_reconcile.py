"""
Full reconciliation: query ALL trades in DB, get IBKR executions,
identify real vs phantom, compute correct PNL.
"""
import sys, os
sys.path.insert(0, '/app')
os.chdir('/app')
from dotenv import load_dotenv
load_dotenv()

import sqlite3
import time
from datetime import datetime

# ── 1. Query ALL trades from DB ──────────────────────────────────────
print("=" * 70)
print("STEP 1: ALL TRADES IN DB (today)")
print("=" * 70)

conn = sqlite3.connect('/app/data/live_0dte_trades.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Show table schema first
cur.execute("PRAGMA table_info(trades)")
cols = [r['name'] for r in cur.fetchall()]
print(f"Columns: {cols}")

cur.execute("SELECT * FROM trades WHERE date(entry_time) >= '2026-03-18' ORDER BY id")
all_trades = [dict(r) for r in cur.fetchall()]
print(f"\nTotal DB trades today: {len(all_trades)}")
for t in all_trades:
    print(f"\n  ID={t['id']} symbol={t['symbol']} qty={t.get('quantity',0)} "
          f"entry=${t.get('entry_price',0)} exit=${t.get('exit_price',0)} "
          f"pnl=${t.get('pnl',0)} status={t.get('status','?')}")
    print(f"    entry_t={str(t.get('entry_time',''))[:25]} "
          f"exit_t={str(t.get('exit_time',''))[:25]}")
    print(f"    entry_order={t.get('entry_order_id','')} "
          f"exit_order={t.get('exit_order_id','')}")
    notes = t.get('notes', '') or ''
    print(f"    notes: {notes[:200]}")

# ── 2. Get IBKR executions (current TWS session) ────────────────────
print("\n" + "=" * 70)
print("STEP 2: IBKR LIVE EXECUTIONS")
print("=" * 70)

from clients.ibkr_adapter import IBKRAdapter

adapter = IBKRAdapter(client_id=88)
try:
    adapter.connect()
    time.sleep(1)

    # Get executions
    execs = adapter.get_executions()
    print(f"IBKR executions in current session: {len(execs)}")
    for ex in execs:
        print(f"  {ex.get('side')} {ex.get('shares')}x {ex.get('trade_symbol',ex.get('localSymbol','?'))} "
              f"@ ${ex.get('price',0):.2f} at {ex.get('time','')} "
              f"order={ex.get('order_id','')} exec_id={ex.get('exec_id','')}")

    # Get positions
    print(f"\nIBKR positions:")
    positions = adapter.get_account_positions()
    if isinstance(positions, list):
        for p in positions:
            sym = p.get('symbol', '?')
            qty = p.get('openQuantity', p.get('quantity', 0))
            avg = p.get('averageEntryPrice', p.get('avg_cost', 0))
            print(f"  {sym} qty={qty} avg_cost=${avg}")
    elif isinstance(positions, dict):
        for sym, p in positions.items():
            print(f"  {sym}: {p}")
    if not positions:
        print("  (none)")

    # Get account summary for realized PNL
    print(f"\nIBKR account:")
    summary = adapter.get_account_summary()
    for k in ['NetLiquidation', 'TotalCashValue', 'RealizedPnL', 'UnrealizedPnL']:
        if k in summary:
            print(f"  {k}: {summary[k]}")

    adapter.disconnect()
except Exception as e:
    print(f"IBKR connection error: {e}")
    try:
        adapter.disconnect()
    except:
        pass

# ── 3. Read gcloud_trades.csv for imported IBKR fills ───────────────
print("\n" + "=" * 70)
print("STEP 3: IBKR IMPORTED FILLS (from gcloud_trades.csv)")
print("=" * 70)

import csv
csv_path = '/app/data/gcloud_trades.csv'
try:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        ibkr_imports = []
        for row in reader:
            if '2026-03-18' in str(row.get('entry_time', '')):
                ibkr_imports.append(row)
        
        print(f"Total rows for today: {len(ibkr_imports)}")
        for i, row in enumerate(ibkr_imports):
            src = "IBKR-IMPORT" if "imported:ibkr" in str(row.get('notes','')) else \
                  "RECOVERED" if "recovered" in str(row.get('strategy_name','')) else \
                  "LIVE-ENGINE"
            print(f"\n  [{src}] {row['symbol']} qty={row['quantity']} "
                  f"entry=${row['entry_price']} exit=${row['exit_price']} "
                  f"pnl=${row.get('pnl',0)} status={row['status']}")
            print(f"    entry_t={str(row['entry_time'])[:30]} exit_t={str(row['exit_time'])[:30]}")
            print(f"    notes: {str(row.get('notes',''))[:200]}")
except FileNotFoundError:
    print("  gcloud_trades.csv not found")

# ── 4. Cross-reference: identify unique real IBKR round-trips ───────
print("\n" + "=" * 70)
print("STEP 4: ACTUAL IBKR ROUND-TRIP ANALYSIS")
print("=" * 70)

# Parse IBKR imported fills (these came from reqExecutions before restart)
ibkr_fills = []
for row in ibkr_imports:
    if "imported:ibkr" in str(row.get('notes', '')):
        ibkr_fills.append(row)
    if "exec_id=" in str(row.get('notes', '')):
        ibkr_fills.append(row)

print(f"IBKR-sourced fills: {len(ibkr_fills)}")
for f in ibkr_fills:
    print(f"  {f['symbol']} qty={f['quantity']} "
          f"entry=${f['entry_price']} @{str(f['entry_time'])[:25]} "
          f"exit=${f['exit_price']} @{str(f['exit_time'])[:25]} "
          f"pnl=${f.get('pnl',0)}")

# ── 5. Compute correct PNL from NLV delta ───────────────────────────
print("\n" + "=" * 70)
print("STEP 5: NLV-BASED P&L CHECK")
print("=" * 70)
print("  NLV before trading (Mar 16): $971,747.82")
print("  NLV now (from IBKR):         see above")
print("  Note: 35x P664 position still shows in IBKR at $1.105 avg")
print("  P664 expired ITM (strike 664, SPY closed at 661.56)")
print("  Intrinsic = $2.44 per contract")
print("  35 x $2.44 x 100 = $8,540 assignment value")

conn.close()
