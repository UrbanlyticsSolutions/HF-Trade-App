#!/usr/bin/env python3
"""Fix 2026-03-18 trades: delete phantoms, fix orphan exit price."""
import sqlite3, os, sys

DB = os.environ.get("TRADE_DB", "/app/data/live_0dte_trades.db")
TODAY = "2026-03-18"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# --- Step 1: Delete phantom trades ---
phantoms = [3, 5]
for tid in phantoms:
    row = cur.execute("SELECT id, symbol, quantity, pnl, notes FROM trades WHERE id=?", (tid,)).fetchone()
    if row:
        print(f"DELETING phantom trade #{row['id']}: {row['symbol']} qty={row['quantity']} fake_pnl=${row['pnl']}")
        print(f"  reason: {row['notes']}")
        cur.execute("DELETE FROM trades WHERE id=?", (tid,))
    else:
        print(f"Trade #{tid} not found (already deleted?)")

# --- Step 2: Fix trade #12 (35x P664 orphan) ---
row12 = cur.execute("SELECT * FROM trades WHERE id=12", ()).fetchone()
if row12:
    strike = 664.0
    spy_close = 661.56
    intrinsic = round(strike - spy_close, 2)  # $2.44 for ITM put
    qty = row12['quantity']
    entry = row12['entry_price']
    new_pnl = round(qty * (intrinsic - entry) * 100, 2)
    
    print(f"\nFIXING trade #12: {row12['symbol']} qty={qty}")
    print(f"  OLD: exit=${row12['exit_price']} pnl=${row12['pnl']}")
    print(f"  NEW: exit=${intrinsic} (ITM intrinsic, strike={strike}, SPY={spy_close})")
    print(f"  NEW pnl: ${new_pnl}")
    
    cur.execute("""
        UPDATE trades 
        SET exit_price=?, pnl=?, pnl_percent=?, 
            notes='Expired ITM, auto-exercise at intrinsic $' || ? || ' (strike=' || ? || ', SPY=' || ? || ')',
            exit_time='2026-03-18T16:00:00'
        WHERE id=12
    """, (intrinsic, new_pnl, round((intrinsic - entry) / entry * 100, 2),
          str(intrinsic), str(strike), str(spy_close)))
else:
    print("Trade #12 not found")

conn.commit()

# --- Step 3: Show corrected trades ---
print("\n" + "=" * 70)
print("CORRECTED TRADES FOR", TODAY)
print("=" * 70)

rows = cur.execute("""
    SELECT id, symbol, quantity, entry_price, exit_price, pnl, status, 
           entry_time, exit_time, notes
    FROM trades 
    WHERE date(entry_time) = ?
    ORDER BY entry_time
""", (TODAY,)).fetchall()

total_pnl = 0
for r in rows:
    pnl = r['pnl'] or 0
    total_pnl += pnl
    print(f"\n  #{r['id']} {r['symbol']} qty={r['quantity']}")
    print(f"     entry=${r['entry_price']} @ {r['entry_time']}")
    print(f"     exit=${r['exit_price']} @ {r['exit_time']}")
    print(f"     PNL: ${pnl:,.2f}  status={r['status']}")
    print(f"     {r['notes']}")

print(f"\n{'=' * 70}")
print(f"TOTAL REALIZED PNL: ${total_pnl:,.2f}")
print(f"{'=' * 70}")

# NLV cross-check
nlv_before = 971747.82
nlv_now = 978757.88
print(f"\nNLV CROSS-CHECK:")
print(f"  NLV Mar 16: ${nlv_before:,.2f}")
print(f"  NLV now:    ${nlv_now:,.2f}")
print(f"  NLV delta:  ${nlv_now - nlv_before:,.2f}")
print(f"  DB PNL:     ${total_pnl:,.2f}")
print(f"  Diff:       ${(nlv_now - nlv_before) - total_pnl:,.2f} (commissions + exercise settlement timing)")

conn.close()
