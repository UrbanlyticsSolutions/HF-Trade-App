"""Fix phantom trades in DB based on IBKR execution verification.

Only 3 orders were actually filled by IBKR on March 10, 2026:
  Order 1: BOT 26 x C680 @ $1.91 (Trade 4 entry) - REAL
  Order 2: SLD 26 x C680 @ $1.79 (Trade 4 exit)  - REAL
  Order 3: BOT 50 x C683 @ $0.65 (Trade 5 entry) - REAL (but exit never filled)

Trades 5 (exit), 6, 7, 8 have phantom PnL because the old engine code
booked PnL at signal.limit_price without fill confirmation.

This script:
  - Trade 5: Reopen (exit never filled, position still held on IBKR)
  - Trade 6: Mark as phantom (entry order 5 never filled)
  - Trade 7: Mark as phantom (entry order 7 never filled)
  - Trade 8: Already closed at $0 by reconciliation — mark as phantom
"""
import sqlite3
from datetime import datetime

DB_PATH = '/app/data/live_0dte_trades.db'
now = datetime.now().isoformat()

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("=== BEFORE FIX ===")
for r in cursor.execute("SELECT id, symbol, quantity, entry_price, exit_price, pnl, status, notes FROM trades WHERE id >= 4"):
    print(dict(r))
print()

# Trade 5: C683 50x - Entry FILLED at $0.65, exit NEVER filled.
# The 50x C683 position is still open on IBKR.
# Reopen this trade so the engine can manage it.
cursor.execute("""
    UPDATE trades SET
        status = 'open',
        exit_price = NULL,
        exit_time = NULL,
        exit_order_id = NULL,
        pnl = NULL,
        pnl_percent = NULL,
        notes = 'Entry filled @ $0.65 (verified). Exit order never filled by IBKR. Reopened for engine mgmt.',
        updated_at = ?
    WHERE id = 5
""", (now,))
print(f"Trade 5 (C683): reopened (exit was phantom, position still on IBKR)")

# Trade 6: C682 37x - Entry order 5 NEVER FILLED by IBKR. Entire trade is phantom.
cursor.execute("""
    UPDATE trades SET
        status = 'closed',
        exit_price = 0,
        pnl = 0,
        pnl_percent = 0,
        notes = 'PHANTOM: entry order never filled by IBKR. PnL zeroed.',
        updated_at = ?
    WHERE id = 6
""", (now,))
print(f"Trade 6 (C682 37x): marked phantom, PnL zeroed")

# Trade 7: C682 32x - Entry order 7 NEVER FILLED by IBKR. Entire trade is phantom.
cursor.execute("""
    UPDATE trades SET
        status = 'closed',
        exit_price = 0,
        pnl = 0,
        pnl_percent = 0,
        notes = 'PHANTOM: entry order never filled by IBKR. PnL zeroed.',
        updated_at = ?
    WHERE id = 7
""", (now,))
print(f"Trade 7 (C682 32x): marked phantom, PnL zeroed")

# Trade 8: P679 31x - Already auto-closed at $0 by reconciliation.
# Entry order 9 was NEVER FILLED. Mark as phantom with $0 PnL.
cursor.execute("""
    UPDATE trades SET
        pnl = 0,
        pnl_percent = 0,
        notes = 'PHANTOM: entry order never filled by IBKR. PnL zeroed. (was auto-closed by reconciliation)',
        updated_at = ?
    WHERE id = 8
""", (now,))
print(f"Trade 8 (P679 31x): marked phantom, PnL zeroed")

# Also fix daily_pnl table to remove phantom PnL
# The real PnL for March 10 is just Trade 4: -$312
cursor.execute("""
    UPDATE daily_pnl SET
        realized_pnl = -312.0,
        total_pnl = -312.0,
        trades_closed = 1,
        win_count = 0,
        loss_count = 1,
        updated_at = ?
    WHERE date = '2026-03-10'
""", (now,))
print(f"daily_pnl for 2026-03-10: corrected to -$312 (1 real closed trade)")

conn.commit()

print()
print("=== AFTER FIX ===")
for r in conn.execute("SELECT id, symbol, quantity, entry_price, exit_price, pnl, status, notes FROM trades WHERE id >= 4"):
    print(dict(r))

print()
print("=== DAILY PnL ===")
for r in conn.execute("SELECT * FROM daily_pnl WHERE date = '2026-03-10'"):
    print(dict(r))

conn.close()
print()
print("DB correction complete.")
