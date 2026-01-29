import sqlite3
import json

conn = sqlite3.connect('live_0dte_trades.db')
c = conn.cursor()

# Delete orphaned trades and exit_records
c.execute("DELETE FROM trades WHERE id IN (3,4,5,6,8,10)")
c.execute("DELETE FROM trades WHERE status='exit_record'")
conn.commit()

# Show remaining
c.execute("SELECT id, action, entry_price, exit_price, pnl, status FROM trades")
print("Remaining trades:")
for r in c.fetchall():
    print(f"  #{r[0]}: {r[1]} @ ${r[2]} -> ${r[3]}, PnL: ${r[4]}, {r[5]}")

# Recalculate
c.execute("SELECT SUM(pnl) FROM trades WHERE status='closed' AND pnl IS NOT NULL")
total_pnl = c.fetchone()[0] or 0
c.execute("SELECT COUNT(*) FROM trades WHERE status='closed'")
total = c.fetchone()[0]

print(f"\nTotal P&L: ${total_pnl}")
print(f"Capital: ${10000 + total_pnl}")

# Update state
state = {
    "initial_capital": 10000.0,
    "current_capital": 10000.0 + total_pnl,
    "high_water_mark": 10000.0 + total_pnl,
    "total_trades": total,
    "total_wins": total,
    "total_losses": 0,
    "total_pnl": total_pnl,
    "max_drawdown": 0.0,
    "daily_records": [],
    "last_updated": "2026-01-28T15:00:00",
    "last_trade_date": "2026-01-28",
    "strategy_state": {}
}
with open("trading_state.json", "w") as f:
    json.dump(state, f, indent=2)

print("Done! Orphaned trades removed.")
conn.close()
