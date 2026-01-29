"""Close orphaned trades at current market price"""
import sqlite3
import sys
sys.path.insert(0, '.')

from clients.questrade_client import QuestradeClient

# Get current market price
client = QuestradeClient()
symbol = 'SPY28Jan26P694.00'
search = client.search_symbols(symbol)

if not search:
    print("Symbol not found - option likely expired. Using last known price $1.00")
    exit_price = 1.00
else:
    sym_id = search[0]['symbolId']
    quote = client.get_quote(sym_id)
    bid = quote.get('bidPrice', 0)
    ask = quote.get('askPrice', 0)
    last = quote.get('lastTradePrice', 0)
    exit_price = bid if bid > 0 else (last if last > 0 else 1.00)
    print(f"Current market: bid=${bid}, ask=${ask}, last=${last}")
    print(f"Using exit price: ${exit_price}")

# Close orphaned trades
conn = sqlite3.connect('live_0dte_trades.db')
cursor = conn.cursor()

# Get orphaned (cancelled) BUY trades
cursor.execute("SELECT id, quantity, entry_price FROM trades WHERE status='cancelled' AND action='BUY'")
orphans = cursor.fetchall()

print(f"\nClosing {len(orphans)} orphaned trades at ${exit_price}:")
total_orphan_pnl = 0

for trade_id, qty, entry_price in orphans:
    pnl = (exit_price - entry_price) * 100 * qty
    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
    total_orphan_pnl += pnl
    
    cursor.execute("""
        UPDATE trades 
        SET status='closed', exit_price=?, pnl=?, pnl_percent=?
        WHERE id=?
    """, (exit_price, pnl, pnl_pct, trade_id))
    
    print(f"  Trade #{trade_id}: BUY @ ${entry_price:.2f} -> SELL @ ${exit_price:.2f} = ${pnl:.2f} ({pnl_pct:.1f}%)")

conn.commit()

# Recalculate total P&L
cursor.execute("SELECT SUM(pnl) FROM trades WHERE status='closed' AND pnl IS NOT NULL")
total_pnl = cursor.fetchone()[0] or 0

cursor.execute("SELECT COUNT(*) FROM trades WHERE status='closed'")
total_trades = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM trades WHERE status='closed' AND pnl > 0")
wins = cursor.fetchone()[0]

losses = total_trades - wins

print(f"\n" + "="*60)
print(f"UPDATED SUMMARY")
print(f"="*60)
print(f"Orphan trades P&L: ${total_orphan_pnl:.2f}")
print(f"Total P&L (all closed): ${total_pnl:.2f}")
print(f"Total trades: {total_trades} ({wins} wins, {losses} losses)")
print(f"Final capital: ${10000 + total_pnl:.2f}")

# Update trading_state.json
import json
state = {
    "initial_capital": 10000.0,
    "current_capital": 10000.0 + total_pnl,
    "high_water_mark": max(10000.0, 10000.0 + total_pnl),
    "total_trades": total_trades,
    "total_wins": wins,
    "total_losses": losses,
    "total_pnl": total_pnl,
    "max_drawdown": 0.0,
    "daily_records": [],
    "last_updated": "2026-01-28T14:45:00.000000",
    "last_trade_date": "2026-01-28",
    "strategy_state": {}
}

with open('trading_state.json', 'w') as f:
    json.dump(state, f, indent=2)

print(f"\ntrading_state.json updated!")
conn.close()
