"""Calculate P&L from all trades in database"""
import sqlite3

conn = sqlite3.connect('live_0dte_trades.db')
cursor = conn.cursor()
cursor.execute('SELECT id, symbol, action, quantity, entry_price, notes, entry_time FROM trades ORDER BY id')
trades = cursor.fetchall()

print("=" * 70)
print("TRADE HISTORY")
print("=" * 70)

buys = []
total_pnl = 0
closed_trades = []

for trade in trades:
    id, symbol, action, qty, price, notes, entry_time = trade
    if action == 'BUY':
        buys.append({'id': id, 'qty': qty, 'price': price, 'time': entry_time})
    elif action == 'SELL' and buys:
        buy = buys.pop(0)
        pnl = (price - buy['price']) * 100 * qty
        total_pnl += pnl
        pnl_pct = (price - buy['price']) / buy['price'] * 100
        closed_trades.append({
            'buy_id': buy['id'],
            'sell_id': id,
            'buy_price': buy['price'],
            'sell_price': price,
            'qty': qty,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })
        print(f"Trade #{buy['id']}->{id}: BUY {qty}x @ ${buy['price']:.2f} -> SELL @ ${price:.2f}")
        print(f"  P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Completed trades: {len(closed_trades)}")
print(f"Orphaned BUYs (not closed): {len(buys)}")
for buy in buys:
    print(f"  - Trade #{buy['id']}: BUY @ ${buy['price']:.2f}")
print()
print(f"Total P&L: ${total_pnl:.2f}")
print(f"Starting Capital: $10,000.00")
print(f"Final Capital: ${10000 + total_pnl:,.2f}")

# Close orphaned trades by marking them cancelled
if buys:
    print()
    print("Note: Orphaned trades are from restarts before position recovery was added.")
    print("Run with --cleanup flag to mark them as cancelled.")
