"""Clean up database - mark orphans and link completed trades"""
import sqlite3

conn = sqlite3.connect('live_0dte_trades.db')
cursor = conn.cursor()

# Mark orphaned trades as cancelled
cursor.execute("UPDATE trades SET status = 'cancelled' WHERE id IN (3, 4, 5, 6, 8, 10)")

# Update trade 1 with proper exit info from trade 7
cursor.execute("UPDATE trades SET exit_price = 1.29, status = 'closed', pnl = 50.00 WHERE id = 1")

# Update trade 2 with proper exit info from trade 9  
cursor.execute("UPDATE trades SET exit_price = 1.20, status = 'closed', pnl = 60.00 WHERE id = 2")

# Mark trade 7 and 9 as exit records (not standalone)
cursor.execute("UPDATE trades SET status = 'exit_record' WHERE id IN (7, 9)")

conn.commit()
print('Database cleaned up:')
print('- 6 orphaned trades marked as cancelled')
print('- 2 completed trades properly linked with P&L')
print('')

# Show current state
cursor.execute("SELECT id, action, entry_price, exit_price, pnl, status FROM trades ORDER BY id")
print("Current trades:")
for row in cursor.fetchall():
    print(f"  {row}")
