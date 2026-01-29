import sqlite3
conn = sqlite3.connect('live_0dte_trades.db')
c = conn.cursor()
c.execute("UPDATE trades SET status='exit_record' WHERE id=11")
conn.commit()
print('Trade 11 marked as exit_record')
c.execute("SELECT COUNT(*) FROM trades WHERE status='open'")
print(f'Open trades: {c.fetchone()[0]}')
conn.close()
