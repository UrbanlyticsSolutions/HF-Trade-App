#!/bin/bash
# Fix orphan trades in database

cd /opt/trading-engine

# Stop engine
sudo systemctl stop trading-engine

# Fix the database
sudo sqlite3 data/live_0dte_trades.db << 'EOSQL'
UPDATE trades SET 
    exit_price = 0.83, 
    exit_time = '2026-01-30T11:26:30',
    pnl = -72.0, 
    pnl_percent = -17.8, 
    status = 'closed', 
    option_type = 'put' 
WHERE id = 16;

DELETE FROM trades WHERE id = 17;

SELECT id, symbol, option_type, entry_price, exit_price, pnl, status 
FROM trades ORDER BY id DESC LIMIT 5;
EOSQL

# Restart engine
sudo systemctl start trading-engine
echo "Done! Engine restarted."
