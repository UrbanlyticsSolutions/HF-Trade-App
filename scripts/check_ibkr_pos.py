#!/usr/bin/env python3
"""Check IBKR positions directly."""
import sys, os, time
sys.path.insert(0, '/app')
from clients.ibkr_client import IBKRClient

client = IBKRClient(host='ib-gateway', port=4004, client_id=99)
client.connect()
time.sleep(2)

print("=== IBKR PORTFOLIO POSITIONS ===")
positions = client._ec.reqPositions()
time.sleep(2)
pos_list = client.wrapper.positions if hasattr(client.wrapper, 'positions') else []
print(f"Wrapper positions attr exists: {hasattr(client.wrapper, 'positions')}")

# Try the adapter approach
print("\n=== VIA ADAPTER get_account_positions ===")
try:
    from clients.ibkr_adapter import IBKRAdapter
    adapter = IBKRAdapter(client, 'DUP964745')
    pos_list = adapter.get_account_positions('DUP964745')
    for p in pos_list:
        sym = p.get('symbol', '')
        qty = p.get('openQuantity', 0)
        avg = p.get('averageEntryPrice', 0)
        print(f"  {sym} qty={qty} avg=${avg}")
    print(f"Total: {len(pos_list)} positions")
except Exception as e:
    import traceback
    traceback.print_exc()

client.disconnect()
