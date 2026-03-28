"""Query IBKR for all positions and executions with full contract details."""
import sys, time, threading
from collections import defaultdict
sys.path.insert(0, '/app')
from clients.ibkr_client import IBKRClient
from ibapi.execution import ExecutionFilter

c = IBKRClient(host='ib-gateway', port=4004, client_id=66)
c.connect()
time.sleep(2)

# Capture ALL positions properly (bypass wrapper bug that keys by underlying)
all_positions = []
pos_done = threading.Event()

def cap_pos(account, contract, pos, avgCost):
    all_positions.append({
        'sym': contract.symbol,
        'local': contract.localSymbol,
        'sec': contract.secType,
        'pos': pos,
        'avg': avgCost,
        'strike': contract.strike,
        'right': contract.right,
        'exp': contract.lastTradeDateOrContractMonth,
    })

def cap_posEnd():
    pos_done.set()

c.wrapper.position = cap_pos
c.wrapper.positionEnd = cap_posEnd
c._ec.reqPositions()
pos_done.wait(timeout=10)
c._ec.cancelPositions()

print("=" * 50)
print("ALL IBKR POSITIONS")
print("=" * 50)
for p in all_positions:
    print(f"  {p['local']}  secType={p['sec']}  qty={p['pos']}  avgCost={p['avg']:.4f}  strike={p['strike']}  right={p['right']}  exp={p['exp']}")
if not all_positions:
    print("  (no positions)")

# Capture ALL executions with full contract details
all_execs = []
exec_done = threading.Event()

def cap_exec(reqId, contract, execution):
    all_execs.append({
        'sym': contract.symbol,
        'local': contract.localSymbol,
        'sec': contract.secType,
        'strike': contract.strike,
        'right': contract.right,
        'exp': contract.lastTradeDateOrContractMonth,
        'oid': execution.orderId,
        'eid': execution.execId,
        'side': execution.side,
        'shares': execution.shares,
        'price': execution.price,
        'time': execution.time,
        'cid': execution.clientId,
    })

def cap_execEnd(reqId):
    exec_done.set()

c.wrapper.execDetails = cap_exec
c.wrapper.execDetailsEnd = cap_execEnd
filt = ExecutionFilter()
c._ec.reqExecutions(10010, filt)
exec_done.wait(timeout=10)

print()
print("=" * 50)
print("ALL IBKR EXECUTIONS (grouped by clientId + orderId)")
print("=" * 50)
print(f"Total executions: {len(all_execs)}")
print()

by_key = defaultdict(list)
for exe in all_execs:
    by_key[(exe['cid'], exe['oid'])].append(exe)

for key in sorted(by_key.keys()):
    fills = by_key[key]
    total_shares = sum(f['shares'] for f in fills)
    side = fills[0]['side']
    local = fills[0]['local']
    total_cost = sum(f['shares'] * f['price'] for f in fills)
    avg_price = total_cost / total_shares if total_shares > 0 else 0
    t = fills[0]['time']
    print(f"  clientId={key[0]} orderId={key[1]}: {side} {total_shares:.0f} x {local} @ avg ${avg_price:.4f}  time={t}  ({len(fills)} fills)")

# Also get account summary for NLV
print()
print("=" * 50)
print("ACCOUNT SUMMARY")
print("=" * 50)
acct_values = {}
acct_done = threading.Event()

def cap_acct(key, val, currency, accountName):
    acct_values[key] = (val, currency)

def cap_acctEnd(accountName):
    acct_done.set()

c.wrapper.updateAccountValue = cap_acct
c.wrapper.accountDownloadEnd = cap_acctEnd
c._ec.reqAccountUpdates(True, "")
acct_done.wait(timeout=10)
c._ec.reqAccountUpdates(False, "")

for k in ['NetLiquidation', 'TotalCashValue', 'GrossPositionValue', 'UnrealizedPnL', 'RealizedPnL']:
    if k in acct_values:
        print(f"  {k}: {acct_values[k][0]} {acct_values[k][1]}")

c.disconnect()
