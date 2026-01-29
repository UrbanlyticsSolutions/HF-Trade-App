"""Check Supabase backend status"""
import httpx
import json

token = 'sbp_437b265321c47ad67470b0d7636e41f11479e70c'
project = 'ncnbasvptocuwgxvyjmw'
anon = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5jbmJhc3ZwdG9jdXdneHZ5am13Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg1NTA1MjEsImV4cCI6MjA4NDEyNjUyMX0.766CPDI3GU-Rt4g5IRdRS-HPR84cCtlV7Ub-vHX0N4Q'

# 1. Check project status
print('=== PROJECT STATUS ===')
r = httpx.get(f'https://api.supabase.com/v1/projects/{project}', headers={'Authorization': f'Bearer {token}'})
data = r.json()
print(f"Name: {data.get('name')}")
print(f"Status: {data.get('status')}")
print(f"Region: {data.get('region')}")
print()

# 2. Check edge functions
print('=== EDGE FUNCTIONS ===')
r = httpx.get(f'https://api.supabase.com/v1/projects/{project}/functions', headers={'Authorization': f'Bearer {token}'})
funcs = r.json()
for f in funcs:
    print(f"{f.get('slug')}: {f.get('status')} (v{f.get('version')})")
print()

# 3. Test trading-api function is responding
print('=== TRADING-API LIVE TEST ===')
r = httpx.get(f'https://{project}.supabase.co/functions/v1/trading-api/state')
print(f'GET /state (no auth): {r.status_code} - Expected 401')

# 4. Check database has trades
print()
print('=== DATABASE TRADES ===')
headers = {'apikey': anon, 'Authorization': f'Bearer {anon}'}
r = httpx.get(f'https://{project}.supabase.co/rest/v1/trades?select=id,symbol,pnl,status,entry_time&order=entry_time.desc&limit=5', headers=headers)
trades = r.json()
print(f'Recent trades: {len(trades)}')
for t in trades:
    entry = t.get('entry_time', '')[:19] if t.get('entry_time') else 'N/A'
    print(f"  {t.get('symbol')}: {t.get('status')} P&L=${t.get('pnl')} @ {entry}")

# 5. Check trading state
print()
print('=== TRADING STATE ===')
r = httpx.get(f'https://{project}.supabase.co/rest/v1/trading_state', headers=headers)
state = r.json()
for s in state:
    print(f"  {s.get('key')}: {s.get('value')}")

print()
print('=== SUMMARY ===')
print(f"Project: {data.get('status')}")
print(f"Functions: {len(funcs)} deployed")
print(f"Trades in DB: {len(trades)}")
print(f"Auth working: {'YES' if r.status_code == 200 else 'NO'}")
