"""Test direct database connection to Supabase"""
import httpx

token = 'sbp_437b265321c47ad67470b0d7636e41f11479e70c'
project_ref = 'ncnbasvptocuwgxvyjmw'

# Try the migrations approach
migration_sql = """
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    trade_id INTEGER UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    option_type TEXT,
    quantity INTEGER,
    entry_price DECIMAL(10, 4),
    exit_price DECIMAL(10, 4),
    entry_time TIMESTAMPTZ,
    exit_time TIMESTAMPTZ,
    pnl DECIMAL(10, 2),
    status TEXT DEFAULT 'open',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""

# Try different endpoints
endpoints = [
    f"https://api.supabase.com/v1/projects/{project_ref}/database/query",
    f"https://api.supabase.com/v1/projects/{project_ref}/database/execute-sql", 
    f"https://api.supabase.com/platform/pg-meta/{project_ref}/query",
]

for endpoint in endpoints:
    print(f"\nTrying: {endpoint}")
    try:
        resp = httpx.post(
            endpoint,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"query": migration_sql},
            timeout=30
        )
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")

# Try to list available endpoints via the Management API
print("\n\nChecking available database endpoints...")
resp = httpx.get(
    f"https://api.supabase.com/v1/projects/{project_ref}",
    headers={"Authorization": f"Bearer {token}"}
)
print(f"Project details: {resp.json()}")
