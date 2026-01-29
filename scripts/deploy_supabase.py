"""
Deploy trading system to Supabase
- Creates database tables
- Deploys edge functions for API access

Usage:
    python scripts/deploy_supabase.py --token YOUR_TOKEN
    python scripts/deploy_supabase.py --token YOUR_TOKEN --project PROJECT_REF
"""
import argparse
import json
import httpx
from pathlib import Path

MANAGEMENT_API = "https://api.supabase.com/v1"


def list_projects(token: str) -> list:
    """List all Supabase projects."""
    resp = httpx.get(
        f"{MANAGEMENT_API}/projects",
        headers={"Authorization": f"Bearer {token}"}
    )
    resp.raise_for_status()
    return resp.json()


def list_organizations(token: str) -> list:
    """List all organizations."""
    resp = httpx.get(
        f"{MANAGEMENT_API}/organizations",
        headers={"Authorization": f"Bearer {token}"}
    )
    resp.raise_for_status()
    return resp.json()


def get_project(token: str, project_ref: str) -> dict:
    """Get project details."""
    resp = httpx.get(
        f"{MANAGEMENT_API}/projects/{project_ref}",
        headers={"Authorization": f"Bearer {token}"}
    )
    resp.raise_for_status()
    return resp.json()


def execute_sql(token: str, project_ref: str, sql: str) -> dict:
    """Execute SQL on the project database."""
    resp = httpx.post(
        f"{MANAGEMENT_API}/projects/{project_ref}/database/query",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"query": sql},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()


def create_tables(token: str, project_ref: str):
    """Create trading tables in Supabase."""
    
    trades_table = """
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

    CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
    CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
    """
    
    equity_curve_table = """
    CREATE TABLE IF NOT EXISTS equity_curve (
        id SERIAL PRIMARY KEY,
        trade_id INTEGER REFERENCES trades(trade_id),
        equity DECIMAL(12, 2) NOT NULL,
        pnl DECIMAL(10, 2),
        trade_type TEXT,
        recorded_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_equity_curve_trade_id ON equity_curve(trade_id);
    """
    
    state_table = """
    CREATE TABLE IF NOT EXISTS trading_state (
        id SERIAL PRIMARY KEY,
        key TEXT UNIQUE NOT NULL,
        value JSONB NOT NULL,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    
    print("Creating trades table...")
    result = execute_sql(token, project_ref, trades_table)
    print(f"  Result: {result}")
    
    print("Creating equity_curve table...")
    result = execute_sql(token, project_ref, equity_curve_table)
    print(f"  Result: {result}")
    
    print("Creating trading_state table...")
    result = execute_sql(token, project_ref, state_table)
    print(f"  Result: {result}")
    
    print("✅ Tables created successfully!")


def create_edge_function(token: str, project_ref: str):
    """Create edge function for API access."""
    
    function_code = '''
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from "https://esm.sh/@supabase/supabase-js@2"

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? ''
    )

    const url = new URL(req.url)
    const path = url.pathname.replace('/trading-api', '')

    // GET /state - Get current trading state
    if (req.method === 'GET' && path === '/state') {
      const { data, error } = await supabase
        .from('trading_state')
        .select('*')
      
      if (error) throw error
      return new Response(JSON.stringify(data), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      })
    }

    // GET /trades - Get all trades
    if (req.method === 'GET' && path === '/trades') {
      const { data, error } = await supabase
        .from('trades')
        .select('*')
        .order('entry_time', { ascending: false })
        .limit(100)
      
      if (error) throw error
      return new Response(JSON.stringify(data), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      })
    }

    // GET /equity - Get equity curve
    if (req.method === 'GET' && path === '/equity') {
      const { data, error } = await supabase
        .from('equity_curve')
        .select('*')
        .order('recorded_at', { ascending: true })
      
      if (error) throw error
      return new Response(JSON.stringify(data), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      })
    }

    // POST /sync - Sync local data to Supabase
    if (req.method === 'POST' && path === '/sync') {
      const body = await req.json()
      
      if (body.trades) {
        const { error } = await supabase
          .from('trades')
          .upsert(body.trades, { onConflict: 'trade_id' })
        if (error) throw error
      }
      
      if (body.equity_curve) {
        const { error } = await supabase
          .from('equity_curve')
          .insert(body.equity_curve)
        if (error) throw error
      }
      
      if (body.state) {
        for (const [key, value] of Object.entries(body.state)) {
          const { error } = await supabase
            .from('trading_state')
            .upsert({ key, value, updated_at: new Date().toISOString() }, { onConflict: 'key' })
          if (error) throw error
        }
      }
      
      return new Response(JSON.stringify({ success: true }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      })
    }

    return new Response(JSON.stringify({ error: 'Not found' }), {
      status: 404,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    })

  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    })
  }
})
'''
    
    # Save function locally for reference
    func_dir = Path(__file__).parent.parent / "supabase" / "functions" / "trading-api"
    func_dir.mkdir(parents=True, exist_ok=True)
    (func_dir / "index.ts").write_text(function_code)
    print(f"Edge function saved to {func_dir / 'index.ts'}")
    
    # Note: Deploying edge functions requires Supabase CLI
    print("\n⚠️ To deploy the edge function, run:")
    print(f"   supabase functions deploy trading-api --project-ref {project_ref}")
    print("\nOr use the Supabase Dashboard to create the function manually.")


def migrate_local_data(token: str, project_ref: str):
    """Migrate local SQLite data to Supabase."""
    import sqlite3
    
    db_path = Path(__file__).parent.parent / "data" / "live_0dte_trades.db"
    if not db_path.exists():
        print(f"⚠️ Local database not found: {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Migrate trades
    cursor.execute("SELECT * FROM trades WHERE status = 'closed'")
    trades = [dict(row) for row in cursor.fetchall()]
    
    if trades:
        print(f"Migrating {len(trades)} trades...")
        # Build insert SQL with proper NULL handling
        for trade in trades:
            # Escape single quotes and handle NULL values
            symbol = trade['symbol'].replace("'", "''") if trade.get('symbol') else 'NULL'
            entry_time = f"'{trade['entry_time']}'" if trade.get('entry_time') else 'NULL'
            exit_time = f"'{trade['exit_time']}'" if trade.get('exit_time') else 'NULL'
            entry_price = trade.get('entry_price') if trade.get('entry_price') is not None else 'NULL'
            exit_price = trade.get('exit_price') if trade.get('exit_price') is not None else 'NULL'
            pnl = trade.get('pnl') if trade.get('pnl') is not None else 'NULL'
            quantity = trade.get('quantity', 1) or 1
            status = trade.get('status', 'closed')
            
            sql = f"""
            INSERT INTO trades (trade_id, symbol, quantity, entry_price, exit_price, 
                               entry_time, exit_time, pnl, status)
            VALUES ({trade['id']}, '{symbol}', {quantity},
                    {entry_price}, {exit_price},
                    {entry_time}, {exit_time},
                    {pnl}, '{status}')
            ON CONFLICT (trade_id) DO UPDATE SET
                pnl = EXCLUDED.pnl,
                exit_price = EXCLUDED.exit_price,
                exit_time = EXCLUDED.exit_time,
                status = EXCLUDED.status;
            """
            try:
                execute_sql(token, project_ref, sql)
                print(f"  ✓ Trade {trade['id']}")
            except Exception as e:
                print(f"  ✗ Error migrating trade {trade['id']}: {e}")
    
    # Migrate state
    state_path = Path(__file__).parent.parent / "trading_state.json"
    if state_path.exists():
        print("Migrating trading state...")
        with open(state_path) as f:
            state = json.load(f)
        
        for key, value in state.items():
            sql = f"""
            INSERT INTO trading_state (key, value)
            VALUES ('{key}', '{json.dumps(value)}')
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW();
            """
            try:
                execute_sql(token, project_ref, sql)
            except Exception as e:
                print(f"  Error migrating state {key}: {e}")
    
    conn.close()
    print("✅ Migration complete!")


def main():
    parser = argparse.ArgumentParser(description="Deploy trading system to Supabase")
    parser.add_argument("--token", required=True, help="Supabase access token")
    parser.add_argument("--project", help="Project reference (will list projects if not provided)")
    parser.add_argument("--create-tables", action="store_true", help="Create database tables")
    parser.add_argument("--create-function", action="store_true", help="Create edge function")
    parser.add_argument("--migrate", action="store_true", help="Migrate local data")
    parser.add_argument("--all", action="store_true", help="Do everything")
    
    args = parser.parse_args()
    
    if not args.project:
        print("\n📋 Your Supabase Projects:\n")
        projects = list_projects(args.token)
        for p in projects:
            print(f"  {p['name']}: {p['id']}")
            print(f"    Region: {p.get('region', 'N/A')}")
            print(f"    Status: {p.get('status', 'N/A')}")
            print()
        
        if projects:
            print(f"Run again with --project {projects[0]['id']} to deploy")
        return
    
    project_ref = args.project
    print(f"\n🚀 Deploying to project: {project_ref}\n")
    
    if args.all or args.create_tables:
        print("\n📊 Creating tables...")
        create_tables(args.token, project_ref)
    
    if args.all or args.create_function:
        print("\n⚡ Creating edge function...")
        create_edge_function(args.token, project_ref)
    
    if args.all or args.migrate:
        print("\n📦 Migrating local data...")
        migrate_local_data(args.token, project_ref)
    
    if not (args.all or args.create_tables or args.create_function or args.migrate):
        print("No action specified. Use --all, --create-tables, --create-function, or --migrate")


if __name__ == "__main__":
    main()
