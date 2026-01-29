"""Deploy Edge Function to Supabase using Management API"""
import httpx
import json

token = 'sbp_437b265321c47ad67470b0d7636e41f11479e70c'
project_ref = 'ncnbasvptocuwgxvyjmw'

# Edge function code - WITH API KEY AUTHENTICATION
# Your secret API key - change this to something only you know!
YOUR_SECRET_API_KEY = "TRADING_SECRET_2026_CHANGE_ME"

function_code = """const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-api-key",
}

// Your secret API key - set via environment variable
const API_KEY = Deno.env.get("TRADING_API_KEY")

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders })
  }

  // AUTHENTICATION: Check API key
  const providedKey = req.headers.get("x-api-key") || new URL(req.url).searchParams.get("key")
  if (!API_KEY || providedKey !== API_KEY) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), { 
      status: 401, 
      headers: { ...corsHeaders, "Content-Type": "application/json" } 
    })
  }

  const SUPABASE_URL = Deno.env.get("SUPABASE_URL")
  const SUPABASE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")
  
  const url = new URL(req.url)
  const path = url.pathname.split("/").pop()

  try {
    const headers = {
      "apikey": SUPABASE_KEY,
      "Authorization": "Bearer " + SUPABASE_KEY,
      "Content-Type": "application/json"
    }

    if (path === "trades" || path === "trading-api") {
      const res = await fetch(SUPABASE_URL + "/rest/v1/trades?order=entry_time.desc&limit=100", { headers })
      const data = await res.json()
      return new Response(JSON.stringify(data), { headers: { ...corsHeaders, "Content-Type": "application/json" } })
    }

    if (path === "state") {
      const res = await fetch(SUPABASE_URL + "/rest/v1/trading_state", { headers })
      const data = await res.json()
      return new Response(JSON.stringify(data), { headers: { ...corsHeaders, "Content-Type": "application/json" } })
    }

    if (path === "equity") {
      const res = await fetch(SUPABASE_URL + "/rest/v1/equity_curve?order=recorded_at.asc", { headers })
      const data = await res.json()
      return new Response(JSON.stringify(data), { headers: { ...corsHeaders, "Content-Type": "application/json" } })
    }

    return new Response(JSON.stringify({ endpoints: ["trades", "state", "equity"] }), { headers: { ...corsHeaders, "Content-Type": "application/json" } })
  } catch (e) {
    return new Response(JSON.stringify({ error: e.message }), { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } })
  }
})
"""

# List existing functions
print("Checking existing edge functions...")
resp = httpx.get(
    f"https://api.supabase.com/v1/projects/{project_ref}/functions",
    headers={"Authorization": f"Bearer {token}"}
)
print(f"Existing functions: {resp.json()}")

# Try to create or update the function
print("\nDeploying edge function 'trading-api'...")

# First, check if function exists
functions = resp.json()
function_exists = any(f.get('slug') == 'trading-api' for f in functions)

if function_exists:
    print("Function exists, updating...")
    resp = httpx.patch(
        f"https://api.supabase.com/v1/projects/{project_ref}/functions/trading-api",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "body": function_code,
            "verify_jwt": False
        },
        timeout=60
    )
else:
    print("Creating new function...")
    resp = httpx.post(
        f"https://api.supabase.com/v1/projects/{project_ref}/functions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "slug": "trading-api",
            "name": "trading-api",
            "body": function_code,
            "verify_jwt": False
        },
        timeout=60
    )

print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:500] if resp.text else 'Empty'}")

if resp.status_code in [200, 201]:
    print(f"\n✅ Edge function deployed!")
    
    # Set the secret API key as environment variable
    print("\nSetting TRADING_API_KEY secret...")
    secret_resp = httpx.post(
        f"https://api.supabase.com/v1/projects/{project_ref}/secrets",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json=[{"name": "TRADING_API_KEY", "value": YOUR_SECRET_API_KEY}],
        timeout=30
    )
    if secret_resp.status_code in [200, 201]:
        print(f"✅ Secret TRADING_API_KEY set!")
    else:
        print(f"⚠️ Secret setup: {secret_resp.status_code} - {secret_resp.text[:200]}")
    
    print(f"\n🔐 YOUR API KEY: {YOUR_SECRET_API_KEY}")
    print(f"   (Change this in the script and redeploy for security!)")
    print(f"\nURL: https://{project_ref}.supabase.co/functions/v1/trading-api")
    print("\nAuthenticated endpoints (add ?key=YOUR_KEY or header x-api-key):")
    print(f"  GET  .../trading-api/trades?key={YOUR_SECRET_API_KEY}")
    print(f"  GET  .../trading-api/state?key={YOUR_SECRET_API_KEY}")
    print(f"  GET  .../trading-api/equity?key={YOUR_SECRET_API_KEY}")
else:
    print(f"\n❌ Deployment failed: {resp.status_code}")
