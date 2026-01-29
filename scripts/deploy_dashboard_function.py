"""Deploy Dashboard Edge Function to Supabase - Full HTML Dashboard"""
import httpx

token = 'sbp_437b265321c47ad67470b0d7636e41f11479e70c'
project_ref = 'ncnbasvptocuwgxvyjmw'

# Dashboard edge function - serves complete HTML page (using string concat to avoid template issues)
dashboard_code = r"""
Deno.serve(async (req) => {
  const SUPABASE_URL = Deno.env.get("SUPABASE_URL")
  const SUPABASE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")
  
  const headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": "Bearer " + SUPABASE_KEY,
    "Content-Type": "application/json"
  }

  const [tradesRes, stateRes] = await Promise.all([
    fetch(SUPABASE_URL + "/rest/v1/trades?order=entry_time.desc&limit=100", { headers }),
    fetch(SUPABASE_URL + "/rest/v1/trading_state", { headers })
  ])

  const trades = await tradesRes.json()
  const stateArr = await stateRes.json()

  const state = {}
  for (const s of stateArr) { state[s.key] = s.value }

  const initialCapital = state.initial_capital || 10000
  const currentCapital = state.current_capital || initialCapital
  const pnl = currentCapital - initialCapital
  const pnlPct = ((pnl / initialCapital) * 100).toFixed(2)
  const winTrades = trades.filter(t => t.pnl > 0).length
  const lossTrades = trades.filter(t => t.pnl < 0).length
  const winRate = trades.length > 0 ? ((winTrades / trades.length) * 100).toFixed(1) : 0

  let tradesRows = ""
  for (const t of trades) {
    const pnlClass = t.pnl >= 0 ? "profit" : "loss"
    const ep = t.entry_price ? t.entry_price.toFixed(2) : "-"
    const xp = t.exit_price ? t.exit_price.toFixed(2) : "-"
    const pl = t.pnl ? t.pnl.toFixed(2) : "-"
    tradesRows += "<tr><td>" + t.trade_id + "</td><td>" + t.symbol + "</td><td>" + t.quantity + "</td><td>$" + ep + "</td><td>$" + xp + "</td><td class='" + pnlClass + "'>$" + pl + "</td><td>" + new Date(t.entry_time).toLocaleString() + "</td><td>" + (t.exit_time ? new Date(t.exit_time).toLocaleString() : "-") + "</td><td>" + t.status + "</td></tr>"
  }
  if (!tradesRows) tradesRows = "<tr><td colspan='9' style='text-align:center;'>No trades yet</td></tr>"

  const pnlClass = pnl >= 0 ? "profit" : "loss"

  const html = `<!DOCTYPE html>
<html>
<head>
  <title>Trading Dashboard</title>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }
    h1 { color: #38bdf8; margin-bottom: 20px; }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }
    .stat-card { background: #1e293b; padding: 20px; border-radius: 10px; border-left: 4px solid #38bdf8; }
    .stat-card h3 { color: #94a3b8; font-size: 14px; margin-bottom: 5px; }
    .stat-card .value { font-size: 28px; font-weight: bold; }
    .profit { color: #22c55e; }
    .loss { color: #ef4444; }
    table { width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 10px; overflow: hidden; margin-top: 20px; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #334155; }
    th { background: #334155; color: #38bdf8; font-weight: 600; }
    tr:hover { background: #334155; }
    .refresh-btn { background: #38bdf8; color: #0f172a; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: bold; }
    .refresh-btn:hover { background: #0ea5e9; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
  </style>
</head>
<body>
  <div class="header">
    <h1>Trading Dashboard</h1>
    <button class="refresh-btn" onclick="location.reload()">Refresh</button>
  </div>
  
  <div class="stats">
    <div class="stat-card">
      <h3>Initial Capital</h3>
      <div class="value">$` + initialCapital.toLocaleString() + `</div>
    </div>
    <div class="stat-card">
      <h3>Current Capital</h3>
      <div class="value">$` + currentCapital.toLocaleString() + `</div>
    </div>
    <div class="stat-card">
      <h3>Total P&L</h3>
      <div class="value ` + pnlClass + `">$` + pnl.toLocaleString() + ` (` + pnlPct + `%)</div>
    </div>
    <div class="stat-card">
      <h3>Win Rate</h3>
      <div class="value">` + winRate + `% (` + winTrades + `W / ` + lossTrades + `L)</div>
    </div>
    <div class="stat-card">
      <h3>Total Trades</h3>
      <div class="value">` + trades.length + `</div>
    </div>
  </div>

  <h2 style="color: #38bdf8;">Trade History</h2>
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Symbol</th>
        <th>Qty</th>
        <th>Entry</th>
        <th>Exit</th>
        <th>P&L</th>
        <th>Entry Time</th>
        <th>Exit Time</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
      ` + tradesRows + `
    </tbody>
  </table>

  <p style="margin-top: 30px; color: #64748b; text-align: center;">
    Last updated: ` + new Date().toLocaleString() + ` | Hosted on Supabase
  </p>
</body>
</html>`

  return new Response(html, {
    headers: { "Content-Type": "text/html; charset=utf-8" }
  })
})
"""

# Deploy the dashboard function
print("Deploying dashboard edge function...")

# Check if exists
resp = httpx.get(
    f"https://api.supabase.com/v1/projects/{project_ref}/functions",
    headers={"Authorization": f"Bearer {token}"}
)
functions = resp.json()
function_exists = any(f.get('slug') == 'dashboard' for f in functions)

if function_exists:
    print("Function exists, updating...")
    resp = httpx.patch(
        f"https://api.supabase.com/v1/projects/{project_ref}/functions/dashboard",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "body": dashboard_code,
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
            "slug": "dashboard",
            "name": "dashboard",
            "body": dashboard_code,
            "verify_jwt": False
        },
        timeout=60
    )

print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:300] if resp.text else 'Empty'}")

if resp.status_code in [200, 201]:
    print(f"\n✅ Dashboard deployed!")
    print(f"\n🌐 ACCESS YOUR DASHBOARD:")
    print(f"   https://{project_ref}.supabase.co/functions/v1/dashboard")
else:
    print(f"\n❌ Deployment failed")
