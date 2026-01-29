const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders })
  }

  const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!
  const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY")!
  const SUPABASE_SERVICE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!

  // Verify JWT token from Authorization header
  const authHeader = req.headers.get("Authorization")
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return new Response(JSON.stringify({ error: "Missing authorization" }), {
      status: 401,
      headers: { ...corsHeaders, "Content-Type": "application/json" }
    })
  }

  const token = authHeader.replace("Bearer ", "")

  // Verify token by calling Supabase Auth API
  try {
    const authResp = await fetch(`${SUPABASE_URL}/auth/v1/user`, {
      headers: {
        "Authorization": `Bearer ${token}`,
        "apikey": SUPABASE_ANON_KEY
      }
    })
    
    if (!authResp.ok) {
      return new Response(JSON.stringify({ error: "Invalid or expired token" }), {
        status: 401,
        headers: { ...corsHeaders, "Content-Type": "application/json" }
      })
    }
  } catch {
    return new Response(JSON.stringify({ error: "Auth verification failed" }), {
      status: 401,
      headers: { ...corsHeaders, "Content-Type": "application/json" }
    })
  }

  // User is authenticated - proceed with database operations
  const url = new URL(req.url)
  const path = url.pathname.split("/").pop()

  const dbHeaders = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": `Bearer ${SUPABASE_SERVICE_KEY}`,
    "Content-Type": "application/json"
  }

  try {
    if (path === "trades" || path === "trading-api") {
      const res = await fetch(`${SUPABASE_URL}/rest/v1/trades?order=entry_time.desc&limit=100`, { headers: dbHeaders })
      const data = await res.json()
      return new Response(JSON.stringify(data), { headers: { ...corsHeaders, "Content-Type": "application/json" } })
    }

    if (path === "state") {
      const res = await fetch(`${SUPABASE_URL}/rest/v1/trading_state`, { headers: dbHeaders })
      const data = await res.json()
      return new Response(JSON.stringify(data), { headers: { ...corsHeaders, "Content-Type": "application/json" } })
    }

    if (path === "equity") {
      const res = await fetch(`${SUPABASE_URL}/rest/v1/equity_curve?order=recorded_at.asc`, { headers: dbHeaders })
      const data = await res.json()
      return new Response(JSON.stringify(data), { headers: { ...corsHeaders, "Content-Type": "application/json" } })
    }

    return new Response(JSON.stringify({ endpoints: ["trades", "state", "equity"] }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" }
    })
  } catch (e) {
    return new Response(JSON.stringify({ error: (e as Error).message }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" }
    })
  }
})
