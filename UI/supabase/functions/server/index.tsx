import { Hono } from "npm:hono";
import { cors } from "npm:hono/cors";
import { logger } from "npm:hono/logger";
import * as kv from "./kv_store.tsx";
import { createClient } from "npm:@supabase/supabase-js@2";

const app = new Hono();

// Create Supabase clients
const getSupabaseAdmin = () => {
  const url = Deno.env.get('SUPABASE_URL');
  const key = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
  
  if (!url || !key) {
    console.error('Missing Supabase credentials:', { url: !!url, key: !!key });
    throw new Error('Supabase credentials not configured');
  }
  
  return createClient(url, key, {
    auth: {
      autoRefreshToken: false,
      persistSession: false
    }
  });
};

const getSupabaseClient = () => createClient(
  Deno.env.get('SUPABASE_URL') ?? '',
  Deno.env.get('SUPABASE_ANON_KEY') ?? '',
);

// Enable logger
app.use('*', logger(console.log));

// Enable CORS for all routes and methods
app.use(
  "/*",
  cors({
    origin: "*",
    allowHeaders: ["Content-Type", "Authorization"],
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    exposeHeaders: ["Content-Length"],
    maxAge: 600,
  }),
);

// Middleware to verify authenticated user
const requireAuth = async (c: any, next: any) => {
  try {
    const accessToken = c.req.header('Authorization')?.split(' ')[1];
    if (!accessToken) {
      return c.json({ error: 'Unauthorized: No token provided' }, 401);
    }

    const supabase = getSupabaseAdmin();
    const { data: { user }, error } = await supabase.auth.getUser(accessToken);
    
    if (error || !user) {
      console.log('Authorization error:', error);
      return c.json({ error: 'Unauthorized: Invalid token' }, 401);
    }

    c.set('userId', user.id);
    await next();
  } catch (error) {
    console.log('Auth middleware error:', error);
    return c.json({ error: 'Unauthorized' }, 401);
  }
};

// Health check endpoint
app.get("/make-server-c1a53bfc/health", (c) => {
  return c.json({ status: "ok" });
});

// Check if user exists endpoint (no auth required)
app.get("/make-server-c1a53bfc/check-user", async (c) => {
  try {
    const email = c.req.query('email') || 'realericzhu@gmail.com';
    
    // Try to get user by email using anon key (won't work for getting user details)
    // Instead, we'll try to sign in with a dummy password to check if user exists
    const supabase = getSupabaseClient();
    
    // This will fail if user doesn't exist, but that's what we want to check
    const { data, error } = await supabase.auth.signInWithPassword({
      email: email,
      password: 'dummy-check-password-that-wont-work'
    });
    
    if (error) {
      // If error is "Invalid login credentials", user might exist but password is wrong
      // If error is "Email not confirmed", user exists
      // If error is something else, user might not exist
      
      if (error.message.includes('Email not confirmed')) {
        return c.json({ exists: true, confirmed: false });
      }
      
      // For "Invalid login credentials", we can't be 100% sure, but likely user exists
      // or doesn't exist. We'll return unknown.
      return c.json({ exists: 'unknown', error: error.message });
    }
    
    return c.json({ exists: true, confirmed: true });
  } catch (error: any) {
    console.log('Check user error:', error);
    return c.json({ exists: 'unknown', error: error.message }, 500);
  }
});

// One-time user setup endpoint (creates the authorized user)
app.post("/make-server-c1a53bfc/setup-user", async (c) => {
  try {
    console.log('=== User Setup Endpoint Called ===');
    
    const supabaseUrl = Deno.env.get('SUPABASE_URL');
    const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
    
    console.log('Environment check:');
    console.log('- SUPABASE_URL:', supabaseUrl ? `${supabaseUrl.substring(0, 20)}...` : 'NOT SET');
    console.log('- SERVICE_ROLE_KEY:', serviceRoleKey ? `${serviceRoleKey.substring(0, 20)}...` : 'NOT SET');
    
    if (!supabaseUrl) {
      return c.json({ 
        error: 'SUPABASE_URL not configured',
        message: 'Server configuration error. SUPABASE_URL environment variable is missing.',
        instructions: [
          '1. Go to your Supabase Dashboard: https://supabase.com/dashboard',
          '2. Navigate to Authentication > Users',
          '3. Click "Add User" or "Invite"',
          '4. Email: realericzhu@gmail.com',
          '5. Password: 1234',
          '6. Check "Auto Confirm Email"',
          '7. Click "Create User"'
        ]
      }, 500);
    }

    if (!serviceRoleKey) {
      return c.json({ 
        error: 'SUPABASE_SERVICE_ROLE_KEY not configured',
        message: 'The service role key is not configured. You need to create the user manually.',
        instructions: [
          '1. Go to your Supabase Dashboard: https://supabase.com/dashboard',
          '2. Navigate to Authentication > Users',
          '3. Click "Add User" or "Invite"',
          '4. Email: realericzhu@gmail.com',
          '5. Password: 1234',
          '6. Check "Auto Confirm Email"',
          '7. Click "Create User"'
        ]
      }, 500);
    }
    
    const supabase = createClient(supabaseUrl, serviceRoleKey, {
      auth: {
        autoRefreshToken: false,
        persistSession: false
      }
    });
    
    // Check if user already exists
    console.log('Checking for existing users...');
    const { data: existingUsers, error: listError } = await supabase.auth.admin.listUsers();
    
    if (listError) {
      console.log('Error listing users:', listError);
      return c.json({
        error: 'Failed to check existing users',
        message: listError.message,
        suggestion: 'Please create the user manually in Supabase Dashboard',
        instructions: [
          'Go to: https://supabase.com/dashboard',
          'Authentication > Users > Add User',
          'Email: realericzhu@gmail.com',
          'Password: 1234',
          'Auto Confirm Email: ON'
        ]
      }, 500);
    }
    
    const userExists = existingUsers?.users.some(u => u.email === 'realericzhu@gmail.com');
    
    if (userExists) {
      console.log('User already exists: realericzhu@gmail.com');
      return c.json({ message: 'User already exists', email: 'realericzhu@gmail.com' });
    }

    console.log('Creating user: realericzhu@gmail.com');
    
    // Create the user
    const { data, error } = await supabase.auth.admin.createUser({
      email: 'realericzhu@gmail.com',
      password: '1234',
      user_metadata: { name: 'Eric Zhu' },
      // Automatically confirm the user's email since an email server hasn't been configured.
      email_confirm: true
    });

    if (error) {
      console.log('Error creating user:', error);
      return c.json({ 
        error: error.message,
        suggestion: 'Please create the user manually in Supabase Dashboard',
        instructions: [
          'Go to: https://supabase.com/dashboard',
          'Authentication > Users > Add User',
          'Email: realericzhu@gmail.com',
          'Password: 1234',
          'Auto Confirm Email: ON'
        ]
      }, 400);
    }

    console.log('User created successfully:', data.user.email, 'ID:', data.user.id);

    // Initialize user's trading state
    const userId = data.user.id;
    await kv.set(`user:${userId}:initial_capital`, '10000');
    await kv.set(`user:${userId}:current_capital`, '10000');

    console.log('User state initialized for:', userId);

    return c.json({ 
      message: 'User created successfully',
      email: data.user.email
    });
  } catch (error: any) {
    console.log('Setup user error:', error);
    return c.json({ 
      error: 'Failed to create user',
      details: error.message || String(error),
      suggestion: 'Please create the user manually in Supabase Dashboard',
      instructions: [
        'Go to: https://supabase.com/dashboard',
        'Authentication > Users > Add User',
        'Email: realericzhu@gmail.com',
        'Password: 1234',
        'Auto Confirm Email: ON'
      ]
    }, 500);
  }
});

// Get trades endpoint (protected)
app.get("/make-server-c1a53bfc/trades", requireAuth, async (c) => {
  try {
    const userId = c.get('userId');
    const trades = await kv.getByPrefix(`user:${userId}:trade:`);
    
    // Transform trades data
    const tradesArray = trades.map((item: any) => JSON.parse(item.value));
    
    return c.json(tradesArray);
  } catch (error) {
    console.log('Error fetching trades:', error);
    return c.json({ error: 'Failed to fetch trades' }, 500);
  }
});

// Get state endpoint (protected)
app.get("/make-server-c1a53bfc/state", requireAuth, async (c) => {
  try {
    const userId = c.get('userId');
    
    const [initialCapital, currentCapital] = await Promise.all([
      kv.get(`user:${userId}:initial_capital`),
      kv.get(`user:${userId}:current_capital`)
    ]);

    const stateData = [
      { key: 'initial_capital', value: initialCapital ? parseFloat(initialCapital) : 10000 },
      { key: 'current_capital', value: currentCapital ? parseFloat(currentCapital) : 10000 }
    ];

    return c.json(stateData);
  } catch (error) {
    console.log('Error fetching state:', error);
    return c.json({ error: 'Failed to fetch state' }, 500);
  }
});

// Create trade endpoint (protected)
app.post("/make-server-c1a53bfc/trades", requireAuth, async (c) => {
  try {
    const userId = c.get('userId');
    const trade = await c.req.json();
    
    const tradeId = trade.trade_id || crypto.randomUUID();
    await kv.set(`user:${userId}:trade:${tradeId}`, JSON.stringify({
      ...trade,
      trade_id: tradeId
    }));

    // Update current capital if trade is closed
    if (trade.status === 'closed' && trade.pnl) {
      const currentCapital = await kv.get(`user:${userId}:current_capital`);
      const newCapital = parseFloat(currentCapital || '10000') + parseFloat(trade.pnl);
      await kv.set(`user:${userId}:current_capital`, newCapital.toString());
    }

    return c.json({ message: 'Trade created successfully', tradeId });
  } catch (error) {
    console.log('Error creating trade:', error);
    return c.json({ error: 'Failed to create trade' }, 500);
  }
});

// Update state endpoint (protected)
app.post("/make-server-c1a53bfc/state", requireAuth, async (c) => {
  try {
    const userId = c.get('userId');
    const body = await c.req.json();
    
    if (body.initial_capital !== undefined) {
      await kv.set(`user:${userId}:initial_capital`, body.initial_capital.toString());
    }
    
    if (body.current_capital !== undefined) {
      await kv.set(`user:${userId}:current_capital`, body.current_capital.toString());
    }

    return c.json({ message: 'State updated successfully' });
  } catch (error) {
    console.log('Error updating state:', error);
    return c.json({ error: 'Failed to update state' }, 500);
  }
});

// Initialize demo data for new users
app.post("/make-server-c1a53bfc/init-demo", requireAuth, async (c) => {
  try {
    const userId = c.get('userId');
    
    // Check if user already has data
    const existingTrades = await kv.getByPrefix(`user:${userId}:trade:`);
    if (existingTrades.length > 0) {
      return c.json({ message: 'Demo data already exists' });
    }

    // Create demo trades
    const demoTrades = [
      {
        trade_id: crypto.randomUUID(),
        symbol: 'AAPL',
        quantity: 10,
        entry_price: 150.00,
        exit_price: 155.00,
        pnl: 50.00,
        entry_time: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
        exit_time: new Date(Date.now() - 6 * 24 * 60 * 60 * 1000).toISOString(),
        status: 'closed'
      },
      {
        trade_id: crypto.randomUUID(),
        symbol: 'GOOGL',
        quantity: 5,
        entry_price: 2800.00,
        exit_price: 2750.00,
        pnl: -250.00,
        entry_time: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
        exit_time: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
        status: 'closed'
      },
      {
        trade_id: crypto.randomUUID(),
        symbol: 'TSLA',
        quantity: 15,
        entry_price: 700.00,
        exit_price: 720.00,
        pnl: 300.00,
        entry_time: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
        exit_time: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
        status: 'closed'
      },
      {
        trade_id: crypto.randomUUID(),
        symbol: 'MSFT',
        quantity: 8,
        entry_price: 300.00,
        exit_price: null,
        pnl: null,
        entry_time: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
        exit_time: null,
        status: 'open'
      }
    ];

    for (const trade of demoTrades) {
      await kv.set(`user:${userId}:trade:${trade.trade_id}`, JSON.stringify(trade));
    }

    // Update capital with demo profits
    const totalPnl = demoTrades
      .filter(t => t.pnl !== null)
      .reduce((sum, t) => sum + (t.pnl || 0), 0);
    
    await kv.set(`user:${userId}:current_capital`, (10000 + totalPnl).toString());

    return c.json({ message: 'Demo data initialized successfully' });
  } catch (error) {
    console.log('Error initializing demo data:', error);
    return c.json({ error: 'Failed to initialize demo data' }, 500);
  }
});

Deno.serve(app.fetch);