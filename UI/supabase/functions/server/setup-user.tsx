// One-time setup script to create the authorized user
// This should be run once to create the user realericzhu@gmail with password 1234

import { createClient } from "npm:@supabase/supabase-js@2";

const setupUser = async () => {
  const supabaseUrl = Deno.env.get('SUPABASE_URL') ?? '';
  const supabaseServiceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '';
  
  const supabase = createClient(supabaseUrl, supabaseServiceRoleKey);

  try {
    // Create the user
    const { data, error } = await supabase.auth.admin.createUser({
      email: 'realericzhu@gmail.com',
      password: '1234',
      user_metadata: { name: 'Eric Zhu' },
      // Automatically confirm the user's email since an email server hasn't been configured.
      email_confirm: true
    });

    if (error) {
      console.error('Error creating user:', error);
      return;
    }

    console.log('User created successfully:', data.user.email);
    console.log('User ID:', data.user.id);

    // Initialize user's trading state
    const userId = data.user.id;
    
    // Note: You would need to import kv_store here if you want to initialize
    // the user's capital. For now, this will be done when they first log in
    // through the init-demo endpoint.
    
    console.log('Setup complete!');
  } catch (err) {
    console.error('Setup error:', err);
  }
};

// Run the setup if this file is executed directly
if (import.meta.main) {
  setupUser();
}