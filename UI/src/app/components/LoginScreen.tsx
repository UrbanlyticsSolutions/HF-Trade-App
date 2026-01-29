import { useState, FormEvent } from 'react';
import { projectId } from '/utils/supabase/info';

const API_BASE_URL = `https://${projectId}.supabase.co/functions/v1/make-server-c1a53bfc`;

interface LoginScreenProps {
  onLogin: (email: string, password: string) => Promise<void>;
}

export function LoginScreen({ onLogin }: LoginScreenProps) {
  const [email, setEmail] = useState('realericzhu@gmail.com');
  const [password, setPassword] = useState('1234');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [setupStatus, setSetupStatus] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setSetupStatus(null);
    setIsLoading(true);

    try {
      await onLogin(email, password);
    } catch (err: any) {
      const errorMessage = err.message || 'An error occurred. Please try again.';
      
      // If credentials are invalid, automatically try to create the user
      if (errorMessage.includes('Invalid login credentials')) {
        console.log('User not found, attempting to create user automatically...');
        setSetupStatus('User not found. Creating account automatically...');
        
        try {
          const response = await fetch(`${API_BASE_URL}/setup-user`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            }
          });
          
          const result = await response.json();
          
          if (!response.ok) {
            console.error('Automatic user creation failed:', result);
            setError(`Could not create user automatically.\n\n${result.error || result.message || 'Unknown error'}\n\nPlease create the user manually in Supabase Dashboard:\n1. Go to https://supabase.com/dashboard\n2. Navigate to Authentication > Users\n3. Click "Add User"\n4. Email: realericzhu@gmail.com\n5. Password: 1234\n6. Check "Auto Confirm Email"\n7. Click "Create User"`);
            setSetupStatus(null);
            setIsLoading(false);
            return;
          }
          
          console.log('User created successfully, attempting login again...');
          setSetupStatus('✅ User created! Logging in...');
          
          // Wait a moment for user to be fully created
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          // Try logging in again
          try {
            await onLogin(email, password);
            // If we get here, login succeeded!
          } catch (retryError: any) {
            console.error('Login failed after user creation:', retryError);
            setError('User was created but login failed. Please try logging in again.');
            setSetupStatus(null);
          }
        } catch (setupError: any) {
          console.error('Setup error:', setupError);
          setError(`Network error: ${setupError.message}\n\nPlease create the user manually in Supabase Dashboard:\n1. Go to https://supabase.com/dashboard\n2. Authentication > Users > Add User\n3. Email: realericzhu@gmail.com\n4. Password: 1234\n5. Auto Confirm Email: ON`);
          setSetupStatus(null);
        }
      } else {
        setError(errorMessage);
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div 
      className="min-h-screen flex items-center justify-center p-5"
      style={{ background: '#0f172a' }}
    >
      <div 
        className="w-full max-w-md p-10 rounded-xl shadow-2xl"
        style={{ background: '#1e293b' }}
      >
        <h1 
          className="text-3xl font-bold mb-2"
          style={{ color: '#38bdf8' }}
        >
          Trading Dashboard
        </h1>
        <p 
          className="mb-8 text-sm"
          style={{ color: '#94a3b8' }}
        >
          Sign in to access the dashboard
        </p>

        <form onSubmit={handleSubmit}>
          <div className="mb-5">
            <label 
              htmlFor="email" 
              className="block mb-2 text-sm font-medium"
              style={{ color: '#e2e8f0' }}
            >
              Email
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email"
              required
              className="w-full px-3 py-3 rounded-md text-sm transition-colors focus:outline-none focus:border-[#38bdf8]"
              style={{
                background: '#0f172a',
                border: '1px solid #334155',
                color: '#e2e8f0'
              }}
            />
          </div>

          <div className="mb-5">
            <label 
              htmlFor="password" 
              className="block mb-2 text-sm font-medium"
              style={{ color: '#e2e8f0' }}
            >
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              required
              className="w-full px-3 py-3 rounded-md text-sm transition-colors focus:outline-none focus:border-[#38bdf8]"
              style={{
                background: '#0f172a',
                border: '1px solid #334155',
                color: '#e2e8f0'
              }}
            />
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 rounded-md font-semibold text-base transition-colors disabled:opacity-50 mb-3"
            style={{
              background: '#38bdf8',
              color: '#0f172a'
            }}
          >
            {isLoading ? (setupStatus || 'Logging in...') : 'Login'}
          </button>

          {error && (
            <div className="mb-3">
              <div 
                className="text-sm p-3 rounded-md"
                style={{ 
                  color: '#ef4444',
                  background: '#7f1d1d20',
                  border: '1px solid #ef444450',
                  whiteSpace: 'pre-line'
                }}
              >
                {error}
              </div>
            </div>
          )}

          {setupStatus && (
            <p 
              className="mb-3 text-sm"
              style={{ color: '#10b981' }}
            >
              {setupStatus}
            </p>
          )}

          <div 
            className="mt-4 p-3 rounded-md text-xs"
            style={{ background: '#0f172a', border: '1px solid #334155', color: '#94a3b8' }}
          >
            <p className="font-semibold mb-1" style={{ color: '#38bdf8' }}>Default Credentials:</p>
            <p>Email: realericzhu@gmail.com</p>
            <p>Password: 1234</p>
          </div>
        </form>
      </div>
    </div>
  );
}