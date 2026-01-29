import { useState } from 'react';
import { projectId } from '/utils/supabase/info';

const API_BASE_URL = `https://${projectId}.supabase.co/functions/v1/make-server-c1a53bfc`;

interface SetupScreenProps {
  onSetupComplete: () => void;
}

export function SetupScreen({ onSetupComplete }: SetupScreenProps) {
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState('');
  const [showManualInstructions, setShowManualInstructions] = useState(false);

  const handleAutomaticSetup = async () => {
    setIsCreating(true);
    setError('');
    
    try {
      const response = await fetch(`${API_BASE_URL}/setup-user`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      const result = await response.json();
      
      if (!response.ok) {
        console.error('Automatic setup failed:', result);
        setError(result.error || result.message || 'Automatic setup failed');
        setShowManualInstructions(true);
        setIsCreating(false);
        return;
      }
      
      // Success!
      alert('✅ User created successfully! Click OK to continue to login.');
      onSetupComplete();
    } catch (error: any) {
      console.error('Setup error:', error);
      setError(`Network error: ${error.message}`);
      setShowManualInstructions(true);
      setIsCreating(false);
    }
  };

  return (
    <div 
      className="min-h-screen flex items-center justify-center p-5"
      style={{ background: '#0f172a' }}
    >
      <div 
        className="w-full max-w-2xl p-10 rounded-xl shadow-2xl"
        style={{ background: '#1e293b' }}
      >
        <h1 
          className="text-3xl font-bold mb-2"
          style={{ color: '#38bdf8' }}
        >
          🔧 First-Time Setup Required
        </h1>
        <p 
          className="mb-8 text-base"
          style={{ color: '#94a3b8' }}
        >
          The trading dashboard user account needs to be created before you can log in.
        </p>

        {!showManualInstructions ? (
          <>
            <div 
              className="mb-6 p-5 rounded-lg"
              style={{ background: '#0f172a', border: '1px solid #334155' }}
            >
              <h2 
                className="text-lg font-semibold mb-3"
                style={{ color: '#38bdf8' }}
              >
                Option 1: Automatic Setup (Try First)
              </h2>
              <p className="text-sm mb-4" style={{ color: '#cbd5e1' }}>
                Click the button below to automatically create the user account.
              </p>
              <button
                onClick={handleAutomaticSetup}
                disabled={isCreating}
                className="w-full py-3 rounded-md font-semibold text-base transition-colors disabled:opacity-50"
                style={{
                  background: '#10b981',
                  color: '#fff'
                }}
              >
                {isCreating ? 'Creating User Account...' : '✨ Create User Automatically'}
              </button>
            </div>

            <div className="text-center mb-4" style={{ color: '#64748b' }}>
              <span className="text-sm">OR</span>
            </div>

            <div 
              className="p-5 rounded-lg"
              style={{ background: '#0f172a', border: '1px solid #334155' }}
            >
              <h2 
                className="text-lg font-semibold mb-3"
                style={{ color: '#38bdf8' }}
              >
                Option 2: Manual Setup
              </h2>
              <p className="text-sm mb-4" style={{ color: '#cbd5e1' }}>
                If automatic setup doesn't work, you can create the user manually.
              </p>
              <button
                onClick={() => setShowManualInstructions(true)}
                className="w-full py-2 rounded-md font-medium text-base transition-colors"
                style={{
                  background: '#334155',
                  color: '#e2e8f0'
                }}
              >
                📖 Show Manual Setup Instructions
              </button>
            </div>

            {error && (
              <div 
                className="mt-6 p-4 rounded-md text-sm"
                style={{ 
                  background: '#7f1d1d20',
                  border: '1px solid #ef444450',
                  color: '#ef4444'
                }}
              >
                <strong>Error:</strong> {error}
              </div>
            )}
          </>
        ) : (
          <div 
            className="p-6 rounded-lg"
            style={{ background: '#0f172a', border: '2px solid #38bdf8' }}
          >
            <h2 
              className="text-xl font-bold mb-4"
              style={{ color: '#38bdf8' }}
            >
              📋 Manual Setup Instructions
            </h2>
            
            <div className="space-y-3 mb-6">
              <div className="flex items-start gap-3">
                <span 
                  className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                  style={{ background: '#38bdf8', color: '#0f172a' }}
                >
                  1
                </span>
                <p className="text-sm" style={{ color: '#e2e8f0' }}>
                  Go to your <strong>Supabase Dashboard</strong>:
                  <br />
                  <a 
                    href="https://supabase.com/dashboard" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="underline"
                    style={{ color: '#38bdf8' }}
                  >
                    https://supabase.com/dashboard
                  </a>
                </p>
              </div>

              <div className="flex items-start gap-3">
                <span 
                  className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                  style={{ background: '#38bdf8', color: '#0f172a' }}
                >
                  2
                </span>
                <p className="text-sm" style={{ color: '#e2e8f0' }}>
                  Navigate to: <strong>Authentication → Users</strong>
                </p>
              </div>

              <div className="flex items-start gap-3">
                <span 
                  className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                  style={{ background: '#38bdf8', color: '#0f172a' }}
                >
                  3
                </span>
                <p className="text-sm" style={{ color: '#e2e8f0' }}>
                  Click <strong>"Add User"</strong> or <strong>"Invite"</strong> button
                </p>
              </div>

              <div className="flex items-start gap-3">
                <span 
                  className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                  style={{ background: '#38bdf8', color: '#0f172a' }}
                >
                  4
                </span>
                <div className="flex-1">
                  <p className="text-sm mb-2" style={{ color: '#e2e8f0' }}>
                    Fill in the form with these <strong>exact values</strong>:
                  </p>
                  <div 
                    className="p-3 rounded-md font-mono text-xs"
                    style={{ background: '#1e293b', border: '1px solid #334155', color: '#38bdf8' }}
                  >
                    <div className="mb-1">
                      <span style={{ color: '#94a3b8' }}>Email:</span> realericzhu@gmail.com
                    </div>
                    <div>
                      <span style={{ color: '#94a3b8' }}>Password:</span> 1234
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <span 
                  className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                  style={{ background: '#ef4444', color: '#fff' }}
                >
                  ⚠️
                </span>
                <p className="text-sm font-semibold" style={{ color: '#ef4444' }}>
                  IMPORTANT: Check "Auto Confirm Email" or "Email Confirmed"
                </p>
              </div>

              <div className="flex items-start gap-3">
                <span 
                  className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                  style={{ background: '#38bdf8', color: '#0f172a' }}
                >
                  5
                </span>
                <p className="text-sm" style={{ color: '#e2e8f0' }}>
                  Click <strong>"Create User"</strong>
                </p>
              </div>

              <div className="flex items-start gap-3">
                <span 
                  className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                  style={{ background: '#10b981', color: '#fff' }}
                >
                  ✓
                </span>
                <p className="text-sm" style={{ color: '#e2e8f0' }}>
                  Come back to this page and click the button below
                </p>
              </div>
            </div>

            <button
              onClick={onSetupComplete}
              className="w-full py-3 rounded-md font-semibold text-base transition-colors"
              style={{
                background: '#38bdf8',
                color: '#0f172a'
              }}
            >
              ✅ I've Created the User - Continue to Login
            </button>

            <button
              onClick={() => setShowManualInstructions(false)}
              className="w-full mt-3 py-2 rounded-md font-medium text-sm transition-colors"
              style={{
                background: '#334155',
                color: '#e2e8f0'
              }}
            >
              ← Back to Automatic Setup
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
