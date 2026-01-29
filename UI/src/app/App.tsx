import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { LoginScreen } from '@/app/components/LoginScreen';
import { DashboardHeader } from '@/app/components/DashboardHeader';
import { StatsCard } from '@/app/components/StatsCard';
import { TradesTable } from '@/app/components/TradesTable';
import { LoadingOverlay } from '@/app/components/LoadingOverlay';
import { formatCurrency, formatPercent } from '@/app/utils/formatters';
import type { Trade, StateData } from '@/app/types';
import { getSupabaseClient } from '@/app/utils/supabase-client';
import { projectId } from '/utils/supabase/info';

const API_BASE_URL = `https://${projectId}.supabase.co/functions/v1/trading-api`;

// Get singleton Supabase client instance
const supabase = getSupabaseClient();

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [accessToken, setAccessToken] = useState<string>('');
  const [trades, setTrades] = useState<Trade[]>([]);
  const [initialCapital, setInitialCapital] = useState(0);
  const [currentCapital, setCurrentCapital] = useState(0);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [initializingDemo, setInitializingDemo] = useState(false);
  const [checkingSession, setCheckingSession] = useState(true);

  // Check for existing session on mount
  useEffect(() => {
    checkSession();
  }, []);

  // Auto-refresh when authenticated
  useEffect(() => {
    if (isAuthenticated && accessToken) {
      loadDashboardData();
      const interval = setInterval(loadDashboardData, 30000);
      return () => clearInterval(interval);
    }
  }, [isAuthenticated, accessToken]);

  const checkSession = async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      
      if (session?.access_token) {
        console.log('Active session found, logging in automatically');
        setAccessToken(session.access_token);
        setIsAuthenticated(true);
        
        // Fetch data
        await Promise.all([
          fetchTrades(session.access_token),
          fetchState(session.access_token),
        ]);
        
        setCheckingSession(false);
        return true;
      }
      
      setCheckingSession(false);
      return false;
    } catch (error) {
      console.error('Session check error:', error);
      setCheckingSession(false);
      return false;
    }
  };

  const apiCall = async (endpoint: string, options: RequestInit = {}) => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`,
        ...options.headers,
      }
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || `API error: ${response.status}`);
    }

    return await response.json();
  };

  const fetchTrades = async (token: string): Promise<Trade[]> => {
    try {
      const response = await fetch(`${API_BASE_URL}/trades`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch trades: ${response.status}`);
      }
      
      const data = await response.json();
      setTrades(data);
      return data;
    } catch (error) {
      console.error('Error fetching trades:', error);
      return [];
    }
  };

  const fetchState = async (token: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/state`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch state: ${response.status}`);
      }
      
      const stateData: StateData[] = await response.json();
      const state: Record<string, number> = {};
      stateData.forEach(item => {
        state[item.key] = item.value;
      });
      
      setInitialCapital(state.initial_capital || 0);
      setCurrentCapital(state.current_capital || 0);
      setLastUpdated(new Date());
      
      return state;
    } catch (error) {
      console.error('Error fetching state:', error);
      return {};
    }
  };

  const initializeDemoData = async (token: string) => {
    try {
      setInitializingDemo(true);
      const response = await fetch(`${API_BASE_URL}/init-demo`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to initialize demo data: ${response.status}`);
      }
      
      // Reload data after initializing demo
      await Promise.all([
        fetchTrades(token),
        fetchState(token),
      ]);
    } catch (error) {
      console.error('Error initializing demo data:', error);
    } finally {
      setInitializingDemo(false);
    }
  };

  const loadDashboardData = async () => {
    try {
      setIsLoading(true);
      const [tradesData, stateData] = await Promise.all([
        apiCall('/trades'),
        apiCall('/state')
      ]);

      const state: Record<string, number> = {};
      (stateData as StateData[]).forEach(item => {
        state[item.key] = item.value;
      });

      setTrades(tradesData);
      setInitialCapital(state.initial_capital || 0);
      setCurrentCapital(state.current_capital || 0);
      setLastUpdated(new Date());

      // Initialize demo data if no trades exist
      if (tradesData.length === 0 && !initializingDemo) {
        setInitializingDemo(true);
        try {
          await apiCall('/init-demo', { method: 'POST' });
          // Reload data after initializing demo
          const [newTradesData, newStateData] = await Promise.all([
            apiCall('/trades'),
            apiCall('/state')
          ]);
          
          const newState: Record<string, number> = {};
          (newStateData as StateData[]).forEach(item => {
            newState[item.key] = item.value;
          });

          setTrades(newTradesData);
          setInitialCapital(newState.initial_capital || 0);
          setCurrentCapital(newState.current_capital || 0);
        } catch (demoError) {
          console.error('Error initializing demo data:', demoError);
        } finally {
          setInitializingDemo(false);
        }
      }
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogin = async (email: string, password: string) => {
    console.log('Attempting login for:', email);
    
    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) {
        console.error('Supabase auth error:', error);
        throw error;
      }

      if (!data.session?.access_token) {
        throw new Error('No access token received');
      }

      console.log('Login successful, token received');
      
      setAccessToken(data.session.access_token);
      setIsAuthenticated(true);

      // Fetch data
      const [tradesResult] = await Promise.all([
        fetchTrades(data.session.access_token),
        fetchState(data.session.access_token),
      ]);

      // Initialize demo data if no trades exist
      if (tradesResult.length === 0) {
        await initializeDemoData(data.session.access_token);
      }
    } catch (error: any) {
      console.error('Login error:', error);
      throw error;
    }
  };

  const handleLogout = async () => {
    try {
      await supabase.auth.signOut();
      setAccessToken('');
      setIsAuthenticated(false);
      setTrades([]);
      setInitialCapital(0);
      setCurrentCapital(0);
      setLastUpdated(null);
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  // Calculate stats
  const totalPnL = currentCapital - initialCapital;
  const totalPnLPercent = initialCapital > 0 ? (totalPnL / initialCapital) * 100 : 0;
  const closedTrades = trades.filter(t => t.status === 'closed');
  const winningTrades = closedTrades.filter(t => t.pnl > 0).length;
  const losingTrades = closedTrades.filter(t => t.pnl < 0).length;
  const winRate = closedTrades.length > 0 ? (winningTrades / closedTrades.length) * 100 : 0;

  // Prepare equity curve data - cumulative P&L from trades
  const sortedTrades = [...trades]
    .filter(t => t.status === 'closed' && t.pnl !== null && t.pnl !== undefined)
    .sort((a, b) => {
      const timeA = new Date(a.exit_time || a.entry_time).getTime();
      const timeB = new Date(b.exit_time || b.entry_time).getTime();
      return timeA - timeB;
    });

  // Build equity curve with cumulative P&L
  let cumulativePnL = 0;
  const equityData = sortedTrades.length > 0
    ? sortedTrades.map((trade) => {
        cumulativePnL += (trade.pnl || 0);
        return {
          x: new Date(trade.exit_time || trade.entry_time),
          y: initialCapital + cumulativePnL
        };
      })
    : [{ x: new Date(), y: initialCapital }];

  // Add starting point at beginning
  if (sortedTrades.length > 0) {
    equityData.unshift({
      x: new Date(sortedTrades[0].entry_time),
      y: initialCapital
    });
  }

  // Calculate Y-axis range - zoom to daily P&L range with padding
  const yValues = equityData.map(d => d.y);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const yRange = maxY - minY;
  const yPadding = Math.max(yRange * 0.1, 50); // 10% padding or at least $50

  // Show nothing while checking session
  if (checkingSession) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: '#0f172a' }}>
        <div style={{ color: '#38bdf8' }}>Loading...</div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginScreen onLogin={handleLogin} />;
  }

  return (
    <div className="min-h-screen p-5 md:p-8" style={{ background: '#0f172a', color: '#e2e8f0' }}>
      {initializingDemo && <LoadingOverlay message="Initializing your demo trading data..." />}
      
      <div className="max-w-[1600px] mx-auto">
        <DashboardHeader 
          lastUpdated={lastUpdated}
          onRefresh={loadDashboardData}
          onLogout={handleLogout}
          isLoading={isLoading}
        />

        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-5 mb-8">
          <StatsCard
            label="Initial Capital"
            value={formatCurrency(initialCapital)}
          />
          <StatsCard
            label="Current Capital"
            value={formatCurrency(currentCapital)}
          />
          <StatsCard
            label="Total P&L"
            value={formatCurrency(totalPnL)}
            subtext={formatPercent(totalPnLPercent)}
            variant={totalPnL >= 0 ? 'positive' : 'negative'}
          />
          <StatsCard
            label="Win Rate"
            value={formatPercent(winRate)}
            subtext={`${winningTrades}W / ${losingTrades}L`}
          />
          <StatsCard
            label="Total Trades"
            value={trades.length.toString()}
          />
        </div>

        {/* Equity Curve Chart */}
        <div className="mb-8 p-6 rounded-xl" style={{ background: '#1e293b', border: '1px solid #334155' }}>
          <div className="flex justify-between items-center mb-5">
            <h2 className="text-lg font-semibold" style={{ color: '#e2e8f0' }}>
              Equity Curve
            </h2>
            <div className="text-sm" style={{ color: totalPnL >= 0 ? '#22c55e' : '#ef4444' }}>
              Daily P&L: {formatCurrency(totalPnL)} ({formatPercent(totalPnLPercent)})
            </div>
          </div>
          <Plot
            data={[
              {
                x: equityData.map(d => d.x),
                y: equityData.map(d => d.y),
                type: 'scatter',
                mode: 'lines+markers',
                line: {
                  color: totalPnL >= 0 ? '#22c55e' : '#ef4444',
                  width: 2
                },
                marker: {
                  size: 6,
                  color: totalPnL >= 0 ? '#22c55e' : '#ef4444'
                },
                fill: 'tonexty',
                fillcolor: totalPnL >= 0 ? 'rgba(34, 197, 94, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                hovertemplate: '%{x|%H:%M:%S}<br>$%{y:,.0f}<extra></extra>'
              },
              // Baseline at initial capital
              {
                x: equityData.map(d => d.x),
                y: equityData.map(() => initialCapital),
                type: 'scatter',
                mode: 'lines',
                line: {
                  color: '#64748b',
                  width: 1,
                  dash: 'dash'
                },
                hoverinfo: 'skip',
                showlegend: false
              }
            ]}
            layout={{
              paper_bgcolor: '#1e293b',
              plot_bgcolor: '#1e293b',
              font: {
                color: '#e2e8f0'
              },
              xaxis: {
                gridcolor: '#334155',
                title: 'Date'
              },
              yaxis: {
                gridcolor: '#334155',
                title: 'Equity ($)',
                range: [minY - yPadding, maxY + yPadding],
                tickformat: '$,.0f'
              },
              margin: { t: 20, r: 20, b: 40, l: 80 },
              hovermode: 'closest',
              autosize: true
            }}
            config={{
              responsive: true,
              displayModeBar: false
            }}
            style={{ width: '100%', height: '400px' }}
            useResizeHandler
          />
        </div>

        {/* Trades Table */}
        <TradesTable trades={trades} />
      </div>
    </div>
  );
}