import { RefreshCcw, LogOut } from 'lucide-react';

interface DashboardHeaderProps {
  lastUpdated: Date | null;
  onRefresh: () => void;
  onLogout: () => void;
  isLoading?: boolean;
}

export function DashboardHeader({ lastUpdated, onRefresh, onLogout, isLoading }: DashboardHeaderProps) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
      <h1 
        className="text-3xl font-bold"
        style={{ color: '#38bdf8' }}
      >
        Trading Dashboard
      </h1>

      <div className="flex flex-wrap items-center gap-3">
        <span 
          className="text-sm"
          style={{ color: '#94a3b8' }}
        >
          Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'Never'}
        </span>
        
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="flex items-center gap-2 px-5 py-2.5 rounded-md text-sm font-medium transition-all disabled:opacity-50"
          style={{
            background: '#1e293b',
            color: '#38bdf8',
            border: '1px solid #38bdf8'
          }}
        >
          <RefreshCcw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
        
        <button
          onClick={onLogout}
          className="flex items-center gap-2 px-5 py-2.5 rounded-md text-sm font-medium transition-colors"
          style={{
            background: '#ef4444',
            color: 'white'
          }}
        >
          <LogOut className="w-4 h-4" />
          Logout
        </button>
      </div>
    </div>
  );
}
