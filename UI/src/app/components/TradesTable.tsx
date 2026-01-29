import { formatCurrency, formatDateTime } from '@/app/utils/formatters';
import type { Trade } from '@/app/types';

interface TradesTableProps {
  trades: Trade[];
}

export function TradesTable({ trades }: TradesTableProps) {
  const sortedTrades = [...trades].sort((a, b) => 
    new Date(b.entry_time).getTime() - new Date(a.entry_time).getTime()
  );

  if (trades.length === 0) {
    return (
      <div 
        className="p-6 rounded-xl"
        style={{ background: '#1e293b', border: '1px solid #334155' }}
      >
        <h2 className="text-lg font-semibold mb-5" style={{ color: '#e2e8f0' }}>
          Trade History
        </h2>
        <div className="text-center py-10" style={{ color: '#94a3b8' }}>
          No trades found
        </div>
      </div>
    );
  }

  return (
    <div 
      className="p-6 rounded-xl overflow-x-auto"
      style={{ background: '#1e293b', border: '1px solid #334155' }}
    >
      <h2 className="text-lg font-semibold mb-5" style={{ color: '#e2e8f0' }}>
        Trade History
      </h2>
      
      <table className="w-full border-collapse">
        <thead style={{ background: '#0f172a' }}>
          <tr>
            <th 
              className="px-3 py-3 text-left text-sm font-semibold"
              style={{ color: '#94a3b8', borderBottom: '1px solid #334155' }}
            >
              ID
            </th>
            <th 
              className="px-3 py-3 text-left text-sm font-semibold"
              style={{ color: '#94a3b8', borderBottom: '1px solid #334155' }}
            >
              Symbol
            </th>
            <th 
              className="px-3 py-3 text-left text-sm font-semibold"
              style={{ color: '#94a3b8', borderBottom: '1px solid #334155' }}
            >
              Qty
            </th>
            <th 
              className="px-3 py-3 text-left text-sm font-semibold"
              style={{ color: '#94a3b8', borderBottom: '1px solid #334155' }}
            >
              Entry Price
            </th>
            <th 
              className="px-3 py-3 text-left text-sm font-semibold"
              style={{ color: '#94a3b8', borderBottom: '1px solid #334155' }}
            >
              Exit Price
            </th>
            <th 
              className="px-3 py-3 text-left text-sm font-semibold"
              style={{ color: '#94a3b8', borderBottom: '1px solid #334155' }}
            >
              P&L
            </th>
            <th 
              className="px-3 py-3 text-left text-sm font-semibold"
              style={{ color: '#94a3b8', borderBottom: '1px solid #334155' }}
            >
              Entry Time
            </th>
            <th 
              className="px-3 py-3 text-left text-sm font-semibold"
              style={{ color: '#94a3b8', borderBottom: '1px solid #334155' }}
            >
              Exit Time
            </th>
            <th 
              className="px-3 py-3 text-left text-sm font-semibold"
              style={{ color: '#94a3b8', borderBottom: '1px solid #334155' }}
            >
              Status
            </th>
          </tr>
        </thead>
        <tbody>
          {sortedTrades.map((trade, index) => (
            <tr 
              key={index}
              className="transition-colors hover:bg-[#0f172a]"
            >
              <td 
                className="px-3 py-3 text-sm"
                style={{ color: '#e2e8f0', borderBottom: '1px solid #334155' }}
              >
                {trade.trade_id || '-'}
              </td>
              <td 
                className="px-3 py-3 text-sm"
                style={{ color: '#e2e8f0', borderBottom: '1px solid #334155' }}
              >
                {trade.symbol || '-'}
              </td>
              <td 
                className="px-3 py-3 text-sm"
                style={{ color: '#e2e8f0', borderBottom: '1px solid #334155' }}
              >
                {trade.quantity || '-'}
              </td>
              <td 
                className="px-3 py-3 text-sm"
                style={{ color: '#e2e8f0', borderBottom: '1px solid #334155' }}
              >
                {formatCurrency(trade.entry_price)}
              </td>
              <td 
                className="px-3 py-3 text-sm"
                style={{ color: '#e2e8f0', borderBottom: '1px solid #334155' }}
              >
                {trade.exit_price ? formatCurrency(trade.exit_price) : '-'}
              </td>
              <td 
                className="px-3 py-3 text-sm font-semibold"
                style={{ 
                  color: trade.pnl > 0 ? '#22c55e' : trade.pnl < 0 ? '#ef4444' : '#e2e8f0',
                  borderBottom: '1px solid #334155' 
                }}
              >
                {formatCurrency(trade.pnl || 0)}
              </td>
              <td 
                className="px-3 py-3 text-sm"
                style={{ color: '#e2e8f0', borderBottom: '1px solid #334155' }}
              >
                {formatDateTime(trade.entry_time)}
              </td>
              <td 
                className="px-3 py-3 text-sm"
                style={{ color: '#e2e8f0', borderBottom: '1px solid #334155' }}
              >
                {formatDateTime(trade.exit_time)}
              </td>
              <td 
                className="px-3 py-3 text-sm"
                style={{ borderBottom: '1px solid #334155' }}
              >
                <span 
                  className="px-2 py-1 rounded text-xs font-medium"
                  style={{
                    background: trade.status === 'open' ? '#38bdf8' : '#334155',
                    color: trade.status === 'open' ? '#0f172a' : '#e2e8f0'
                  }}
                >
                  {trade.status || '-'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
