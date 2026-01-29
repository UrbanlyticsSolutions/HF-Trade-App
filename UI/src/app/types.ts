export interface Trade {
  trade_id: string;
  symbol: string;
  quantity: number;
  entry_price: number;
  exit_price: number | null;
  pnl: number;
  entry_time: string;
  exit_time: string | null;
  status: 'open' | 'closed';
}

export interface StateData {
  key: 'initial_capital' | 'current_capital';
  value: number;
}
