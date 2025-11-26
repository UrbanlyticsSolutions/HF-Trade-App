import logging
import csv
import os
from datetime import datetime
from typing import List, Dict, Any
from .base_strategy import BaseStrategy

logger = logging.getLogger('BacktestEngine')

class BacktestEngine:
    """
    Engine to run a strategy against historical data.
    """

    def __init__(self, strategy: BaseStrategy, data: List[Dict[str, Any]]):
        self.strategy = strategy
        self.data = data
        self.trades = []
        
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """
        Replays the historical data through the strategy.
        """
        logger.info(f"Starting backtest for {self.strategy.symbol} with {len(self.data)} ticks...")
        
        # Sort data by date just in case
        sorted_data = sorted(self.data, key=lambda x: x.get('date', ''))

        for tick in sorted_data:
            # Normalize data structure to match what live client returns
            # Live client (quote) returns: {'price': ..., 'volume': ...}
            # Historical (1min) returns: {'close': ..., 'volume': ..., 'date': ...}
            # We'll use 'close' as the current price
            
            price = tick.get('close')
            if price is None:
                continue

            # Create a standardized tick object
            tick_data = {
                'price': price,
                'volume': tick.get('volume'),
                'timestamp': tick.get('date')
            }

            # Update Strategy
            order = self.strategy.on_tick(tick_data)

            # Execute Order
            if order:
                self._execute_order(order, price, tick.get('date'))

        self._print_summary()
        self.save_results()

    def _execute_order(self, order: dict, price: float, timestamp: str):
        action = order.get('action')
        quantity = order.get('quantity')
        
        if action and quantity:
            self.strategy.update_position(action, quantity, price)
            self.trades.append({
                'timestamp': timestamp,
                'action': action,
                'quantity': quantity,
                'price': price,
                'cash': self.strategy.cash,
                'position': self.strategy.position
            })

    def _print_summary(self):
        print("\n=== Backtest Summary ===")
        print(f"Initial Cash: $100,000.00")
        
        final_value = self.strategy.cash
        # Add value of current position
        if self.data:
            last_price = self.data[-1].get('close', 0)
            final_value += self.strategy.position * last_price
            
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Return: {((final_value - 100000) / 100000) * 100:.2f}%")
        print(f"Total Trades: {len(self.trades)}")
        print("========================")

    def save_results(self):
        """
        Saves backtest results (summary and trades) to CSV.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"backtest_{self.strategy.symbol}_{timestamp}.csv")
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write Summary Section
                final_value = self.strategy.cash
                if self.data:
                    last_price = self.data[-1].get('close', 0)
                    final_value += self.strategy.position * last_price
                
                writer.writerow(['=== Backtest Summary ==='])
                writer.writerow(['Symbol', self.strategy.symbol])
                writer.writerow(['Initial Cash', 100000.0])
                writer.writerow(['Final Value', final_value])
                writer.writerow(['Return %', ((final_value - 100000) / 100000) * 100])
                writer.writerow(['Total Trades', len(self.trades)])
                writer.writerow([]) # Empty line
                
                # Write Trades Header
                writer.writerow(['=== Trade Log ==='])
                writer.writerow(['Timestamp', 'Action', 'Quantity', 'Price', 'Cash', 'Position'])
                
                # Write Trades
                for trade in self.trades:
                    writer.writerow([
                        trade['timestamp'],
                        trade['action'],
                        trade['quantity'],
                        trade['price'],
                        trade['cash'],
                        trade['position']
                    ])
            
            print(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")
