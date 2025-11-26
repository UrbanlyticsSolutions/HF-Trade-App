import time
import logging
import csv
import os
from datetime import datetime
from typing import Optional
from .base_strategy import BaseStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TradingEngine')

class TradingEngine:
    """
    Main execution engine for the HFT strategy.
    Handles data fetching loop and passes data to the strategy.
    """

    def __init__(self, client, strategy: BaseStrategy, interval: float = 1.0):
        """
        Args:
            client: API client instance (must have a quote() method)
            strategy: Instance of a class inheriting from BaseStrategy
            interval: Polling interval in seconds
        """
        self.client = client
        self.strategy = strategy
        self.interval = interval
        self.is_running = False
        
        # Setup CSV logging
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.output_dir, f"live_trades_{strategy.symbol}_{timestamp}.csv")
        
        # Initialize CSV with header
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Symbol', 'Action', 'Quantity', 'Price', 'Cost_Revenue'])
            
        logger.info(f"Logging trades to {self.log_file}")

    def run_live(self):
        """
        Starts the main trading loop.
        """
        self.is_running = True
        logger.info(f"Starting trading engine for {self.strategy.symbol} with {self.interval}s interval...")

        try:
            while self.is_running:
                try:
                    # 1. Fetch Data
                    start_time = time.time()
                    quotes = self.client.quote(self.strategy.symbol)
                    
                    if not quotes:
                        logger.warning(f"No quote data received for {self.strategy.symbol}")
                        continue
                        
                    # Assuming the client returns a list of dicts, take the first one
                    current_data = quotes[0]
                    price = current_data.get('price')
                    
                    if price is None:
                        logger.warning(f"Price missing in data: {current_data}")
                        continue

                    logger.info(f"Tick: {self.strategy.symbol} @ ${price}")

                    # 2. Update Strategy
                    order = self.strategy.on_tick(current_data)

                    # 3. Execute Order (Mock)
                    if order:
                        self._execute_order(order, price)

                    # 4. Sleep for remainder of interval
                    elapsed = time.time() - start_time
                    sleep_time = max(0, self.interval - elapsed)
                    time.sleep(sleep_time)

                except Exception as e:
                    logger.error(f"Error in trading loop: {e}", exc_info=True)
                    time.sleep(self.interval) # Sleep on error to avoid rapid looping

        except KeyboardInterrupt:
            logger.info("Stopping trading engine...")
        finally:
            self.is_running = False

    def _execute_order(self, order: dict, price: float):
        """
        Simulates order execution.
        """
        action = order.get('action')
        quantity = order.get('quantity')
        
        if action and quantity:
            logger.info(f"*** EXECUTING ORDER: {action} {quantity} @ ${price} ***")
            # Update strategy state (simulating fill)
            self.strategy.update_position(action, quantity, price)
            
            # Log to CSV
            try:
                cost_revenue = quantity * price
                if action == 'BUY':
                    cost_revenue = -cost_revenue
                
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        self.strategy.symbol,
                        action,
                        quantity,
                        price,
                        cost_revenue
                    ])
            except Exception as e:
                logger.error(f"Failed to log trade to CSV: {e}")
