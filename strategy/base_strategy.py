from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.position = 0  # Current position size (positive for long, negative for short)
        self.cash = 100000.0  # Starting simulated cash
        self.holdings = 0.0 # Current holdings value

    @abstractmethod
    def on_tick(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Called when new market data is received.
        
        Args:
            data: Dictionary containing market data (price, volume, etc.)
            
        Returns:
            A dictionary representing an order (e.g., {'action': 'BUY', 'quantity': 10})
            or None if no action is taken.
        """
        pass

    def update_position(self, action: str, quantity: int, price: float):
        """
        Updates the internal position state based on executed trades.
        """
        if action == 'BUY':
            cost = quantity * price
            if self.cash >= cost:
                self.cash -= cost
                self.position += quantity
                print(f"[STRATEGY] Bought {quantity} {self.symbol} at {price}. Cash: {self.cash:.2f}, Pos: {self.position}")
            else:
                print(f"[STRATEGY] Insufficient funds to buy {quantity} {self.symbol} at {price}")
        elif action == 'SELL':
            if self.position >= quantity:
                revenue = quantity * price
                self.cash += revenue
                self.position -= quantity
                print(f"[STRATEGY] Sold {quantity} {self.symbol} at {price}. Cash: {self.cash:.2f}, Pos: {self.position}")
            else:
                print(f"[STRATEGY] Insufficient position to sell {quantity} {self.symbol}")
