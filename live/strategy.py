"""
Strategy Base Class - Abstract base for trading strategies

Provides:
- Standard interface for strategies
- Event hooks for quotes, fills, and signals
- Position and order management integration
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal from strategy"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: int = 1
    order_type: str = "LIMIT"  # MARKET, LIMIT
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    confidence: float = 0.0  # 0-1
    strategy_name: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptionQuote:
    """Option quote data"""
    symbol: str
    underlying: str
    underlying_price: float
    strike: float
    expiration: str
    option_type: str  # call/put
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    iv: Optional[float] = None
    timestamp: str = ""


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Subclass and implement the abstract methods to create a strategy.
    """
    
    def __init__(self, name: str, symbols: List[str] = None):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
        """
        self.name = name
        self.symbols = symbols or []
        self.is_active = True
        self._position_manager = None
        self._order_manager = None
        self._current_positions: Dict[str, int] = {}
        self._pending_signals: List[Signal] = []
        
    def set_managers(self, position_manager, order_manager):
        """Set the position and order managers"""
        self._position_manager = position_manager
        self._order_manager = order_manager
    
    @abstractmethod
    def on_quote(self, symbol: str, quote: Dict[str, Any]) -> Optional[Signal]:
        """
        Called when a new quote arrives.
        
        Args:
            symbol: Symbol for the quote
            quote: Quote data dict
            
        Returns:
            Signal if action should be taken, None otherwise
        """
        pass
    
    @abstractmethod
    def on_option_quote(self, quote: OptionQuote) -> Optional[Signal]:
        """
        Called when a new option quote arrives.
        
        Args:
            quote: OptionQuote object
            
        Returns:
            Signal if action should be taken, None otherwise
        """
        pass
    
    def on_fill(self, order: Any) -> None:
        """
        Called when an order is filled.
        
        Args:
            order: Filled order object
        """
        logger.info(f"[{self.name}] Fill: {order.side} {order.filled_quantity} {order.symbol} @ ${order.avg_fill_price:.2f}")
    
    def on_start(self) -> None:
        """Called when strategy starts"""
        logger.info(f"Strategy {self.name} started")
    
    def on_stop(self) -> None:
        """Called when strategy stops"""
        logger.info(f"Strategy {self.name} stopped")
    
    def on_market_open(self) -> None:
        """Called at market open"""
        pass
    
    def on_market_close(self) -> None:
        """Called at market close"""
        pass
    
    def get_position(self, symbol: str) -> int:
        """Get current position for symbol"""
        if self._position_manager:
            pos = self._position_manager.get_position(symbol)
            return pos.quantity if pos else 0
        return self._current_positions.get(symbol, 0)
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in symbol"""
        return self.get_position(symbol) != 0
    
    def create_signal(
        self,
        symbol: str,
        action: str,
        quantity: int = 1,
        limit_price: Optional[float] = None,
        reason: str = "",
        confidence: float = 1.0,
        **metadata
    ) -> Signal:
        """
        Create a trading signal.
        
        Args:
            symbol: Symbol to trade
            action: BUY, SELL, HOLD
            quantity: Quantity to trade
            limit_price: Limit price (None for market)
            reason: Reason for the signal
            confidence: Confidence level 0-1
            **metadata: Additional metadata
            
        Returns:
            Signal object
        """
        return Signal(
            symbol=symbol,
            action=action,
            quantity=quantity,
            order_type="LIMIT" if limit_price else "MARKET",
            limit_price=limit_price,
            reason=reason,
            confidence=confidence,
            strategy_name=self.name,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        )


class OptionStrategy(Strategy):
    """
    Base class for option-specific strategies.
    
    Provides additional helpers for option trading.
    """
    
    def __init__(self, name: str, underlyings: List[str] = None):
        """
        Initialize option strategy.
        
        Args:
            name: Strategy name
            underlyings: List of underlying symbols to trade options on
        """
        super().__init__(name, underlyings)
        self.underlyings = underlyings or []
        self.max_delta = 0.30  # Max delta for entries
        self.min_dte = 7  # Minimum days to expiration
        self.max_dte = 45  # Maximum days to expiration
        self.min_iv = 0.15  # Minimum IV for entries
        self.max_iv = 1.0  # Maximum IV for entries
    
    @abstractmethod
    def on_option_quote(self, quote: OptionQuote) -> Optional[Signal]:
        """
        Called when a new option quote arrives.
        
        Must be implemented by subclasses.
        """
        pass
    
    def on_quote(self, symbol: str, quote: Dict[str, Any]) -> Optional[Signal]:
        """Default implementation - override if needed"""
        return None
    
    def filter_options(
        self,
        options: List[OptionQuote],
        option_type: Optional[str] = None,
        min_delta: Optional[float] = None,
        max_delta: Optional[float] = None,
        min_dte: Optional[int] = None,
        max_dte: Optional[int] = None,
        min_iv: Optional[float] = None,
        max_iv: Optional[float] = None
    ) -> List[OptionQuote]:
        """
        Filter options based on criteria.
        
        Args:
            options: List of OptionQuote objects
            option_type: 'call' or 'put' filter
            min_delta: Minimum absolute delta
            max_delta: Maximum absolute delta
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            min_iv: Minimum implied volatility
            max_iv: Maximum implied volatility
            
        Returns:
            Filtered list of options
        """
        filtered = options
        
        if option_type:
            filtered = [o for o in filtered if o.option_type == option_type]
        
        if min_delta is not None:
            filtered = [o for o in filtered if o.delta and abs(o.delta) >= min_delta]
        
        if max_delta is not None:
            filtered = [o for o in filtered if o.delta and abs(o.delta) <= max_delta]
        
        if min_iv is not None:
            filtered = [o for o in filtered if o.iv and o.iv >= min_iv]
        
        if max_iv is not None:
            filtered = [o for o in filtered if o.iv and o.iv <= max_iv]
        
        # DTE filtering requires parsing expiration
        if min_dte is not None or max_dte is not None:
            today = datetime.now().date()
            filtered_dte = []
            for o in filtered:
                try:
                    exp_date = datetime.strptime(o.expiration, "%Y-%m-%d").date()
                    dte = (exp_date - today).days
                    if min_dte is not None and dte < min_dte:
                        continue
                    if max_dte is not None and dte > max_dte:
                        continue
                    filtered_dte.append(o)
                except:
                    pass
            filtered = filtered_dte
        
        return filtered
    
    def find_atm_options(
        self,
        options: List[OptionQuote],
        underlying_price: float,
        option_type: Optional[str] = None
    ) -> List[OptionQuote]:
        """
        Find at-the-money options.
        
        Args:
            options: List of options
            underlying_price: Current underlying price
            option_type: Optional 'call' or 'put' filter
            
        Returns:
            Options sorted by closeness to ATM
        """
        if option_type:
            options = [o for o in options if o.option_type == option_type]
        
        # Sort by distance from ATM
        return sorted(options, key=lambda o: abs(o.strike - underlying_price))
    
    def find_otm_puts(
        self,
        options: List[OptionQuote],
        underlying_price: float,
        min_otm_percent: float = 5.0
    ) -> List[OptionQuote]:
        """
        Find out-of-the-money puts.
        
        Args:
            options: List of options
            underlying_price: Current underlying price
            min_otm_percent: Minimum OTM percentage
            
        Returns:
            OTM puts sorted by strike (highest first)
        """
        puts = [o for o in options if o.option_type == 'put']
        max_strike = underlying_price * (1 - min_otm_percent / 100)
        otm_puts = [p for p in puts if p.strike <= max_strike]
        return sorted(otm_puts, key=lambda o: o.strike, reverse=True)
    
    def find_otm_calls(
        self,
        options: List[OptionQuote],
        underlying_price: float,
        min_otm_percent: float = 5.0
    ) -> List[OptionQuote]:
        """
        Find out-of-the-money calls.
        
        Args:
            options: List of options
            underlying_price: Current underlying price
            min_otm_percent: Minimum OTM percentage
            
        Returns:
            OTM calls sorted by strike (lowest first)
        """
        calls = [o for o in options if o.option_type == 'call']
        min_strike = underlying_price * (1 + min_otm_percent / 100)
        otm_calls = [c for c in calls if c.strike >= min_strike]
        return sorted(otm_calls, key=lambda o: o.strike)
    
    def calculate_credit(self, sell_option: OptionQuote, buy_option: Optional[OptionQuote] = None) -> float:
        """
        Calculate credit for a spread.
        
        Args:
            sell_option: Option to sell
            buy_option: Option to buy (for spreads)
            
        Returns:
            Net credit per contract
        """
        credit = sell_option.bid
        if buy_option:
            credit -= buy_option.ask
        return max(0, credit)
    
    def calculate_max_profit(self, credit: float, contracts: int = 1) -> float:
        """Calculate max profit for credit strategy"""
        return credit * 100 * contracts
    
    def calculate_max_loss(
        self,
        sell_option: OptionQuote,
        buy_option: Optional[OptionQuote],
        credit: float,
        contracts: int = 1
    ) -> float:
        """
        Calculate max loss for a spread.
        
        Args:
            sell_option: Short option
            buy_option: Long option (for defined risk)
            credit: Credit received
            contracts: Number of contracts
            
        Returns:
            Max loss (positive number)
        """
        if buy_option:
            # Vertical spread - defined risk
            width = abs(sell_option.strike - buy_option.strike)
            return (width - credit) * 100 * contracts
        else:
            # Naked - undefined risk
            return float('inf')


# Example concrete strategies

class CoveredCallStrategy(OptionStrategy):
    """
    Covered call strategy.
    
    Sells OTM calls against long stock positions.
    """
    
    def __init__(self, underlyings: List[str], target_delta: float = 0.30):
        super().__init__("CoveredCall", underlyings)
        self.target_delta = target_delta
    
    def on_option_quote(self, quote: OptionQuote) -> Optional[Signal]:
        """Check if we should sell a covered call"""
        if quote.option_type != 'call':
            return None
        
        if quote.underlying not in self.underlyings:
            return None
        
        # Check if we have stock
        stock_position = self.get_position(quote.underlying)
        if stock_position < 100:
            return None
        
        # Check if we already have a call
        if self.has_position(quote.symbol):
            return None
        
        # Check delta
        if quote.delta and abs(quote.delta) <= self.target_delta:
            # Good candidate for covered call
            return self.create_signal(
                symbol=quote.symbol,
                action="SELL",
                quantity=stock_position // 100,
                limit_price=quote.bid,
                reason=f"Covered call: delta={quote.delta:.2f}, IV={quote.iv:.2%}"
            )
        
        return None


class PutCreditSpreadStrategy(OptionStrategy):
    """
    Bull put spread strategy.
    
    Sells OTM put spreads for credit.
    """
    
    def __init__(self, underlyings: List[str], target_credit: float = 0.30, spread_width: float = 5.0):
        super().__init__("PutCreditSpread", underlyings)
        self.target_credit = target_credit  # Minimum credit as % of width
        self.spread_width = spread_width
    
    def on_option_quote(self, quote: OptionQuote) -> Optional[Signal]:
        """Check if we should enter a put credit spread"""
        # This is a simplified example - full implementation would
        # require tracking pairs of options
        if quote.option_type != 'put':
            return None
        
        if quote.underlying not in self.underlyings:
            return None
        
        # Check if already have position
        if self.has_position(quote.symbol):
            return None
        
        # Check criteria
        if quote.delta and abs(quote.delta) <= 0.30:
            if quote.iv and quote.iv >= self.min_iv:
                return self.create_signal(
                    symbol=quote.symbol,
                    action="SELL",
                    quantity=1,
                    limit_price=quote.bid,
                    reason=f"Put spread candidate: delta={quote.delta:.2f}, IV={quote.iv:.2%}"
                )
        
        return None
