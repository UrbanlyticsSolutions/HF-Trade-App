"""
Live Trading System Package

Complete live trading system using IBKR TWS API (or Questrade) for real-time option data.
Includes:
- Trade database for storing all trades
- Position manager for tracking positions
- Order manager for executing orders
- Live trading engine
- Strategy base class

Usage:
    from clients.ibkr_adapter import create_ibkr_client
    from live import create_engine
    
    client = create_ibkr_client()
    engine = create_engine(
        client=client,
        account_id="YOUR_ACCOUNT_ID",
        option_underlyings=["SPY", "QQQ"],
        mode="monitor"  # "monitor", "paper", or "live"
    )
    engine.run()
"""

from .trade_database import TradeDatabase, Trade, QuoteSnapshot
from .position_manager import PositionManager, Position
from .order_manager import OrderManager, Order, OrderSide, OrderType, OrderStatus, TimeInForce
from .engine import LiveTradingEngine, EngineConfig, create_engine
from .strategy import Strategy, OptionStrategy, OptionQuote, Signal
from .strategy import CoveredCallStrategy, PutCreditSpreadStrategy
from .strategy_0dte import Live0DTEStrategy, create_0dte_strategy
from .state_persistence import StatePersistence, get_persistence, TradingState

__all__ = [
    # Database
    'TradeDatabase',
    'Trade',
    'QuoteSnapshot',
    # Position Management
    'PositionManager',
    'Position',
    # Order Management
    'OrderManager',
    'Order',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'TimeInForce',
    # Strategy
    'Strategy',
    'OptionStrategy',
    'OptionQuote',
    'Signal',
    'CoveredCallStrategy',
    'PutCreditSpreadStrategy',
    # 0DTE Strategy
    'Live0DTEStrategy',
    'create_0dte_strategy',
    # State Persistence
    'StatePersistence',
    'get_persistence',
    'TradingState',
    # Engine
    'LiveTradingEngine',
    'EngineConfig',
    'create_engine',
]
