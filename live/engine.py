"""
Live Trading Engine - Main orchestration for live trading

Coordinates:
- Real-time data fetching from Questrade
- Strategy execution
- Order management
- Position tracking
- Database logging
"""
import logging
import time
import threading
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_eastern_time() -> datetime:
    """Get current time in US Eastern timezone."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except ImportError:
        # Fallback: Calculate ET offset manually
        utc_now = datetime.now(timezone.utc)
        month = utc_now.month
        if 3 <= month <= 11:
            et_offset = timedelta(hours=-4)
        else:
            et_offset = timedelta(hours=-5)
        return utc_now + et_offset


@dataclass
class EngineConfig:
    """Configuration for live trading engine"""
    account_id: str
    symbols: List[str] = None
    option_underlyings: List[str] = None
    quote_interval: float = 5.0  # Seconds between quote updates
    position_sync_interval: float = 30.0  # Seconds between position syncs
    market_open: dt_time = dt_time(9, 30)
    market_close: dt_time = dt_time(16, 0)
    mode: str = "monitor"  # "monitor" (no orders), "paper" (local simulation), "live" (real orders)
    max_daily_loss: float = 1000.0  # Max daily loss before stopping
    max_position_value: float = 10000.0  # Max value per position


class LiveTradingEngine:
    """
    Main live trading engine.
    
    Orchestrates all components of the trading system.
    """
    
    def __init__(
        self,
        questrade_client,
        trade_db,
        position_manager,
        order_manager,
        config: EngineConfig
    ):
        """
        Initialize live trading engine.
        
        Args:
            questrade_client: QuestradeClient for API access
            trade_db: TradeDatabase for persistence
            position_manager: PositionManager instance
            order_manager: OrderManager instance
            config: EngineConfig with settings
        """
        self.client = questrade_client
        self.db = trade_db
        self.positions = position_manager
        self.orders = order_manager
        self.config = config
        
        # State
        self._running = False
        self._strategies: List = []
        self._quote_thread: Optional[threading.Thread] = None
        self._position_thread: Optional[threading.Thread] = None
        
        # Metrics
        self._daily_pnl = 0.0
        self._trade_count = 0
        self._start_time: Optional[datetime] = None
        
        # Callbacks
        self._on_quote_callbacks: List[Callable] = []
        self._on_signal_callbacks: List[Callable] = []
        
        # Set account for managers
        self.positions.set_account(config.account_id)
        self.orders.set_account(config.account_id)
        
        # Register fill callback
        self.orders.on_fill(self._on_fill)
    
    def add_strategy(self, strategy) -> None:
        """
        Add a strategy to the engine.
        
        Args:
            strategy: Strategy instance
        """
        strategy.set_managers(self.positions, self.orders)
        # Pass trade database for position recovery
        if hasattr(strategy, 'set_trade_db'):
            strategy.set_trade_db(self.db)
        self._strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.name}")
    
    def on_quote(self, callback: Callable) -> None:
        """Register quote callback"""
        self._on_quote_callbacks.append(callback)
    
    def on_signal(self, callback: Callable) -> None:
        """Register signal callback"""
        self._on_signal_callbacks.append(callback)
    
    def start(self) -> None:
        """Start the trading engine"""
        if self._running:
            logger.warning("Engine already running")
            return
        
        logger.info("=" * 60)
        logger.info("Starting Live Trading Engine")
        logger.info(f"Account: {self.config.account_id}")
        logger.info(f"Symbols: {self.config.symbols}")
        logger.info(f"Option Underlyings: {self.config.option_underlyings}")
        logger.info(f"Mode: {self.config.mode}")
        logger.info("=" * 60)
        
        self._running = True
        self._start_time = get_eastern_time()
        
        # Initial sync
        self.positions.sync_positions()
        self.orders.sync_orders()
        
        # Start strategies
        for strategy in self._strategies:
            strategy.on_start()
        
        # Start background threads
        self._quote_thread = threading.Thread(target=self._quote_loop, daemon=True)
        self._quote_thread.start()
        
        self._position_thread = threading.Thread(target=self._position_loop, daemon=True)
        self._position_thread.start()
        
        logger.info("Engine started successfully")
    
    def stop(self) -> None:
        """Stop the trading engine"""
        if not self._running:
            return
        
        logger.info("Stopping engine...")
        self._running = False
        
        # Stop strategies
        for strategy in self._strategies:
            strategy.on_stop()
        
        # Wait for threads
        if self._quote_thread:
            self._quote_thread.join(timeout=5)
        if self._position_thread:
            self._position_thread.join(timeout=5)
        
        # Final position sync
        self.positions.sync_positions()
        
        # Log summary
        self._print_summary()
        
        logger.info("Engine stopped")
    
    def run(self) -> None:
        """Run the engine (blocking)"""
        self.start()
        
        try:
            while self._running:
                time.sleep(1)
                
                # Check for max loss
                if self._daily_pnl < -self.config.max_daily_loss:
                    logger.warning(f"Max daily loss reached: ${self._daily_pnl:.2f}")
                    self._running = False
                    break
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()
    
    def _quote_loop(self) -> None:
        """Background thread for fetching quotes"""
        logger.info("Quote loop started")
        
        while self._running:
            try:
                # Check market hours
                if not self._is_market_open():
                    time.sleep(60)  # Check every minute outside market hours
                    continue
                
                # Fetch stock quotes
                if self.config.symbols:
                    self._process_stock_quotes()
                
                # Fetch option quotes
                if self.config.option_underlyings:
                    self._process_option_quotes()
                
            except Exception as e:
                logger.error(f"Quote loop error: {e}")
            
            time.sleep(self.config.quote_interval)
        
        logger.info("Quote loop stopped")
    
    def _position_loop(self) -> None:
        """Background thread for syncing positions"""
        logger.info("Position loop started")
        
        while self._running:
            try:
                # Skip Questrade sync in paper mode
                if self.config.mode == "live":
                    self.positions.sync_positions()
                    self.positions.update_quotes()
                    self.orders.sync_orders()
                
                # Calculate daily P&L
                exposure = self.positions.get_total_exposure()
                self._daily_pnl = exposure.get('total_unrealized_pnl', 0)
                
            except Exception as e:
                logger.error(f"Position loop error: {e}")
            
            time.sleep(self.config.position_sync_interval)
        
        logger.info("Position loop stopped")
    
    def _process_stock_quotes(self) -> None:
        """Process stock quotes and run strategies"""
        for symbol in self.config.symbols:
            try:
                quote = self.client.get_quote_by_symbol(symbol)
                if not quote:
                    continue
                
                # Notify callbacks
                for callback in self._on_quote_callbacks:
                    callback(symbol, quote)
                
                # Run strategies
                for strategy in self._strategies:
                    if not strategy.is_active:
                        continue
                    
                    signal = strategy.on_quote(symbol, quote)
                    if signal:
                        self._process_signal(signal)
                        
            except Exception as e:
                logger.error(f"Error processing quote for {symbol}: {e}")
    
    def _process_option_quotes(self) -> None:
        """Process option quotes and run strategies"""
        from .strategy import OptionQuote
        
        for underlying in self.config.option_underlyings:
            try:
                # Get 0DTE options (today's expiry)
                today = get_eastern_time().strftime("%Y-%m-%d")
                logger.info(f"Fetching 0DTE options for {underlying}, expiry={today}")
                options_data = self.client.get_atm_options(underlying, expiry_date=today, num_strikes=10)
                
                # Combine calls and puts
                all_options = options_data.get('calls', []) + options_data.get('puts', [])
                logger.info(f"Got {len(all_options)} options for {underlying}")
                
                for opt in all_options:
                    # Convert to OptionQuote
                    # Use the expiry we requested since Questrade may not return it
                    expiry = opt.get('expiryDate', '') or today
                    # Parse option type (case-insensitive, check symbol as fallback)
                    opt_type_raw = str(opt.get('optionType', '')).lower()
                    symbol = opt.get('symbol', '')
                    if opt_type_raw == 'call':
                        opt_type = 'call'
                    elif opt_type_raw == 'put':
                        opt_type = 'put'
                    else:
                        # Parse from symbol: SPY28Jan26C695.00 or SPY28Jan26P695.00
                        # Use rfind to get the LAST occurrence (not the P in SPY)
                        c_pos = symbol.rfind('C')
                        p_pos = symbol.rfind('P')
                        if c_pos > 5:
                            opt_type = 'call'
                        elif p_pos > 5:
                            opt_type = 'put'
                        else:
                            opt_type = 'unknown'
                    quote = OptionQuote(
                        symbol=symbol,
                        underlying=underlying,
                        underlying_price=opt.get('underlyingPrice', 0),
                        strike=opt.get('strikePrice', opt.get('strike', 0)),
                        expiration=expiry,
                        option_type=opt_type,
                        bid=opt.get('bidPrice', 0),
                        ask=opt.get('askPrice', 0),
                        last=opt.get('lastTradePrice', opt.get('lastTradePriceTrHrs', 0)),
                        volume=opt.get('volume', 0),
                        open_interest=opt.get('openInterest', 0),
                        delta=opt.get('delta'),
                        gamma=opt.get('gamma'),
                        theta=opt.get('theta'),
                        vega=opt.get('vega'),
                        iv=opt.get('volatility'),
                        timestamp=get_eastern_time().isoformat()
                    )
                    
                    # Store quote snapshot (non-blocking)
                    if self.db:
                        try:
                            from .trade_database import QuoteSnapshot
                            snapshot = QuoteSnapshot(
                                symbol=quote.symbol,
                                timestamp=quote.timestamp,
                                bid_price=quote.bid,
                                ask_price=quote.ask,
                                last_price=quote.last,
                                volume=quote.volume,
                                open_interest=quote.open_interest,
                                delta=quote.delta,
                                gamma=quote.gamma,
                                theta=quote.theta,
                                vega=quote.vega,
                                iv=quote.iv,
                                underlying_price=quote.underlying_price
                            )
                            self.db.insert_quote_snapshot(snapshot)
                        except Exception as db_err:
                            logger.debug(f"Quote snapshot insert skipped: {db_err}")
                    
                    # Run option strategies
                    for strategy in self._strategies:
                        if not strategy.is_active:
                            continue
                        
                        signal = strategy.on_option_quote(quote)
                        if signal:
                            self._process_signal(signal)
                            
            except Exception as e:
                logger.error(f"Error processing options for {underlying}: {e}")
    
    def _process_signal(self, signal) -> None:
        """Process a trading signal"""
        logger.info(f"Signal: {signal.action} {signal.quantity} {signal.symbol} - {signal.reason}")
        
        # Notify callbacks
        for callback in self._on_signal_callbacks:
            callback(signal)
        
        # Execute based on mode
        if self.config.mode == "monitor":
            logger.info("Monitor mode - signal not executed")
            return
        
        if signal.action == "HOLD":
            return
        
        # PENDING means waiting for option selection - don't execute yet
        if signal.action == "PENDING":
            logger.debug(f"Pending signal - waiting for option selection: {signal.reason}")
            return
        
        try:
            # Paper trading - simulate fills locally
            if self.config.mode == "paper":
                self._simulate_paper_trade(signal)
                return
            
            # Live trading - send to Questrade
            if signal.action == "BUY":
                order = self.orders.buy(
                    symbol=signal.symbol,
                    quantity=signal.quantity,
                    limit_price=signal.limit_price
                )
            elif signal.action == "SELL":
                order = self.orders.sell(
                    symbol=signal.symbol,
                    quantity=signal.quantity,
                    limit_price=signal.limit_price
                )
            else:
                logger.warning(f"Unknown signal action: {signal.action}")
                return
            
            self._trade_count += 1
            
            # Log to database
            if self.db and order.order_id:
                self.db.insert_trade(
                    symbol=signal.symbol,
                    side=signal.action,
                    quantity=signal.quantity,
                    entry_price=signal.limit_price or 0,
                    strategy=signal.strategy_name,
                    notes=signal.reason
                )
                
        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
    
    def _on_fill(self, order) -> None:
        """Handle order fill event"""
        logger.info(f"Fill: {order.side} {order.filled_quantity} {order.symbol} @ ${order.avg_fill_price:.2f}")
        
        # Notify strategies
        for strategy in self._strategies:
            strategy.on_fill(order)
        
        # Update database
        if self.db:
            self.db.update_order_status(
                str(order.order_id),
                "Filled",
                filled_at=get_eastern_time().isoformat(),
                fill_price=order.avg_fill_price,
                commission=order.commission
            )
    
    def _simulate_paper_trade(self, signal) -> None:
        """
        Simulate a paper trade locally without sending to Questrade.
        
        Uses signal's limit_price as fill price, tracks in database.
        """
        fill_price = signal.limit_price or 0.0
        
        logger.info(f"[PAPER] Simulated {signal.action}: {signal.quantity} {signal.symbol} @ ${fill_price:.2f}")
        
        self._trade_count += 1
        
        # Create a mock order for the strategy callback
        from live.order_manager import Order
        mock_order = Order(
            order_id=int(get_eastern_time().timestamp() * 1000),  # Fake ID
            symbol=signal.symbol,
            quantity=signal.quantity,
            filled_quantity=signal.quantity,
            side=signal.action,
            order_type="Limit",
            limit_price=fill_price,
            avg_fill_price=fill_price,
            status="Filled",
            notes="[PAPER TRADE]"
        )
        
        # Notify strategies of the simulated fill
        for strategy in self._strategies:
            strategy.on_fill(mock_order)
        
        # Log to database with [PAPER] tag
        if self.db:
            from live.trade_database import Trade
            trade = Trade(
                symbol=signal.symbol,
                action=signal.action,
                quantity=signal.quantity,
                entry_price=fill_price,
                entry_time=get_eastern_time().isoformat(),
                strategy_name=signal.strategy_name,
                notes=f"[PAPER] {signal.reason}",
                status="open"
            )
            self.db.insert_trade(trade)
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = get_eastern_time()
        
        # Check day of week (0=Monday, 6=Sunday)
        if now.weekday() >= 5:
            return False
        
        # Check time
        current_time = now.time()
        return self.config.market_open <= current_time <= self.config.market_close
    
    def _print_summary(self) -> None:
        """Print trading session summary"""
        runtime = get_eastern_time() - self._start_time if self._start_time else None
        
        print("\n" + "=" * 60)
        print("TRADING SESSION SUMMARY")
        print("=" * 60)
        print(f"Account: {self.config.account_id}")
        print(f"Runtime: {runtime}")
        print(f"Trades: {self._trade_count}")
        print(f"Daily P&L: ${self._daily_pnl:,.2f}")
        print("-" * 60)
        
        # Position summary
        self.positions.print_positions()
        
        # Open orders
        self.orders.print_orders()
        
        print("=" * 60)
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        exposure = self.positions.get_total_exposure()
        risk = self.positions.get_risk_metrics()
        
        return {
            "running": self._running,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "account_id": self.config.account_id,
            "mode": self.config.mode,
            "market_open": self._is_market_open(),
            "trade_count": self._trade_count,
            "daily_pnl": self._daily_pnl,
            "strategy_count": len(self._strategies),
            "positions": exposure,
            "risk": risk
        }


def create_engine(
    questrade_client,
    account_id: str,
    symbols: List[str] = None,
    option_underlyings: List[str] = None,
    mode: str = "monitor",
    db_path: str = "live_trades.db"
):
    """
    Factory function to create a fully configured trading engine.
    
    Args:
        questrade_client: QuestradeClient instance
        account_id: Questrade account ID
        symbols: Stock symbols to track
        option_underlyings: Option underlying symbols
        mode: Trading mode - "monitor", "paper", or "live"
        db_path: Path to database file
        
    Returns:
        Configured LiveTradingEngine
    """
    from .trade_database import TradeDatabase
    from .position_manager import PositionManager
    from .order_manager import OrderManager
    
    # Create components
    db = TradeDatabase(db_path)
    positions = PositionManager(questrade_client, db)
    orders = OrderManager(questrade_client, db)
    
    config = EngineConfig(
        account_id=account_id,
        symbols=symbols or [],
        option_underlyings=option_underlyings or [],
        mode=mode
    )
    
    engine = LiveTradingEngine(
        questrade_client=questrade_client,
        trade_db=db,
        position_manager=positions,
        order_manager=orders,
        config=config
    )
    
    return engine
