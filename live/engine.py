"""
Live Trading Engine - Main orchestration for live trading

Coordinates:
- Real-time data fetching from broker (IBKR or Questrade)
- Strategy execution
- Order management
- Position tracking
- Database logging
"""
import logging
import re
import time
import threading
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _send_trade_alert(subject: str, body: str) -> None:
    """Fire-and-forget email alert. Runs in a daemon thread to avoid blocking the engine."""
    def _send():
        try:
            from .health_report import send_alert
            send_alert(f"{subject}\n\n{body}")
        except Exception as e:
            logger.debug(f"Alert email failed (non-critical): {e}")
    t = threading.Thread(target=_send, daemon=True)
    t.start()


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
    mode: str = "monitor"  # "monitor" (no orders), "paper" (IBKR paper account), "live" (IBKR live account)
    max_daily_loss: float = 1000.0  # Max daily loss before stopping
    max_position_value: float = None  # Max value per position

    def __post_init__(self):
        if self.max_position_value is None:
            from config import defaults as cfg
            self.max_position_value = cfg.max_position_value()


class LiveTradingEngine:
    """
    Main live trading engine.
    
    Orchestrates all components of the trading system.
    """
    
    def __init__(
        self,
        client,
        trade_db,
        position_manager,
        order_manager,
        config: EngineConfig,
        quote_client=None,
        chains_client=None
    ):
        """
        Initialize live trading engine.
        
        Args:
            client: Broker client for order execution (IBKRAdapter or QuestradeClient)
            trade_db: TradeDatabase for persistence
            position_manager: PositionManager instance
            order_manager: OrderManager instance
            config: EngineConfig with settings
            quote_client: Client for real-time stock/option quotes.
                          If None, uses the main client.
            chains_client: Client for option chain discovery (get_atm_options).
                           If None, uses quote_client.
        """
        self.client = client
        self.quote_client = quote_client or client
        self.chains_client = chains_client or self.quote_client
        self.db = trade_db
        self.positions = position_manager
        self.orders = order_manager
        self.config = config
        
        # Initialize Flex client for historical trade data from IBKR
        self._flex_client = None
        self._flex_query_id = None
        self._flex_trades_cache = None       # cached Flex trades
        self._flex_cache_time = 0            # timestamp of last Flex fetch
        self._FLEX_CACHE_TTL = 120           # re-fetch Flex every 2 minutes
        try:
            import os
            from clients.ibkr_flex import IBKRFlexClient
            token = os.environ.get("IBKR_FLEX_TOKEN", "")
            qid = os.environ.get("IBKR_FLEX_QUERY_ID", "")
            if token and qid:
                self._flex_client = IBKRFlexClient(token=token)
                self._flex_query_id = int(qid)
                logger.info("Flex Web Service configured for historical trade reconciliation")
            else:
                logger.info("IBKR_FLEX_TOKEN / IBKR_FLEX_QUERY_ID not set — reconciliation uses session executions only")
        except Exception as flex_err:
            logger.warning(f"Flex client init failed (will use session executions only): {flex_err}")
        
        # State
        self._running = False
        self._strategies: List = []
        self._quote_thread: Optional[threading.Thread] = None
        self._position_thread: Optional[threading.Thread] = None
        
        # Metrics
        self._daily_pnl = 0.0
        self._baseline_unrealized_pnl: Optional[float] = None  # Snapshot at startup to isolate engine-session P&L
        self._trade_count = 0
        self._start_time: Optional[datetime] = None
        self._broker_nlv: Optional[float] = None  # Real broker NetLiquidation
        self._broker_buying_power: Optional[float] = None  # Cached from account summary
        self._MIN_BUYING_POWER = 500.0  # Block new entries below this threshold
        self._max_loss_reached = False  # Set when daily loss limit hit; blocks new entries but keeps engine alive
        
        # Callbacks
        self._on_quote_callbacks: List[Callable] = []
        self._on_signal_callbacks: List[Callable] = []
        
        # Pending exit orders: order_id -> {trade_id, symbol, signal_reason}
        # Used to defer trade closure until fill is confirmed by broker
        self._pending_exit_orders: Dict[int, Dict] = {}
        # Pending entry orders: order_id -> Trade info
        # Used to defer entry recording until fill is confirmed
        self._pending_entry_orders: Dict[int, Dict] = {}
        # Stranded positions: trades where all exit retries are exhausted.
        # The position loop watchdog will attempt a final MARKET sweep.
        self._stranded_positions: List[Dict] = []  # [{trade_id, symbol, stranded_at}]
        
        # Circuit breaker: track consecutive Inactive/rejected orders
        self._consecutive_rejects = 0
        self._REJECT_CIRCUIT_BREAKER = 5  # Block new entries after 5 consecutive rejects
        self._circuit_breaker_tripped = False
        
        # EOD forced liquidation
        self._eod_liquidation_done = False
        self._eod_emergency_done = False
        self._EOD_LIQUIDATION_TIME = dt_time(15, 45)  # Force-close all positions at 3:45 PM ET
        self._EOD_LAST_CHANCE_TIME = dt_time(15, 55)   # Emergency MKT sweep at 3:55 PM ET
        
        # EOD Flex reconciliation — rebuild DB from IBKR Flex as source of truth
        self._eod_flex_reconcile_done = False
        self._EOD_FLEX_RECONCILE_TIME = dt_time(16, 5)  # Reconcile at 4:05 PM ET (after market close)
        
        # Set account for managers
        self.positions.set_account(config.account_id)
        self.orders.set_account(config.account_id)
        
        # Register fill callback
        self.orders.on_fill(self._on_fill)
        # Register rejection/cancellation callback
        self.orders.on_reject(self._on_reject)
    
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
        
        # Cancel any stale orders from a previous session before doing anything
        if self.config.mode in ("paper", "live"):
            self._cancel_stale_orders()
            self._reconcile_broker_positions()
            self._reconcile_executions()
            # Re-recover positions in strategies after reconciliation may have
            # created new DB entries for orphaned IBKR positions
            for strategy in self._strategies:
                if hasattr(strategy, '_recover_open_positions') and not getattr(strategy, 'trade_state', None) or (hasattr(strategy, 'trade_state') and not strategy.trade_state.in_trade):
                    strategy._recover_open_positions()
        
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
        
        # Flex reconciliation — rebuild DB from IBKR Flex before summary
        self._run_flex_reconcile(source="SHUTDOWN")
        
        # Log summary
        self._print_summary()
        
        # Send daily summary email
        self._send_daily_summary()
        
        logger.info("Engine stopped")
    
    def run(self) -> None:
        """Run the engine (blocking)"""
        self.start()
        
        _consecutive_conn_errors = 0
        _MAX_CONN_ERRORS = 30  # only stop after sustained failures (was 10)
        
        try:
            while self._running:
                time.sleep(1)
                
                # Check for max loss — block new entries but keep running for pending exits
                if self._daily_pnl < -self.config.max_daily_loss:
                    if not self._max_loss_reached:
                        logger.warning(f"Max daily loss reached: ${self._daily_pnl:.2f} — blocking new entries")
                        self._max_loss_reached = True
                        _send_trade_alert(
                            "MAX DAILY LOSS REACHED",
                            f"Daily P&L: ${self._daily_pnl:.2f}\nLimit: ${self.config.max_daily_loss:.2f}\nNew entries BLOCKED. Engine stays alive for pending exits.",
                        )
                    
                    # Only stop if no pending exit orders remain
                    if not self._pending_exit_orders:
                        logger.info("Max daily loss reached and no pending exit orders — stopping engine")
                        self._running = False
                        break
                
                # Connection watchdog: proactively check broker connection health
                try:
                    if hasattr(self.client, 'ensure_connected'):
                        self.client.ensure_connected()
                    if self.quote_client is not self.client and hasattr(self.quote_client, 'ensure_connected'):
                        self.quote_client.ensure_connected()
                    if self.chains_client is not self.quote_client and self.chains_client is not self.client and hasattr(self.chains_client, 'ensure_connected'):
                        self.chains_client.ensure_connected()
                    _consecutive_conn_errors = 0
                except ConnectionError as ce:
                    _consecutive_conn_errors += 1
                    # Exponential backoff: 5s, 10s, 20s, 30s, 30s, ...
                    backoff = min(5 * (2 ** min(_consecutive_conn_errors - 1, 3)), 30)
                    logger.error(f"Connection watchdog: {ce} "
                                 f"(attempt {_consecutive_conn_errors}/{_MAX_CONN_ERRORS}, "
                                 f"retry in {backoff}s)")
                    if _consecutive_conn_errors >= _MAX_CONN_ERRORS:
                        logger.critical("Too many consecutive connection failures — stopping engine.")
                        _send_trade_alert(
                            "ENGINE STOPPING — CONNECTION FAILURE",
                            f"{_consecutive_conn_errors} consecutive connection failures.\nEngine is shutting down. Manual intervention required.",
                        )
                        self._running = False
                        break
                    time.sleep(backoff)
                
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
                
            except ConnectionError as ce:
                logger.warning(f"Quote loop connection error (will retry): {ce}")
                time.sleep(10)  # longer sleep on connection issues
                continue
            except Exception as e:
                logger.error(f"Quote loop error: {e}")
            
            time.sleep(self.config.quote_interval)
        
        logger.info("Quote loop stopped")
    
    def _position_loop(self) -> None:
        """Background thread for syncing positions and recording broker balance"""
        logger.info("Position loop started")
        
        # Track last balance record time — record every 5 minutes
        _last_balance_record = 0
        _BALANCE_RECORD_INTERVAL = 300  # seconds
        _last_exec_reconcile = 0
        _EXEC_RECONCILE_INTERVAL = 30  # reconcile executions every sync cycle
        _last_pos_reconcile = 0
        _POS_RECONCILE_INTERVAL = 30  # reconcile broker positions every sync cycle
        
        while self._running:
            try:
                # Sync broker positions for both paper and live modes.
                # Paper mode connects to IBKR paper account, so sync is valid.
                if self.config.mode in ("paper", "live"):
                    self.positions.sync_positions()
                    self.positions.update_quotes()
                    self.orders.sync_orders()
                    # Persist current positions to DB for dashboard
                    if self.db and hasattr(self.db, 'update_current_positions'):
                        self.db.update_current_positions(self.positions.get_all_positions())
                
                # Calculate daily P&L (only change since engine started, excludes pre-existing positions)
                exposure = self.positions.get_total_exposure()
                raw_unrealized = exposure.get('total_unrealized_pnl', 0)
                if self._baseline_unrealized_pnl is None:
                    self._baseline_unrealized_pnl = raw_unrealized
                    logger.info(f"Baseline unrealized P&L snapshot: ${raw_unrealized:,.2f}")
                self._daily_pnl = raw_unrealized - self._baseline_unrealized_pnl
                
                # Periodically fetch and record real broker balance
                now_ts = time.time()
                if self.config.mode in ("paper", "live") and (now_ts - _last_balance_record) >= _BALANCE_RECORD_INTERVAL:
                    try:
                        summary = self.client.get_account_balances(self.config.account_id)
                        # Extract NetLiquidation — try USD first, then CAD
                        nlv = None
                        nlv_info = summary.get('NetLiquidation', {})
                        if isinstance(nlv_info, dict):
                            nlv = nlv_info.get('USD') or nlv_info.get('CAD')
                        if nlv is not None:
                            nlv = float(nlv)
                            # Store in DB balance_history table
                            if self.db and hasattr(self.db, 'record_balance'):
                                cash_info = summary.get('TotalCashValue', {})
                                cash = None
                                if isinstance(cash_info, dict):
                                    cash = cash_info.get('USD') or cash_info.get('CAD')
                                cash = float(cash) if cash is not None else None
                                gpv_info = summary.get('GrossPositionValue', {})
                                gpv = None
                                if isinstance(gpv_info, dict):
                                    gpv = gpv_info.get('USD') or gpv_info.get('CAD')
                                gpv = float(gpv) if gpv is not None else None
                                unrealized_info = summary.get('UnrealizedPnL', {})
                                unrealized = None
                                if isinstance(unrealized_info, dict):
                                    unrealized = unrealized_info.get('USD') or unrealized_info.get('CAD')
                                unrealized = float(unrealized) if unrealized is not None else None
                                self.db.record_balance(
                                    account_id=self.config.account_id,
                                    net_liquidation=nlv,
                                    cash=cash,
                                    positions_value=gpv,
                                    unrealized_pnl=unrealized,
                                )
                            # Cache buying power for pre-order margin check
                            bp_info = summary.get('BuyingPower', {})
                            if isinstance(bp_info, dict):
                                bp_val = bp_info.get('USD') or bp_info.get('CAD')
                                if bp_val is not None:
                                    self._broker_buying_power = float(bp_val)
                                    logger.debug(f"Broker BuyingPower: ${self._broker_buying_power:,.2f}")
                            # Update state persistence with real broker balance
                            self._broker_nlv = nlv
                            # Write to trading_state.json so dashboard can read it
                            self._update_state_broker_balance(nlv, cash)
                            # Sync broker equity into strategies for position sizing
                            self._sync_strategy_capital(nlv)
                            logger.debug(f"Broker NLV: ${nlv:,.2f}")
                        _last_balance_record = now_ts
                    except Exception as bal_err:
                        logger.debug(f"Balance fetch skipped: {bal_err}")
                
                # Check pending exit orders for timeouts
                self._check_pending_orders()

                # Quote-independent time exit: force-close positions that
                # exceeded max hold or EOD exit time, even if no option
                # quotes are flowing (illiquid 0DTE in the afternoon).
                if self.config.mode in ("paper", "live"):
                    self._check_time_based_exits()

                # EOD forced liquidation check — prevent 0DTE exercise into stock
                if self.config.mode in ("paper", "live"):
                    self._check_eod_liquidation()

                # EOD Flex reconciliation — rebuild DB from IBKR Flex after market close
                if self.config.mode in ("paper", "live"):
                    self._check_eod_flex_reconcile()

                # Periodically reconcile IBKR executions with local DB
                if (now_ts - _last_exec_reconcile) >= _EXEC_RECONCILE_INTERVAL:
                    try:
                        self._reconcile_executions()
                    except Exception as recon_err:
                        logger.debug(f"Execution reconciliation skipped: {recon_err}")
                    _last_exec_reconcile = now_ts

                # Periodically reconcile broker positions vs DB (remove phantoms)
                if (now_ts - _last_pos_reconcile) >= _POS_RECONCILE_INTERVAL:
                    try:
                        self._reconcile_broker_positions()
                    except Exception as pos_err:
                        logger.debug(f"Position reconciliation skipped: {pos_err}")
                    _last_pos_reconcile = now_ts
            
            except ConnectionError as ce:
                logger.warning(f"Position loop connection error (will retry): {ce}")
                time.sleep(10)
                continue
            except Exception as e:
                logger.error(f"Position loop error: {e}")
            
            time.sleep(self.config.position_sync_interval)
        
        logger.info("Position loop stopped")
    
    def _check_pending_orders(self) -> None:
        """Check pending entry/exit orders for timeouts or rejections.
        
        Called every position_sync_interval (~30s) from _position_loop.
        If an exit order hasn't filled within 90 seconds, log a warning
        so the operator (or a future auto-escalation) can intervene.
        """
        if not self._pending_exit_orders and not self._pending_entry_orders and not self._stranded_positions:
            return
        
        now = time.time()
        _ORDER_TIMEOUT_SECS = 90   # warn after 90s without fill
        _EXIT_EXPIRE_SECS  = 180  # auto-expire stale exits after 180s
        
        # Check exit orders
        stale_exits = []
        for order_id, info in self._pending_exit_orders.items():
            elapsed = now - info.get("submitted_at", now)
            if elapsed > _ORDER_TIMEOUT_SECS:
                stale_exits.append((order_id, info, elapsed))
        
        for order_id, info, elapsed in stale_exits:
            logger.warning(
                f"EXIT order {order_id} for trade {info['trade_id']} "
                f"({info['symbol']}) has not filled after {elapsed:.0f}s — "
                f"check IBKR TWS for order status"
            )
            # Check if IBKR reports the order as Cancelled/Inactive
            try:
                order_statuses = getattr(self.client, '_ibkr', self.client)
                if hasattr(order_statuses, 'order_statuses'):
                    status_info = order_statuses.order_statuses.get(order_id, {})
                    status = status_info.get('status', '') if isinstance(status_info, dict) else ''
                    if status in ('Cancelled', 'Inactive', 'ApiCancelled'):
                        logger.error(
                            f"EXIT order {order_id} was {status} by broker — "
                            f"trade {info['trade_id']} remains OPEN"
                        )
                        self._pending_exit_orders.pop(order_id, None)
            except Exception:
                pass

            # OE-R3 fix: Auto-expire stale pending exits after 180s.
            # Move them to _stranded_positions for the watchdog to handle.
            if elapsed > _EXIT_EXPIRE_SECS and order_id in self._pending_exit_orders:
                logger.error(
                    f"EXIT order {order_id} for trade {info['trade_id']} "
                    f"({info['symbol']}) expired after {elapsed:.0f}s — "
                    f"promoting to stranded position for watchdog retry"
                )
                self._stranded_positions.append({
                    "trade_id": info["trade_id"],
                    "symbol": info["symbol"],
                    "stranded_at": now,
                })
                self._pending_exit_orders.pop(order_id, None)
        
        # Check entry orders — detect Cancelled/Inactive and clean up phantom trades
        stale_entries = []
        for order_id, info in list(self._pending_entry_orders.items()):
            elapsed = now - info.get("submitted_at", now)
            if elapsed > _ORDER_TIMEOUT_SECS:
                stale_entries.append((order_id, info, elapsed))
        
        for order_id, info, elapsed in stale_entries:
            symbol = info.get('symbol', '')
            logger.warning(
                f"ENTRY order {order_id} ({symbol}) has not filled after {elapsed:.0f}s"
            )
            # Check if IBKR reports the order as Cancelled/Inactive
            try:
                order_statuses = getattr(self.client, '_ibkr', self.client)
                if hasattr(order_statuses, 'order_statuses'):
                    status_info = order_statuses.order_statuses.get(order_id, {})
                    status = status_info.get('status', '') if isinstance(status_info, dict) else ''
                    if status in ('Cancelled', 'Inactive', 'ApiCancelled'):
                        logger.warning(
                            f"ENTRY order {order_id} was {status} by broker — "
                            f"no trade recorded (order never filled)"
                        )
                        self._pending_entry_orders.pop(order_id, None)
                        # Notify strategy to clear in-flight state
                        for strategy in self._strategies:
                            if hasattr(strategy, 'on_trade_cancelled'):
                                strategy.on_trade_cancelled(None, symbol)
            except Exception as e:
                logger.debug(f"Failed to check entry order {order_id} status: {e}")
        
        # OE-R1 fix: Watchdog for stranded positions (all exit retries exhausted).
        # Attempt one final MARKET sweep per stranded position per cycle.
        if self._stranded_positions:
            _STRANDED_RETRY_INTERVAL = 60  # Retry every 60s
            remaining = []
            for sp in self._stranded_positions:
                elapsed = now - sp.get("stranded_at", now)
                trade_id = sp["trade_id"]
                symbol = sp["symbol"]
                
                if elapsed < _STRANDED_RETRY_INTERVAL:
                    remaining.append(sp)
                    continue
                
                # Check if broker still has this position
                try:
                    broker_positions = self.positions.get_all_positions()
                    has_position = any(p.symbol == symbol for p in broker_positions)
                except Exception:
                    has_position = True  # Assume still open if we can't check
                
                if not has_position:
                    # Position already closed (broker cleared it, exercised, etc.)
                    logger.info(f"Stranded trade {trade_id} ({symbol}) no longer held by broker — closing in DB")
                    if self.db:
                        try:
                            self.db.close_trade(
                                trade_id=trade_id,
                                exit_price=0.01,
                                exit_time=get_eastern_time().isoformat(),
                                notes="[STRANDED] Position not found at broker — forced close",
                            )
                        except Exception:
                            pass
                    continue
                
                # Still held — attempt one more MARKET order
                logger.warning(f"Stranded watchdog: retrying MARKET sell for trade {trade_id} ({symbol})")
                try:
                    qty = 1
                    if self.db:
                        trade = self.db.get_trade(trade_id)
                        if trade:
                            qty = trade.get('quantity', 1)
                    mkt_order = self.orders.sell(symbol=symbol, quantity=qty, limit_price=None)
                    if mkt_order and mkt_order.order_id:
                        self._pending_exit_orders[mkt_order.order_id] = {
                            "trade_id": trade_id,
                            "symbol": symbol,
                            "signal_reason": f"[STRANDED WATCHDOG] final MARKET sweep",
                            "submitted_at": time.time(),
                            "retry_count": 0,  # Fresh retry count for the new order
                        }
                        logger.info(f"Stranded watchdog: MARKET order {mkt_order.order_id} for trade {trade_id}")
                    else:
                        # Order failed, keep stranded for next cycle
                        sp["stranded_at"] = now
                        remaining.append(sp)
                except Exception as e:
                    logger.error(f"Stranded watchdog MARKET sell failed for {symbol}: {e}")
                    sp["stranded_at"] = now
                    remaining.append(sp)
            
            self._stranded_positions = remaining
    
    def _process_stock_quotes(self) -> None:
        """Process stock quotes and run strategies"""
        for symbol in self.config.symbols:
            try:
                quote = self.quote_client.get_quote_by_symbol(symbol)
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
        
        today = get_eastern_time().strftime("%Y-%m-%d")
        
        # ===== STEP 1: Fetch quotes for OPEN POSITIONS first =====
        # This ensures we can exit positions even if they drift away from ATM
        open_position_symbols = set()
        for strategy in self._strategies:
            if hasattr(strategy, 'trade_state') and strategy.trade_state.in_trade:
                if strategy.trade_state.symbol:
                    open_position_symbols.add(strategy.trade_state.symbol)
        
        if open_position_symbols:
            logger.info(f"Fetching quotes for {len(open_position_symbols)} open positions: {open_position_symbols}")
            for symbol in open_position_symbols:
                try:
                    quote_data = self.quote_client.get_quote_by_symbol(symbol)
                    if quote_data:
                        # Parse option type from symbol
                        c_pos = symbol.rfind('C')
                        p_pos = symbol.rfind('P')
                        opt_type = 'call' if c_pos > p_pos else 'put'
                        
                        quote = OptionQuote(
                            symbol=symbol,
                            underlying="SPY",
                            underlying_price=quote_data.get('underlyingPrice', 0),
                            strike=quote_data.get('strikePrice', 0),
                            expiration=today,
                            option_type=opt_type,
                            bid=quote_data.get('bidPrice', 0),
                            ask=quote_data.get('askPrice', 0),
                            last=quote_data.get('lastTradePrice', quote_data.get('lastTradePriceTrHrs', 0)),
                            volume=quote_data.get('volume', 0),
                            open_interest=quote_data.get('openInterest', 0),
                            delta=quote_data.get('delta'),
                            gamma=quote_data.get('gamma'),
                            theta=quote_data.get('theta'),
                            vega=quote_data.get('vega'),
                            iv=quote_data.get('volatility'),
                            timestamp=get_eastern_time().isoformat()
                        )
                        
                        logger.info(f"Open position quote: {symbol} bid=${quote.bid:.2f} ask=${quote.ask:.2f} last=${quote.last:.2f}")
                        
                        # Process for exit check
                        for strategy in self._strategies:
                            if not strategy.is_active:
                                continue
                            signal = strategy.on_option_quote(quote)
                            if signal:
                                self._process_signal(signal)
                    else:
                        logger.warning(f"Could not fetch quote for open position: {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching quote for open position {symbol}: {e}")
        
        # ===== STEP 2: Fetch ATM options for new entries =====
        for underlying in self.config.option_underlyings:
            try:
                # Get 0DTE options (today's expiry)
                today = get_eastern_time().strftime("%Y-%m-%d")
                logger.info(f"Fetching 0DTE options for {underlying}, expiry={today}")
                options_data = None
                for _attempt in range(3):
                    try:
                        options_data = self.chains_client.get_atm_options(underlying, expiry_date=today, num_strikes=10)
                        break
                    except Exception as fetch_err:
                        err_str = str(fetch_err)
                        if "504" in err_str or "502" in err_str or "timeout" in err_str.lower():
                            logger.warning(f"Option fetch attempt {_attempt+1}/3 failed (transient): {fetch_err}")
                            time.sleep(2 * (_attempt + 1))
                        else:
                            raise
                if options_data is None:
                    logger.error(f"Option fetch failed after 3 retries for {underlying}")
                    continue
                
                # Combine calls and puts
                all_options = options_data.get('calls', []) + options_data.get('puts', [])
                logger.info(f"Got {len(all_options)} options for {underlying}")
                
                # Refresh pending direction timestamp so expiry timer starts
                # from when options are available, not from stock quote time
                for strategy in self._strategies:
                    if hasattr(strategy, '_pending_direction') and strategy._pending_direction:
                        if hasattr(strategy, '_pending_direction_time'):
                            from datetime import datetime
                            strategy._pending_direction_time = datetime.now(strategy._pending_direction_time.tzinfo) if strategy._pending_direction_time else None
                            logger.info(f"Refreshed pending direction timestamp for {strategy.__class__.__name__}")
                
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
                        
                        try:
                            signal = strategy.on_option_quote(quote)
                            if signal:
                                self._process_signal(signal)
                        except Exception as strat_err:
                            logger.error(f"Strategy error on {quote.symbol}: {strat_err}", exc_info=True)
                            
            except Exception as e:
                logger.error(f"Error processing options for {underlying}: {e}")
    
    def _process_signal(self, signal) -> None:
        """Process a trading signal"""
        logger.info(f"Signal: {signal.action} {signal.quantity} {signal.symbol} - {signal.reason}")
        
        # Block new entries when max daily loss is reached (exits still allowed)
        if self._max_loss_reached and signal.action == "BUY":
            logger.warning(f"Max daily loss active — blocking new entry: {signal.symbol}")
            return
        
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
        
        tag = "[PAPER]" if self.config.mode == "paper" else "[LIVE]"
        
        try:
            # Circuit breaker: block new entries when too many consecutive rejects
            if signal.action == "BUY" and self._circuit_breaker_tripped:
                logger.warning(f"Circuit breaker TRIPPED — blocking new BUY for {signal.symbol}. "
                               f"{self._consecutive_rejects} consecutive rejects.")
                return

            # Margin pre-check: block BUY if cached buying power is too low
            if signal.action == "BUY" and self._broker_buying_power is not None:
                if self._broker_buying_power < self._MIN_BUYING_POWER:
                    logger.warning(f"Insufficient buying power ${self._broker_buying_power:,.2f} "
                                   f"(min ${self._MIN_BUYING_POWER:,.2f}) — blocking BUY {signal.symbol}")
                    return

            # OE-R4 fix: Duplicate BUY guard — block if we already have a
            # pending entry order for the same symbol.
            if signal.action == "BUY":
                dup = any(
                    v.get('symbol') == signal.symbol
                    for v in self._pending_entry_orders.values()
                )
                if dup:
                    logger.warning(
                        f"Duplicate BUY blocked for {signal.symbol} — "
                        f"pending entry order already in flight"
                    )
                    return

            # Both paper and live modes send real orders to the broker.
            # In paper mode the broker is IBKR paper-trading account (port 4002/4004)
            # so orders get real simulated fills from IBKR's paper engine.
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
            
            logger.info(f"{tag} Order sent: {signal.action} {signal.quantity} {signal.symbol} @ ${signal.limit_price or 'MKT'}")
            self._trade_count += 1
            
            # DON'T insert to DB yet — only create the trade record
            # when the fill is confirmed by broker (in _on_fill).
            # This prevents phantom $0 P&L trades when orders are rejected.
            if order.order_id:
                if signal.action == "BUY":
                    # Determine option type for later DB insert
                    option_type = None
                    sym = signal.symbol
                    if "C" in sym and sym.split("C")[-1].replace(".", "").isdigit():
                        option_type = "call"
                    elif "P" in sym and sym.split("P")[-1].replace(".", "").isdigit():
                        option_type = "put"
                    # Track for fill confirmation — trade is created in _on_fill
                    self._pending_entry_orders[order.order_id] = {
                        "symbol": signal.symbol,
                        "quantity": signal.quantity,
                        "limit_price": signal.limit_price,
                        "strategy_name": signal.strategy_name,
                        "option_type": option_type,
                        "tag": tag,
                        "reason": signal.reason,
                        "submitted_at": time.time(),
                    }
                elif signal.action == "SELL":
                    # DON'T close the trade yet — wait for fill confirmation.
                    # Record the pending exit so _on_fill can close it.
                    if self.db:
                        open_trades = self.db.get_open_trades(symbol=signal.symbol)
                        if open_trades:
                            trade_id = open_trades[0]['id']
                            self._pending_exit_orders[order.order_id] = {
                                "trade_id": trade_id,
                                "symbol": signal.symbol,
                                "signal_reason": f"{tag} {signal.reason}",
                                "submitted_at": time.time(),
                            }
                            logger.info(f"Exit order {order.order_id} pending fill for trade {trade_id}")
                        else:
                            logger.warning(f"No open trade found for {signal.symbol} to close")
                
        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
    
    def _on_fill(self, order) -> None:
        """Handle order fill event — this is where PnL is actually booked."""
        logger.info(f"Fill CONFIRMED: {order.side} {order.filled_quantity} {order.symbol} @ ${order.avg_fill_price:.2f}")
        
        # Notify strategies
        for strategy in self._strategies:
            strategy.on_fill(order)
        
        # Reset circuit breaker on successful fill
        if self._consecutive_rejects > 0:
            logger.info(f"Order filled — resetting reject counter (was {self._consecutive_rejects})")
        self._consecutive_rejects = 0
        if self._circuit_breaker_tripped:
            self._circuit_breaker_tripped = False
            logger.info("Circuit breaker reset — order fills resumed")
        
        # Update order status in DB
        if self.db:
            self.db.update_order_status(
                order.order_id,
                "Filled",
                filled_quantity=order.filled_quantity,
                avg_fill_price=order.avg_fill_price,
            )
        
        # If this was a BUY fill, NOW create the trade in DB with real fill price
        if order.order_id in self._pending_entry_orders:
            entry_info = self._pending_entry_orders.pop(order.order_id)
            if self.db and order.avg_fill_price > 0:
                from .trade_database import Trade
                tag = entry_info.get('tag', '[PAPER]')
                reason = entry_info.get('reason', '')
                trade = Trade(
                    symbol=entry_info["symbol"],
                    action="BUY",
                    quantity=entry_info.get("quantity", order.filled_quantity or 1),
                    entry_price=order.avg_fill_price,
                    entry_time=get_eastern_time().isoformat(),
                    strategy_name=entry_info.get("strategy_name"),
                    option_type=entry_info.get("option_type"),
                    notes=f"{tag} {reason} [FILLED]",
                    status="open",
                    entry_order_id=order.order_id,
                )
                trade_id = self.db.insert_trade(trade)
                logger.info(f"Entry fill confirmed — trade {trade_id} created: "
                           f"{entry_info['symbol']} @ ${order.avg_fill_price:.2f}")
                _send_trade_alert(
                    f"ENTRY FILL: {entry_info['symbol']}",
                    f"Trade #{trade_id}\n{entry_info['symbol']} @ ${order.avg_fill_price:.2f}\nQty: {order.filled_quantity}",
                )
        if order.order_id in self._pending_exit_orders:
            exit_info = self._pending_exit_orders.pop(order.order_id)
            trade_id = exit_info["trade_id"]
            real_exit_price = order.avg_fill_price if order.avg_fill_price > 0 else 0
            if self.db and real_exit_price > 0:
                closed = self.db.close_trade(
                    trade_id=trade_id,
                    exit_price=real_exit_price,
                    exit_time=get_eastern_time().isoformat(),
                    exit_order_id=order.order_id,
                    notes=exit_info["signal_reason"],
                )
                if closed:
                    logger.info(f"Trade {trade_id} CONFIRMED closed: "
                               f"P&L ${closed['pnl']:.2f} ({closed['pnl_percent']:.2f}%) "
                               f"@ ${real_exit_price:.2f}")
                    pnl_sign = '+' if closed['pnl'] >= 0 else ''
                    _send_trade_alert(
                        f"EXIT FILL: {exit_info['symbol']} {pnl_sign}${closed['pnl']:.2f}",
                        f"Trade #{trade_id}\n{exit_info['symbol']} @ ${real_exit_price:.2f}\n"
                        f"P&L: {pnl_sign}${closed['pnl']:.2f} ({closed['pnl_percent']:.1f}%)\n"
                        f"Reason: {exit_info.get('signal_reason', 'N/A')}",
                    )
            elif real_exit_price == 0:
                # OE-R2 fix: fall back to last known bid when fill price is 0
                fallback_price = None
                try:
                    quote = self.quote_client.get_quote_by_symbol(exit_info["symbol"])
                    if quote:
                        fallback_price = quote.get('bidPrice') or quote.get('lastTradePrice')
                except Exception:
                    pass
                
                if fallback_price and fallback_price > 0:
                    logger.warning(f"Exit fill for trade {trade_id} had zero price — "
                                  f"using last-known bid ${fallback_price:.2f}")
                    closed = self.db.close_trade(
                        trade_id=trade_id,
                        exit_price=fallback_price,
                        exit_time=get_eastern_time().isoformat(),
                        exit_order_id=order.order_id,
                        notes=f"{exit_info['signal_reason']} [ZERO_FILL: bid fallback ${fallback_price:.2f}]",
                    )
                    if closed:
                        logger.info(f"Trade {trade_id} closed at bid fallback: P&L ${closed['pnl']:.2f}")
                else:
                    logger.error(f"Exit fill for trade {trade_id} has zero price and no bid fallback — "
                                f"closing at $0.01 to prevent permanent open state")
                    self.db.close_trade(
                        trade_id=trade_id,
                        exit_price=0.01,
                        exit_time=get_eastern_time().isoformat(),
                        exit_order_id=order.order_id,
                        notes=f"{exit_info['signal_reason']} [ZERO_FILL: forced $0.01 close]",
                    )
    
    def _on_reject(self, order) -> None:
        """Handle order rejection/cancellation — clean up phantom trades immediately."""
        logger.warning(f"Order REJECTED/CANCELLED: {order.side} {order.quantity} {order.symbol} "
                       f"orderId={order.order_id} status={order.status}")
        
        # Track consecutive rejects for circuit breaker
        self._consecutive_rejects += 1
        if self._consecutive_rejects >= self._REJECT_CIRCUIT_BREAKER and not self._circuit_breaker_tripped:
            self._circuit_breaker_tripped = True
            logger.critical(f"CIRCUIT BREAKER TRIPPED: {self._consecutive_rejects} consecutive order rejections — "
                           f"blocking new entries (likely margin/buying power issue)")
            _send_trade_alert(
                "CIRCUIT BREAKER TRIPPED",
                f"{self._consecutive_rejects} consecutive order rejections.\nNew entries BLOCKED.\nLikely margin/buying power issue — check IBKR.",
            )
        
        # If this was a BUY entry that was rejected, just discard — no DB trade was created
        if order.order_id in self._pending_entry_orders:
            entry_info = self._pending_entry_orders.pop(order.order_id)
            symbol = entry_info.get("symbol", "")
            logger.warning(f"ENTRY order {order.order_id} ({symbol}) was {order.status} — "
                          f"no trade recorded (order never filled)")
            # Notify strategy to clear any in-flight state
            for strategy in self._strategies:
                if hasattr(strategy, 'on_trade_cancelled'):
                    strategy.on_trade_cancelled(None, symbol)
        
        # If this was a SELL exit that was rejected, resubmit as MARKET order (max 2 retries)
        # LIMIT exits can miss in fast-moving 0DTE markets; MARKET ensures closure
        if order.order_id in self._pending_exit_orders:
            exit_info = self._pending_exit_orders.pop(order.order_id)
            trade_id = exit_info.get("trade_id")
            symbol = exit_info.get("symbol", "")
            retry_count = exit_info.get("retry_count", 0)
            max_retries = 2
            
            if retry_count >= max_retries:
                logger.critical(f"EXIT order for trade {trade_id} ({symbol}) FAILED after {retry_count} retries — "
                               f"POSITION MAY REMAIN OPEN — adding to stranded watchdog")
                _send_trade_alert(
                    f"STRANDED POSITION: {symbol}",
                    f"Trade #{trade_id} exit FAILED after {retry_count} retries.\nPosition may remain OPEN.\nAdded to watchdog for final MARKET sweep.",
                )
                # OE-R1 fix: hand off to position-loop watchdog for one final sweep attempt
                self._stranded_positions.append({
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "stranded_at": time.time(),
                })
                return
            
            logger.error(f"EXIT order {order.order_id} for trade {trade_id} ({symbol}) "
                        f"was {order.status} — resubmitting as MARKET order (retry {retry_count + 1}/{max_retries})")
            try:
                # Get quantity from DB trade
                qty = 1
                if self.db:
                    trade = self.db.get_trade(trade_id)
                    if trade:
                        qty = trade.get('quantity', 1)
                market_order = self.orders.sell(symbol=symbol, quantity=qty, limit_price=None)
                if market_order and market_order.order_id:
                    self._pending_exit_orders[market_order.order_id] = {
                        "trade_id": trade_id,
                        "symbol": symbol,
                        "signal_reason": f"MARKET RETRY #{retry_count + 1}: original exit {order.order_id} was {order.status}",
                        "submitted_at": time.time(),
                        "retry_count": retry_count + 1,
                    }
                    logger.info(f"MARKET exit order {market_order.order_id} submitted for trade {trade_id}")
            except Exception as retry_err:
                logger.error(f"Failed to resubmit MARKET exit for {symbol}: {retry_err}")
    
    def _sync_strategy_capital(self, nlv: float) -> None:
        """Push real broker equity into strategy account_capital for position sizing."""
        for strategy in self._strategies:
            if hasattr(strategy, 'account_capital'):
                old = strategy.account_capital
                strategy.account_capital = nlv
                if hasattr(strategy, 'risk_manager'):
                    strategy.risk_manager.capital = nlv
                if hasattr(strategy, 'persistence'):
                    strategy.persistence.state.current_capital = nlv
                if abs(old - nlv) > 1.0:
                    logger.info(f"Strategy capital synced to broker NLV: ${old:,.2f} -> ${nlv:,.2f}")

    def _update_state_broker_balance(self, nlv: float, cash: float = None) -> None:
        """Write real broker balance to trading_state.json for dashboard display."""
        import json, os, tempfile
        state_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trading_state.json')
        try:
            state = {}
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
            state['broker_nlv'] = nlv
            if cash is not None:
                state['broker_cash'] = cash
            state['broker_balance_time'] = get_eastern_time().isoformat()
            # Atomic write: temp file + rename to avoid dashboard reading partial JSON
            dir_name = os.path.dirname(state_file)
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
            try:
                with os.fdopen(fd, 'w') as tf:
                    json.dump(state, tf, indent=2)
                os.replace(tmp_path, state_file)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        except Exception as e:
            logger.debug(f"Failed to write broker balance to state: {e}")

    def _cancel_stale_orders(self) -> None:
        """Cancel all open IBKR orders from a previous session on startup."""
        try:
            if hasattr(self.client, 'cancel_all_open_orders'):
                logger.info("Cancelling any stale orders from previous session...")
                self.client.cancel_all_open_orders()
                logger.info("Stale orders cancelled")
            elif hasattr(self.client, 'get_account_orders'):
                orders = self.client.get_account_orders(self.config.account_id)
                open_orders = [o for o in orders if o.get('state') in
                               ('Submitted', 'PreSubmitted', 'Pending', 'Open')]
                for o in open_orders:
                    oid = o.get('id')
                    logger.warning(f"Cancelling stale order {oid}: "
                                   f"{o.get('side')} {o.get('totalQuantity')} {o.get('symbol')}")
                    self.client.cancel_order(self.config.account_id, oid)
                if open_orders:
                    logger.info(f"Cancelled {len(open_orders)} stale order(s)")
        except Exception as e:
            logger.error(f"Failed to cancel stale orders: {e}")

    def _get_flex_trades(self) -> List[Dict]:
        """Fetch historical trades from IBKR Flex Web Service (cached).

        Returns a list of Flex trade dicts, or empty list if Flex is not
        configured or the request fails.  Results are cached for
        ``_FLEX_CACHE_TTL`` seconds so the periodic reconciliation loop
        doesn't hammer the Flex API.
        """
        if not self._flex_client or not self._flex_query_id:
            return []
        now = time.time()
        if self._flex_trades_cache is not None and (now - self._flex_cache_time) < self._FLEX_CACHE_TTL:
            return self._flex_trades_cache
        try:
            trades = self._flex_client.fetch_trades(self._flex_query_id)
            self._flex_trades_cache = trades
            self._flex_cache_time = now
            logger.info(f"Flex: fetched {len(trades)} historical trades from IBKR")
            return trades
        except Exception as e:
            logger.warning(f"Flex trade fetch failed: {e}")
            return self._flex_trades_cache or []

    def _get_all_ibkr_executions(self, account_id: str = "") -> List[Dict]:
        """Get IBKR executions from ALL available sources.

        Priority:
          1. Flex Web Service (real historical trades — survives TWS restarts)
          2. TWS session executions (fallback — only current session)

        For Flex trades, the dict is normalised to the same keys the rest
        of the engine expects (trade_symbol, side, shares, price, time,
        order_id, exec_id).
        """
        # --- Flex historical trades (authoritative) ---
        flex_trades = self._get_flex_trades()
        normalised: List[Dict] = []
        if flex_trades:
            for ft in flex_trades:
                # Map Flex XML fields → engine execution format
                buy_sell = ft.get("buySell", ft.get("side", ""))
                # Flex uses BUY/SELL; adapter uses BOT/SLD
                if buy_sell.upper() in ("BUY", "BOT"):
                    side = "BOT"
                elif buy_sell.upper() in ("SELL", "SLD"):
                    side = "SLD"
                else:
                    side = buy_sell

                # Build internal trade symbol
                asset = ft.get("assetCategory", "")
                sym = ft.get("symbol", "")
                put_call = ft.get("putCall", "")
                strike = ft.get("strike", "")
                expiry = ft.get("expiry", ft.get("lastTradeDateOrContractMonth", ""))
                if asset == "OPT" and put_call and expiry:
                    strike_str = str(int(float(strike))) if strike else ""
                    right = put_call[0].upper() if put_call else ""
                    trade_sym = f"{sym}{expiry}{right}{strike_str}"
                else:
                    trade_sym = sym

                normalised.append({
                    "symbol": sym,
                    "trade_symbol": trade_sym,
                    "side": side,
                    "shares": abs(int(float(ft.get("quantity", 0)))),
                    "price": float(ft.get("tradePrice", ft.get("price", 0))),
                    "time": ft.get("dateTime", ft.get("tradeDate", "")),
                    "order_id": int(ft.get("orderID", ft.get("ibOrderID", 0)) or 0),
                    "exec_id": ft.get("ibExecID", ft.get("execID", "")),
                    "acct_number": ft.get("accountId", ""),
                    "commission": float(ft.get("ibCommission", ft.get("commission", 0)) or 0),
                    "realized_pnl": float(ft.get("fifoPnlRealized", ft.get("realizedPnl", 0)) or 0),
                })
            logger.info(f"Using {len(normalised)} Flex historical trades for reconciliation")
            return normalised

        # --- Fallback: TWS session executions ---
        try:
            execs = self.client.get_executions(account_id)
            if execs:
                logger.info(f"Flex unavailable — using {len(execs)} TWS session executions")
            return execs or []
        except Exception as e:
            logger.warning(f"TWS get_executions failed: {e}")
            return []

    def _check_time_based_exits(self) -> None:
        """Quote-independent exit: force-close positions past max hold or EOD exit time.

        The strategy's _check_exit only runs when an option quote arrives.
        If quotes stop flowing (illiquid 0DTE in the afternoon), positions
        can survive past max_hold and into exercise.  This method runs
        every ~30 s from the position loop and submits a MARKET exit
        for any strategy position that is overdue.
        """
        now = get_eastern_time()
        current_time = now.time()

        for strategy in self._strategies:
            if not hasattr(strategy, 'trade_state'):
                continue
            ts = strategy.trade_state
            if not ts.in_trade or not ts.symbol:
                continue

            # Determine max hold and EOD exit time from strategy config
            direction = ts.direction
            if direction == 'CALL':
                max_hold = getattr(strategy, 'call_max_hold_minutes', 80)
            else:
                max_hold = getattr(strategy, 'put_max_hold_minutes', 80)
            exit_time = getattr(strategy, 'exit_time', dt_time(15, 0))

            # Check max hold
            overdue = False
            reason = ""
            if ts.entry_time:
                try:
                    entry_dt = datetime.fromisoformat(ts.entry_time)
                    if entry_dt.tzinfo is None and now.tzinfo is not None:
                        entry_dt = entry_dt.replace(tzinfo=now.tzinfo)
                    hold_minutes = (now - entry_dt).total_seconds() / 60
                    if hold_minutes >= max_hold:
                        overdue = True
                        reason = f"QUOTE-INDEPENDENT MAX HOLD ({int(hold_minutes)}min >= {max_hold}min)"
                except (ValueError, TypeError):
                    pass

            # Check EOD exit time
            if not overdue and current_time >= exit_time:
                overdue = True
                reason = f"QUOTE-INDEPENDENT EOD EXIT (time={current_time} >= {exit_time})"

            if overdue:
                symbol = ts.symbol
                qty = ts.quantity
                logger.warning(f"TIME-BASED EXIT: {reason} — submitting MARKET sell for {qty}x {symbol}")

                # Check if there's already a pending exit
                already_pending = any(
                    info.get('symbol') == symbol
                    for info in self._pending_exit_orders.values()
                )
                if already_pending:
                    logger.info(f"TIME-BASED EXIT: {symbol} already has pending exit — skipping")
                    continue

                try:
                    order = self.orders.sell(symbol=symbol, quantity=qty, limit_price=None)
                    if order and order.order_id:
                        trade_id = None
                        if self.db:
                            open_trades = self.db.get_open_trades(symbol=symbol)
                            if open_trades:
                                trade_id = open_trades[0]['id']
                        self._pending_exit_orders[order.order_id] = {
                            "trade_id": trade_id,
                            "symbol": symbol,
                            "signal_reason": reason,
                            "submitted_at": time.time(),
                        }
                        logger.warning(f"TIME-BASED EXIT order {order.order_id} submitted for {symbol}")
                    else:
                        logger.error(f"TIME-BASED EXIT: failed to submit MARKET sell for {symbol}")
                except Exception as e:
                    logger.error(f"TIME-BASED EXIT: error submitting exit for {symbol}: {e}")

    def _eod_force_close_positions(self, use_market: bool = False) -> None:
        """
        Force-close ALL open option positions before expiry to prevent exercise.
        
        Called at 15:45 ET with LIMIT orders, then at 15:55 ET with MARKET orders
        as a last-resort emergency sweep.
        
        This prevents ITM 0DTE options from being exercised into massive
        short/long stock positions that blow up account margin.
        """
        if not self.db:
            return
        
        open_trades = self.db.get_open_trades()
        if not open_trades:
            logger.info("EOD liquidation: no open trades to close")
            return
        
        order_type = "MARKET" if use_market else "LIMIT"
        logger.warning(f"EOD FORCED LIQUIDATION ({order_type}): {len(open_trades)} open positions to close")
        
        for trade in open_trades:
            symbol = trade.get('symbol', '')
            trade_id = trade.get('id')
            qty = trade.get('quantity', 1)
            
            # Skip if already has a pending exit order
            already_pending = any(
                info.get('trade_id') == trade_id 
                for info in self._pending_exit_orders.values()
            )
            if already_pending:
                logger.info(f"EOD: trade {trade_id} ({symbol}) already has pending exit — skipping")
                continue
            
            try:
                if use_market:
                    # Emergency: MARKET order, get out at any price
                    order = self.orders.sell(symbol=symbol, quantity=qty, limit_price=None)
                else:
                    # Fetch live bid from broker for a realistic LIMIT price.
                    # Falls back to $0.01 only if the quote fetch fails.
                    limit = 0.01
                    try:
                        quote_data = self.quote_client.get_quote_by_symbol(symbol)
                        if quote_data:
                            bid = quote_data.get('bidPrice', 0) or 0
                            if bid > 0.01:
                                limit = round(bid * 0.95, 2)  # Slight discount for fast fill
                    except Exception as q_err:
                        logger.warning(f"EOD: could not fetch quote for {symbol}: {q_err}")
                    limit = max(limit, 0.01)
                    order = self.orders.sell(symbol=symbol, quantity=qty, limit_price=limit)
                
                if order and order.order_id:
                    self._pending_exit_orders[order.order_id] = {
                        "trade_id": trade_id,
                        "symbol": symbol,
                        "signal_reason": f"EOD FORCED LIQUIDATION ({order_type}) — prevent 0DTE exercise",
                        "submitted_at": time.time(),
                    }
                    logger.warning(f"EOD {order_type} exit order {order.order_id} for trade {trade_id} "
                                  f"({symbol} x{qty})")
                else:
                    logger.error(f"EOD: failed to submit {order_type} exit for {symbol}")
            except Exception as e:
                logger.error(f"EOD: failed to close {symbol}: {e}")

    def _check_eod_liquidation(self) -> None:
        """Check if we need to force-close positions before expiry. Called from position loop."""
        now = get_eastern_time()
        current_time = now.time()
        today_str = now.strftime("%Y%m%d")
        
        # Reset the flag at midnight for a new day
        if hasattr(self, '_eod_liquidation_date') and self._eod_liquidation_date != today_str:
            self._eod_liquidation_done = False
            self._eod_emergency_done = False
        
        # Phase 1: 15:45 ET — LIMIT orders to close everything
        if current_time >= self._EOD_LIQUIDATION_TIME and not self._eod_liquidation_done:
            self._eod_liquidation_done = True
            self._eod_liquidation_date = today_str
            logger.warning("=" * 60)
            logger.warning("EOD LIQUIDATION PHASE 1: Force-closing all positions (LIMIT)")
            logger.warning("=" * 60)
            _send_trade_alert(
                "EOD LIQUIDATION PHASE 1",
                "15:45 ET — Force-closing all positions with LIMIT orders.",
            )
            self._eod_force_close_positions(use_market=False)
        
        # Phase 2: 15:55 ET — MARKET orders for anything still open
        if current_time >= self._EOD_LAST_CHANCE_TIME and not self._eod_emergency_done:
            self._eod_emergency_done = True
            open_trades = self.db.get_open_trades() if self.db else []
            if open_trades:
                logger.critical("=" * 60)
                logger.critical(f"EOD EMERGENCY: {len(open_trades)} positions STILL OPEN at 15:55 — MARKET sweep")
                logger.critical("=" * 60)
                _send_trade_alert(
                    f"EOD EMERGENCY: {len(open_trades)} POSITIONS STILL OPEN",
                    f"15:55 ET — {len(open_trades)} positions still open!\nMARKET sweep executing NOW.\nManual check recommended.",
                )
                self._eod_force_close_positions(use_market=True)

    def _check_eod_flex_reconcile(self) -> None:
        """EOD Flex reconciliation — re-sync DB from IBKR Flex as source of truth.
        
        Runs once at 16:05 ET after market close. Performs a full_refresh
        (deletes all previously imported IBKR rows, re-imports from Flex)
        then reconciles the strategy state to match.
        """
        now = get_eastern_time()
        current_time = now.time()
        today_str = now.strftime("%Y%m%d")

        # Reset flag at midnight
        if hasattr(self, '_eod_flex_reconcile_date') and self._eod_flex_reconcile_date != today_str:
            self._eod_flex_reconcile_done = False

        if current_time >= self._EOD_FLEX_RECONCILE_TIME and not self._eod_flex_reconcile_done:
            self._eod_flex_reconcile_done = True
            self._eod_flex_reconcile_date = today_str
            self._run_flex_reconcile(source="EOD")

    def _run_flex_reconcile(self, source: str = "EOD") -> None:
        """Execute Flex full-refresh reconciliation."""
        if not self.db:
            return
        try:
            from live.trade_sync import TradeSync
            syncer = TradeSync(self.db)
            n = syncer.sync_from_ibkr(full_refresh=True)
            logger.info(f"Flex reconciliation ({source}): {n} trades imported from IBKR Flex")

            # Reconcile strategy state with updated DB
            for strategy in self._strategies:
                if hasattr(strategy, 'persistence') and strategy.persistence:
                    strategy.persistence.reconcile_with_db()
                    logger.info(f"Flex reconciliation ({source}): strategy state reconciled with DB")

            _send_trade_alert(
                f"FLEX RECONCILIATION ({source})",
                f"DB rebuilt from IBKR Flex — {n} trades imported.\n"
                f"Source of truth: IBKR Flex Web Service.",
            )
        except Exception as e:
            logger.warning(f"Flex reconciliation ({source}) failed: {e}")

    def _reconcile_broker_positions(self) -> None:
        """
        Check IBKR for real positions vs DB open trades.
        Two-way reconciliation:
          1. IBKR has position, DB doesn't → create DB entry (orphaned broker position)
          2. DB has open trade, IBKR doesn't → find real exit price from IBKR fills, then close

        Also syncs live market price from IBKR positions into open DB trades
        so the dashboard always shows the real current value.

        Uses normalized (underlying, expiry, strike, right) keys so that
        different symbol formats (Questrade vs IBKR) still match.
        """
        try:
            positions = self.client.get_account_positions(self.config.account_id)
            # Build set of normalized keys for broker positions with qty > 0
            broker_keys: Dict[Tuple, str] = {}  # normalized_key -> raw symbol
            broker_data: Dict[Tuple, Dict] = {}  # normalized_key -> position data
            
            if positions:
                for pos in positions:
                    symbol = pos.get('symbol', '')
                    qty = pos.get('openQuantity', 0)
                    if qty == 0:
                        continue
                    key = self._normalize_option_key(symbol)
                    if key:
                        broker_keys[key] = symbol
                        broker_data[key] = pos
            
            # Get open trades from DB
            if not self.db:
                return
            open_trades = self.db.get_open_trades()
            # Build normalized keys for DB trades
            db_keys: Dict[Tuple, dict] = {}  # normalized_key -> trade dict
            for t in open_trades:
                sym = t.get('symbol', '')
                key = self._normalize_option_key(sym)
                if key:
                    db_keys[key] = t

            # ── Direction 1: IBKR has position, DB doesn't → create + sync price ──
            for key, symbol in broker_keys.items():
                pos_data = broker_data.get(key, {})
                qty = pos_data.get('openQuantity', 1)
                avg_cost = pos_data.get('averageEntryPrice', 0)

                if key not in db_keys:
                    underlying, expiry, strike, right = key
                    option_type = 'call' if right == 'C' else 'put'

                    # Try to find the real entry price from IBKR executions
                    real_entry = avg_cost
                    try:
                        all_execs = self._get_all_ibkr_executions(self.config.account_id)
                        for ex in all_execs:
                            ex_sym = ex.get("trade_symbol", "")
                            ex_key = self._normalize_option_key(ex_sym)
                            if ex_key == key and ex.get("side") == "BOT":
                                real_entry = ex.get("price", avg_cost)
                                break
                    except Exception:
                        pass

                    logger.warning(
                        f"ORPHANED BROKER POSITION: {symbol} "
                        f"(not in DB) — creating DB entry for exit management "
                        f"({qty}x @ ${real_entry:.2f})"
                    )
                    try:
                        from .trade_database import Trade
                        trade = Trade(
                            symbol=symbol,
                            underlying=underlying,
                            trade_type="option",
                            option_type=option_type,
                            strike=strike,
                            expiration=expiry,
                            action="BUY",
                            quantity=qty,
                            entry_price=real_entry,
                            entry_time=get_eastern_time().isoformat(),
                            status="open",
                            strategy_name="recovered",
                            account_id=self.config.account_id,
                            notes="AUTO-CREATED: orphaned IBKR position adopted for exit management",
                        )
                        self.db.insert_trade(trade)
                        logger.info(f"Created DB entry for orphaned position: {symbol}")
                    except Exception as ins_err:
                        logger.error(f"Failed to create DB entry for orphaned {symbol}: {ins_err}")
                else:
                    # Position exists in both — sync entry price from IBKR if it differs
                    db_trade = db_keys[key]
                    trade_id = db_trade['id']
                    db_entry = db_trade.get('entry_price', 0)
                    if avg_cost > 0 and db_entry > 0 and abs(db_entry - avg_cost) > 0.005:
                        # IBKR avgCost is the truth — correct DB
                        logger.info(
                            f"Position price sync: trade {trade_id} "
                            f"DB entry=${db_entry:.4f} → IBKR avg=${avg_cost:.4f}"
                        )
                        self.db.update_trade(trade_id, entry_price=avg_cost)

            # ── Direction 2: DB has open trade, IBKR doesn't → phantom / filled / expired ──
            # Pre-fetch all IBKR executions once for exit price lookups
            all_execs = []
            try:
                all_execs = self._get_all_ibkr_executions(self.config.account_id)
            except Exception:
                pass

            for key, trade in db_keys.items():
                if key not in broker_keys:
                    trade_id = trade.get('id')
                    trade_symbol = trade.get('symbol', '')
                    entry_price = trade.get('entry_price', 0)
                    entry_order_id = trade.get('entry_order_id')

                    # Skip if there's a pending entry order — BUY not yet filled/confirmed
                    has_pending_entry = any(
                        info.get("symbol") == trade_symbol
                        for info in self._pending_entry_orders.values()
                    )
                    # Also check by entry_order_id directly
                    if not has_pending_entry and entry_order_id:
                        has_pending_entry = entry_order_id in self._pending_entry_orders
                    if has_pending_entry:
                        logger.info(
                            f"Trade {trade_id} ({trade_symbol}) has no IBKR position "
                            f"but has pending ENTRY order — waiting for fill/cancel"
                        )
                        continue

                    # Skip if there's a pending exit order — wait for fill confirmation
                    has_pending_exit = any(
                        info.get("trade_id") == trade_id
                        for info in self._pending_exit_orders.values()
                    )
                    if has_pending_exit:
                        logger.info(
                            f"Trade {trade_id} ({trade_symbol}) has no IBKR position "
                            f"but has pending exit order — waiting for fill confirmation"
                        )
                        continue

                    # Skip if trade was created very recently (< 120s) — give order time to fill
                    trade_entry_time = trade.get('entry_time', '')
                    if trade_entry_time:
                        try:
                            from datetime import datetime as _dt
                            if 'T' in trade_entry_time:
                                entry_dt = _dt.fromisoformat(trade_entry_time.replace('Z', '+00:00'))
                                now_dt = get_eastern_time()
                                if hasattr(entry_dt, 'tzinfo') and entry_dt.tzinfo and hasattr(now_dt, 'tzinfo') and now_dt.tzinfo:
                                    age_secs = (now_dt - entry_dt).total_seconds()
                                else:
                                    age_secs = 999
                                if age_secs < 120:
                                    logger.info(
                                        f"Trade {trade_id} ({trade_symbol}) has no IBKR position "
                                        f"but was created {age_secs:.0f}s ago — waiting for order fill"
                                    )
                                    continue
                        except Exception:
                            pass

                    # Search ALL IBKR executions for a SELL fill on this contract
                    exit_price = 0.0
                    exit_time = ""
                    exit_commission = 0.0
                    for ex in reversed(all_execs):
                        ex_sym = ex.get("trade_symbol", "")
                        ex_key = self._normalize_option_key(ex_sym)
                        if ex_key == key and ex.get("side") == "SLD":
                            exit_price = ex.get("price", 0)
                            exit_time = ex.get("time", "")
                            exit_commission = abs(ex.get("commission", 0))
                            break

                    if exit_price > 0:
                        logger.warning(
                            f"PHANTOM TRADE FIXED: trade {trade_id} ({trade_symbol}) "
                            f"no IBKR position but found SELL fill @ ${exit_price:.2f} — closing"
                        )
                        if exit_commission > 0:
                            old_comm = trade.get("commission", 0) or 0
                            self.db.update_trade(trade_id, commission=old_comm + exit_commission)
                        self.db.close_trade(
                            trade_id=trade_id,
                            exit_price=exit_price,
                            exit_time=exit_time or get_eastern_time().isoformat(),
                            notes="AUTO-CLOSED: no IBKR position, found real SELL fill (reconciliation)",
                        )
                    else:
                        # No SELL fill found anywhere — truly phantom or expired worthless
                        logger.error(
                            f"PHANTOM TRADE: trade {trade_id} ({trade_symbol}) "
                            f"has NO IBKR position and NO SELL fill in history — "
                            f"closing at $0.01 (expired/cancelled)"
                        )
                        self.db.close_trade(
                            trade_id=trade_id,
                            exit_price=0.01,
                            exit_time=get_eastern_time().isoformat(),
                            notes="AUTO-CLOSED: no IBKR position, no SELL fill (expired/phantom)",
                        )

            # ── Direction 3: Detect STOCK positions from option exercise/assignment ──
            # If IBKR shows a stock position for an option underlying (e.g. SPY)
            # that has no matching open DB trade, it was likely created by exercise.
            # Auto-close it with a MARKET order to prevent holding unintended stock.
            option_underlyings = set()
            if self.config.option_underlyings:
                option_underlyings = {u.upper() for u in self.config.option_underlyings}

            if positions:
                for pos in positions:
                    symbol = pos.get('symbol', '')
                    qty = pos.get('openQuantity', 0)
                    asset_class = pos.get('assetCategory', pos.get('secType', ''))

                    if qty == 0:
                        continue

                    # Only flag stock positions for known option underlyings
                    sym_upper = symbol.upper()
                    if sym_upper not in option_underlyings:
                        continue

                    # Skip if this is an option (already handled above)
                    if self._normalize_option_key(symbol) is not None:
                        continue

                    # This is a STOCK position for an option underlying — exercise/assignment
                    logger.critical(
                        f"EXERCISE STOCK DETECTED: {qty} shares of {symbol} "
                        f"(likely from 0DTE option exercise) — submitting MARKET cover"
                    )

                    # Check if there's already a pending order for this stock
                    already_pending = any(
                        info.get('symbol') == symbol
                        for info in self._pending_exit_orders.values()
                    )
                    if already_pending:
                        logger.info(f"EXERCISE STOCK: {symbol} already has pending exit — skipping")
                        continue

                    try:
                        if qty > 0:
                            # Long stock from call exercise → sell
                            order = self.orders.sell(symbol=symbol, quantity=qty, limit_price=None)
                        else:
                            # Short stock from put exercise → buy to cover
                            order = self.orders.buy(symbol=symbol, quantity=abs(qty), limit_price=None)

                        if order and order.order_id:
                            self._pending_exit_orders[order.order_id] = {
                                "trade_id": None,
                                "symbol": symbol,
                                "signal_reason": f"EXERCISE STOCK LIQUIDATION: {qty} shares of {symbol}",
                                "submitted_at": time.time(),
                            }
                            logger.critical(
                                f"EXERCISE STOCK: {'SELL' if qty > 0 else 'BUY'} order {order.order_id} "
                                f"submitted for {abs(qty)} shares of {symbol}"
                            )
                        else:
                            logger.error(f"EXERCISE STOCK: failed to submit cover order for {symbol}")
                    except Exception as ex_err:
                        logger.error(f"EXERCISE STOCK: error submitting cover for {symbol}: {ex_err}")

        except Exception as e:
            logger.error(f"Position reconciliation failed: {e}", exc_info=True)

    @staticmethod
    def _normalize_option_key(symbol: str) -> Optional[Tuple[str, str, float, str]]:
        """Normalize an option symbol to (underlying, expiry_yyyymmdd, strike, right).

        Handles both Questrade format (SPY16Mar26C669.00) and
        IBKR/OCC format (SPY20260316C669).
        Returns None for non-option symbols.
        """
        # Questrade-style: SPY16Mar26C669.00
        m = re.match(
            r'^([A-Z]+)(\d{2})([A-Za-z]{3})(\d{2})([CP])(\d+\.?\d*)$',
            symbol,
        )
        if m:
            underlying, day, mon_str, yr, right, strike = m.groups()
            months = {
                "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
            }
            month = months.get(mon_str, "01")
            return (underlying, f"20{yr}{month}{day}", float(strike), right)

        # IBKR / OCC fallback: SPY20260316C669
        m2 = re.match(r'^([A-Z]+)(\d{8})([CP])(\d+\.?\d*)$', symbol)
        if m2:
            underlying, expiry, right, strike = m2.groups()
            return (underlying, expiry, float(strike), right)

        return None

    def _reconcile_executions(self) -> None:
        """
        Fetch IBKR execution reports and fully reconcile against local trade DB.

        Uses Flex Web Service for historical trades when available (survives
        TWS restarts), falling back to TWS session executions.

        Full reconciliation:
          1. BUY fill exists but no DB trade → create missing trade entry
          2. BUY fill price differs from DB entry_price → fix to real IBKR price
          3. SELL fill exists but DB trade still 'open' → close with real fill price
          4. Already-closed DB trade has wrong exit_price (e.g. $0.01 phantom close)
             and a real SELL fill exists → correct exit_price and recalculate PnL
          5. Commission from IBKR → sync into DB trade
        """
        if not self.db:
            return

        try:
            executions = self._get_all_ibkr_executions(self.config.account_id)
            if not executions:
                return

            logger.info(f"Execution reconciliation: {len(executions)} IBKR execution(s)")

            # ── Build indexes of IBKR fills by normalized key ──
            # Each key → list of fills, sorted by time
            ibkr_buys: Dict[Tuple, List[Dict]] = {}   # norm_key → [buy fills]
            ibkr_sells: Dict[Tuple, List[Dict]] = {}   # norm_key → [sell fills]
            for ex in executions:
                trade_sym = ex.get("trade_symbol", "")
                norm = self._normalize_option_key(trade_sym)
                if not norm:
                    continue
                side = ex.get("side", "")
                if side == "BOT":
                    ibkr_buys.setdefault(norm, []).append(ex)
                elif side == "SLD":
                    ibkr_sells.setdefault(norm, []).append(ex)

            # Sort all fill lists by time
            for fills in ibkr_buys.values():
                fills.sort(key=lambda f: f.get("time", ""))
            for fills in ibkr_sells.values():
                fills.sort(key=lambda f: f.get("time", ""))

            # ── Get ALL DB trades (open + closed today) for matching ──
            open_trades = self.db.get_open_trades()
            today_str = get_eastern_time().strftime("%Y-%m-%d")
            closed_today = self.db.get_trades_by_date(today_str)

            all_db_trades = {t['id']: t for t in open_trades}
            for t in closed_today:
                all_db_trades[t['id']] = t

            # Index DB trades by normalized key
            db_by_norm: Dict[Tuple, List[Dict]] = {}
            for t in all_db_trades.values():
                sym = t.get('symbol', '')
                norm = self._normalize_option_key(sym)
                if norm:
                    db_by_norm.setdefault(norm, []).append(t)

            # Sort DB trades per key by entry_time
            for trades in db_by_norm.values():
                trades.sort(key=lambda t: t.get('entry_time', ''))

            fixes_applied = 0

            # ── CHECK 1: Sync BUY fill prices into DB entry_price ──
            for norm_key, buy_fills in ibkr_buys.items():
                db_trades = db_by_norm.get(norm_key, [])
                for i, fill in enumerate(buy_fills):
                    fill_price = fill.get("price", 0)
                    fill_time = fill.get("time", "")
                    fill_shares = fill.get("shares", 0)
                    fill_oid = fill.get("order_id", 0)
                    trade_sym = fill.get("trade_symbol", "")
                    commission = fill.get("commission", 0)

                    # Find matching DB trade: by order_id first, then positional (FIFO)
                    matched_trade = None
                    if fill_oid:
                        matched_trade = self.db.get_trade_by_order_id(fill_oid)
                    if not matched_trade and i < len(db_trades):
                        matched_trade = db_trades[i]

                    if not matched_trade:
                        # Missing entry — create it
                        logger.warning(
                            f"RECONCILE — MISSING ENTRY: IBKR BUY {fill_shares}x "
                            f"{trade_sym} @ ${fill_price:.2f} at {fill_time} — creating DB trade"
                        )
                        try:
                            underlying, expiry, strike, right = norm_key
                            option_type = 'call' if right == 'C' else 'put'
                            from .trade_database import Trade
                            new_trade = Trade(
                                symbol=trade_sym,
                                underlying=underlying,
                                trade_type="option",
                                option_type=option_type,
                                strike=strike,
                                expiration=expiry,
                                action="BUY",
                                quantity=fill_shares,
                                entry_price=fill_price,
                                entry_time=fill_time or get_eastern_time().isoformat(),
                                status="open",
                                commission=abs(commission),
                                entry_order_id=fill_oid if fill_oid else None,
                                strategy_name="recovered",
                                account_id=self.config.account_id,
                                notes="AUTO-CREATED: IBKR BUY fill had no DB trade (reconciliation)",
                            )
                            tid = self.db.insert_trade(new_trade)
                            # Add to index so SELL matching can find it
                            created = self.db.get_trade(tid)
                            if created:
                                db_trades.append(created)
                                all_db_trades[tid] = created
                            fixes_applied += 1
                        except Exception as ins_err:
                            logger.error(f"Failed to create missing trade for {trade_sym}: {ins_err}")
                        continue

                    # Entry exists — fix price if wrong
                    db_entry_price = matched_trade.get("entry_price", 0)
                    trade_id = matched_trade['id']
                    if fill_price > 0 and abs(db_entry_price - fill_price) > 0.005:
                        logger.warning(
                            f"RECONCILE — ENTRY PRICE FIX: trade {trade_id} "
                            f"DB=${db_entry_price:.4f} → IBKR=${fill_price:.4f}"
                        )
                        updates = {"entry_price": fill_price}
                        # If trade is already closed, recalculate PnL with correct entry
                        if matched_trade.get("status") == "closed" and matched_trade.get("exit_price"):
                            exit_p = matched_trade["exit_price"]
                            qty = matched_trade.get("quantity", 1)
                            comm = matched_trade.get("commission", 0)
                            pnl = (exit_p - fill_price) * qty * 100 - comm
                            pnl_pct = (pnl / (fill_price * qty * 100)) * 100 if fill_price > 0 else 0
                            updates["pnl"] = pnl
                            updates["pnl_percent"] = pnl_pct
                            logger.info(f"  Recalculated PnL for trade {trade_id}: ${pnl:.2f}")
                        self.db.update_trade(trade_id, **updates)
                        fixes_applied += 1

                    # Sync commission if we have it and DB doesn't
                    if commission and abs(commission) > 0:
                        db_comm = matched_trade.get("commission", 0) or 0
                        if abs(db_comm) < 0.01:
                            self.db.update_trade(trade_id, commission=abs(commission))

            # ── CHECK 2: Sync SELL fills into DB — close open trades, fix wrong exit prices ──
            for norm_key, sell_fills in ibkr_sells.items():
                db_trades = db_by_norm.get(norm_key, [])
                for i, fill in enumerate(sell_fills):
                    fill_price = fill.get("price", 0)
                    fill_time = fill.get("time", "")
                    fill_oid = fill.get("order_id", 0)
                    trade_sym = fill.get("trade_symbol", "")
                    commission = fill.get("commission", 0)

                    # Find matching DB trade: by order_id, then positional (FIFO)
                    matched_trade = None
                    if fill_oid:
                        matched_trade = self.db.get_trade_by_order_id(fill_oid)
                    if not matched_trade and i < len(db_trades):
                        matched_trade = db_trades[i]

                    if not matched_trade:
                        # SELL fill with no matching DB trade — already fully handled or
                        # was a position we never tracked. Log and skip.
                        logger.info(
                            f"RECONCILE — IBKR SELL {trade_sym} @ ${fill_price:.2f} "
                            f"has no matching DB trade (may already be closed)"
                        )
                        continue

                    trade_id = matched_trade['id']
                    status = matched_trade.get("status", "")

                    if status == "open":
                        # DB trade still open but IBKR has a SELL fill → close it now
                        # Skip if pending exit order (fill callback will handle)
                        has_pending = any(
                            info.get("trade_id") == trade_id
                            for info in self._pending_exit_orders.values()
                        )
                        if has_pending:
                            continue

                        logger.warning(
                            f"RECONCILE — MISSED EXIT: trade {trade_id} ({trade_sym}) "
                            f"has IBKR SELL @ ${fill_price:.2f} but DB still open — closing"
                        )
                        close_notes = f"AUTO-CLOSED: IBKR SELL fill (reconciliation, exec_id={fill.get('exec_id', '')})"
                        # Add commission to existing
                        if commission and abs(commission) > 0:
                            old_comm = matched_trade.get("commission", 0) or 0
                            self.db.update_trade(trade_id, commission=old_comm + abs(commission))
                        self.db.close_trade(
                            trade_id=trade_id,
                            exit_price=fill_price,
                            exit_time=fill_time or get_eastern_time().isoformat(),
                            exit_order_id=fill_oid if fill_oid else None,
                            notes=close_notes,
                        )
                        fixes_applied += 1

                    elif status == "closed":
                        # Already closed — verify exit_price matches IBKR
                        db_exit = matched_trade.get("exit_price", 0) or 0
                        if fill_price > 0 and abs(db_exit - fill_price) > 0.005:
                            # Check if the DB exit looks like a phantom ($0.01) or truly wrong
                            is_phantom = db_exit < 0.02
                            notes = matched_trade.get("notes", "") or ""
                            is_auto_closed = "AUTO-CLOSED" in notes and "no IBKR position" in notes

                            if is_phantom or is_auto_closed or abs(db_exit - fill_price) > 0.10:
                                entry_p = matched_trade.get("entry_price", 0)
                                qty = matched_trade.get("quantity", 1)
                                comm = matched_trade.get("commission", 0) or 0
                                if commission and abs(commission) > 0:
                                    comm = abs(commission)
                                pnl = (fill_price - entry_p) * qty * 100 - comm
                                pnl_pct = (pnl / (entry_p * qty * 100)) * 100 if entry_p > 0 else 0
                                logger.warning(
                                    f"RECONCILE — EXIT PRICE FIX: trade {trade_id} "
                                    f"DB exit=${db_exit:.4f} → IBKR=${fill_price:.4f} "
                                    f"(PnL ${matched_trade.get('pnl', 0):.2f} → ${pnl:.2f})"
                                )
                                self.db.update_trade(
                                    trade_id,
                                    exit_price=fill_price,
                                    exit_time=fill_time or matched_trade.get("exit_time", ""),
                                    exit_order_id=fill_oid if fill_oid else matched_trade.get("exit_order_id"),
                                    pnl=pnl,
                                    pnl_percent=pnl_pct,
                                    commission=comm,
                                    notes=f"PRICE-CORRECTED by reconciliation (was ${db_exit:.4f})",
                                )
                                fixes_applied += 1

            if fixes_applied:
                logger.info(f"Execution reconciliation: {fixes_applied} fix(es) applied")

        except Exception as e:
            logger.error(f"Execution reconciliation failed: {e}", exc_info=True)

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
    
    def _send_daily_summary(self) -> None:
        """Send end-of-day summary email with trade details, P&L, and account status."""
        try:
            from .health_report import check_database, check_trading_state
            db = check_database()
            state = check_trading_state()
            today = db.get("today", {})

            trades = today.get("trades", 0)
            wins = today.get("wins", 0)
            losses = today.get("losses", 0)
            pnl = today.get("pnl", 0)
            commissions = today.get("commissions", 0)
            net_pnl = today.get("net_pnl", 0)
            open_pos = db.get("open_positions", [])

            capital = state.get("current_capital", 0)
            total_pnl = state.get("total_pnl", 0)
            total_wins = state.get("total_wins", 0)
            total_losses = state.get("total_losses", 0)
            total_count = total_wins + total_losses
            win_rate = (total_wins / total_count * 100) if total_count > 0 else 0

            runtime = get_eastern_time() - self._start_time if self._start_time else None

            pnl_sign = "+" if net_pnl >= 0 else ""
            total_sign = "+" if total_pnl >= 0 else ""

            lines = [
                "DAILY TRADING SUMMARY",
                f"Date: {get_eastern_time().strftime('%Y-%m-%d')}",
                f"Runtime: {runtime}",
                "",
                "--- Today ---",
                f"Trades: {trades}  ({wins}W / {losses}L)",
                f"Gross P&L: ${pnl:,.2f}",
                f"Commissions: ${commissions:,.2f}",
                f"Net P&L: {pnl_sign}${net_pnl:,.2f}",
                "",
                "--- Account ---",
                f"Capital: ${capital:,.2f}",
                f"Total P&L: {total_sign}${total_pnl:,.2f}",
                f"Win Rate: {win_rate:.1f}% ({total_wins}W / {total_losses}L)",
                f"Max Drawdown: ${state.get('max_drawdown', 0):,.2f}",
            ]

            if open_pos:
                lines.append("")
                lines.append(f"--- Open Positions ({len(open_pos)}) ---")
                for p in open_pos:
                    lines.append(f"  {p['symbol']}  qty={p['qty']}  entry=${p['entry_price']:.2f}")

            body = "\n".join(lines)
            result_emoji = "+" if net_pnl >= 0 else ""
            _send_trade_alert(
                f"DAILY SUMMARY: {trades}T {wins}W/{losses}L {result_emoji}${net_pnl:,.2f}",
                body,
            )
        except Exception as e:
            logger.debug(f"Daily summary email failed (non-critical): {e}")

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
    client,
    account_id: str,
    symbols: List[str] = None,
    option_underlyings: List[str] = None,
    mode: str = "monitor",
    db_path: str = "live_trades.db",
    quote_client=None,
    chains_client=None
):
    """
    Factory function to create a fully configured trading engine.
    
    Args:
        client: Broker client for order execution and positions (IBKR)
        account_id: Account ID
        symbols: Stock symbols to track
        option_underlyings: Option underlying symbols
        mode: Trading mode - "monitor", "paper", or "live"
        db_path: Path to database file
        quote_client: Client for real-time stock/option quotes.
                      If None, uses the main client.
        chains_client: Client for option chain discovery (get_atm_options).
                       If None, uses quote_client.
        
    Returns:
        Configured LiveTradingEngine
    """
    from .trade_database import TradeDatabase
    from .position_manager import PositionManager
    from .order_manager import OrderManager
    
    # Create components
    # OrderManager uses the execution client (IBKR)
    # PositionManager uses IBKR for position sync, quote_client for price updates
    db = TradeDatabase(db_path)
    positions = PositionManager(client, trade_db=db, quote_client=quote_client)
    orders = OrderManager(client, db)
    
    config = EngineConfig(
        account_id=account_id,
        symbols=symbols or [],
        option_underlyings=option_underlyings or [],
        mode=mode
    )
    
    engine = LiveTradingEngine(
        client=client,
        trade_db=db,
        position_manager=positions,
        order_manager=orders,
        config=config,
        quote_client=quote_client,
        chains_client=chains_client
    )
    
    return engine
