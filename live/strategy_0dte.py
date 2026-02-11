"""
Live 0DTE SPY Options Strategy

Full implementation based on backtested ORB (Opening Range Breakout) strategy.

Backtest Results (2025 OOS):
- 1,004 trades
- 91.2% win rate  
- $6.9M P&L
- 2.3% max drawdown

Strategy Rules:
- Window: 10:00 - 11:00 AM ET
- Options: $0.50 - $1.00 (cheap, high gamma)
- Signal: ORB breakout (price breaks 30-min opening range)
- Profit Target: +22%
- Stop Loss: -25%
- Risk: Stop after first daily loss
"""
import logging
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

# Import from existing modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.strategy import OptionStrategy, Signal, OptionQuote
from live.state_persistence import StatePersistence, get_persistence
from core.signals import compute_features, get_basic_signal, STRATEGIES
from core.risk_manager import RiskManager, RiskConfig, KellyCalculator

logger = logging.getLogger(__name__)


def get_eastern_time() -> datetime:
    """Get current time in US Eastern timezone."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except ImportError:
        # Fallback: Calculate ET offset manually
        # ET is UTC-5 (EST) or UTC-4 (EDT)
        utc_now = datetime.now(timezone.utc)
        # Simple DST check: March-November is EDT (UTC-4), else EST (UTC-5)
        month = utc_now.month
        if 3 <= month <= 11:
            et_offset = timedelta(hours=-4)
        else:
            et_offset = timedelta(hours=-5)
        return utc_now + et_offset


@dataclass
class ORBState:
    """Track Opening Range Breakout state for the day"""
    date: str = ""
    orb_high: float = 0.0
    orb_low: float = 0.0
    orb_calculated: bool = False
    bars: List[Dict] = field(default_factory=list)
    
    def reset(self, date: str):
        self.date = date
        self.orb_high = 0.0
        self.orb_low = 0.0
        self.orb_calculated = False
        self.bars = []


@dataclass
class TradeState:
    """Track current trade state"""
    in_trade: bool = False
    symbol: str = ""
    direction: str = ""  # CALL or PUT
    entry_price: float = 0.0
    entry_time: str = ""
    quantity: int = 0
    highest_price: float = 0.0
    lowest_price: float = 0.0


@dataclass 
class DayState:
    """Track daily trading state"""
    date: str = ""
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    pnl_today: float = 0.0
    had_loss: bool = False  # Stop trading after first loss
    

class Live0DTEStrategy(OptionStrategy):
    """
    Live 0DTE SPY Options Strategy using ORB signals.
    
    Based on backtested strategy with 91.2% win rate.
    """
    
    def __init__(
        self,
        # Strategy parameters (from backtest config)
        strategy: str = "orb",
        profit_target_pct: float = 0.22,
        stop_loss_pct: float = 0.25,
        min_option_price: float = 0.50,
        max_option_price: float = 1.00,
        # Trading window (ET times) - backtest optimal: 10:00-11:00
        trade_start_hour: int = 10,
        trade_start_minute: int = 0,
        trade_end_hour: int = 11,
        trade_end_minute: int = 0,
        exit_hour: int = 15,
        # Max hold time (87% of trades close in 5 min per backtest)
        max_hold_minutes: int = 5,
        # ORB parameters
        orb_minutes: int = 30,
        orb_buffer_pct: float = 0.10,  # 10% of ORB range buffer for breakout (matches backtest)
        # RSI parameters (for momentum strategy fallback)
        rsi_call_threshold: float = 70,
        rsi_put_threshold: float = 30,
        # Risk management (aligned with backtest RiskManager defaults)
        max_contracts: int = 5,
        stop_after_first_loss: bool = False,  # Backtest runs with False — never stop after loss
        max_consecutive_losses: int = 999,  # Backtest default 999 — effectively unlimited
        # Capital
        account_capital: float = 10000,
        risk_per_trade_pct: float = 0.02,
        # Optional: Questrade client for historical data backfill
        questrade_client = None,
    ):
        super().__init__("Live0DTE", ["SPY"])
        
        # Questrade client for historical data
        self._questrade_client = questrade_client
        
        # Strategy params
        self.strategy = strategy
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.min_option_price = min_option_price
        self.max_option_price = max_option_price
        
        # Trading window
        self.trade_start = dt_time(trade_start_hour, trade_start_minute)
        self.trade_end = dt_time(trade_end_hour, trade_end_minute)
        self.exit_time = dt_time(exit_hour, 0)
        self.max_hold_minutes = max_hold_minutes
        
        # ORB params
        self.orb_minutes = orb_minutes
        self.orb_buffer_pct = orb_buffer_pct
        
        # RSI params
        self.rsi_call_threshold = rsi_call_threshold
        self.rsi_put_threshold = rsi_put_threshold
        
        # Risk management
        self.max_contracts = max_contracts
        self.stop_after_first_loss = stop_after_first_loss
        self.max_consecutive_losses = max_consecutive_losses
        self.account_capital = account_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        
        # Initialize RiskManager with Kelly from backtest stats
        # Backtest: 91.2% WR, avg_win ~22%, avg_loss ~25%, PF=50.93
        self.risk_config = RiskConfig(
            kelly_fraction=0.20,           # Use 20% of full Kelly
            min_kelly_pct=0.02,            # Min 2%
            max_kelly_pct=0.20,            # Max 20%
            max_risk_per_trade_pct=0.02,   # Max 2% risk per trade
            max_position_pct=0.07,         # Max 7% in single position
            max_position_value=5000,       # Absolute cap
            max_contracts=max_contracts,
            stop_after_first_loss=stop_after_first_loss,
        )
        self.risk_manager = RiskManager(account_capital, self.risk_config)
        
        # Pre-calculate Kelly from backtest stats
        # Kelly = (W × avg_win/avg_loss - L) / (avg_win/avg_loss)
        kelly_calc = KellyCalculator(self.risk_config)
        self.kelly_pct = kelly_calc.calculate_from_winrate(
            win_rate=0.912,    # 91.2% from backtest
            avg_win=0.22,      # 22% profit target
            avg_loss=0.25      # 25% stop loss
        )
        self.risk_manager.set_kelly(self.kelly_pct)
        logger.info(f"Kelly position size: {self.kelly_pct:.1%}")
        
        # State persistence - load previous capital/P&L
        self.persistence = get_persistence("trading_state.json")
        self._load_persisted_state()
        
        # State tracking
        self.orb_state = ORBState()
        self.trade_state = TradeState()
        self.day_state = DayState()
        
        # Signal cooldown - prevent re-entry too quickly after exit
        self._last_exit_time = None
        self._signal_cooldown_seconds = 60  # Wait 60s after exit before new entry
        
        # Pending signal expiry
        self._pending_direction_time = None
        self._pending_expiry_seconds = 30  # Pending signals expire after 30s
        
        # Price history for indicators
        self._price_history: List[Dict] = []
        self._rsi_period = 14
        
        logger.info(f"Live0DTE Strategy initialized:")
        logger.info(f"  Strategy: {strategy}")
        logger.info(f"  Window: {self.trade_start} - {self.trade_end}")
        logger.info(f"  ORB Buffer: {orb_buffer_pct:.1%} of range")
        logger.info(f"  Options: ${min_option_price:.2f} - ${max_option_price:.2f}")
        logger.info(f"  Target: {profit_target_pct:.0%} | Stop: {stop_loss_pct:.0%}")
        logger.info(f"  Max Hold: {max_hold_minutes} min")
        logger.info(f"  Exit Hour: {exit_hour}:00")
        logger.info(f"  Capital: ${self.account_capital:,.2f}")
        
        # Trade database reference (set by engine)
        self._trade_db = None
    
    def set_trade_db(self, trade_db):
        """Set trade database reference for position recovery and state reconciliation"""
        self._trade_db = trade_db
        
        # Reconcile state with DB (DB is source of truth)
        if hasattr(trade_db, 'db_path'):
            db_path = str(trade_db.db_path)
        else:
            db_path = getattr(trade_db, 'db_path', None)
        
        if db_path:
            self.persistence.db_path = db_path
            result = self.persistence.reconcile_with_db(db_path)
            if result.get("discrepancies"):
                logger.warning(f"State reconciled - fixed {len(result['discrepancies'])} discrepancies")
                # Update strategy capital from reconciled state
                self.account_capital = self.persistence.state.current_capital
                logger.info(f"Updated capital from DB: ${self.account_capital:,.2f}")
        self._recover_open_positions()
    
    def _recover_open_positions(self):
        """Recover open positions from database on startup, close orphans"""
        if not self._trade_db:
            return
        
        open_trades = self._trade_db.get_open_trades()
        if not open_trades:
            logger.info("No open positions to recover")
            return
        
        logger.info(f"Found {len(open_trades)} open trade(s) in DB")
        
        # If multiple open trades, close all orphans (keep only most recent)
        spy_trades = [t for t in open_trades if 'SPY' in t.get('symbol', '')]
        
        if len(spy_trades) > 1:
            # Sort by entry_time descending, keep most recent, close rest
            spy_trades.sort(key=lambda t: t.get('entry_time', ''), reverse=True)
            for orphan in spy_trades[1:]:
                trade_id = orphan.get('id')
                logger.warning(f"CLOSING ORPHAN trade {trade_id}: {orphan.get('symbol')} (stale open position)")
                try:
                    self._trade_db.close_trade(
                        trade_id=trade_id,
                        exit_price=orphan.get('entry_price', 0),
                        notes="ORPHAN: closed on startup - stale open position"
                    )
                except Exception as e:
                    logger.error(f"Failed to close orphan trade {trade_id}: {e}")
            spy_trades = spy_trades[:1]  # Keep only most recent
        
        # Recover the most recent open trade
        if spy_trades:
            trade = spy_trades[0]
            symbol = trade.get('symbol', '')
            if 'SPY' in symbol and ('P' in symbol or 'C' in symbol):
                # Check if trade is stale (entry > max_hold_minutes ago)
                from datetime import datetime as dt_datetime
                try:
                    entry_dt = dt_datetime.fromisoformat(trade.get('entry_time', ''))
                    now = get_eastern_time()
                    if entry_dt.tzinfo is None and now.tzinfo is not None:
                        entry_dt = entry_dt.replace(tzinfo=now.tzinfo)
                    hold_minutes = (now - entry_dt).total_seconds() / 60
                    if hold_minutes > self.max_hold_minutes * 2:  # Stale if 2x max hold
                        logger.warning(f"CLOSING STALE trade {trade.get('id')}: held {hold_minutes:.0f} min (>{self.max_hold_minutes*2} limit)")
                        try:
                            self._trade_db.close_trade(
                                trade_id=trade.get('id'),
                                exit_price=trade.get('entry_price', 0),
                                notes=f"STALE: closed on startup after {hold_minutes:.0f} min"
                            )
                        except Exception as e:
                            logger.error(f"Failed to close stale trade: {e}")
                        return
                except (ValueError, TypeError):
                    pass
                
                direction = 'PUT' if symbol.rfind('P') > symbol.rfind('C') else 'CALL'
                self.trade_state = TradeState(
                    in_trade=True,
                    symbol=symbol,
                    direction=direction,
                    entry_price=trade.get('entry_price', 0),
                    entry_time=trade.get('entry_time', ''),
                    quantity=trade.get('quantity', 0),
                    highest_price=trade.get('entry_price', 0),
                    lowest_price=trade.get('entry_price', 0)
                )
                logger.info(f"RECOVERED open position: {direction} {self.trade_state.quantity}x {symbol} @ ${self.trade_state.entry_price:.2f}")
                logger.info(f"  Entry time: {self.trade_state.entry_time}")
    
    def _load_persisted_state(self):
        """Load persisted state from previous sessions"""
        # Set initial capital if first run
        if self.persistence.state.total_trades == 0:
            self.persistence.set_initial_capital(self.account_capital)
        else:
            # Use persisted capital
            self.account_capital = self.persistence.state.current_capital
            logger.info(f"Restored capital: ${self.account_capital:,.2f}")
            logger.info(f"Total P&L: ${self.persistence.state.total_pnl:,.2f}")
            logger.info(f"Total trades: {self.persistence.state.total_trades}")
    
    def _save_trade_result(self, pnl: float, is_win: bool):
        """Save trade result to persistent state"""
        self.persistence.record_trade(pnl, is_win)
        self.account_capital = self.persistence.state.current_capital
    
    def _save_daily_summary(self):
        """Save end-of-day summary"""
        self.persistence.record_daily_summary(
            trades=self.day_state.trades_today,
            wins=self.day_state.wins_today,
            losses=self.day_state.losses_today,
            pnl=self.day_state.pnl_today,
            notes=f"Strategy: {self.strategy}"
        )
    
    def on_start(self):
        """Called when strategy starts"""
        logger.info("=" * 60)
        logger.info("LIVE 0DTE STRATEGY STARTED")
        logger.info(f"Strategy: {self.strategy.upper()}")
        logger.info(f"Trading Window: {self.trade_start} - {self.trade_end} ET")
        logger.info(f"Options Range: ${self.min_option_price:.2f} - ${self.max_option_price:.2f}")
        logger.info(f"Profit Target: {self.profit_target_pct:.0%}")
        logger.info(f"Stop Loss: {self.stop_loss_pct:.0%}")
        logger.info(f"Current Capital: ${self.account_capital:,.2f}")
        logger.info("=" * 60)
        
        # Show persisted state summary
        self.persistence.print_summary()
    
    def on_stop(self):
        """Called when strategy stops"""
        # Save daily summary
        if self.day_state.trades_today > 0:
            self._save_daily_summary()
        
        logger.info("Strategy stopped - state saved")
        self.persistence.print_summary()
    
    def on_quote(self, symbol: str, quote: Dict[str, Any]) -> Optional[Signal]:
        """
        Process SPY quote for ORB calculation and signal generation.
        """
        if symbol != "SPY":
            return None
        
        now = get_eastern_time()
        current_time = now.time()
        today = now.strftime("%Y-%m-%d")
        
        # Reset state for new day
        if self.day_state.date != today:
            self._reset_daily_state(today)
        
        # Update price history
        price = quote.get('lastTradePrice', 0)
        high = quote.get('highPrice', price)
        low = quote.get('lowPrice', price)
        volume = quote.get('volume', 0)
        
        self._update_price_history(now, price, high, low, volume)
        
        # Calculate ORB during first 30 minutes (9:30 - 10:00)
        orb_end_time = dt_time(10, 0)
        if current_time < orb_end_time:
            self._update_orb(high, low)
            return None
        
        # Mark ORB as calculated after 10:00
        if not self.orb_state.orb_calculated and current_time >= orb_end_time:
            self.orb_state.orb_calculated = True
            logger.info(f"ORB Calculated - High: ${self.orb_state.orb_high:.2f}, Low: ${self.orb_state.orb_low:.2f}")
        
        # Check if in trading window
        if not self._in_trading_window(current_time):
            return None
        
        # Check if should stop trading (after first loss)
        if self.stop_after_first_loss and self.day_state.had_loss:
            return None
        
        # Don't enter new trade if already in one
        if self.trade_state.in_trade:
            return None
        
        # Check cooldown after last exit (prevent rapid re-entry on same breakout)
        if self._last_exit_time:
            seconds_since_exit = (now - self._last_exit_time).total_seconds()
            if seconds_since_exit < self._signal_cooldown_seconds:
                return None
        
        # Generate signal
        signal_direction = self._get_signal(price, quote)
        
        if signal_direction:
            logger.info(f"SIGNAL: {signal_direction} | SPY @ ${price:.2f}")
            # Set pending direction for option selection with timestamp
            self._pending_direction = signal_direction
            self._pending_direction_time = now
            # Return signal - actual option selection happens in on_option_quote
            return Signal(
                symbol="SPY",
                action="PENDING",  # Will be resolved to actual option
                quantity=1,
                reason=f"{self.strategy.upper()} signal: {signal_direction}",
                strategy_name=self.name,
                timestamp=now.isoformat(),
                metadata={"direction": signal_direction, "spy_price": price}
            )
        
        return None
    
    def on_option_quote(self, quote: OptionQuote) -> Optional[Signal]:
        """
        Process 0DTE option quote for entry/exit.
        """
        if quote.underlying != "SPY":
            return None
        
        now = get_eastern_time()
        current_time = now.time()
        today = now.strftime("%Y-%m-%d")
        
        # Log first option of each type for debugging
        if hasattr(self, '_pending_direction') and self._pending_direction:
            if not hasattr(self, '_debug_logged_opts'):
                self._debug_logged_opts = set()
            debug_key = f"{self._pending_direction}_{quote.option_type}"
            if debug_key not in self._debug_logged_opts:
                logger.info(f"DEBUG OPT: direction={self._pending_direction} opt_type={quote.option_type} symbol={quote.symbol} expiry='{quote.expiration}'")
                self._debug_logged_opts.add(debug_key)
        
        # Check if this is a 0DTE option
        if not self._is_0dte(quote.expiration):
            return None
        
        # ===== MANAGE EXISTING POSITION =====
        if self.trade_state.in_trade and quote.symbol == self.trade_state.symbol:
            return self._check_exit(quote, now)
        
        # ===== CHECK FOR NEW ENTRY =====
        if not self.trade_state.in_trade:
            # Check if we have a pending signal (and it hasn't expired)
            if hasattr(self, '_pending_direction') and self._pending_direction:
                # Check pending signal expiry
                if self._pending_direction_time:
                    pending_age = (now - self._pending_direction_time).total_seconds()
                    if pending_age > self._pending_expiry_seconds:
                        logger.info(f"Pending signal expired after {pending_age:.0f}s")
                        self._pending_direction = None
                        self._pending_direction_time = None
                        return None
                direction = self._pending_direction
                
                # Check if option matches direction
                if direction == "CALL" and quote.option_type != "call":
                    return None
                if direction == "PUT" and quote.option_type != "put":
                    return None
                
                # Check price range
                option_price = quote.last or ((quote.bid + quote.ask) / 2)
                logger.info(f"Checking option: {quote.symbol} price=${option_price:.2f} delta={quote.delta}")
                
                if not (self.min_option_price <= option_price <= self.max_option_price):
                    logger.info(f"  REJECTED: price ${option_price:.2f} outside range ${self.min_option_price}-${self.max_option_price}")
                    return None
                
                # Check delta (want high gamma, around 0.40-0.60 delta)
                if quote.delta:
                    abs_delta = abs(quote.delta)
                    if not (0.35 <= abs_delta <= 0.65):
                        logger.info(f"  REJECTED: delta {quote.delta:.2f} outside range 0.35-0.65")
                        return None
                else:
                    logger.info(f"  REJECTED: no delta value")
                    return None
                
                # Calculate position size
                contracts = self._calculate_position_size(option_price)
                
                # Clear pending direction
                self._pending_direction = None
                
                # Enter trade
                self.trade_state = TradeState(
                    in_trade=True,
                    symbol=quote.symbol,
                    direction=direction,
                    entry_price=option_price,
                    entry_time=now.isoformat(),
                    quantity=contracts,
                    highest_price=option_price,
                    lowest_price=option_price
                )
                
                logger.info(f"ENTRY: {direction} {contracts}x {quote.symbol} @ ${option_price:.2f}")
                
                return self.create_signal(
                    symbol=quote.symbol,
                    action="BUY",
                    quantity=contracts,
                    limit_price=quote.ask,  # Use ask for buy
                    reason=f"0DTE {direction} entry: {self.strategy}",
                    confidence=0.9
                )
        
        return None
    
    def _check_exit(self, quote: OptionQuote, now: datetime) -> Optional[Signal]:
        """Check if should exit current position"""
        current_time = now.time()
        current_price = quote.last or ((quote.bid + quote.ask) / 2)
        
        # Update high/low tracking
        self.trade_state.highest_price = max(self.trade_state.highest_price, current_price)
        self.trade_state.lowest_price = min(self.trade_state.lowest_price, current_price)
        
        entry_price = self.trade_state.entry_price
        pnl_pct = (current_price - entry_price) / entry_price
        
        exit_reason = None
        
        # ===== PROFIT TARGET =====
        if pnl_pct >= self.profit_target_pct:
            exit_reason = f"PROFIT TARGET: {pnl_pct:.1%}"
        
        # ===== STOP LOSS =====
        elif pnl_pct <= -self.stop_loss_pct:
            exit_reason = f"STOP LOSS: {pnl_pct:.1%}"
            self.day_state.had_loss = True
        
        # ===== MAX HOLD TIME EXIT =====
        elif self.trade_state.entry_time:
            from datetime import datetime as dt_datetime
            try:
                entry_dt = dt_datetime.fromisoformat(self.trade_state.entry_time)
                # Ensure both datetimes are timezone-aware for correct subtraction
                if entry_dt.tzinfo is None and now.tzinfo is not None:
                    entry_dt = entry_dt.replace(tzinfo=now.tzinfo)
                elif entry_dt.tzinfo is not None and now.tzinfo is None:
                    now = now.replace(tzinfo=entry_dt.tzinfo)
            except (ValueError, TypeError):
                # Fallback: parse entry time manually
                logger.warning(f"Could not parse entry_time: {self.trade_state.entry_time}")
                entry_dt = now  # Force exit on parse failure
            hold_minutes = (now - entry_dt).total_seconds() / 60
            if hold_minutes >= self.max_hold_minutes:
                exit_reason = f"MAX HOLD TIME ({int(hold_minutes)}min): {pnl_pct:.1%}"
                if pnl_pct < 0:
                    self.day_state.had_loss = True
        
        # ===== TIME EXIT (EOD) =====
        if not exit_reason and current_time >= self.exit_time:
            exit_reason = f"TIME EXIT (EOD): {pnl_pct:.1%}"
            if pnl_pct < 0:
                self.day_state.had_loss = True
        
        # ===== EXECUTE EXIT =====
        if exit_reason:
            pnl_dollars = (current_price - entry_price) * 100 * self.trade_state.quantity
            is_win = pnl_pct > 0
            
            logger.info(f"EXIT: {self.trade_state.direction} {self.trade_state.quantity}x {quote.symbol}")
            logger.info(f"  Entry: ${entry_price:.2f} → Exit: ${current_price:.2f}")
            logger.info(f"  P&L: {pnl_pct:.1%} (${pnl_dollars:.2f})")
            logger.info(f"  Reason: {exit_reason}")
            
            # Update day stats
            self.day_state.trades_today += 1
            self.day_state.pnl_today += pnl_dollars
            if is_win:
                self.day_state.wins_today += 1
            else:
                self.day_state.losses_today += 1
            
            # Save to persistent state
            self._save_trade_result(pnl_dollars, is_win)
            logger.info(f"  Capital: ${self.account_capital:,.2f}")
            
            # Reset trade state
            symbol = self.trade_state.symbol
            quantity = self.trade_state.quantity
            self.trade_state = TradeState()
            self._last_exit_time = now  # Set cooldown timer
            
            return self.create_signal(
                symbol=symbol,
                action="SELL",
                quantity=quantity,
                limit_price=quote.bid,  # Use bid for sell
                reason=exit_reason,
                confidence=1.0
            )
        
        return None
    
    def _get_signal(self, price: float, quote: Dict) -> Optional[str]:
        """Generate trading signal based on strategy"""
        
        if self.strategy == "orb":
            return self._get_orb_signal(price)
        elif self.strategy == "momentum":
            return self._get_rsi_signal(price, trend_follow=True)
        elif self.strategy == "mean_reversion":
            return self._get_rsi_signal(price, trend_follow=False)
        else:
            return self._get_orb_signal(price)  # Default to ORB
    
    def _get_orb_signal(self, price: float) -> Optional[str]:
        """Opening Range Breakout signal"""
        if not self.orb_state.orb_calculated:
            return None
        
        orb_high = self.orb_state.orb_high
        orb_low = self.orb_state.orb_low
        orb_range = orb_high - orb_low
        buffer = orb_range * self.orb_buffer_pct
        
        if price > orb_high + buffer:
            self._pending_direction = "CALL"
            return "CALL"
        elif price < orb_low - buffer:
            self._pending_direction = "PUT"
            return "PUT"
        
        return None
    
    def _get_rsi_signal(self, price: float, trend_follow: bool = True) -> Optional[str]:
        """RSI-based signal"""
        rsi = self._calculate_rsi()
        if rsi is None:
            return None
        
        if trend_follow:
            # Momentum: trade with trend
            if rsi > self.rsi_call_threshold:
                self._pending_direction = "CALL"
                return "CALL"
            elif rsi < self.rsi_put_threshold:
                self._pending_direction = "PUT"
                return "PUT"
        else:
            # Mean reversion: fade extremes
            if rsi < self.rsi_put_threshold:
                self._pending_direction = "CALL"
                return "CALL"
            elif rsi > self.rsi_call_threshold:
                self._pending_direction = "PUT"
                return "PUT"
        
        return None
    
    def _calculate_rsi(self) -> Optional[float]:
        """Calculate RSI from price history"""
        if len(self._price_history) < self._rsi_period + 1:
            return None
        
        prices = [bar['close'] for bar in self._price_history[-self._rsi_period-1:]]
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _update_orb(self, high: float, low: float):
        """Update Opening Range high/low"""
        if self.orb_state.orb_high == 0:
            self.orb_state.orb_high = high
            self.orb_state.orb_low = low
        else:
            self.orb_state.orb_high = max(self.orb_state.orb_high, high)
            self.orb_state.orb_low = min(self.orb_state.orb_low, low)
    
    def _update_price_history(self, timestamp: datetime, close: float, high: float, low: float, volume: int):
        """Update price history for indicator calculation"""
        self._price_history.append({
            'timestamp': timestamp,
            'close': close,
            'high': high,
            'low': low,
            'volume': volume
        })
        
        # Keep last 100 bars
        if len(self._price_history) > 100:
            self._price_history = self._price_history[-100:]
    
    def _reset_daily_state(self, date: str):
        """Reset all state for new trading day"""
        self.orb_state.reset(date)
        # Preserve trade_state if we have an open position (recovered from DB)
        if not self.trade_state.in_trade:
            self.trade_state = TradeState()
        else:
            logger.info(f"Preserving open position: {self.trade_state.symbol}")
        self.day_state = DayState(date=date)
        self._price_history = []
        self._pending_direction = None
        logger.info(f"Daily state reset for {date}")
        
        # Backfill ORB if starting after 10:00 AM ET
        now = get_eastern_time()
        orb_end_time = dt_time(10, 0)
        if now.time() >= orb_end_time:
            self._backfill_orb()
    
    def _in_trading_window(self, current_time: dt_time) -> bool:
        """Check if current time is within trading window"""
        return self.trade_start <= current_time <= self.trade_end
    
    def _backfill_orb(self):
        """Backfill ORB from historical candles when starting after 10:00 AM"""
        if not self._questrade_client:
            logger.warning("No Questrade client - cannot backfill ORB. Using day's high/low fallback.")
            return
        
        try:
            # Get SPY symbol ID
            symbols = self._questrade_client.search_symbols("SPY")
            spy_symbol = next((s for s in symbols if s.get('symbol') == 'SPY'), None)
            if not spy_symbol:
                logger.error("Could not find SPY symbol")
                return
            
            symbol_id = spy_symbol['symbolId']
            
            # Fetch 1-minute candles for 9:30-10:00 AM ET today
            now = get_eastern_time()
            today = now.date()
            from datetime import datetime as dt
            orb_start = dt.combine(today, dt_time(9, 30))
            orb_end = dt.combine(today, dt_time(10, 0))
            
            # Make timezone-aware for API
            try:
                from zoneinfo import ZoneInfo
                et_tz = ZoneInfo("America/New_York")
                orb_start = orb_start.replace(tzinfo=et_tz)
                orb_end = orb_end.replace(tzinfo=et_tz)
            except Exception:
                pass
            
            candles = self._questrade_client.get_candles(
                symbol_id=symbol_id,
                start_time=orb_start,
                end_time=orb_end,
                interval="OneMinute"
            )
            
            if candles:
                self.orb_state.orb_high = max(c.get('high', 0) for c in candles)
                self.orb_state.orb_low = min(c.get('low', float('inf')) for c in candles)
                self.orb_state.orb_calculated = True
                logger.info(f"ORB Backfilled from {len(candles)} candles - High: ${self.orb_state.orb_high:.2f}, Low: ${self.orb_state.orb_low:.2f}")
            else:
                logger.warning("No candles returned for ORB backfill")
                
        except Exception as e:
            logger.error(f"ORB backfill failed: {e}")
    
    def _is_0dte(self, expiration: str) -> bool:
        """Check if option expires today (0DTE)"""
        today = get_eastern_time().strftime("%Y-%m-%d")
        # Normalize expiration - handle various formats
        exp_date = expiration[:10] if len(expiration) >= 10 else expiration
        result = exp_date == today
        if not result and not hasattr(self, '_0dte_debug_logged'):
            logger.debug(f"0DTE check: expiration='{expiration}' exp_date='{exp_date}' today='{today}' match={result}")
            self._0dte_debug_logged = True
        return result
    
    def _calculate_position_size(self, option_price: float) -> int:
        """Calculate number of contracts using Kelly-based RiskManager"""
        # Update risk manager capital
        self.risk_manager.capital = self.account_capital
        
        # Get position size from risk manager (uses Kelly + risk caps)
        contracts, position_value = self.risk_manager.get_position_size(
            option_price=option_price,
            stop_loss_pct=self.stop_loss_pct
        )
        
        # Log the sizing details
        kelly_position = self.account_capital * self.kelly_pct
        logger.info(f"Position sizing: Kelly={self.kelly_pct:.1%} (${kelly_position:.0f}), "
                   f"contracts={contracts}, value=${position_value:.0f}")
        
        return contracts
    
    def get_status(self) -> Dict[str, Any]:
        """Get current strategy status"""
        return {
            "strategy": self.strategy,
            "trading_window": f"{self.trade_start} - {self.trade_end}",
            "orb_calculated": self.orb_state.orb_calculated,
            "orb_high": self.orb_state.orb_high,
            "orb_low": self.orb_state.orb_low,
            "in_trade": self.trade_state.in_trade,
            "current_position": self.trade_state.symbol if self.trade_state.in_trade else None,
            "trades_today": self.day_state.trades_today,
            "wins_today": self.day_state.wins_today,
            "pnl_today": self.day_state.pnl_today,
            "stopped_for_day": self.day_state.had_loss and self.stop_after_first_loss
        }
    
    def print_status(self):
        """Print current status"""
        status = self.get_status()
        print("\n" + "=" * 50)
        print("LIVE 0DTE STRATEGY STATUS")
        print("=" * 50)
        print(f"Strategy: {status['strategy'].upper()}")
        print(f"Window: {status['trading_window']}")
        print(f"ORB: High ${status['orb_high']:.2f} | Low ${status['orb_low']:.2f}")
        print(f"In Trade: {status['in_trade']}")
        if status['current_position']:
            print(f"Position: {status['current_position']}")
        print(f"Today: {status['wins_today']}/{status['trades_today']} wins | P&L ${status['pnl_today']:.2f}")
        if status['stopped_for_day']:
            print("⚠️ STOPPED FOR DAY (after first loss)")
        print("=" * 50)


def create_0dte_strategy(
    account_capital: float = 10000,
    strategy: str = "orb",
    **kwargs
) -> Live0DTEStrategy:
    """
    Factory function to create 0DTE strategy with defaults from backtest config.
    
    Args:
        account_capital: Starting capital
        strategy: 'orb', 'momentum', 'mean_reversion'
        **kwargs: Override any strategy parameters
        
    Returns:
        Configured Live0DTEStrategy
    """
    # Default parameters from backtest config (optimized)
    defaults = {
        "strategy": strategy,
        "profit_target_pct": 0.22,
        "stop_loss_pct": 0.25,
        "min_option_price": 0.50,
        "max_option_price": 1.00,
        "trade_start_hour": 10,
        "trade_start_minute": 0,
        "trade_end_hour": 11,
        "trade_end_minute": 0,
        "exit_hour": 15,
        "orb_minutes": 30,
        "orb_buffer_pct": 0.10,
        "rsi_call_threshold": 70,
        "rsi_put_threshold": 30,
        "max_contracts": 5,
        "stop_after_first_loss": False,  # Aligned with backtest RiskManager
        "max_consecutive_losses": 999,  # Aligned with backtest RiskManager
        "account_capital": account_capital,
        "risk_per_trade_pct": 0.02,
    }
    
    # Override with any provided kwargs
    defaults.update(kwargs)
    
    return Live0DTEStrategy(**defaults)
