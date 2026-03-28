"""
Live 0DTE SPY Options Strategy

Phase 8 Momentum strategy (optimized Feb 2026).

Backtest Results (Jan-Feb 2026 OOS):
- 86 trades
- 75.6% win rate
- +187.3% return ($10K -> $28.7K)
- 5.8% max drawdown
- Sharpe 10.60

Strategy Rules:
- Window: 10:00 - 11:00 AM ET
- Options: $0.50 - $2.00
- Signal: RSI Momentum (RSI > 70 -> CALL, RSI < 30 -> PUT)
- Profit Target: +50%
- Stop Loss: -35%
- Max Hold: 80 min (16 bars)
- Risk: Stop after first loss (SFL) + 0.8% daily loss limit + CL=3
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
from core.risk_manager import RiskManager, RiskConfig
from core.regime_classifier import classify_regime_incremental, _compute_day_stats
from config import defaults as cfg

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
    # Smart exit tracking
    option_price_history: List[float] = field(default_factory=list)
    max_pnl_pct: float = 0.0


@dataclass 
class DayState:
    """Track daily trading state"""
    date: str = ""
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    pnl_today: float = 0.0
    

class Live0DTEStrategy(OptionStrategy):
    """
    Live 0DTE SPY Options Strategy using ORB signals.
    
    Based on backtested strategy with 91.2% win rate.
    """
    
    def __init__(
        self,
        # Strategy parameters (defaults from config/strategy.json)
        strategy: str = None,
        profit_target_pct: float = None,
        stop_loss_pct: float = None,
        min_option_price: float = None,
        max_option_price: float = None,
        # Trading window (ET times)
        trade_start_hour: int = None,
        trade_start_minute: int = None,
        trade_end_hour: int = None,
        trade_end_minute: int = None,
        exit_hour: int = None,
        # Max hold time
        max_hold_minutes: int = None,
        # ORB parameters
        orb_minutes: int = None,
        orb_buffer_pct: float = None,
        # RSI parameters
        rsi_call_threshold: float = None,
        rsi_put_threshold: float = None,
        # Asymmetric CALL/PUT exits
        call_profit_target_pct: float = None,
        put_profit_target_pct: float = None,
        call_stop_loss_pct: float = None,
        put_stop_loss_pct: float = None,
        call_max_hold_minutes: int = None,
        put_max_hold_minutes: int = None,
        # Regime detection
        use_regime_detection: bool = None,
        regime_lookback_days: int = None,
        regime_vol_percentile: float = None,
        regime_trend_percentile: float = None,
        regime_size_reduction: float = None,
        regime_skip_first_bar: bool = None,
        regime_rsi_buffer: int = None,
        regime_tighter_stop_pct: float = None,
        # Risk management
        max_contracts: int = None,
        max_daily_losses: int = None,
        max_consecutive_losses: int = None,
        max_daily_loss_pct: float = None,
        # Capital
        account_capital: float = None,
        risk_per_trade_pct: float = None,
        # Optional: Broker client for historical data backfill
        broker_client = None,
    ):
        super().__init__("Live0DTE", ["SPY"])
        
        # Broker client for historical data
        self._broker_client = broker_client
        
        # Resolve all defaults from config/strategy.json
        self.strategy = strategy or cfg.get_trade_config().get('strategy', 'momentum')
        self.profit_target_pct = profit_target_pct if profit_target_pct is not None else cfg.profit_target_pct()
        self.stop_loss_pct = stop_loss_pct if stop_loss_pct is not None else cfg.stop_loss_pct()
        self.min_option_price = min_option_price if min_option_price is not None else cfg.min_option_price()
        self.max_option_price = max_option_price if max_option_price is not None else cfg.max_option_price()
        
        # Trading window
        self.trade_start = dt_time(
            trade_start_hour if trade_start_hour is not None else cfg.trade_start_hour(),
            trade_start_minute if trade_start_minute is not None else cfg.trade_start_minute()
        )
        self.trade_end = dt_time(
            trade_end_hour if trade_end_hour is not None else cfg.trade_end_hour(),
            trade_end_minute if trade_end_minute is not None else cfg.trade_end_minute()
        )
        self.exit_time = dt_time(exit_hour if exit_hour is not None else cfg.exit_hour(), 0)
        self.max_hold_minutes = max_hold_minutes if max_hold_minutes is not None else cfg.max_hold_minutes()
        
        # ORB params
        self.orb_minutes = orb_minutes if orb_minutes is not None else cfg.orb_minutes()
        self.orb_buffer_pct = orb_buffer_pct if orb_buffer_pct is not None else cfg.orb_buffer_pct()
        
        # RSI params
        self.rsi_call_threshold = rsi_call_threshold if rsi_call_threshold is not None else cfg.rsi_call_threshold()
        self.rsi_put_threshold = rsi_put_threshold if rsi_put_threshold is not None else cfg.rsi_put_threshold()
        
        # Asymmetric CALL/PUT exits (fall back to generic PT/SL if not set)
        tc = cfg.get_trade_config()
        self.call_profit_target_pct = call_profit_target_pct if call_profit_target_pct is not None else tc.get('call_profit_target_pct', self.profit_target_pct)
        self.put_profit_target_pct = put_profit_target_pct if put_profit_target_pct is not None else tc.get('put_profit_target_pct', self.profit_target_pct)
        self.call_stop_loss_pct = call_stop_loss_pct if call_stop_loss_pct is not None else tc.get('call_stop_loss_pct', self.stop_loss_pct)
        self.put_stop_loss_pct = put_stop_loss_pct if put_stop_loss_pct is not None else tc.get('put_stop_loss_pct', self.stop_loss_pct)
        # Hold time: bars * 5 min = minutes (backtest uses 5-min bars)
        _call_hold_bars = tc.get('call_max_hold_bars')
        _put_hold_bars = tc.get('put_max_hold_bars')
        self.call_max_hold_minutes = call_max_hold_minutes if call_max_hold_minutes is not None else (_call_hold_bars * 5 if _call_hold_bars else self.max_hold_minutes)
        self.put_max_hold_minutes = put_max_hold_minutes if put_max_hold_minutes is not None else (_put_hold_bars * 5 if _put_hold_bars else self.max_hold_minutes)
        
        # Regime detection params — classification thresholds
        self._use_regime_detection = use_regime_detection if use_regime_detection is not None else tc.get('use_regime_detection', False)
        self._regime_lookback_days = regime_lookback_days if regime_lookback_days is not None else tc.get('regime_lookback_days', 5)
        self._regime_vol_percentile = regime_vol_percentile if regime_vol_percentile is not None else tc.get('regime_vol_percentile', 0.30)
        self._regime_trend_percentile = regime_trend_percentile if regime_trend_percentile is not None else tc.get('regime_trend_percentile', 0.25)
        self._regime_up_day_pct = tc.get('regime_up_day_pct', 0.70)
        self._regime_dn_day_pct = tc.get('regime_dn_day_pct', 0.70)
        self._regime_momentum_threshold = tc.get('regime_momentum_threshold', 0.012)
        self._regime_high_vol_percentile = tc.get('regime_high_vol_percentile', 0.75)
        self._regime_adx_trend_threshold = tc.get('regime_adx_trend_threshold', 25.0)

        # Per-regime adjustments — STEADY_UP
        self._steady_up_size_reduction = tc.get('steady_up_size_reduction', 0.30)
        self._steady_up_call_pt_override = tc.get('steady_up_call_pt_override', None)
        self._steady_up_skip_puts = tc.get('steady_up_skip_puts', True)
        self._steady_up_rsi_buffer = tc.get('steady_up_rsi_buffer', 5)

        # Per-regime adjustments — STEADY_DN
        self._steady_dn_size_reduction = tc.get('steady_dn_size_reduction', 0.30)
        self._steady_dn_put_pt_override = tc.get('steady_dn_put_pt_override', None)
        self._steady_dn_skip_calls = tc.get('steady_dn_skip_calls', True)
        self._steady_dn_rsi_buffer = tc.get('steady_dn_rsi_buffer', 5)

        # Per-regime adjustments — CHOPPY
        self._choppy_size_reduction = tc.get('choppy_size_reduction', 0.50)
        self._choppy_skip_first_bar = tc.get('choppy_skip_first_bar', True)
        self._choppy_rsi_buffer = tc.get('choppy_rsi_buffer', 5)
        self._choppy_tighter_stop_pct = tc.get('choppy_tighter_stop_pct', None)

        # Per-regime adjustments — VOLATILE
        self._volatile_size_reduction = tc.get('volatile_size_reduction', 0.0)
        self._volatile_stop_buffer_pct = tc.get('volatile_stop_buffer_pct', 0.10)
        self._volatile_pt_buffer_pct = tc.get('volatile_pt_buffer_pct', 0.10)

        # Per-regime adjustments — TRENDING
        self._trending_skip_counter = tc.get('trending_skip_counter', True)
        self._trending_hold_buffer = tc.get('trending_hold_buffer', 4)

        # Backward-compat legacy params
        self._regime_size_reduction = regime_size_reduction if regime_size_reduction is not None else tc.get('regime_size_reduction', 0.50)
        self._regime_skip_first_bar = regime_skip_first_bar if regime_skip_first_bar is not None else tc.get('regime_skip_first_bar', True)
        self._regime_rsi_buffer = regime_rsi_buffer if regime_rsi_buffer is not None else tc.get('regime_rsi_buffer', 5)
        self._regime_tighter_stop_pct = regime_tighter_stop_pct if regime_tighter_stop_pct is not None else tc.get('regime_tighter_stop_pct', None)

        # Regime state: full 6-regime detection
        self._regime_type = 'NORMAL'
        self._regime_direction = 0
        self._regime_is_choppy = False  # backward compat
        self._regime_daily_vols = []
        self._regime_daily_trends = []
        self._regime_day_stats = []  # accumulated day stats for incremental classifier
        
        # Risk management
        self.max_contracts = max_contracts if max_contracts is not None else cfg.max_contracts()
        self.max_daily_losses = max_daily_losses if max_daily_losses is not None else cfg.max_daily_losses()
        self.max_consecutive_losses = max_consecutive_losses if max_consecutive_losses is not None else cfg.max_consecutive_losses()
        self.max_daily_loss_pct = max_daily_loss_pct if max_daily_loss_pct is not None else cfg.max_daily_loss_pct()
        self.account_capital = account_capital if account_capital is not None else cfg.initial_capital()
        self.risk_per_trade_pct = risk_per_trade_pct if risk_per_trade_pct is not None else cfg.max_risk_per_trade_pct()
        
        # Limit order pricing: mid + offset (cents)
        self._limit_offset_cents = cfg.limit_offset_cents()
        
        # Initialize RiskManager with risk config from strategy.json
        self.risk_config = RiskConfig(
            kelly_fraction=cfg.kelly_fraction(),
            min_kelly_pct=cfg.max_risk_per_trade_pct(),
            max_kelly_pct=cfg.kelly_fraction(),
            max_risk_per_trade_pct=cfg.max_risk_per_trade_pct(),
            max_position_pct=cfg.max_position_pct(),
            max_position_value=cfg.max_position_value(),
            max_contracts=self.max_contracts,
            max_daily_losses=self.max_daily_losses,
            max_consecutive_losses=self.max_consecutive_losses,
            max_daily_loss_pct=self.max_daily_loss_pct,
            consec_loss_reduction=cfg.consec_loss_reduction(),
            wins_to_reset_streak=cfg.wins_to_reset_streak(),
        )
        self.risk_manager = RiskManager(account_capital, self.risk_config)
        
        # Use kelly_pct from config (set via backtest sensitivity analysis)
        self.kelly_pct = cfg.kelly_pct()
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
        self._pending_expiry_seconds = 180  # Pending signals expire after 180s (IBKR option loading takes ~75s)
        
        # Price history for indicators (5-minute bars, matching backtest)
        self._price_history: List[Dict] = []
        self._current_bar: Optional[Dict] = None  # Accumulating 5-min bar
        self._bar_interval_minutes = 5  # Match backtest 5-min bars
        self._rsi_period = 14
        
        # Adaptive PUT filter: throttle PUTs based on recent performance
        tc = cfg.get_trade_config()
        self._put_adaptive = tc.get('put_adaptive_filter', True)
        self._put_streak_thr = tc.get('put_loss_streak_threshold', 2)
        self._put_cooldown_n = tc.get('put_adaptive_cooldown', 3)
        self._put_consec_losses: int = 0
        self._put_cooldown_remaining: int = 0
        
        # Adaptive CALL filter: throttle CALLs based on recent performance
        self._call_adaptive = tc.get('call_adaptive_filter', False)
        self._call_streak_thr = tc.get('call_loss_streak_threshold', 2)
        self._call_cooldown_n = tc.get('call_adaptive_cooldown', 3)
        self._call_consec_losses: int = 0
        self._call_cooldown_remaining: int = 0
        
        # Smart exit: configurable for live trading
        # Provides real-time assessment beyond fixed PT/SL/TIME
        self._smart_exit_enabled = cfg.get_trade_config().get('use_smart_exit', False)
        
        logger.info(f"Live0DTE Strategy initialized:")
        logger.info(f"  Strategy: {self.strategy}")
        logger.info(f"  Window: {self.trade_start} - {self.trade_end}")
        logger.info(f"  Options: ${self.min_option_price:.2f} - ${self.max_option_price:.2f}")
        logger.info(f"  CALL: PT={self.call_profit_target_pct:.0%} | SL={self.call_stop_loss_pct:.0%} | Hold={self.call_max_hold_minutes}min")
        logger.info(f"  PUT:  PT={self.put_profit_target_pct:.0%} | SL={self.put_stop_loss_pct:.0%} | Hold={self.put_max_hold_minutes}min")
        logger.info(f"  RSI: CALL>{self.rsi_call_threshold} | PUT<{self.rsi_put_threshold}")
        logger.info(f"  MDL: {self.max_daily_losses} | CL: {self.max_consecutive_losses} | DLL: {self.max_daily_loss_pct:.1%}")
        logger.info(f"  Capital: ${self.account_capital:,.2f}")
        if self._use_regime_detection:
            logger.info(f"  Regime Detection: ON (lookback={self._regime_lookback_days}, vol_pctl={self._regime_vol_percentile}, "
                        f"trend_pctl={self._regime_trend_percentile}, rsi_buf={self._regime_rsi_buffer}, "
                        f"size_red={self._regime_size_reduction:.0%}, skip_1st={self._regime_skip_first_bar})")
        else:
            logger.info(f"  Regime Detection: OFF")
        
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
        # Always set capital from CLI arg — resets state if capital changed
        self.persistence.set_initial_capital(self.account_capital)
    
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
        logger.info(f"CALL: PT={self.call_profit_target_pct:.0%} SL={self.call_stop_loss_pct:.0%} Hold={self.call_max_hold_minutes}min")
        logger.info(f"PUT:  PT={self.put_profit_target_pct:.0%} SL={self.put_stop_loss_pct:.0%} Hold={self.put_max_hold_minutes}min")
        logger.info(f"RSI: CALL>{self.rsi_call_threshold} PUT<{self.rsi_put_threshold}")
        logger.info(f"Regime Detection: {'ON' if self._use_regime_detection else 'OFF'}")
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
        
        # Check if should stop trading (max daily losses reached)
        if self.day_state.losses_today >= self.max_daily_losses:
            return None
        
        # ===== REGIME DETECTION: Skip first bar in choppy regime =====
        if self._use_regime_detection and self._regime_type == 'CHOPPY':
            if self._choppy_skip_first_bar:
                trade_start_minutes = self.trade_start.hour * 60 + self.trade_start.minute
                current_minutes = current_time.hour * 60 + current_time.minute
                if current_minutes <= trade_start_minutes + 5:
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
        
        # ===== REGIME DIRECTIONAL FILTER: skip counter-regime entries =====
        if signal_direction and self._use_regime_detection:
            if signal_direction == 'PUT' and self._regime_type == 'STEADY_UP' and self._steady_up_skip_puts:
                logger.info(f"REGIME SKIP: PUT blocked in STEADY_UP regime")
                return None
            if signal_direction == 'CALL' and self._regime_type == 'STEADY_DN' and self._steady_dn_skip_calls:
                logger.info(f"REGIME SKIP: CALL blocked in STEADY_DN regime")
                return None
            if self._regime_type == 'TRENDING' and self._trending_skip_counter:
                if self._regime_direction > 0 and signal_direction == 'PUT':
                    logger.info(f"REGIME SKIP: PUT blocked in TRENDING-UP regime")
                    return None
                elif self._regime_direction < 0 and signal_direction == 'CALL':
                    logger.info(f"REGIME SKIP: CALL blocked in TRENDING-DN regime")
                    return None
        
        # PUT ADAPTIVE FILTER: Skip PUTs during cooldown after loss streak
        if signal_direction == "PUT" and self._put_adaptive:
            if self._put_cooldown_remaining > 0:
                self._put_cooldown_remaining -= 1
                logger.info(f"PUT ADAPTIVE: cooldown skip ({self._put_cooldown_remaining} remaining)")
                return None
        
        # CALL ADAPTIVE FILTER: Skip CALLs during cooldown after loss streak
        if signal_direction == "CALL" and self._call_adaptive:
            if self._call_cooldown_remaining > 0:
                self._call_cooldown_remaining -= 1
                logger.info(f"CALL ADAPTIVE: cooldown skip ({self._call_cooldown_remaining} remaining)")
                return None
        
        # PUT ENTRY FILTER: Skip low-quality PUT signals (only during uptrends)
        if signal_direction == "PUT":
            put_filter_require_uptrend = cfg.get_trade_config().get('put_filter_require_uptrend', True)
            
            # Check if filter should engage: only during uptrends if required
            _put_filter_active = True
            if put_filter_require_uptrend:
                _put_filter_active = self._is_uptrend()
            
            if _put_filter_active:
                put_min_rsi = cfg.get_trade_config().get('put_min_rsi', 0)
                put_skip_days = cfg.get_trade_config().get('put_skip_days', None)
                put_min_entry_minutes = cfg.get_trade_config().get('put_min_entry_minutes', 0)
                
                current_rsi = self._calculate_rsi()
                
                # Skip PUTs with RSI below minimum (very oversold → bounce risk)
                if put_min_rsi > 0 and current_rsi is not None and current_rsi < put_min_rsi:
                    logger.info(f"PUT FILTERED (uptrend): RSI={current_rsi:.1f} < min {put_min_rsi}")
                    return None
                
                # Skip PUTs on specific weekdays (e.g. Monday=0)
                if put_skip_days and now.weekday() in put_skip_days:
                    logger.info(f"PUT FILTERED (uptrend): weekday={now.weekday()} in skip_days={put_skip_days}")
                    return None
                
                # Skip PUTs before minimum entry time
                if put_min_entry_minutes > 0:
                    entry_minutes = current_time.hour * 60 + current_time.minute
                    if entry_minutes < put_min_entry_minutes:
                        logger.info(f"PUT FILTERED (uptrend): time {entry_minutes}min < min {put_min_entry_minutes}min")
                        return None
        
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
        
        # Diagnostic: log pending state on first call per cycle
        if not hasattr(self, '_diag_logged'):
            self._diag_logged = False
        if not self._diag_logged:
            has_pd = hasattr(self, '_pending_direction')
            pd_val = getattr(self, '_pending_direction', 'MISSING')
            logger.info(f"DIAG on_option_quote: has_pending={has_pd} pending_dir={pd_val} in_trade={self.trade_state.in_trade} opt={quote.symbol}")
            self._diag_logged = True
        
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
            # SFL hard gate: block ALL entries once daily loss limit is reached,
            # even if a pending direction was set before the loss was recorded.
            if self.day_state.losses_today >= self.max_daily_losses:
                if hasattr(self, '_pending_direction') and self._pending_direction:
                    logger.warning(
                        f"SFL HARD GATE: blocking pending {self._pending_direction} entry "
                        f"(losses_today={self.day_state.losses_today} >= max={self.max_daily_losses})"
                    )
                    self._pending_direction = None
                    self._pending_direction_time = None
                return None

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
                
                # Mid + offset limit: saves spread vs paying full ask
                mid = (quote.bid + quote.ask) / 2
                offset = self._limit_offset_cents * 0.01
                buy_limit = round(mid + offset, 2)
                
                return self.create_signal(
                    symbol=quote.symbol,
                    action="BUY",
                    quantity=contracts,
                    limit_price=buy_limit,
                    reason=f"0DTE {direction} entry: {self.strategy}",
                    confidence=0.9
                )
        
        return None
    
    def _check_exit(self, quote: OptionQuote, now: datetime) -> Optional[Signal]:
        """Check if should exit current position (includes smart exit assessment)"""
        current_time = now.time()
        current_price = quote.last or ((quote.bid + quote.ask) / 2)
        
        # Update high/low tracking
        self.trade_state.highest_price = max(self.trade_state.highest_price, current_price)
        self.trade_state.lowest_price = min(self.trade_state.lowest_price, current_price)
        
        entry_price = self.trade_state.entry_price
        pnl_pct = (current_price - entry_price) / entry_price
        
        # Direction-specific PT/SL/hold
        direction = self.trade_state.direction
        if direction == 'CALL':
            _pt = self.call_profit_target_pct
            _sl = self.call_stop_loss_pct
            _max_hold = self.call_max_hold_minutes
        else:
            _pt = self.put_profit_target_pct
            _sl = self.put_stop_loss_pct
            _max_hold = self.put_max_hold_minutes
        
        # ===== REGIME ADJUSTMENTS: Per-regime PT/SL/hold overrides =====
        if self._use_regime_detection:
            if self._regime_type == 'CHOPPY' and self._choppy_tighter_stop_pct is not None:
                _sl = self._choppy_tighter_stop_pct

            if self._regime_type == 'STEADY_UP':
                if direction == 'CALL' and self._steady_up_call_pt_override is not None:
                    _pt = self._steady_up_call_pt_override

            if self._regime_type == 'STEADY_DN':
                if direction == 'PUT' and self._steady_dn_put_pt_override is not None:
                    _pt = self._steady_dn_put_pt_override

            if self._regime_type == 'VOLATILE':
                _sl = _sl + self._volatile_stop_buffer_pct
                _pt = _pt + self._volatile_pt_buffer_pct

            # TRENDING: extend max hold time (let winners run)
            if self._regime_type == 'TRENDING' and self._trending_hold_buffer > 0:
                _max_hold += self._trending_hold_buffer * 5  # bars -> minutes
        
        # Track option price history and max unrealized P&L for smart exit
        self.trade_state.option_price_history.append(current_price)
        if pnl_pct > self.trade_state.max_pnl_pct:
            self.trade_state.max_pnl_pct = pnl_pct
        
        exit_reason = None
        
        # ===== PROFIT TARGET (direction-specific) =====
        if pnl_pct >= _pt:
            exit_reason = f"PROFIT TARGET: {pnl_pct:.1%} (target {_pt:.0%})"
        
        # ===== STOP LOSS (direction-specific) =====
        elif pnl_pct <= -_sl:
            exit_reason = f"STOP LOSS: {pnl_pct:.1%} (limit -{_sl:.0%})"
            self.day_state.had_loss = True
        
        # ===== SMART EXIT ASSESSMENT =====
        elif self._smart_exit_enabled:
            prices = self.trade_state.option_price_history
            ticks_held = len(prices)
            
            # 1. ADVERSE VELOCITY — single-tick large drop (25%+ = genuine crash)
            if len(prices) >= 2:
                tick_return = (prices[-1] - prices[-2]) / prices[-2]
                if tick_return < -0.25:
                    exit_reason = f"SMART ADVERSE: {pnl_pct:.1%} (bar drop {tick_return:.1%})"
                    # Smart exits never cascade to post-loss correction
            
            # 2. PROFIT PROTECTION — had +20%, now giving it back
            #    Skip if near profit target (let it ride to PT)
            if not exit_reason and self.trade_state.max_pnl_pct >= 0.20:
                _near_pt = pnl_pct >= (_pt * 0.50)
                if not _near_pt:
                    floor_pct = self.trade_state.max_pnl_pct * 0.50
                    if pnl_pct < floor_pct:
                        # Require 2 consecutive ticks below floor
                        self.trade_state._protect_below_count = getattr(self.trade_state, '_protect_below_count', 0) + 1
                        if self.trade_state._protect_below_count >= 2:
                            exit_reason = f"SMART PROTECT: {pnl_pct:.1%} (peak was {self.trade_state.max_pnl_pct:.1%})"
                    else:
                        self.trade_state._protect_below_count = 0
            
            # 3. STALL DETECTION — no movement over recent history (theta eating)
            #    Only when underwater
            if not exit_reason and len(prices) >= 4 and pnl_pct < 0:
                recent = prices[-4:]
                range_pct = (max(recent) - min(recent)) / recent[-1]
                if range_pct < 0.05:
                    exit_reason = f"SMART STALL: {pnl_pct:.1%} (range {range_pct:.1%} over 4 ticks)"
                    # Smart exits never cascade to post-loss correction
            
            # 4. UNDERLYING RSI REVERSAL — check if our thesis broke
            #    Require minimum 3 ticks held; use signal thresholds + 5pt buffer
            if not exit_reason and ticks_held >= 3 and pnl_pct < 0 and len(self._price_history) >= 14:
                current_rsi = self._compute_rsi()
                if current_rsi is not None:
                    if self.trade_state.direction == 'CALL' and current_rsi < (self.rsi_put_threshold - 5):
                        exit_reason = f"SMART REVERSAL: RSI={current_rsi:.0f} (CALL thesis broken, RSI<{self.rsi_put_threshold - 5:.0f})"
                        # Smart exits never cascade to post-loss correction
                    elif self.trade_state.direction == 'PUT' and current_rsi > (self.rsi_call_threshold + 5):
                        exit_reason = f"SMART REVERSAL: RSI={current_rsi:.0f} (PUT thesis broken, RSI>{self.rsi_call_threshold + 5:.0f})"
                        # Smart exits never cascade to post-loss correction
        
        # ===== MAX HOLD TIME EXIT (direction-specific) =====
        # NOTE: This is a standalone `if` (not elif) so it fires regardless
        # of whether smart exit was evaluated but produced no exit_reason.
        if not exit_reason and self.trade_state.entry_time:
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
            if hold_minutes >= _max_hold:
                exit_reason = f"MAX HOLD TIME ({int(hold_minutes)}min >= {int(_max_hold)}min): {pnl_pct:.1%}"
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
            
            # Update adaptive PUT filter state
            if self.trade_state.direction == 'PUT' and self._put_adaptive:
                if is_win:
                    self._put_consec_losses = 0
                else:
                    self._put_consec_losses += 1
                    if self._put_consec_losses >= self._put_streak_thr:
                        self._put_cooldown_remaining = self._put_cooldown_n
                        self._put_consec_losses = 0
                        logger.info(f"PUT ADAPTIVE: cooldown triggered, skipping next {self._put_cooldown_n} PUT signals")
            
            # Update adaptive CALL filter state
            if self.trade_state.direction == 'CALL' and self._call_adaptive:
                if is_win:
                    self._call_consec_losses = 0
                else:
                    self._call_consec_losses += 1
                    if self._call_consec_losses >= self._call_streak_thr:
                        self._call_cooldown_remaining = self._call_cooldown_n
                        self._call_consec_losses = 0
                        logger.info(f"CALL ADAPTIVE: cooldown triggered, skipping next {self._call_cooldown_n} CALL signals")
            
            # Reset trade state
            symbol = self.trade_state.symbol
            quantity = self.trade_state.quantity
            self.trade_state = TradeState()
            self._last_exit_time = now  # Set cooldown timer
            
            # Mid - offset limit: gets better fill than full bid
            mid = (quote.bid + quote.ask) / 2
            offset = self._limit_offset_cents * 0.01
            sell_limit = round(mid - offset, 2)
            
            return self.create_signal(
                symbol=symbol,
                action="SELL",
                quantity=quantity,
                limit_price=sell_limit,
                reason=exit_reason,
                confidence=1.0
            )
        
        return None
    
    def on_trade_cancelled(self, trade_id: int, symbol: str) -> None:
        """Called by engine when a BUY entry order is cancelled/rejected.
        
        Resets trade_state so the strategy does NOT send SELL signals
        for a position that was never filled. Without this, a cancelled
        BUY followed by a SELL creates a naked short on IBKR.
        """
        if self.trade_state.in_trade and self.trade_state.symbol == symbol:
            logger.warning(
                f"Trade cancelled: resetting trade_state for {symbol} "
                f"(was {self.trade_state.direction} {self.trade_state.quantity}x)"
            )
            self.trade_state = TradeState()
    
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
        
        # Per-regime RSI buffer (matches backtest engine logic)
        call_thr = self.rsi_call_threshold
        put_thr = self.rsi_put_threshold
        if self._use_regime_detection:
            _rsi_buf = 0
            if self._regime_type == 'CHOPPY' and self._choppy_rsi_buffer > 0:
                _rsi_buf = self._choppy_rsi_buffer
            elif self._regime_type == 'STEADY_UP' and self._steady_up_rsi_buffer > 0:
                _rsi_buf = self._steady_up_rsi_buffer
            elif self._regime_type == 'STEADY_DN' and self._steady_dn_rsi_buffer > 0:
                _rsi_buf = self._steady_dn_rsi_buffer
            if _rsi_buf > 0:
                call_thr += _rsi_buf
                put_thr -= _rsi_buf
        
        if trend_follow:
            # Momentum: trade with trend
            if rsi > call_thr:
                self._pending_direction = "CALL"
                return "CALL"
            elif rsi < put_thr:
                self._pending_direction = "PUT"
                return "PUT"
        else:
            # Mean reversion: fade extremes
            if rsi < put_thr:
                self._pending_direction = "CALL"
                return "CALL"
            elif rsi > call_thr:
                self._pending_direction = "PUT"
                return "PUT"
        
        return None
    
    def _calculate_rsi(self) -> Optional[float]:
        """Calculate RSI from 5-minute bars (matches backtest granularity).
        
        Includes the current incomplete bar so RSI updates within bars,
        but each bar covers 5 minutes of price action instead of one tick.
        """
        # Include current incomplete bar for live updates
        bars = list(self._price_history)
        if self._current_bar:
            bars.append(self._current_bar)
        
        if len(bars) < self._rsi_period + 1:
            return None
        
        prices = [bar['close'] for bar in bars[-self._rsi_period-1:]]
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
    
    def _compute_rsi(self) -> Optional[float]:
        """Compute current RSI for smart exit assessment (alias for _calculate_rsi)"""
        return self._calculate_rsi()
    
    def _is_uptrend(self) -> bool:
        """
        Check if SPY is in a sustained uptrend using recent price history.
        Uses the same logic as backtest trend filter: count up/down days
        and cumulative return over a lookback window.
        """
        tc = cfg.get_trade_config()
        lookback = tc.get('trend_lookback_days', 5)
        up_thr = tc.get('trend_up_days_threshold', 4)
        ret_thr = tc.get('trend_return_threshold', 0.015)
        
        if len(self._price_history) < lookback + 1:
            return False
        
        # Get daily closes from recent bars (approximate: use last close per unique date)
        recent = self._price_history[-(lookback * 8):]  # ~8 bars per day in trading window
        daily_closes = {}
        for bar in recent:
            ts = bar.get('timestamp')
            if ts:
                d = ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10]
                daily_closes[d] = bar['close']
        
        dates = sorted(daily_closes.keys())
        if len(dates) < lookback:
            return False
        
        closes = [daily_closes[d] for d in dates[-lookback:]]
        
        up_days = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        cum_return = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
        
        return up_days >= up_thr and cum_return > ret_thr
    
    def _compute_regime(self):
        """
        Compute regime from recent price history using shared classifier.
        Mirrors backtest engine.compute_regime_features() exactly via
        core.regime_classifier.classify_regime_incremental().

        Detects 6 regimes: VOLATILE, STEADY_UP, STEADY_DN, TRENDING, CHOPPY, NORMAL.
        Uses ONLY past data — no look-ahead bias.
        """
        if not self._use_regime_detection:
            self._regime_type = 'NORMAL'
            self._regime_direction = 0
            self._regime_is_choppy = False
            return

        lookback = self._regime_lookback_days

        # Build daily close prices from price history
        daily_bars = {}
        for bar in self._price_history:
            ts = bar.get('timestamp')
            if ts:
                d = ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10]
                if d not in daily_bars:
                    daily_bars[d] = []
                daily_bars[d].append(bar['close'])

        dates = sorted(daily_bars.keys())
        if len(dates) < 2:
            self._regime_type = 'NORMAL'
            self._regime_direction = 0
            self._regime_is_choppy = False
            return

        # Compute day stats for completed days only (exclude today)
        recent_dates = dates[:-1] if len(dates) > 1 else dates
        for d in recent_dates[-lookback:]:
            prices = daily_bars[d]
            day_stat = _compute_day_stats(prices)
            if day_stat is not None:
                day_stat['date'] = d
                # Avoid duplicate entries for the same date
                if not self._regime_day_stats or self._regime_day_stats[-1].get('date') != d:
                    self._regime_day_stats.append(day_stat)

        if len(self._regime_day_stats) < 2:
            self._regime_type = 'NORMAL'
            self._regime_direction = 0
            self._regime_is_choppy = False
            return

        # Build config dict matching strategy.json params
        regime_config = {
            'lookback': lookback,
            'vol_percentile': self._regime_vol_percentile,
            'trend_percentile': self._regime_trend_percentile,
            'up_day_pct': self._regime_up_day_pct,
            'dn_day_pct': self._regime_dn_day_pct,
            'momentum_threshold': self._regime_momentum_threshold,
            'high_vol_percentile': self._regime_high_vol_percentile,
            'adx_trend_threshold': self._regime_adx_trend_threshold,
        }

        result = classify_regime_incremental(
            self._regime_day_stats,
            self._regime_daily_vols,
            self._regime_daily_trends,
            regime_config,
        )

        old_type = self._regime_type
        self._regime_type = result['regime_type']
        self._regime_direction = result['direction']
        self._regime_is_choppy = result['is_choppy']

        if self._regime_type != old_type:
            logger.info(f"REGIME CHANGE: {old_type} -> {self._regime_type} "
                       f"(dir={self._regime_direction}, vol_pctl={result['vol_pctl']:.2f}, "
                       f"trend_pctl={result['trend_pctl']:.2f})")
    
    def _update_orb(self, high: float, low: float):
        """Update Opening Range high/low"""
        if self.orb_state.orb_high == 0:
            self.orb_state.orb_high = high
            self.orb_state.orb_low = low
        else:
            self.orb_state.orb_high = max(self.orb_state.orb_high, high)
            self.orb_state.orb_low = min(self.orb_state.orb_low, low)
    
    def _get_bar_start(self, timestamp: datetime) -> datetime:
        """Get the start of the 5-minute bar containing this timestamp"""
        bar_minute = (timestamp.minute // self._bar_interval_minutes) * self._bar_interval_minutes
        return timestamp.replace(minute=bar_minute, second=0, microsecond=0)

    def _update_price_history(self, timestamp: datetime, close: float, high: float, low: float, volume: int):
        """Update price history resampled to 5-minute bars (matches backtest RSI).
        
        Raw quote ticks (~5-8s) are aggregated into 5-min OHLCV bars.
        Without this, RSI-14 covers ~112 seconds instead of ~70 minutes,
        making signals hyper-sensitive to micro-noise.
        """
        bar_start = self._get_bar_start(timestamp)

        if self._current_bar is None:
            # Start first bar
            self._current_bar = {
                'timestamp': bar_start,
                'open': close, 'high': high, 'low': low,
                'close': close, 'volume': volume,
            }
        elif bar_start > self._current_bar['timestamp']:
            # New 5-min period — close previous bar and start new one
            self._price_history.append(self._current_bar)
            logger.debug(f"5-min bar closed: {self._current_bar['timestamp'].strftime('%H:%M')} "
                         f"O={self._current_bar['open']:.2f} H={self._current_bar['high']:.2f} "
                         f"L={self._current_bar['low']:.2f} C={self._current_bar['close']:.2f}")
            self._current_bar = {
                'timestamp': bar_start,
                'open': close, 'high': high, 'low': low,
                'close': close, 'volume': volume,
            }
        else:
            # Same bar — update OHLCV
            self._current_bar['close'] = close
            self._current_bar['high'] = max(self._current_bar['high'], high)
            self._current_bar['low'] = min(self._current_bar['low'], low)
            self._current_bar['volume'] += volume

        # Keep last 100 completed bars
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
        self._current_bar = None
        self._pending_direction = None
        
        # Compute regime for new day based on accumulated history
        self._compute_regime()
        if self._use_regime_detection:
            logger.info(f"Regime: {self._regime_type} (dir={self._regime_direction})")
        
        logger.info(f"Daily state reset for {date}")
        
        # Backfill price history for RSI when starting mid-day
        now = get_eastern_time()
        if now.time() >= dt_time(9, 35):
            self._backfill_price_history()
        
        # Backfill ORB if starting after 10:00 AM ET
        orb_end_time = dt_time(10, 0)
        if now.time() >= orb_end_time:
            self._backfill_orb()
    
    def _in_trading_window(self, current_time: dt_time) -> bool:
        """Check if current time is within trading window"""
        return self.trade_start <= current_time <= self.trade_end
    
    def _backfill_price_history(self):
        """Backfill 5-min price history for RSI when starting mid-day.
        
        Fetches FiveMinute candles from market open to now so that
        RSI-14 has proper 70-minute context on first signal.
        """
        if not self._broker_client:
            logger.warning("No broker client - cannot backfill price history for RSI")
            return
        
        try:
            symbols = self._broker_client.search_symbols("SPY")
            spy_symbol = next((s for s in symbols if s.get('symbol') == 'SPY'), None)
            if not spy_symbol:
                logger.error("Could not find SPY symbol for price backfill")
                return
            
            symbol_id = spy_symbol['symbolId']
            now = get_eastern_time()
            today = now.date()
            from datetime import datetime as dt
            
            backfill_start = dt.combine(today, dt_time(9, 30))
            backfill_end = now
            
            try:
                from zoneinfo import ZoneInfo
                et_tz = ZoneInfo("America/New_York")
                backfill_start = backfill_start.replace(tzinfo=et_tz)
                if backfill_end.tzinfo is None:
                    backfill_end = backfill_end.replace(tzinfo=et_tz)
            except Exception:
                pass
            
            candles = self._broker_client.get_candles(
                symbol_id=symbol_id,
                start_time=backfill_start,
                end_time=backfill_end,
                interval="FiveMinutes"
            )
            
            if candles:
                for c in candles:
                    ts = c.get('start', c.get('end', backfill_start))
                    if isinstance(ts, str):
                        try:
                            ts = dt.fromisoformat(ts.replace('Z', '+00:00'))
                        except Exception:
                            ts = backfill_start
                    bar = {
                        'timestamp': ts,
                        'open': c.get('open', 0),
                        'high': c.get('high', 0),
                        'low': c.get('low', 0),
                        'close': c.get('close', 0),
                        'volume': c.get('volume', 0),
                    }
                    self._price_history.append(bar)
                logger.info(f"RSI backfill: {len(candles)} five-min bars loaded "
                            f"({backfill_start.strftime('%H:%M')}-{backfill_end.strftime('%H:%M')})")
            else:
                logger.warning("No candles returned for RSI price history backfill")
        except Exception as e:
            logger.error(f"RSI price history backfill failed: {e}")

    def _backfill_orb(self):
        """Backfill ORB from historical candles when starting after 10:00 AM"""
        if not self._broker_client:
            logger.warning("No broker client - cannot backfill ORB. Using day's high/low fallback.")
            return
        
        try:
            # Get SPY symbol ID
            symbols = self._broker_client.search_symbols("SPY")
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
            
            candles = self._broker_client.get_candles(
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
        # Normalize expiration - handle YYYYMMDD and YYYY-MM-DD formats
        exp = expiration.replace("-", "").strip()[:8]  # "20260303"
        today_nodash = today.replace("-", "")           # "20260303"
        result = exp == today_nodash
        if not result and not hasattr(self, '_0dte_debug_logged'):
            logger.info(f"0DTE check FAIL: expiration='{expiration}' normalized='{exp}' today='{today_nodash}'")
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
        
        # Per-regime position size reduction (matches backtest engine logic)
        if self._use_regime_detection:
            _regime_size_red = 0.0
            if self._regime_type == 'CHOPPY':
                _regime_size_red = self._choppy_size_reduction
            elif self._regime_type == 'STEADY_UP':
                _regime_size_red = self._steady_up_size_reduction
            elif self._regime_type == 'STEADY_DN':
                _regime_size_red = self._steady_dn_size_reduction
            elif self._regime_type == 'VOLATILE':
                _regime_size_red = self._volatile_size_reduction

            if _regime_size_red > 0:
                reduced = max(1, int(contracts * (1 - _regime_size_red)))
                if reduced < contracts:
                    logger.info(f"REGIME: Reducing position {contracts} -> {reduced} contracts ({self._regime_type} regime, -{_regime_size_red:.0%})")
                    contracts = reduced
        
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
            "stopped_for_day": self.day_state.losses_today >= self.max_daily_losses
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
    account_capital: float = None,
    strategy: str = None,
    **kwargs
) -> Live0DTEStrategy:
    """
    Factory function to create 0DTE strategy.
    
    All defaults come from config/strategy.json via config.defaults.
    
    Args:
        account_capital: Starting capital (default: from config)
        strategy: 'momentum', 'orb', 'mean_reversion' (default: from config)
        **kwargs: Override any strategy parameters
        
    Returns:
        Configured Live0DTEStrategy
    """
    params = {"strategy": strategy} if strategy else {}
    if account_capital is not None:
        params["account_capital"] = account_capital
    params.update(kwargs)
    
    return Live0DTEStrategy(**params)
