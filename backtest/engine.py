"""
0DTE Options Backtest Engine
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import sqlite3
import sys

from config import defaults as cfg
sys.path.insert(0, '.')

from core.signals import (
    compute_features, 
    compute_rolling_volatility,
    get_basic_signal,
    TradingMLModel,
    DayFilterModel,
)
from core.risk_manager import RiskManager, RiskConfig
from core.regime_classifier import classify_regimes, _compute_day_stats
from clients.database import MarketDatabase


@dataclass
class TradeConfig:
    """0DTE trading configuration"""
    # Strategy type: momentum, mean_reversion, bb_breakout, vwap_reversion, orb
    strategy: str = "momentum"
    
    # Entry timing
    trade_start_hour: int = 9
    trade_start_minute: int = 35
    trade_end_hour: int = 15
    trade_end_minute: int = 0
    
    # RSI thresholds (for momentum and mean_reversion strategies)
    rsi_call_threshold: float = 70.0
    rsi_put_threshold: float = 30.0
    
    # Bollinger Band Breakout params
    bb_buffer_pct: float = 0.0       # Buffer beyond BB for signal (0 = touch band)
    
    # VWAP Reversion params  
    vwap_dev_threshold: float = 0.3  # % deviation from VWAP to trigger (0.3%)
    
    # Opening Range Breakout params
    orb_minutes: int = 30            # Minutes to define opening range
    orb_buffer_pct: float = 0.1      # Buffer beyond ORB for signal (10% of range)
    
    # Option selection
    min_option_price: float = 0.50
    max_option_price: float = 5.00
    
    # Exit parameters (defaults when adaptive disabled)
    profit_target_pct: float = 0.25
    stop_loss_pct: float = 0.35
    max_hold_bars: int = 4  # 4 bars = 20 min max hold time
    
    # ============================================================
    # ADAPTIVE EXIT SYSTEM (Profit Taking + Stop Loss)
    # ============================================================
    use_adaptive_exits: bool = False   # Master switch for adaptive exits
    
    # Volatility-based PROFIT TARGETS
    profit_low_vol: float = 0.15       # Profit target when VIX < 10 (calm - cash out quickly)
    profit_mid_vol: float = 0.55       # Profit target when VIX 10-25 (normal)
    profit_high_vol: float = 0.55      # Profit target when VIX > 25 (volatile)
    
    # Volatility-based STOP LOSSES
    stop_low_vol: float = 0.20         # Stop loss when VIX < 10 (calm - tighter)
    stop_mid_vol: float = 0.40         # Stop loss when VIX 10-25 (normal)
    stop_high_vol: float = 0.40        # Stop loss when VIX > 25 (volatile)
    
    # VIX thresholds
    vix_low_threshold: float = 10.0    # VIX below this = low vol
    vix_high_threshold: float = 25.0   # VIX above this = high vol
    
    # Entry price adjustment (cheap options need wider targets)
    cheap_option_threshold: float = 1.00  # Options below $1
    cheap_option_bonus: float = 0.05      # Add 5% to targets for cheap options
    
    # Time-based adjustment (tighter exits later in day)
    use_time_decay_exits: bool = False
    time_decay_factor: float = 0.02    # Tighten by 2% per hour after start
    
    # ============================================================
    # TRAILING STOP SYSTEM
    # ============================================================
    use_trailing_stop: bool = False    # Enable trailing stop
    trail_activation_pct: float = 0.10 # Activate trail after +10% profit
    trail_distance_pct: float = 0.50   # Trail at 50% of max gain
    breakeven_activation: float = 0.08 # Move stop to breakeven after +8%
    
    # ============================================================
    # TIME-DECAY EXIT SYSTEM
    # ============================================================
    use_time_decay_exit: bool = False   # Enable time-based target decay
    time_decay_profit_per_bar: float = 0.03  # Reduce profit target 3% per bar
    min_profit_target: float = 0.10     # Don't go below 10% profit target
    
    # Quick-exit escalation: tighten stop if underwater after bar 1
    use_quick_exit: bool = False        # Enable quick-exit escalation
    underwater_stop_tighten: float = 0.15  # Tighten stop to 15% if underwater after bar 1
    quick_exit_profit_threshold: float = 0.05  # Require +5% profit before moving to breakeven
    breakeven_buffer_pct: float = 0.02  # Lock in +2% profit instead of 0% (covers slippage)
    
    # ML filtering
    ml_confidence_threshold: float = 0.0  # DISABLED - ML hurts performance
    use_ml_filter: bool = False  # Skip ML entirely
    
    # Day filter (ML-based)
    use_ml_day_filter: bool = False  # Disabled - costs more profit than it saves
    day_filter_threshold: float = 0.50  # Probability threshold
    
    # Fallback volatility filter (if not using ML day filter)
    skip_day_filter: bool = True  # Skip all day filtering
    volatility_threshold: float = 0.80  # Percentile of historical vol
    volatility_lookback_days: int = 5

    # ============================================================
    # ASYMMETRIC CALL/PUT EXITS (None = use shared PT/SL)
    # ============================================================
    call_profit_target_pct: float = None
    put_profit_target_pct: float = None
    call_stop_loss_pct: float = None
    put_stop_loss_pct: float = None
    call_max_hold_bars: int = None
    put_max_hold_bars: int = None
    
    # Minimum contracts to enter a trade (skip if position too small)
    min_contracts_per_trade: int = 1
    
    # Maximum contracts per trade (cap tail risk from oversized positions)
    max_contracts_per_trade: int = 0  # 0 = no cap
    
    # ============================================================
    # POST-LOSS STRATEGY: what to do after first daily loss
    # ============================================================
    # Options:
    #   "none"             - no change (original behaviour)
    #   "flip"             - blind flip: invert every subsequent signal
    #   "momentum_confirm" - flip only when 3-bar momentum confirms the reversal
    #   "multi_confirm"    - re-derive direction from momentum+VWAP+trend consensus;
    #                        skip trade if market is ambiguous/choppy
    #   "adaptive"         - context-aware: adapts threshold based on loss exit reason,
    #                        volatility regime, volume confirmation, and cooldown
    post_loss_strategy: str = "none"
    post_loss_momentum_threshold: float = 0.10  # min |momentum_3| to confirm flip (% change)
    
    # Adaptive post-loss parameters (used when post_loss_strategy == "adaptive")
    post_loss_cooldown_bars: int = 2        # min bars to wait after loss before re-entering
    post_loss_stop_factor: float = 0.5      # threshold multiplier for STOP losses (lower = easier flip)
    post_loss_time_factor: float = 1.5      # threshold multiplier for TIME losses (harder to flip)
    post_loss_min_vol_ratio: float = 0.8    # minimum volume ratio to confirm post-loss entry
    
    # ============================================================
    # REGIME DETECTION: Multi-regime classifier
    # ============================================================
    # Detects 5 market regimes from prior-day technicals:
    #   STEADY_UP  — Persistent uptrend + low vol (the May/Aug killer)
    #   STEADY_DN  — Persistent downtrend + low vol
    #   TRENDING   — Strong directional move with high momentum
    #   CHOPPY     — Low vol, no direction, range-bound
    #   VOLATILE   — High vol, big swings (good for momentum)
    #   NORMAL     — None of the above (no adjustments)
    #
    use_regime_detection: bool = False      # Master switch
    regime_lookback_days: int = 5           # Rolling window for regime features

    # --- Classification thresholds (calibrate these) ---
    regime_vol_percentile: float = 0.30     # Bottom N% of intraday vol = "low vol"
    regime_trend_percentile: float = 0.25   # Bottom N% of trend strength = "no trend"
    regime_up_day_pct: float = 0.70         # >= 70% up-days in window = steady uptrend
    regime_dn_day_pct: float = 0.70         # >= 70% down-days in window = steady downtrend
    regime_momentum_threshold: float = 0.012  # |return| over window > 1.2% = directional
    regime_high_vol_percentile: float = 0.75  # Top 25% of intraday vol = "high vol"
    regime_adx_trend_threshold: float = 25.0  # ADX-proxy above this = strong trend

    # --- Per-regime adjustments: STEADY_UP ---
    #   (Grinds higher daily, options decay before hitting PT)
    steady_up_size_reduction: float = 0.30  # Cut size 30%
    steady_up_call_pt_override: float = None  # Lower CALL PT (e.g. 0.30 vs 0.50)
    steady_up_skip_puts: bool = True          # PUTs lose in steady uptrend
    steady_up_rsi_buffer: int = 5             # Require stronger RSI for entry

    # --- Per-regime adjustments: STEADY_DN ---
    steady_dn_size_reduction: float = 0.30
    steady_dn_put_pt_override: float = None   # Lower PUT PT
    steady_dn_skip_calls: bool = True          # CALLs lose in steady downtrend
    steady_dn_rsi_buffer: int = 5

    # --- Per-regime adjustments: CHOPPY ---
    #   (Low vol + no direction = theta decay eats premiums)
    choppy_size_reduction: float = 0.50
    choppy_skip_first_bar: bool = True
    choppy_rsi_buffer: int = 5
    choppy_tighter_stop_pct: float = None     # Override stop loss (None=no change)

    # --- Per-regime adjustments: VOLATILE ---
    #   (Big swings = good for momentum, but widen stops)
    volatile_size_reduction: float = 0.0      # No size cut (vol is good)
    volatile_stop_buffer_pct: float = 0.10    # Widen stop by 10% (let it breathe)
    volatile_pt_buffer_pct: float = 0.10      # Widen PT by 10% (bigger moves)

    # --- Per-regime adjustments: TRENDING ---
    #   (Strong directional — ride the trend, skip counter-trend)
    trending_skip_counter: bool = True        # Skip counter-trend entries
    trending_hold_buffer: int = 4             # Add N bars to max hold (let it run)

    # --- Backward-compat aliases (used if code references old fields) ---
    regime_size_reduction: float = 0.50
    regime_skip_first_bar: bool = True
    regime_rsi_buffer: int = 5
    regime_tighter_stop_pct: float = None
    
    # ============================================================
    # DIRECTIONAL TREND FILTER: Skip counter-trend entries
    # ============================================================
    # Detects sustained multi-day uptrends/downtrends and skips
    # counter-trend entries (PUTs in uptrend, CALLs in downtrend).
    # Uses ONLY prior-day data — no look-ahead bias.
    #
    # August 2025 case study: SPY grinded up +3.71% with multiple
    # consecutive up days. PUTs went 5/20 (25% WR), losing $7,045.
    # This filter would have skipped most of those PUT entries.
    #
    use_trend_filter: bool = False
    trend_lookback_days: int = 5            # Rolling window of trading days
    trend_up_days_threshold: int = 4        # Skip PUTs if >= N of last lookback were up
    trend_down_days_threshold: int = 4      # Skip CALLs if >= N of last lookback were down
    trend_return_threshold: float = 0.015   # Also require |return| > 1.5% over window
    trend_filter_action: str = 'skip'       # 'skip' = no trade, 'reduce' = cut size
    trend_size_reduction: float = 0.50      # If action='reduce', cut position by 50%
    
    # ============================================================
    # PUT ENTRY FILTERS: Skip low-quality PUT signals
    # ============================================================
    # Based on August 2025 analysis: very oversold PUTs bounce,
    # Monday PUTs have 0% WR, and early entries (10:00-10:05) lose.
    #
    put_min_rsi: float = 25.0               # Skip PUTs with RSI < 25 (oversold bounces)
    put_skip_days: list = None              # Skip PUTs on these weekdays [0=Mon] (None=disabled)
    put_min_entry_minutes: int = 0          # Skip PUTs before this time (minutes since midnight, 610=10:10)
    put_filter_require_uptrend: bool = True  # Only apply PUT filters when market is in uptrend
    
    # ============================================================
    # ADAPTIVE PUT FILTER: Auto-throttle PUTs based on recent performance
    # ============================================================
    # Rather than detecting market regime with indicators (which fire too
    # broadly in a bull market), this adapts to outcomes. If PUTs are
    # consistently losing, the regime is unfavorable — stop taking PUTs.
    #
    # Backtest: +$10,636 full-year net gain vs baseline (not a cost).
    # August 2025: reduced -$2,237 to -$134 (saved $2,102).
    #
    put_adaptive_filter: bool = True        # Master switch for adaptive PUT filter
    put_loss_streak_threshold: int = 2      # Skip PUTs after N consecutive PUT losses
    put_adaptive_cooldown: int = 3          # Skip next N PUT signals after streak hit
    
    # ============================================================
    # ADAPTIVE CALL FILTER: Auto-throttle CALLs based on recent performance
    # ============================================================
    # Mirrors PUT adaptive filter. After N consecutive CALL losses,
    # skip next M CALL signals. Analysis shows CALL WR drops to 46.4%
    # after 3 consecutive losses, and CALLs are 64% of loss streaks.
    #
    call_adaptive_filter: bool = False       # Master switch for adaptive CALL filter
    call_loss_streak_threshold: int = 2      # Skip CALLs after N consecutive CALL losses
    call_adaptive_cooldown: int = 3          # Skip next N CALL signals after streak hit
    
    # ============================================================
    # DIRECTION-AWARE LOSS ESCALATION: Cross-direction streak cooldown
    # ============================================================
    # Tracks ALL consecutive losses (regardless of direction). When a
    # direction dominates the recent loss window, cooldown that direction.
    # Addresses mixed-direction streaks (19/21 streaks are mixed CALL/PUT).
    #
    # Analysis: After 3 consecutive losses, CALL WR drops to 46.4% (PF 0.74)
    # while PUT stays positive (66.7% WR). CALLs are 64% of loss streaks.
    #
    use_direction_loss_escalation: bool = False  # Master switch
    direction_loss_window: int = 3               # Look at last N consecutive losses
    direction_loss_threshold: int = 2            # Cooldown if >= N of window are same direction
    direction_loss_cooldown: int = 3             # Skip next N signals of the losing direction
    
    # ============================================================
    # POST-LOSS RSI TIGHTENING: Raise signal quality bar after streaks
    # ============================================================
    # When RiskManager enters reduced mode (after max_consecutive_losses),
    # tighten RSI thresholds to filter out weak signals that extend streaks.
    # Analysis: streak entries have RSI=71.6 vs normal 76.8 for CALLs.
    #
    consec_loss_rsi_buffer: int = 0  # Extra RSI buffer when in consecutive loss mode
                                     # CALL: need RSI > threshold + buffer
                                     # PUT:  need RSI < threshold - buffer
    
    # ============================================================
    # SMART EXIT SYSTEM: Intelligent mid-trade assessment
    # ============================================================
    # Adds real-time market assessment during trades. Each feature can be
    # independently enabled. The system is conservative by default: stall
    # and reversal checks only fire when underwater to avoid cutting winners.
    #
    # Key design principle: Smart exits NEVER cascade into post-loss
    # correction. They are tactical retreats, not market regime changes.
    # This prevents the cascade problem where early smart exits change
    # risk state and block subsequent (often winning) trades.
    #
    # Backtest notes (2025 in-sample, v3):
    #   ProfitProtect alone:  +786% (98% of baseline, PF 2.34)
    #   AdverseVelocity alone: +773% (96% of baseline, best risk-adjusted)
    #   StallDetect alone:    +756% (94% of baseline, PF 2.39)
    #   All combined:         +443% (55% of baseline, MaxDD 9.7%)
    #   Trade-off: individual features retain 94-98% of upside; combining
    #   all features still suffers from interaction/cascade effects.
    #
    use_smart_exit: bool = False            # Master switch (off by default)
    
    # 1. STALL DETECTION — 0DTE theta is decaying; if price isn't moving, exit early
    #    Only fires when position is underwater (pct_change < 0) — let winners ride
    smart_stall_bars: int = 5              # After N bars of no movement, signal stall
    smart_stall_threshold: float = 0.03    # ±3% range = "stalled"
    smart_stall_only_underwater: bool = True  # Only stall-exit if losing money
    
    # 2. UNDERLYING REVERSAL — exit if the signal that got us in has flipped
    #    Only fires when position is underwater — profitable trades can ride the reversal
    smart_underlying_reversal: bool = True  # Check if underlying RSI/momentum reversed
    smart_rsi_reversal_band: float = 30.0  # RSI must move 30pts past midline (65->35 for CALL)
    smart_reversal_only_underwater: bool = True  # Only exit on reversal if losing
    
    # 3. ADVERSE VELOCITY — big single-bar adverse move = early warning
    smart_adverse_bar_pct: float = 0.25    # Single bar drops >25% of prev close = emergency exit
    smart_adverse_only_underwater: bool = False  # Fire on all trades (25% crash is always serious)
    
    # 4. PROFIT PROTECTION RATCHET — once profitable, don't give it all back
    smart_profit_protect_trigger: float = 0.30   # Activate after reaching +30% unrealized
    smart_profit_protect_floor: float = 0.60     # Keep at least 60% of max unrealized profit
    smart_profit_protect_min_bars: int = 2       # Require N consecutive bars below floor to confirm
    smart_profit_protect_near_pt_pct: float = 0.50  # Skip protect if pct_change > PT * this (near profit target)
    
    # 5. MOMENTUM-BASED HOLD EXTENSION — don't force TIME exit if momentum is strong
    smart_momentum_extend: bool = True     # Extend hold if momentum is in our favor
    smart_momentum_extend_bars: int = 3    # Max extra bars to hold beyond max_hold
    smart_momentum_extend_threshold: float = 0.08  # Require option rising >8% per bar to extend
    
    # 6. SMART EXIT CASCADE CONTROL
    smart_exit_loss_threshold: float = 0.10   # Only arm post-loss correction if smart exit loss > this %
    smart_reversal_min_bars: int = 3           # Min bars held before reversal can fire
    
    @classmethod
    def from_json(cls, json_path: str = "config/strategy.json") -> "TradeConfig":
        """Load TradeConfig from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        cfg_data = data.get('trade_config', {})
        return cls(**{k: v for k, v in cfg_data.items() if hasattr(cls, k) or k in cls.__dataclass_fields__})


@dataclass
class Trade0DTE:
    """0DTE trade record"""
    date: str
    time: str
    direction: str
    strike: float
    option_ticker: str
    rsi: float
    ml_prob: float
    kelly_pct: float
    entry: float
    exit: float
    exit_reason: str
    bars_held: int
    num_contracts: int
    pnl: float
    capital: float
    
    def to_dict(self) -> dict:
        return self.__dict__


class Backtest0DTE:
    """
    0DTE Options Backtesting Engine
    
    Key features:
    - NO look-ahead bias (uses ML day filter or rolling historical volatility)
    - Uses core ML model for signal filtering
    - Uses ML day filter for day selection
    - Uses core RiskManager for position sizing
    """
    
    def __init__(
        self,
        trade_config: TradeConfig = None,
        risk_config: RiskConfig = None,
        initial_capital: float = None
    ):
        if initial_capital is None:
            initial_capital = cfg.initial_capital()
        self.trade_config = trade_config or TradeConfig()
        self.risk_config = risk_config or RiskConfig()
        self.initial_capital = initial_capital
        
        # Components
        self.risk_manager = RiskManager(initial_capital, self.risk_config)
        self.ml_model: Optional[TradingMLModel] = None
        self.day_filter: Optional[DayFilterModel] = None
        
        # Pre-built indexes for fast lookups
        self._opt_by_date_time: Dict = {}     # (date,time) -> DataFrame slice
        self._opt_by_ticker_date: Dict = {}   # (ticker,date) -> DataFrame slice
        
        # Data
        self.db = MarketDatabase("data/market_data.db")
        
    def load_data(
        self,
        start_date: str,
        end_date: str,
        underlying: str = 'SPY'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load underlying and options data.
        
        Returns:
            Tuple of (underlying_df, options_df, features_df)
        """
        print(f"Loading data from {start_date} to {end_date}...")
        
        # Load options data
        conn = sqlite3.connect(self.db.db_path)
        query = f"""
        SELECT option_ticker, underlying, timestamp, date, time,
               open, high, low, close, volume, expiration, strike, option_type
        FROM options_intraday
        WHERE underlying = '{underlying}' AND date = expiration
              AND date >= '{start_date}' AND date <= '{end_date}'
        ORDER BY date, time
        """
        options_df = pd.read_sql_query(query, conn)
        options_df = options_df[options_df['close'] > 0].copy()
        conn.close()
        
        print(f"  Options: {len(options_df):,} bars, {options_df['date'].nunique()} days")
        
        # Load underlying data
        data = self.db.get_intraday_5min(underlying)
        underlying_df = pd.DataFrame(data)
        underlying_df = underlying_df.rename(columns={'date': 'timestamp'})
        underlying_df['date'] = pd.to_datetime(underlying_df['timestamp']).dt.strftime('%Y-%m-%d')
        underlying_df['time'] = pd.to_datetime(underlying_df['timestamp']).dt.strftime('%H:%M:%S')
        underlying_df['hour'] = pd.to_datetime(underlying_df['timestamp']).dt.hour
        underlying_df = underlying_df.sort_values('timestamp').reset_index(drop=True)
        
        # Filter to dates with options data
        option_dates = set(options_df['date'].unique())
        underlying_df = underlying_df[underlying_df['date'].isin(option_dates)].copy()
        
        print(f"  Underlying: {len(underlying_df)} bars")
        
        # Compute features (pass orb_minutes from config)
        features_df = compute_features(underlying_df, orb_minutes=self.trade_config.orb_minutes)
        
        # Build indexes for fast lookups
        self._build_option_index(options_df)
        
        return underlying_df, options_df, features_df
    
    def _build_option_index(self, options_df: pd.DataFrame):
        """Pre-index options data for O(1) lookups instead of O(n) scans."""
        # Index by (date, time) for _find_option
        self._opt_by_date_time = {}
        for (date, time), group in options_df.groupby(['date', 'time']):
            self._opt_by_date_time[(date, time)] = group
        
        # Index by (option_ticker, date) for _get_future_bars
        self._opt_by_ticker_date = {}
        for (ticker, date), group in options_df.groupby(['option_ticker', 'date']):
            self._opt_by_ticker_date[(ticker, date)] = group.sort_values('time')
        
        print(f"  Indexes built: {len(self._opt_by_date_time):,} date-time slots, "
              f"{len(self._opt_by_ticker_date):,} ticker-date combos")
    
    def compute_regime_features(self, underlying_df: pd.DataFrame) -> Dict[str, dict]:
        """
        Multi-regime classifier from underlying intraday data.
        Delegates to shared core.regime_classifier for classification logic.
        Uses ONLY past data (rolling window) — no look-ahead bias.

        Returns:
            Dict[date_str, dict] with keys:
              regime_type, is_choppy, direction, intra_vol, trend_strength,
              vol_pctl, trend_pctl, up_day_pct, window_return, adx_proxy
        """
        cfg_t = self.trade_config

        # Build daily_bars list from DataFrame
        daily_bars = []
        for date, grp in underlying_df.groupby('date'):
            prices = grp['close'].values.tolist()
            daily_bars.append({'date': date, 'prices': prices})

        # Sort chronologically
        daily_bars.sort(key=lambda x: x['date'])

        # Build config dict from TradeConfig
        regime_config = {
            'lookback': cfg_t.regime_lookback_days,
            'vol_percentile': cfg_t.regime_vol_percentile,
            'trend_percentile': cfg_t.regime_trend_percentile,
            'up_day_pct': cfg_t.regime_up_day_pct,
            'dn_day_pct': cfg_t.regime_dn_day_pct,
            'momentum_threshold': cfg_t.regime_momentum_threshold,
            'high_vol_percentile': cfg_t.regime_high_vol_percentile,
            'adx_trend_threshold': cfg_t.regime_adx_trend_threshold,
        }

        return classify_regimes(daily_bars, regime_config)
    
    def compute_trend_filter(self, underlying_df: pd.DataFrame) -> Dict[str, dict]:
        """
        Compute per-day directional trend features for counter-trend filtering.
        Uses ONLY prior-day data — no look-ahead bias.
        
        Detects sustained uptrends (unfavorable for PUTs) and downtrends
        (unfavorable for CALLs) by counting up/down days and cumulative
        return over a rolling lookback window.
        
        Returns:
            Dict mapping date -> {
                up_days: int, down_days: int, cum_return: float,
                put_unfavorable: bool, call_unfavorable: bool
            }
        """
        lookback = self.trade_config.trend_lookback_days
        up_thr = self.trade_config.trend_up_days_threshold
        down_thr = self.trade_config.trend_down_days_threshold
        ret_thr = self.trade_config.trend_return_threshold
        
        # Build daily close series
        daily_close = (
            underlying_df.groupby('date')['close']
            .last()
            .sort_index()
        )
        
        dates = daily_close.index.tolist()
        closes = daily_close.values
        
        trend_data: Dict[str, dict] = {}
        
        for i, date in enumerate(dates):
            if i < lookback:
                trend_data[date] = {
                    'up_days': 0, 'down_days': 0, 'cum_return': 0.0,
                    'put_unfavorable': False, 'call_unfavorable': False,
                }
                continue
            
            # Use prior-day data only: window is [i-lookback, i-1]
            # i.e., the lookback days BEFORE today
            window_closes = closes[i - lookback:i]
            
            # Count up/down days (day-over-day changes within window)
            up_days = 0
            down_days = 0
            for j in range(1, len(window_closes)):
                if window_closes[j] > window_closes[j - 1]:
                    up_days += 1
                elif window_closes[j] < window_closes[j - 1]:
                    down_days += 1
            
            # Cumulative return over the window
            cum_return = (window_closes[-1] - window_closes[0]) / window_closes[0]
            
            put_unfavorable = up_days >= up_thr and cum_return > ret_thr
            call_unfavorable = down_days >= down_thr and cum_return < -ret_thr
            
            trend_data[date] = {
                'up_days': up_days,
                'down_days': down_days,
                'cum_return': cum_return,
                'put_unfavorable': put_unfavorable,
                'call_unfavorable': call_unfavorable,
            }
        
        return trend_data
    
    def compute_historical_volatility(self, underlying_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute historical volatility using PAST DATA ONLY.
        No look-ahead bias!
        
        This uses rolling volatility from previous N days,
        NOT the day's actual range.
        """
        return compute_rolling_volatility(
            underlying_df, 
            lookback_days=self.trade_config.volatility_lookback_days
        )
    
    def compute_morning_vol_for_vix(self, underlying_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute morning volatility as VIX proxy.
        Uses ONLY data available at trade time (9:30-10:00 range).
        
        VIX Proxy Logic:
        - Morning range * scaling factor to approximate VIX
        - A 0.5% morning range ≈ VIX 15
        - A 1.0% morning range ≈ VIX 30
        
        Returns:
            Dictionary mapping date -> VIX proxy value
        """
        # Filter to morning session only (before trade window)
        morning = underlying_df[underlying_df['time'] <= '10:00:00'].copy()
        
        if len(morning) == 0:
            return {}
        
        # Calculate morning range per day
        morning_stats = morning.groupby('date').agg({
            'high': 'max',
            'low': 'min',
            'open': 'first',
        })
        
        # Morning range as percentage
        morning_stats['morning_range_pct'] = (
            (morning_stats['high'] - morning_stats['low']) / morning_stats['open'] * 100
        )
        
        # Scale to VIX-like values:
        # 0.3% morning range → VIX ~12 (low vol)
        # 0.5% morning range → VIX ~18 (normal)
        # 0.8% morning range → VIX ~28 (elevated)
        # 1.5% morning range → VIX ~50 (panic)
        # Formula: VIX ≈ morning_range * 35
        morning_stats['vix_proxy'] = morning_stats['morning_range_pct'] * 35
        
        return morning_stats['vix_proxy'].to_dict()
    
    def get_dynamic_stop_loss(
        self,
        date: str,
        hour: int,
        rolling_volatility: Dict[str, float]
    ) -> float:
        """
        Calculate dynamic stop loss based on volatility and time.
        
        Logic:
        - Low volatility (VIX < 15): Tighter stop (20%)
        - Normal volatility (VIX 15-25): Standard stop (28%)  
        - High volatility (VIX > 25): Wider stop (40%)
        - Time decay: Tighter stop later in day (optional)
        
        Returns:
            Stop loss percentage (e.g., 0.28 for 28%)
        """
        cfg = self.trade_config
        
        if not cfg.use_dynamic_stop:
            return cfg.stop_loss_pct
        
        # Get volatility proxy (use rolling vol as VIX proxy)
        # Scale rolling vol to approximate VIX range
        vol = rolling_volatility.get(date, 0.015)
        vix_proxy = vol * 100 * 10  # Rough scaling to VIX-like number
        
        # Determine stop based on volatility regime
        if vix_proxy < cfg.vix_low_threshold:
            base_stop = cfg.stop_low_vol   # Tight stop in calm markets
        elif vix_proxy > cfg.vix_high_threshold:
            base_stop = cfg.stop_high_vol  # Wide stop in volatile markets
        else:
            base_stop = cfg.stop_mid_vol   # Normal stop
        
        # Optional: Time decay adjustment (tighter stop later in day)
        if cfg.use_time_decay_stop:
            hours_from_start = hour - cfg.trade_start_hour
            time_adjustment = hours_from_start * cfg.time_decay_factor
            base_stop = max(0.15, base_stop - time_adjustment)  # Min 15% stop
        
        return base_stop
    
    def get_adaptive_exits(
        self,
        date: str,
        hour: int,
        entry_price: float,
        vix_proxy: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Calculate ADAPTIVE profit target AND stop loss.
        
        Factors:
        1. VIX proxy (from morning volatility) - wider in high vol, tighter in low vol
        2. Entry price - cheap options get wider targets
        3. Time of day - tighter exits later in day (0DTE decay)
        
        Args:
            vix_proxy: Dictionary mapping date -> VIX proxy value (from morning vol)
        
        Returns:
            Tuple of (profit_target_pct, stop_loss_pct)
        """
        cfg = self.trade_config
        
        # If adaptive exits disabled, return fixed values
        if not cfg.use_adaptive_exits:
            return cfg.profit_target_pct, cfg.stop_loss_pct
        
        # Get VIX proxy for this date (default to 18 = normal)
        vix = vix_proxy.get(date, 18.0)
        
        # 1. VOLATILITY-BASED exits
        if vix < cfg.vix_low_threshold:
            # Low vol: Tighter targets and stops (less movement expected)
            profit_target = cfg.profit_low_vol
            stop_loss = cfg.stop_low_vol
        elif vix > cfg.vix_high_threshold:
            # High vol: Wider targets and stops (more movement expected)
            profit_target = cfg.profit_high_vol
            stop_loss = cfg.stop_high_vol
        else:
            # Normal vol
            profit_target = cfg.profit_mid_vol
            stop_loss = cfg.stop_mid_vol
        
        # 2. ENTRY PRICE adjustment (cheap options need wider targets)
        if entry_price < cfg.cheap_option_threshold:
            profit_target += cfg.cheap_option_bonus
            # Don't widen stop for cheap options - more risk there
        
        # 3. TIME DECAY adjustment (tighter later in day)
        if cfg.use_time_decay_exits:
            hours_from_start = max(0, hour - cfg.trade_start_hour)
            time_adjustment = hours_from_start * cfg.time_decay_factor
            profit_target = max(0.15, profit_target - time_adjustment)
            stop_loss = max(0.12, stop_loss - time_adjustment)
        
        return profit_target, stop_loss
    
    def is_volatile_enough(
        self, 
        date: str, 
        rolling_volatility: Dict[str, float],
        all_volatilities: List[float]
    ) -> bool:
        """
        Check if day is volatile enough based on HISTORICAL volatility.
        Uses percentile ranking against past volatility, not future.
        """
        vol = rolling_volatility.get(date, 0)
        if len(all_volatilities) == 0:
            return vol > 0.5  # Default threshold
        
        # Calculate percentile rank of this day's historical vol
        percentile = sum(1 for v in all_volatilities if v <= vol) / len(all_volatilities)
        
        return percentile >= self.trade_config.volatility_threshold
    
    def generate_training_samples(
        self,
        underlying_df: pd.DataFrame,
        options_df: pd.DataFrame,
        features_df: pd.DataFrame,
        rolling_volatility: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Generate training data by simulating trades.
        Used for ML model training.
        """
        samples = []
        cfg = self.trade_config
        
        # Get volatility values for percentile calc (handle both Series and dict)
        if hasattr(rolling_volatility, 'values') and callable(getattr(rolling_volatility, 'values', None)):
            all_vols = list(rolling_volatility.values())
        else:
            all_vols = list(rolling_volatility.dropna().values) if hasattr(rolling_volatility, 'dropna') else list(rolling_volatility)
        
        # Pre-group options
        options_by_date = {date: group for date, group in options_df.groupby('date')}
        
        for idx in range(50, len(underlying_df) - cfg.max_hold_bars - 1, 5):
            row = underlying_df.iloc[idx]
            current_date = row['date']
            current_time = row['time']
            current_hour = row['hour']
            underlying_price = row['close']
            
            if current_date not in options_by_date:
                continue
            
            if current_hour < cfg.trade_start_hour or current_hour > cfg.trade_end_hour:
                continue
            
            # Skip volatility filter when day filter is disabled (skip_day_filter=True)
            # so we get enough samples for Kelly calibration
            if not cfg.skip_day_filter:
                if not self.is_volatile_enough(current_date, rolling_volatility, all_vols):
                    continue
            
            feat = features_df.iloc[idx]
            rsi = feat['rsi']
            momentum = feat.get('momentum', 0) if hasattr(feat, 'get') else feat['momentum']
            
            for direction in ['CALL', 'PUT']:
                if direction == 'CALL' and momentum < 0.1:
                    continue
                if direction == 'PUT' and momentum > -0.1:
                    continue
                
                option_type = 'call' if direction == 'CALL' else 'put'
                
                option = self._find_option(
                    options_df, underlying_price, option_type, 
                    current_date, current_time
                )
                if option is None:
                    continue
                
                entry_price = option['close'] * (1 + self.risk_config.slippage_pct)
                option_ticker = option['option_ticker']
                
                future_bars = self._get_future_bars(
                    options_df, option_ticker, current_date, current_time
                )
                if len(future_bars) == 0:
                    continue
                
                outcome = self._simulate_outcome(future_bars, entry_price)
                if outcome is None:
                    continue
                
                sample = {
                    'date': current_date,
                    'rsi': rsi,
                    'momentum': momentum,
                    'entry_price': entry_price,
                    'bb_position': feat.get('bb_position', 0.5) if hasattr(feat, 'get') else feat['bb_position'],
                    'vwap_dev_pct': feat.get('vwap_dev_pct', 0) if hasattr(feat, 'get') else feat['vwap_dev_pct'],
                    'atr_pct': feat.get('atr_pct', 0) if hasattr(feat, 'get') else feat['atr_pct'],
                    'vol_ratio': feat.get('vol_ratio', 1) if hasattr(feat, 'get') else feat['vol_ratio'],
                    'hour': current_hour,
                    'direction_put': 1 if direction == 'PUT' else 0,
                    'win': outcome['win'],
                    'pnl_pct': outcome['pnl_pct'],
                }
                samples.append(sample)
        
        return pd.DataFrame(samples)
    
    def generate_day_labels(
        self,
        underlying_df: pd.DataFrame,
        options_df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate PnL labels for ALL trading days (for day filter training).
        Simulates trades on every day regardless of volatility filter.
        """
        day_pnl = {}
        cfg = self.trade_config
        
        # Reset indices to ensure alignment
        underlying = underlying_df.reset_index(drop=True)
        features = features_df.reset_index(drop=True)
        
        # Pre-group options by date
        options_by_date = {date: group for date, group in options_df.groupby('date')}
        
        # Get unique dates
        dates = underlying['date'].unique()
        
        for current_date in dates:
            if current_date not in options_by_date:
                continue
            
            day_pnl[current_date] = 0
            day_options = options_by_date[current_date]
            
            # Get indices for this day
            day_mask = underlying['date'] == current_date
            day_indices = underlying[day_mask].index.tolist()
            
            for idx in day_indices[::3]:  # Every 3rd bar for speed
                if idx >= len(features):
                    continue
                    
                row = underlying.iloc[idx]
                current_hour = row['hour']
                current_time = row['time']
                underlying_price = row['close']
                
                if current_hour < cfg.trade_start_hour or current_hour > cfg.trade_end_hour:
                    continue
                if current_hour == cfg.trade_end_hour and int(current_time.split(':')[1]) > cfg.trade_end_minute:
                    continue
                
                feat = features.iloc[idx]
                momentum = feat.get('momentum', 0) if hasattr(feat, 'get') else feat['momentum']
                
                for direction in ['CALL', 'PUT']:
                    if direction == 'CALL' and momentum < 0.1:
                        continue
                    if direction == 'PUT' and momentum > -0.1:
                        continue
                    
                    option_type = 'call' if direction == 'CALL' else 'put'
                    
                    option = self._find_option(
                        day_options, underlying_price, option_type,
                        current_date, current_time
                    )
                    if option is None:
                        continue
                    
                    entry_price = option['close'] * (1 + self.risk_config.slippage_pct)
                    option_ticker = option['option_ticker']
                    
                    future_bars = self._get_future_bars(
                        day_options, option_ticker, current_date, current_time
                    )
                    if len(future_bars) == 0:
                        continue
                    
                    outcome = self._simulate_outcome(future_bars, entry_price)
                    if outcome is not None:
                        day_pnl[current_date] += outcome['pnl_pct']
        
        # Convert to DataFrame
        result = pd.DataFrame([
            {'date': date, 'pnl': pnl} 
            for date, pnl in day_pnl.items()
        ])
        print(f"  Generated PnL labels for {len(result)} days")
        return result
    
    def train_model(
        self, 
        training_data: pd.DataFrame,
        underlying_df: pd.DataFrame = None,
        options_df: pd.DataFrame = None,
        features_df: pd.DataFrame = None
    ) -> Tuple[TradingMLModel, float]:
        """
        Train ML model, day filter, and calculate Kelly from training data.
        
        Args:
            training_data: Trade samples with features and outcomes
            underlying_df: Underlying data for day filter training (optional)
            options_df: Options data for day labeling (optional)
            features_df: Features data for day labeling (optional)
        
        Returns:
            Tuple of (model, kelly_pct)
        """
        print(f"\nTraining ML model on {len(training_data)} samples...")
        print(f"  Training win rate: {training_data['win'].mean()*100:.1f}%")
        
        self.ml_model = TradingMLModel()
        self.ml_model.train(training_data)
        
        # Calculate Kelly from training data
        kelly_pct, stats = self.risk_manager.setup_kelly(training_data)
        print(f"\n  Kelly Calculation:")
        print(f"    Win Rate: {stats.get('win_rate', 0):.1%}")
        print(f"    W/L Ratio: {stats.get('b_ratio', 0):.2f}")
        print(f"    Kelly (25% fractional): {kelly_pct:.1%}")
        
        # Train day filter if enabled and underlying data provided
        if self.trade_config.use_ml_day_filter and underlying_df is not None:
            print(f"\n  Training ML Day Filter...")
            day_features = compute_day_features(underlying_df)
            
            # Generate PnL labels for ALL days (not just days in training_data)
            trade_results = self.generate_day_labels(
                underlying_df, options_df, features_df
            ) if options_df is not None else None
            
            self.day_filter = DayFilterModel(threshold=self.trade_config.day_filter_threshold)
            self.day_filter.train(day_features, trade_results)
        
        return self.ml_model, kelly_pct
    
    def run(
        self,
        underlying_df: pd.DataFrame,
        options_df: pd.DataFrame,
        features_df: pd.DataFrame,
        rolling_volatility: Dict[str, float] = None,
        day_features: pd.DataFrame = None,
        verbose: bool = True
    ) -> List[Trade0DTE]:
        """
        Run backtest using trained model.
        
        Args:
            underlying_df: Underlying OHLCV data
            options_df: Options data
            features_df: Computed features
            rolling_volatility: Historical volatility (fallback if no day filter)
            day_features: Day features for ML day filter
            verbose: Print trade details
        """
        if self.ml_model is None or not self.ml_model.is_trained:
            raise ValueError("Model not trained. Call train_model first.")
        
        trades = []
        cfg = self.trade_config
        option_dates = set(options_df['date'].unique())
        
        # Prepare day filter features if using ML day filter
        day_features_dict = {}
        if cfg.use_ml_day_filter and day_features is not None:
            for _, row in day_features.iterrows():
                day_features_dict[row['date']] = row.to_dict()
        
        # Fallback: volatility percentiles
        cumulative_vols = {}
        if rolling_volatility:
            sorted_dates = sorted(rolling_volatility.keys())
            running_vols = []
            for d in sorted_dates:
                running_vols.append(rolling_volatility[d])
                cumulative_vols[d] = running_vols.copy()
        
        rejected_by_ml = 0
        passed_ml = 0
        rejected_by_day_filter = 0
        passed_day_filter = 0
        
        # Track which days we've checked
        checked_days = set()
        good_days = set()
        
        for idx in range(50, len(underlying_df) - cfg.max_hold_bars - 1):
            row = underlying_df.iloc[idx]
            current_date = row['date']
            current_time = row['time']
            current_hour = row['hour']
            underlying_price = row['close']
            
            if current_date not in option_dates:
                continue
            
            # Time filter
            if current_hour < cfg.trade_start_hour:
                continue
            try:
                current_minute = int(current_time.split(':')[1])
            except:
                current_minute = 0
            # Skip first X minutes of start hour
            if current_hour == cfg.trade_start_hour and current_minute < cfg.trade_start_minute:
                continue
            if current_hour > cfg.trade_end_hour:
                continue
            if current_hour == cfg.trade_end_hour:
                if current_minute >= cfg.trade_end_minute:
                    continue
            
            # Print progress every 1000 bars
            if idx % 1000 == 0 and verbose:
                print(f"Processing {current_date} {current_time}...", end='\r')
            
            # Risk check
            can_trade, reason = self.risk_manager.can_trade(current_date)
            if not can_trade:
                continue
            
            # DAY FILTER - ML-based or fallback to volatility
            if current_date not in checked_days:
                checked_days.add(current_date)
                
                if cfg.skip_day_filter:
                    # No filter, allow all days
                    good_days.add(current_date)
                    passed_day_filter += 1
                elif cfg.use_ml_day_filter and self.day_filter is not None and current_date in day_features_dict:
                    # ML day filter
                    day_feat = day_features_dict[current_date]
                    should_trade, day_prob = self.day_filter.should_trade_today(day_feat)
                    if should_trade:
                        good_days.add(current_date)
                        passed_day_filter += 1
                    else:
                        rejected_by_day_filter += 1
                elif rolling_volatility:
                    # Fallback to volatility filter
                    past_vols = cumulative_vols.get(current_date, [])
                    if self.is_volatile_enough(current_date, rolling_volatility, past_vols):
                        good_days.add(current_date)
                        passed_day_filter += 1
                    else:
                        rejected_by_day_filter += 1
                else:
                    # No filter, allow all days
                    good_days.add(current_date)
            
            # Skip if not a good day
            if current_date not in good_days:
                continue
            
            # Get basic signal
            feat = features_df.iloc[idx]
            direction = get_basic_signal(
                feat,
                rsi_call_threshold=self.trade_config.rsi_call_threshold,
                rsi_put_threshold=self.trade_config.rsi_put_threshold,
                strategy=self.trade_config.strategy,
                bb_buffer_pct=self.trade_config.bb_buffer_pct,
                vwap_dev_threshold=self.trade_config.vwap_dev_threshold,
                orb_buffer_pct=self.trade_config.orb_buffer_pct,
            )
            if direction is None:
                continue
            
            option_type = 'call' if direction == 'CALL' else 'put'
            
            # Find option
            option = self._find_option(
                options_df, underlying_price, option_type,
                current_date, current_time
            )
            if option is None:
                continue
            
            entry_price = self.risk_manager.apply_slippage(option['close'], is_entry=True)
            
            # ML filtering (can be disabled)
            if cfg.use_ml_filter:
                ml_prob = self.ml_model.predict(feat, entry_price, direction)
                
                if ml_prob < cfg.ml_confidence_threshold:
                    rejected_by_ml += 1
                    continue
            else:
                ml_prob = 0.70  # Default value when ML disabled
            
            passed_ml += 1
            
            strike = option['strike']
            option_ticker = option['option_ticker']
            
            # Get future bars
            future_bars = self._get_future_bars(
                options_df, option_ticker, current_date, current_time
            )
            if len(future_bars) == 0:
                continue
            
            # Position sizing
            num_contracts, _ = self.risk_manager.get_position_size(entry_price, ml_prob)
            
            # Simulate exit
            exit_price = None
            exit_reason = None
            bars_held = 0
            
            for bar_idx, (_, bar) in enumerate(future_bars.iterrows()):
                bars_held = bar_idx + 1
                bar_price = bar['close']
                
                pct_change = (bar_price - entry_price) / entry_price
                
                if pct_change >= cfg.profit_target_pct:
                    exit_price = self.risk_manager.apply_slippage(bar_price, is_entry=False)
                    exit_reason = 'PROFIT'
                    break
                
                if pct_change <= -cfg.stop_loss_pct:
                    exit_price = self.risk_manager.apply_slippage(bar_price, is_entry=False)
                    exit_reason = 'STOP'
                    break
                
                if bars_held >= cfg.max_hold_bars:
                    exit_price = self.risk_manager.apply_slippage(bar_price, is_entry=False)
                    exit_reason = 'TIME'
                    break
            
            if exit_price is None:
                continue
            
            # Calculate P&L
            gross_pnl, commission, net_pnl = self.risk_manager.calculate_trade_pnl(
                entry_price, exit_price, num_contracts
            )
            
            # Record trade
            self.risk_manager.record_trade(current_date, net_pnl)
            
            rsi = feat['rsi']
            kelly_pct = self.risk_manager.position_sizer.kelly_pct
            
            if verbose:
                emoji = "+" if net_pnl > 0 else "x"
                print(f"  {emoji} {current_date} {current_time} | K={kelly_pct:.0%} | ML={ml_prob:.0%} | "
                      f"RSI={rsi:.0f} | {direction} {strike:.0f} | {exit_reason} | ${net_pnl:+.2f}")
            
            trades.append(Trade0DTE(
                date=current_date,
                time=current_time,
                direction=direction,
                strike=strike,
                option_ticker=option_ticker,
                rsi=rsi,
                ml_prob=ml_prob,
                kelly_pct=kelly_pct,
                entry=entry_price,
                exit=exit_price,
                exit_reason=exit_reason,
                bars_held=bars_held,
                num_contracts=num_contracts,
                pnl=net_pnl,
                capital=self.risk_manager.capital,
            ))
        
        print(f"\nDay Filter: {passed_day_filter} days passed, {rejected_by_day_filter} days rejected")
        print(f"ML Filter: {passed_ml} trades passed, {rejected_by_ml} rejected")
        
        return trades
    
    def _find_option(
        self,
        options_df: pd.DataFrame,
        underlying_price: float,
        option_type: str,
        date: str,
        entry_time: str
    ) -> Optional[pd.Series]:
        """Find best option contract (slightly ITM) using pre-built index."""
        cfg = self.trade_config
        
        # Fast O(1) lookup instead of scanning full DataFrame
        slot = self._opt_by_date_time.get((date, entry_time))
        if slot is None or slot.empty:
            return None
        
        # Filter by option type and price range
        mask = (
            (slot['option_type'] == option_type) &
            (slot['close'] >= cfg.min_option_price) &
            (slot['close'] <= cfg.max_option_price)
        )
        available = slot[mask]
        
        if available.empty:
            return None
        
        # Prefer slightly ITM
        if option_type == 'call':
            itm = available[available['strike'] < underlying_price]
            if not itm.empty:
                strike_diff = underlying_price - itm['strike']
                return itm.iloc[strike_diff.values.argmin()]
        else:
            itm = available[available['strike'] > underlying_price]
            if not itm.empty:
                strike_diff = itm['strike'] - underlying_price
                return itm.iloc[strike_diff.values.argmin()]
        
        strike_diff = (available['strike'] - underlying_price).abs()
        return available.iloc[strike_diff.values.argmin()]
    
    def _get_future_bars(
        self,
        options_df: pd.DataFrame,
        option_ticker: str,
        date: str,
        entry_time: str
    ) -> pd.DataFrame:
        """Get future bars for an option after entry using pre-built index."""
        # Fast O(1) lookup instead of scanning full DataFrame
        ticker_day = self._opt_by_ticker_date.get((option_ticker, date))
        if ticker_day is None or ticker_day.empty:
            return pd.DataFrame()
        
        # Already sorted by time in index build
        return ticker_day[ticker_day['time'] > entry_time]
    
    def _simulate_outcome(
        self,
        future_bars: pd.DataFrame,
        entry_price: float
    ) -> Optional[dict]:
        """Simulate trade outcome for training data"""
        cfg = self.trade_config
        
        for bar_idx, (_, bar) in enumerate(future_bars.iterrows()):
            if bar_idx >= cfg.max_hold_bars:
                break
            
            bar_price = bar['close']
            pct_change = (bar_price - entry_price) / entry_price
            
            if pct_change >= cfg.profit_target_pct:
                return {'win': 1, 'pnl_pct': pct_change, 'exit': 'PROFIT'}
            
            if pct_change <= -cfg.stop_loss_pct:
                return {'win': 0, 'pnl_pct': pct_change, 'exit': 'STOP'}
        
        if len(future_bars) > 0:
            final_price = future_bars.iloc[min(cfg.max_hold_bars-1, len(future_bars)-1)]['close']
            pct_change = (final_price - entry_price) / entry_price
            return {'win': 1 if pct_change > 0 else 0, 'pnl_pct': pct_change, 'exit': 'TIME'}
        
        return None
    
    def print_results(self, trades: List[Trade0DTE]):
        """Print backtest results summary"""
        if not trades:
            print("\nNo trades executed")
            return
        
        df = pd.DataFrame([t.to_dict() for t in trades])
        
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        total_pnl = df['pnl'].sum()
        win_rate = len(wins) / len(df) * 100
        
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0
        
        win_total = wins['pnl'].sum() if len(wins) > 0 else 0
        loss_total = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        profit_factor = win_total / loss_total if loss_total > 0 else float('inf')
        
        # Max drawdown
        df['peak'] = df['capital'].cummax()
        df['drawdown'] = (df['peak'] - df['capital']) / df['peak'] * 100
        max_dd = df['drawdown'].max()
        
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        
        print(f"\nTrades: {len(df)}")
        print(f"  CALL: {len(df[df['direction'] == 'CALL'])}")
        print(f"  PUT: {len(df[df['direction'] == 'PUT'])}")
        
        print(f"\nWin Rate: {win_rate:.1f}%")
        
        exit_counts = df['exit_reason'].value_counts().to_dict()
        print(f"Exits: {exit_counts}")
        
        print(f"\nStart: ${self.initial_capital:,.2f}")
        print(f"End: ${self.risk_manager.capital:,.2f}")
        print(f"P&L: ${total_pnl:+,.2f} ({total_pnl/self.initial_capital*100:+.1f}%)")
        print(f"Max DD: {max_dd:.1f}%")
        
        print(f"\nAvg Win: ${avg_win:.2f}, Avg Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        # Save trades
        df.to_csv('output/backtest_trades.csv', index=False)
        print(f"\nTrades saved to: output/backtest_trades.csv")
        
        return df
    
    def save_models(self, model_dir: str = 'models'):
        """Save all trained models (trade ML + day filter)"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        if self.ml_model is None:
            raise ValueError("No trade model to save")
        
        kelly_pct = self.risk_manager.position_sizer.kelly_pct
        
        # Save trade model
        trade_model_path = f"{model_dir}/trade_model.joblib"
        self.ml_model.save(trade_model_path, kelly_pct)
        
        # Save day filter if trained
        if self.day_filter is not None and self.day_filter.is_trained:
            day_filter_path = f"{model_dir}/day_filter.joblib"
            self.day_filter.save(day_filter_path)
            print(f"Day filter saved to: {day_filter_path}")
        
        print(f"All models saved to: {model_dir}/")
    
    def load_models(self, model_dir: str = 'models'):
        """Load all trained models (trade ML + day filter)"""
        import os
        
        # Load trade model
        trade_model_path = f"{model_dir}/trade_model.joblib"
        if os.path.exists(trade_model_path):
            self.ml_model, kelly_pct = TradingMLModel.load(trade_model_path)
            self.risk_manager.set_kelly(kelly_pct)
            print(f"Trade model loaded from {trade_model_path}")
            print(f"  Kelly: {kelly_pct:.1%}")
        else:
            raise ValueError(f"Trade model not found: {trade_model_path}")
        
        # Load day filter if exists
        day_filter_path = f"{model_dir}/day_filter.joblib"
        if os.path.exists(day_filter_path):
            self.day_filter = DayFilterModel.load(day_filter_path)
            print(f"Day filter loaded from {day_filter_path}")
            print(f"  Threshold: {self.day_filter.threshold:.1%}")
        else:
            print(f"  No day filter found at {day_filter_path}")
    
    # Keep old methods for backward compatibility
    def save_model(self, path: str):
        """Save trained model (deprecated - use save_models)"""
        if self.ml_model is None:
            raise ValueError("No model to save")
        kelly_pct = self.risk_manager.position_sizer.kelly_pct
        self.ml_model.save(path, kelly_pct)
    
    def load_model(self, path: str):
        """Load trained model (deprecated - use load_models)"""
        self.ml_model, kelly_pct = TradingMLModel.load(path)
        self.risk_manager.set_kelly(kelly_pct)
        print(f"Loaded model from {path}")
        print(f"  Kelly: {kelly_pct:.1%}")

    # ============================================================
    # NO-ML METHODS (V3 - Simplified)
    # ============================================================
    
    def calculate_kelly_only(self, training_data: pd.DataFrame) -> float:
        """
        Calculate Kelly from training data without training ML model.
        
        Args:
            training_data: DataFrame with 'win' and 'pnl_pct' columns
            
        Returns:
            kelly_pct
        """
        if len(training_data) < 10:
            print(f"  Warning: Only {len(training_data)} samples, using default Kelly")
            self.risk_manager.set_kelly(0.08)
            return 0.08
        
        wins = training_data[training_data['win'] == 1]
        losses = training_data[training_data['win'] == 0]
        
        win_rate = len(wins) / len(training_data)
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0.25
        avg_loss = abs(losses['pnl_pct'].mean()) if len(losses) > 0 else 0.35
        
        # Kelly formula
        if avg_loss == 0:
            kelly = 0.10
        else:
            b = avg_win / avg_loss
            kelly = (b * win_rate - (1 - win_rate)) / b
        
        # Fractional Kelly (20%)
        kelly_frac = kelly * 0.20
        kelly_frac = max(0.02, min(0.20, kelly_frac))
        
        self.risk_manager.set_kelly(kelly_frac)
        
        print(f"\n  Kelly Calculation (No ML):")
        print(f"    Win Rate: {win_rate:.1%}")
        print(f"    W/L Ratio: {avg_win/avg_loss:.2f}")
        print(f"    Kelly (20% fractional): {kelly_frac:.1%}")
        
        return kelly_frac
    
    def run_no_ml(
        self,
        underlying_df: pd.DataFrame,
        options_df: pd.DataFrame,
        features_df: pd.DataFrame,
        rolling_volatility: dict = None,
        verbose: bool = True
    ) -> list:
        """
        Run backtest WITHOUT ML filtering.
        Uses RSI calibration filter only.
        
        Args:
            underlying_df: Underlying price data
            options_df: Options data
            features_df: Technical features
            rolling_volatility: Historical volatility (optional, for day filter)
            verbose: Print trade details
            
        Returns:
            List of Trade0DTE objects
        """
        cfg = self.trade_config
        trades = []
        
        # Track stats
        total_signals = 0
        regime_skipped = 0
        regime_reduced = 0
        put_filtered = 0
        
        # Adaptive PUT filter state
        _put_consec_losses: int = 0     # current consecutive PUT loss count
        _put_cooldown_remaining: int = 0  # PUT signals left to skip in cooldown
        put_adaptive_skipped = 0
        
        # Adaptive CALL filter state
        _call_consec_losses: int = 0     # current consecutive CALL loss count
        _call_cooldown_remaining: int = 0  # CALL signals left to skip in cooldown
        call_adaptive_skipped = 0
        
        # Direction-aware loss escalation state
        _recent_loss_dirs: list = []          # rolling list of directions for consecutive losses
        _dir_cooldown_call: int = 0           # remaining CALL signals to skip
        _dir_cooldown_put: int = 0            # remaining PUT signals to skip
        dir_escalation_skipped = 0
        
        # Regime detection
        _regime_data: Dict[str, dict] = {}
        if cfg.use_regime_detection:
            _regime_data = self.compute_regime_features(underlying_df)
            # Count each regime type
            _regime_counts: Dict[str, int] = {}
            for v in _regime_data.values():
                rt = v.get('regime_type', 'NORMAL')
                _regime_counts[rt] = _regime_counts.get(rt, 0) + 1
            choppy_days = sum(1 for v in _regime_data.values() if v.get('is_choppy', False))
            if verbose:
                parts = ', '.join(f'{k}={v}' for k, v in sorted(_regime_counts.items()))
                print(f"\n  Regime Detection: {parts}  (choppy-equiv={choppy_days})")

        # Directional trend filter (also needed for PUT filter gating)
        _trend_data: Dict[str, dict] = {}
        trend_skipped = 0
        trend_reduced = 0
        _need_trend = cfg.use_trend_filter or cfg.put_filter_require_uptrend
        if _need_trend:
            _trend_data = self.compute_trend_filter(underlying_df)
            put_blocked = sum(1 for v in _trend_data.values() if v.get('put_unfavorable', False))
            call_blocked = sum(1 for v in _trend_data.values() if v.get('call_unfavorable', False))
            if verbose:
                print(f"\n  Trend Data: {put_blocked} days PUT-unfavorable, {call_blocked} days CALL-unfavorable")

        # Post-loss correction state
        _post_loss_active: dict = {}    # date -> bool  (correction mode active)
        _loss_direction: dict = {}      # date -> 'CALL'|'PUT'  (direction that lost)
        _first_loss_seen: dict = {}
        _loss_exit_reason: dict = {}    # date -> exit_reason ('STOP','TIME',etc.)
        _loss_bar_idx: dict = {}        # date -> bar index when loss occurred
        
        # Compute VIX proxy from morning volatility (no look-ahead)
        vix_proxy = self.compute_morning_vol_for_vix(underlying_df)
        _vix_median = float(np.median(list(vix_proxy.values()))) if vix_proxy else 15.0
        
        if verbose and cfg.use_adaptive_exits:
            vix_values = list(vix_proxy.values())
            if vix_values:
                low_vol_days = sum(1 for v in vix_values if v < cfg.vix_low_threshold)
                high_vol_days = sum(1 for v in vix_values if v > cfg.vix_high_threshold)
                mid_vol_days = len(vix_values) - low_vol_days - high_vol_days
                print(f"\n  VIX Proxy Distribution:")
                print(f"    Low (<{cfg.vix_low_threshold}): {low_vol_days} days → 20%/18% exits")
                print(f"    Mid ({cfg.vix_low_threshold}-{cfg.vix_high_threshold}): {mid_vol_days} days → 25%/28% exits")
                print(f"    High (>{cfg.vix_high_threshold}): {high_vol_days} days → 35%/40% exits")
        
        # Smart exit: Build (date, time) -> features lookup for underlying assessment
        _feat_by_dt: Dict[tuple, pd.Series] = {}
        if cfg.use_smart_exit and cfg.smart_underlying_reversal:
            for i in range(len(features_df)):
                row_u = underlying_df.iloc[i]
                _feat_by_dt[(row_u['date'], row_u['time'])] = features_df.iloc[i]
            if verbose:
                print(f"\n  Smart Exit: Built {len(_feat_by_dt):,} underlying feature lookups")
        
        total_rows = len(underlying_df)
        total_days = underlying_df['date'].nunique()
        last_date = None
        days_count = 0
        for idx in range(total_rows):
            row = underlying_df.iloc[idx]
            current_date = row['date']
            
            # Progress indicator - print every 50 days
            if current_date != last_date:
                days_count += 1
                if verbose and days_count % 50 == 1:
                    print(f"        Day {days_count}/{total_days}: {current_date} ({len(trades)} trades so far)", flush=True)
                last_date = current_date
            
            current_time = row['time']
            underlying_price = row['close']
            
            # Time filter
            hour = row['hour']
            try:
                minute = int(str(current_time).split(':')[1])
            except (IndexError, ValueError):
                minute = 0
            
            if hour < cfg.trade_start_hour:
                continue
            if hour > cfg.trade_end_hour:
                continue
            if hour == cfg.trade_end_hour and minute > cfg.trade_end_minute:
                continue
            
            # Risk check
            can_trade, reason = self.risk_manager.can_trade(current_date)
            if not can_trade:
                continue
            
            # ============================================================
            # REGIME DETECTION: Multi-regime adjustments
            # ============================================================
            _is_choppy = False
            _regime_type = 'NORMAL'
            _regime_direction = 0
            _regime_skip_puts = False
            _regime_skip_calls = False

            if cfg.use_regime_detection and current_date in _regime_data:
                _rd = _regime_data[current_date]
                _is_choppy = _rd.get('is_choppy', False)
                _regime_type = _rd.get('regime_type', 'NORMAL')
                _regime_direction = _rd.get('direction', 0)

                # --- CHOPPY: skip first bar ---
                if _regime_type == 'CHOPPY' and cfg.choppy_skip_first_bar:
                    if hour == cfg.trade_start_hour and minute <= cfg.trade_start_minute:
                        regime_skipped += 1
                        continue

                # --- STEADY_UP: optionally skip PUTs entirely ---
                if _regime_type == 'STEADY_UP' and cfg.steady_up_skip_puts:
                    _regime_skip_puts = True

                # --- STEADY_DN: optionally skip CALLs entirely ---
                if _regime_type == 'STEADY_DN' and cfg.steady_dn_skip_calls:
                    _regime_skip_calls = True

                # --- TRENDING: skip counter-trend entries ---
                if _regime_type == 'TRENDING' and cfg.trending_skip_counter:
                    if _regime_direction > 0:
                        _regime_skip_puts = True
                    elif _regime_direction < 0:
                        _regime_skip_calls = True
            
            # Get signal from strategy
            feat = features_df.iloc[idx]
            
            # RSI buffer per regime
            _call_thr = self.trade_config.rsi_call_threshold
            _put_thr = self.trade_config.rsi_put_threshold
            if _regime_type == 'CHOPPY' and cfg.choppy_rsi_buffer > 0:
                _call_thr += cfg.choppy_rsi_buffer
                _put_thr -= cfg.choppy_rsi_buffer
            elif _regime_type == 'STEADY_UP' and cfg.steady_up_rsi_buffer > 0:
                _call_thr += cfg.steady_up_rsi_buffer
                _put_thr -= cfg.steady_up_rsi_buffer
            elif _regime_type == 'STEADY_DN' and cfg.steady_dn_rsi_buffer > 0:
                _call_thr += cfg.steady_dn_rsi_buffer
                _put_thr -= cfg.steady_dn_rsi_buffer
            
            # Post-loss RSI tightening: raise bar when in consecutive loss mode
            if cfg.consec_loss_rsi_buffer > 0 and self.risk_manager.in_reduced_mode:
                _call_thr += cfg.consec_loss_rsi_buffer
                _put_thr -= cfg.consec_loss_rsi_buffer
            
            direction = get_basic_signal(
                feat,
                rsi_call_threshold=_call_thr,
                rsi_put_threshold=_put_thr,
                strategy=self.trade_config.strategy,
                bb_buffer_pct=self.trade_config.bb_buffer_pct,
                vwap_dev_threshold=self.trade_config.vwap_dev_threshold,
                orb_buffer_pct=self.trade_config.orb_buffer_pct,
            )
            if direction is None:
                continue

            # REGIME DIRECTIONAL FILTER: skip counter-regime entries
            if direction == 'PUT' and _regime_skip_puts:
                regime_skipped += 1
                continue
            if direction == 'CALL' and _regime_skip_calls:
                regime_skipped += 1
                continue
            
            # POST-LOSS SIGNAL CORRECTION
            _pls = cfg.post_loss_strategy
            if _pls != 'none' and _post_loss_active.get(current_date, False):
                lost_dir = _loss_direction.get(current_date)
                flipped_dir = 'PUT' if direction == 'CALL' else 'CALL'

                if _pls == 'flip':
                    # Blind flip every signal after first loss
                    direction = flipped_dir

                elif _pls == 'momentum_confirm':
                    # Flip only if 3-bar momentum confirms the reversal direction
                    momentum = feat.get('momentum_3', 0) if hasattr(feat, 'get') else feat['momentum_3']
                    _mthr = cfg.post_loss_momentum_threshold
                    if lost_dir == 'CALL':
                        # Market went DOWN vs our CALL — flip to PUT only if momentum negative
                        if momentum < -_mthr:
                            direction = 'PUT'
                        else:
                            continue  # market not confirming — skip
                    else:  # lost_dir == 'PUT'
                        # Market went UP vs our PUT — flip to CALL only if momentum positive
                        if momentum > _mthr:
                            direction = 'CALL'
                        else:
                            continue

                elif _pls == 'multi_confirm':
                    # Ignore original RSI signal; re-derive from multi-indicator consensus
                    momentum = feat.get('momentum_3', 0) if hasattr(feat, 'get') else feat['momentum_3']
                    vwap_d   = feat.get('vwap_distance', 0) if hasattr(feat, 'get') else feat.get('vwap_distance', 0)
                    trend    = feat.get('trend_strength', 0) if hasattr(feat, 'get') else feat.get('trend_strength', 0)
                    _mthr = cfg.post_loss_momentum_threshold

                    bull = int(momentum > _mthr) + int(vwap_d > 0) + int(trend > 0)
                    bear = int(momentum < -_mthr) + int(vwap_d < 0) + int(trend < 0)

                    if bull >= 2:
                        direction = 'CALL'
                    elif bear >= 2:
                        direction = 'PUT'
                    else:
                        continue  # ambiguous / choppy — skip trade

                elif _pls == 'adaptive':
                    # Context-aware post-loss: adapts threshold by exit reason, VIX, volume
                    
                    # 1. COOLDOWN: skip if too soon after loss
                    loss_idx = _loss_bar_idx.get(current_date, 0)
                    if idx - loss_idx < cfg.post_loss_cooldown_bars:
                        continue
                    
                    # 2. VOLUME GATE: require sufficient volume participation
                    vol_ratio = feat.get('vol_ratio', 1.0) if hasattr(feat, 'get') else feat.get('vol_ratio', 1.0)
                    if vol_ratio < cfg.post_loss_min_vol_ratio:
                        continue
                    
                    # 3. ADAPTIVE THRESHOLD based on exit reason + VIX
                    _base_thr = cfg.post_loss_momentum_threshold
                    loss_reason = _loss_exit_reason.get(current_date, 'STOP')
                    
                    # Exit-reason scaling: STOP = market moved hard → easier to flip
                    #                      TIME = choppy → harder to flip
                    if loss_reason == 'STOP':
                        _mthr = _base_thr * cfg.post_loss_stop_factor
                    elif loss_reason == 'TIME':
                        _mthr = _base_thr * cfg.post_loss_time_factor
                    else:
                        _mthr = _base_thr  # TRAIL/BREAKEVEN — use base
                    
                    # VIX scaling: normalize by median VIX so threshold adapts to regime
                    day_vix = vix_proxy.get(current_date, _vix_median)
                    if _vix_median > 0:
                        vix_scale = max(0.5, min(2.0, day_vix / _vix_median))
                        _mthr *= vix_scale
                    
                    # 4. MOMENTUM CONFIRMATION (same logic as momentum_confirm but with adaptive threshold)
                    momentum = feat.get('momentum_3', 0) if hasattr(feat, 'get') else feat['momentum_3']
                    if lost_dir == 'CALL':
                        if momentum < -_mthr:
                            direction = 'PUT'
                        else:
                            continue
                    else:  # lost_dir == 'PUT'
                        if momentum > _mthr:
                            direction = 'CALL'
                        else:
                            continue

            # ============================================================
            # ADAPTIVE PUT FILTER: Skip PUTs when recent PUT performance is poor
            # ============================================================
            if direction == 'PUT' and cfg.put_adaptive_filter:
                if _put_cooldown_remaining > 0:
                    _put_cooldown_remaining -= 1
                    put_adaptive_skipped += 1
                    continue
            
            # ============================================================
            # ADAPTIVE CALL FILTER: Skip CALLs when recent CALL performance is poor
            # ============================================================
            if direction == 'CALL' and cfg.call_adaptive_filter:
                if _call_cooldown_remaining > 0:
                    _call_cooldown_remaining -= 1
                    call_adaptive_skipped += 1
                    continue
            
            # ============================================================
            # DIRECTION-AWARE LOSS ESCALATION: Cooldown dominant losing direction
            # ============================================================
            if cfg.use_direction_loss_escalation:
                if direction == 'CALL' and _dir_cooldown_call > 0:
                    _dir_cooldown_call -= 1
                    dir_escalation_skipped += 1
                    continue
                elif direction == 'PUT' and _dir_cooldown_put > 0:
                    _dir_cooldown_put -= 1
                    dir_escalation_skipped += 1
                    continue
            
            # ============================================================
            # PUT ENTRY FILTERS: Skip low-quality PUT signals
            # Only active when market is in uptrend (put_filter_require_uptrend)
            # or unconditionally if put_filter_require_uptrend=False
            # ============================================================
            if direction == 'PUT':
                # Check if PUT filters should engage
                _put_filter_active = True
                if cfg.put_filter_require_uptrend:
                    td = _trend_data.get(current_date, {})
                    _put_filter_active = td.get('put_unfavorable', False)
                
                if _put_filter_active:
                    # Skip PUTs with RSI below minimum (very oversold → bounce risk)
                    if cfg.put_min_rsi > 0:
                        rsi_val = feat.get('rsi', 50)
                        if rsi_val < cfg.put_min_rsi:
                            put_filtered += 1
                            continue
                    
                    # Skip PUTs on specific weekdays (e.g. Monday=0)
                    if cfg.put_skip_days:
                        dow = pd.to_datetime(current_date).dayofweek
                        if dow in cfg.put_skip_days:
                            put_filtered += 1
                            continue
                    
                    # Skip PUTs before minimum entry time
                    if cfg.put_min_entry_minutes > 0:
                        entry_minutes = hour * 60 + minute
                        if entry_minutes < cfg.put_min_entry_minutes:
                            put_filtered += 1
                            continue
            
            # ============================================================
            # DIRECTIONAL TREND FILTER: Skip counter-trend entries
            # ============================================================
            _trend_blocked = False
            if cfg.use_trend_filter and current_date in _trend_data:
                td = _trend_data[current_date]
                if direction == 'PUT' and td.get('put_unfavorable', False):
                    _trend_blocked = True
                elif direction == 'CALL' and td.get('call_unfavorable', False):
                    _trend_blocked = True
                
                if _trend_blocked and cfg.trend_filter_action == 'skip':
                    trend_skipped += 1
                    continue

            total_signals += 1
            option_type = 'call' if direction == 'CALL' else 'put'
            
            # Find option
            option = self._find_option(
                options_df, underlying_price, option_type,
                current_date, current_time
            )
            if option is None:
                continue
            
            entry_price = self.risk_manager.apply_slippage(option['close'], is_entry=True)
            strike = option['strike']
            option_ticker = option['option_ticker']
            
            # Get future bars
            future_bars = self._get_future_bars(
                options_df, option_ticker, current_date, current_time
            )
            if len(future_bars) == 0:
                continue
            
            # ============================================================
            # ADAPTIVE EXITS: Get dynamic profit target AND stop loss
            # ============================================================
            profit_target, stop_loss = self.get_adaptive_exits(
                current_date, hour, entry_price, vix_proxy
            )
            
            # Direction-specific overrides (skip when adaptive exits override)
            if not cfg.use_adaptive_exits:
                if direction == 'CALL' and cfg.call_profit_target_pct is not None:
                    profit_target = cfg.call_profit_target_pct
                elif direction == 'PUT' and cfg.put_profit_target_pct is not None:
                    profit_target = cfg.put_profit_target_pct
                if direction == 'CALL' and cfg.call_stop_loss_pct is not None:
                    stop_loss = cfg.call_stop_loss_pct
                elif direction == 'PUT' and cfg.put_stop_loss_pct is not None:
                    stop_loss = cfg.put_stop_loss_pct
            
            # ============================================================
            # REGIME ADJUSTMENTS: Per-regime PT/SL/hold overrides
            # ============================================================
            if _regime_type == 'CHOPPY' and cfg.choppy_tighter_stop_pct is not None:
                stop_loss = cfg.choppy_tighter_stop_pct

            if _regime_type == 'STEADY_UP':
                if direction == 'CALL' and cfg.steady_up_call_pt_override is not None:
                    profit_target = cfg.steady_up_call_pt_override

            if _regime_type == 'STEADY_DN':
                if direction == 'PUT' and cfg.steady_dn_put_pt_override is not None:
                    profit_target = cfg.steady_dn_put_pt_override

            if _regime_type == 'VOLATILE':
                stop_loss = stop_loss + cfg.volatile_stop_buffer_pct
                profit_target = profit_target + cfg.volatile_pt_buffer_pct
            
            # Position sizing (PASS STOP LOSS for proper risk calculation)
            num_contracts, _ = self.risk_manager.get_position_size(
                entry_price, 
                ml_confidence=None,
                stop_loss_pct=stop_loss
            )
            
            # ============================================================
            # REGIME ADJUSTMENT: Per-regime position size reduction
            # ============================================================
            _regime_size_red = 0.0
            if _regime_type == 'CHOPPY':
                _regime_size_red = cfg.choppy_size_reduction
            elif _regime_type == 'STEADY_UP':
                _regime_size_red = cfg.steady_up_size_reduction
            elif _regime_type == 'STEADY_DN':
                _regime_size_red = cfg.steady_dn_size_reduction
            elif _regime_type == 'VOLATILE':
                _regime_size_red = cfg.volatile_size_reduction

            if _regime_size_red > 0:
                reduced = max(1, int(num_contracts * (1 - _regime_size_red)))
                if reduced < num_contracts:
                    regime_reduced += 1
                num_contracts = reduced
            
            # ============================================================
            # TREND FILTER: Reduce position size for counter-trend entries
            # ============================================================
            if _trend_blocked and cfg.trend_filter_action == 'reduce':
                reduced = max(1, int(num_contracts * (1 - cfg.trend_size_reduction)))
                if reduced < num_contracts:
                    trend_reduced += 1
                num_contracts = reduced
            
            # Cap max contracts per trade
            if cfg.max_contracts_per_trade > 0:
                num_contracts = min(num_contracts, cfg.max_contracts_per_trade)
            
            # Skip if position too small or below minimum
            if num_contracts == 0:
                continue
            if num_contracts < cfg.min_contracts_per_trade:
                continue
            
            # ============================================================
            # SIMULATE EXIT with ADAPTIVE TARGETS + TRAILING STOP + SMART EXIT
            # ============================================================
            exit_price = None
            exit_reason = None
            bars_held = 0
            
            # Trailing stop tracking
            max_profit_pct = 0.0
            trailing_stop_price = None
            breakeven_activated = False
            
            # Smart exit tracking
            _opt_prices = []           # option price history for momentum/stall
            _smart_profit_floor = None # profit protection ratchet floor
            _protect_below_count = 0   # consecutive bars below profit floor
            
            for bar_idx, (_, bar) in enumerate(future_bars.iterrows()):
                bars_held = bar_idx + 1
                bar_price = bar['close']
                
                pct_change = (bar_price - entry_price) / entry_price
                
                # Track option price history for smart exit
                _opt_prices.append(bar_price)
                
                # Update max profit for trailing stop
                if pct_change > max_profit_pct:
                    max_profit_pct = pct_change
                
                # ============================================================
                # TIME-DECAY EXIT: Reduce profit target each bar
                # ============================================================
                current_profit_target = profit_target
                current_stop_loss = stop_loss
                
                if cfg.use_time_decay_exit and bars_held > 1:
                    # Reduce profit target by X% per bar after first bar
                    decay = (bars_held - 1) * cfg.time_decay_profit_per_bar
                    current_profit_target = max(cfg.min_profit_target, profit_target - decay)
                
                # ============================================================
                # QUICK-EXIT ESCALATION: Tighten stop if underwater after bar 1
                # ============================================================
                if cfg.use_quick_exit and bars_held > 1:
                    if pct_change < 0:  # Underwater - tighten stop
                        current_stop_loss = cfg.underwater_stop_tighten
                    elif pct_change >= cfg.quick_exit_profit_threshold:  # Profitable enough - lock in profit
                        breakeven_activated = True
                        if trailing_stop_price is None:
                            # Lock in buffer profit instead of breakeven (covers slippage)
                            trailing_stop_price = entry_price * (1 + cfg.breakeven_buffer_pct)
                
                # === TRAILING STOP LOGIC ===
                if cfg.use_trailing_stop:
                    # Activate breakeven stop after initial profit
                    if not breakeven_activated and pct_change >= cfg.breakeven_activation:
                        breakeven_activated = True
                        trailing_stop_price = entry_price * 1.001  # Tiny profit to cover fees
                    
                    # Activate trailing stop after larger profit
                    if max_profit_pct >= cfg.trail_activation_pct:
                        # Trail at X% of max gain
                        trail_level = entry_price * (1 + max_profit_pct * (1 - cfg.trail_distance_pct))
                        if trailing_stop_price is None or trail_level > trailing_stop_price:
                            trailing_stop_price = trail_level
                
                # === CHECK EXITS ===
                
                # 1. PROFIT TARGET (with time decay)
                if pct_change >= current_profit_target:
                    exit_price = self.risk_manager.apply_slippage(bar_price, is_entry=False)
                    exit_reason = 'PROFIT'
                    break
                
                # 2. TRAILING STOP / BREAKEVEN (if activated and hit)
                if trailing_stop_price is not None and bar_price <= trailing_stop_price:
                    exit_price = self.risk_manager.apply_slippage(bar_price, is_entry=False)
                    exit_reason = 'BREAKEVEN' if not cfg.use_trailing_stop else 'TRAIL'
                    break
                
                # ============================================================
                # SMART EXIT ASSESSMENT (intelligent mid-trade evaluation)
                # ============================================================
                if cfg.use_smart_exit:
                    _smart_exit = False
                    
                    # 2a. ADVERSE VELOCITY — single bar drops too hard
                    #     Skip if position is profitable (winning trades tolerate dips)
                    if len(_opt_prices) >= 2:
                        _adverse_allowed = (not cfg.smart_adverse_only_underwater) or (pct_change < 0)
                        if _adverse_allowed:
                            bar_return = (_opt_prices[-1] - _opt_prices[-2]) / _opt_prices[-2]
                            if bar_return < -cfg.smart_adverse_bar_pct:
                                exit_price = self.risk_manager.apply_slippage(bar_price, is_entry=False)
                                exit_reason = 'SMART_ADVERSE'
                                _smart_exit = True
                    
                    # 2b. PROFIT PROTECTION RATCHET — once profitable, lock in a floor
                    #     Skip if price is near the profit target (let it ride to PT)
                    if not _smart_exit and max_profit_pct >= cfg.smart_profit_protect_trigger:
                        _near_pt = pct_change >= (profit_target * cfg.smart_profit_protect_near_pt_pct)
                        if not _near_pt:
                            _smart_profit_floor = max_profit_pct * cfg.smart_profit_protect_floor
                            if pct_change < _smart_profit_floor:
                                _protect_below_count += 1
                                if _protect_below_count >= cfg.smart_profit_protect_min_bars:
                                    exit_price = self.risk_manager.apply_slippage(bar_price, is_entry=False)
                                    exit_reason = 'SMART_PROTECT'
                                    _smart_exit = True
                            else:
                                _protect_below_count = 0  # reset if recovered above floor
                    
                    # 2c. STALL DETECTION — theta is eating the position if no movement
                    #     Only fire if underwater (or flag disabled) — don't cut profitable stalls
                    if not _smart_exit and bars_held >= cfg.smart_stall_bars:
                        _stall_allowed = (not cfg.smart_stall_only_underwater) or (pct_change < 0)
                        if _stall_allowed:
                            recent_prices = _opt_prices[-cfg.smart_stall_bars:]
                            range_pct = (max(recent_prices) - min(recent_prices)) / recent_prices[-1]
                            if range_pct < cfg.smart_stall_threshold:
                                exit_price = self.risk_manager.apply_slippage(bar_price, is_entry=False)
                                exit_reason = 'SMART_STALL'
                                _smart_exit = True
                    
                    # 2d. UNDERLYING REVERSAL — the signal thesis has broken
                    #     Only fire if underwater (or flag disabled) — profitable reversals are fine
                    #     Require minimum bars held to avoid reacting to entry-bar noise
                    if not _smart_exit and cfg.smart_underlying_reversal and bars_held >= cfg.smart_reversal_min_bars:
                        _reversal_allowed = (not cfg.smart_reversal_only_underwater) or (pct_change < 0)
                        if _reversal_allowed:
                            bar_time = bar.get('time', '') if hasattr(bar, 'get') else bar['time'] if 'time' in bar.index else ''
                            bar_date = bar.get('date', current_date) if hasattr(bar, 'get') else bar['date'] if 'date' in bar.index else current_date
                            uf = _feat_by_dt.get((bar_date, bar_time))
                            if uf is not None:
                                current_rsi = uf.get('rsi', 50) if hasattr(uf, 'get') else uf['rsi']
                                # Use actual signal thresholds with extra buffer:
                                # CALL thesis breaks when RSI drops BELOW put_threshold - 5
                                # PUT thesis breaks when RSI rises ABOVE call_threshold + 5
                                # This ensures genuine reversal, not just noise around midline
                                if direction == 'CALL' and current_rsi < (cfg.rsi_put_threshold - 5):
                                    exit_price = self.risk_manager.apply_slippage(bar_price, is_entry=False)
                                    exit_reason = 'SMART_REVERSAL'
                                    _smart_exit = True
                                elif direction == 'PUT' and current_rsi > (cfg.rsi_call_threshold + 5):
                                    exit_price = self.risk_manager.apply_slippage(bar_price, is_entry=False)
                                    exit_reason = 'SMART_REVERSAL'
                                    _smart_exit = True
                    
                    if _smart_exit:
                        break
                
                # 3. STOP LOSS (with quick-exit tightening)
                if pct_change <= -current_stop_loss:
                    exit_price = self.risk_manager.apply_slippage(bar_price, is_entry=False)
                    exit_reason = 'STOP'
                    break
                
                # 4. TIME EXIT (direction-specific hold bars) with SMART HOLD EXTENSION
                _hold_limit = cfg.max_hold_bars
                if direction == 'CALL' and cfg.call_max_hold_bars is not None:
                    _hold_limit = cfg.call_max_hold_bars
                elif direction == 'PUT' and cfg.put_max_hold_bars is not None:
                    _hold_limit = cfg.put_max_hold_bars

                # REGIME: extend hold in TRENDING regime (let winners run)
                if _regime_type == 'TRENDING' and cfg.trending_hold_buffer > 0:
                    _hold_limit += cfg.trending_hold_buffer
                
                if bars_held >= _hold_limit:
                    # SMART MOMENTUM EXTENSION: if option is surging, hold a bit longer
                    _force_time_exit = True
                    if cfg.use_smart_exit and cfg.smart_momentum_extend and pct_change > 0:
                        extra_allowed = _hold_limit + cfg.smart_momentum_extend_bars
                        if bars_held < extra_allowed and len(_opt_prices) >= 2:
                            recent_momentum = (_opt_prices[-1] - _opt_prices[-2]) / _opt_prices[-2]
                            if recent_momentum >= cfg.smart_momentum_extend_threshold:
                                _force_time_exit = False  # still surging — hold
                    
                    if _force_time_exit:
                        exit_price = self.risk_manager.apply_slippage(bar_price, is_entry=False)
                        exit_reason = 'TIME'
                        break
            
            if exit_price is None:
                continue
            
            # Calculate P&L
            gross_pnl, commission, net_pnl = self.risk_manager.calculate_trade_pnl(
                entry_price, exit_price, num_contracts
            )
            
            # Record trade
            self.risk_manager.record_trade(current_date, net_pnl)

            # Post-loss correction: arm after first loss this day
            # Smart exits NEVER cascade into post-loss — they are tactical retreats,
            # not evidence of a market regime change that warrants strategy correction
            if cfg.post_loss_strategy != 'none':
                _is_smart_exit = exit_reason.startswith('SMART_') if exit_reason else False
                
                if net_pnl < 0 and not _first_loss_seen.get(current_date, False) and not _is_smart_exit:
                    _first_loss_seen[current_date] = True
                    _post_loss_active[current_date] = True
                    _loss_direction[current_date] = direction  # record which direction lost
                    _loss_exit_reason[current_date] = exit_reason  # STOP, TIME, TRAIL, etc.
                    _loss_bar_idx[current_date] = idx  # bar index for cooldown
            
            rsi = feat['rsi']
            kelly_pct = self.risk_manager.position_sizer.kelly_pct
            
            if verbose:
                emoji = "+" if net_pnl > 0 else "x"
                pl_tag = f"[{cfg.post_loss_strategy.upper()}]" if _post_loss_active.get(current_date, False) else ""
                print(f"  {emoji} {current_date} {current_time} | K={kelly_pct:.0%} | "
                      f"RSI={rsi:.0f} | {direction} {strike:.0f} | {exit_reason} | ${net_pnl:+.2f}{pl_tag}")
            
            trades.append(Trade0DTE(
                date=current_date,
                time=current_time,
                direction=direction,
                strike=strike,
                option_ticker=option_ticker,
                rsi=rsi,
                ml_prob=0.0,  # No ML
                kelly_pct=kelly_pct,
                entry=entry_price,
                exit=exit_price,
                exit_reason=exit_reason,
                bars_held=bars_held,
                num_contracts=num_contracts,
                pnl=net_pnl,
                capital=self.risk_manager.capital
            ))
            
            # Update adaptive PUT filter state after trade completes
            if direction == 'PUT' and cfg.put_adaptive_filter:
                if net_pnl > 0:
                    _put_consec_losses = 0
                else:
                    _put_consec_losses += 1
                    if _put_consec_losses >= cfg.put_loss_streak_threshold:
                        _put_cooldown_remaining = cfg.put_adaptive_cooldown
                        _put_consec_losses = 0
            
            # Update adaptive CALL filter state after trade completes
            if direction == 'CALL' and cfg.call_adaptive_filter:
                if net_pnl > 0:
                    _call_consec_losses = 0
                else:
                    _call_consec_losses += 1
                    if _call_consec_losses >= cfg.call_loss_streak_threshold:
                        _call_cooldown_remaining = cfg.call_adaptive_cooldown
                        _call_consec_losses = 0
            
            # Update direction-aware loss escalation state
            if cfg.use_direction_loss_escalation:
                if net_pnl < 0:
                    _recent_loss_dirs.append(direction)
                    # Check if window is full and a direction dominates
                    if len(_recent_loss_dirs) >= cfg.direction_loss_window:
                        window = _recent_loss_dirs[-cfg.direction_loss_window:]
                        call_count = window.count('CALL')
                        put_count = window.count('PUT')
                        if call_count >= cfg.direction_loss_threshold:
                            _dir_cooldown_call = cfg.direction_loss_cooldown
                        if put_count >= cfg.direction_loss_threshold:
                            _dir_cooldown_put = cfg.direction_loss_cooldown
                else:
                    # Win breaks the consecutive loss streak
                    _recent_loss_dirs.clear()
        
        if verbose:
            print()  # Clear progress line
        print(f"\nTotal signals: {total_signals}")
        print(f"Trades executed: {len(trades)}")
        if cfg.use_regime_detection:
            print(f"Regime: {regime_skipped} signals skipped, {regime_reduced} trades size-reduced")
        if cfg.use_trend_filter:
            print(f"Trend Filter: {trend_skipped} signals skipped, {trend_reduced} trades size-reduced")
        if put_filtered > 0:
            mode = 'uptrend-only' if cfg.put_filter_require_uptrend else 'always'
            print(f"PUT Filter ({mode}): {put_filtered} PUT signals skipped (min_rsi={cfg.put_min_rsi}, skip_days={cfg.put_skip_days}, min_entry_min={cfg.put_min_entry_minutes})")
        if put_adaptive_skipped > 0:
            print(f"PUT Adaptive: {put_adaptive_skipped} PUT signals skipped (streak>={cfg.put_loss_streak_threshold}, cooldown={cfg.put_adaptive_cooldown})")
        if call_adaptive_skipped > 0:
            print(f"CALL Adaptive: {call_adaptive_skipped} CALL signals skipped (streak>={cfg.call_loss_streak_threshold}, cooldown={cfg.call_adaptive_cooldown})")
        if dir_escalation_skipped > 0:
            print(f"Direction Escalation: {dir_escalation_skipped} signals skipped (window={cfg.direction_loss_window}, threshold={cfg.direction_loss_threshold}, cooldown={cfg.direction_loss_cooldown})")
        
        return trades
