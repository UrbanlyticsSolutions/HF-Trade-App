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
sys.path.insert(0, '.')

from core.signals import (
    compute_features, 
    compute_rolling_volatility,
    get_basic_signal,
    TradingMLModel,
    DayFilterModel,
)
from core.risk_manager import RiskManager, RiskConfig
from clients.database import MarketDatabase


@dataclass
class TradeConfig:
    """0DTE trading configuration"""
    # Strategy type: momentum, mean_reversion, bb_breakout, vwap_reversion, orb
    strategy: str = "momentum"
    
    # Entry timing
    trade_start_hour: int = 10
    trade_start_minute: int = 5      # Skip first 5 mins (10:00 has 44% WR vs 75%+ later)
    trade_end_hour: int = 11
    trade_end_minute: int = 30
    
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
    profit_low_vol: float = 0.20       # Profit target when VIX < 15
    profit_mid_vol: float = 0.25       # Profit target when VIX 15-25
    profit_high_vol: float = 0.35      # Profit target when VIX > 25
    
    # Volatility-based STOP LOSSES
    stop_low_vol: float = 0.18         # Stop loss when VIX < 15 (calm)
    stop_mid_vol: float = 0.28         # Stop loss when VIX 15-25 (normal)
    stop_high_vol: float = 0.40        # Stop loss when VIX > 25 (volatile)
    
    # VIX thresholds
    vix_low_threshold: float = 15.0    # VIX below this = low vol
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
        initial_capital: float = 10000
    ):
        self.trade_config = trade_config or TradeConfig()
        self.risk_config = risk_config or RiskConfig()
        self.initial_capital = initial_capital
        
        # Components
        self.risk_manager = RiskManager(initial_capital, self.risk_config)
        self.ml_model: Optional[TradingMLModel] = None
        self.day_filter: Optional[DayFilterModel] = None
        
        # Data
        self.db = MarketDatabase()
        
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
        
        return underlying_df, options_df, features_df
    
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
        
        # Get volatility values for percentile calc
        all_vols = list(rolling_volatility.values())
        
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
            
            # Use HISTORICAL volatility filter (no look-ahead)
            if not self.is_volatile_enough(current_date, rolling_volatility, all_vols):
                continue
            
            feat = features_df.iloc[idx]
            rsi = feat['rsi']
            momentum_3 = feat['momentum_3']
            
            for direction in ['CALL', 'PUT']:
                if direction == 'CALL' and momentum_3 < 0.1:
                    continue
                if direction == 'PUT' and momentum_3 > -0.1:
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
                    'date': current_date,  # Include date for day filter training
                    # Top features (kept)
                    'momentum_3': momentum_3,
                    'trend_strength': feat['trend_strength'],
                    'entry_price': entry_price,
                    'volatility_short': feat['volatility_short'],
                    'stoch_d': feat['stoch_d'],
                    'bb_width': feat['bb_width'],
                    'volatility': feat['volatility'],
                    'dist_from_low': feat['dist_from_low'],
                    'price_vs_sma20': feat['price_vs_sma20'],
                    'stoch_k': feat['stoch_k'],
                    'rsi': rsi,
                    'cci': feat['cci'],
                    'roc_5': feat['roc_5'],
                    'dist_from_high': feat['dist_from_high'],
                    'volume_ratio': feat['volume_ratio'],
                    'hour': current_hour,
                    # NEW features
                    'vwap_distance': feat.get('vwap_distance', 0),
                    'atr_ratio': feat.get('atr_ratio', 1),
                    'direction_put': 1 if direction == 'PUT' else 0,  # PUT has higher win rate
                    # Output
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
                momentum_3 = feat['momentum_3']
                
                for direction in ['CALL', 'PUT']:
                    if direction == 'CALL' and momentum_3 < 0.1:
                        continue
                    if direction == 'PUT' and momentum_3 > -0.1:
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
        """Find best option contract (slightly ITM)"""
        cfg = self.trade_config
        
        mask = (
            (options_df['date'] == date) &
            (options_df['option_type'] == option_type) &
            (options_df['close'] >= cfg.min_option_price) &
            (options_df['close'] <= cfg.max_option_price) &
            (options_df['time'] <= entry_time)
        )
        available = options_df[mask].copy()
        
        if available.empty:
            return None
        
        # Prefer slightly ITM
        if option_type == 'call':
            itm = available[available['strike'] < underlying_price]
            if not itm.empty:
                itm = itm.copy()
                itm['strike_diff'] = underlying_price - itm['strike']
                return itm.loc[itm['strike_diff'].idxmin()]
        else:
            itm = available[available['strike'] > underlying_price]
            if not itm.empty:
                itm = itm.copy()
                itm['strike_diff'] = itm['strike'] - underlying_price
                return itm.loc[itm['strike_diff'].idxmin()]
        
        available['strike_diff'] = abs(available['strike'] - underlying_price)
        return available.loc[available['strike_diff'].idxmin()]
    
    def _get_future_bars(
        self,
        options_df: pd.DataFrame,
        option_ticker: str,
        date: str,
        entry_time: str
    ) -> pd.DataFrame:
        """Get future bars for an option after entry"""
        mask = (
            (options_df['option_ticker'] == option_ticker) &
            (options_df['date'] == date) &
            (options_df['time'] > entry_time)
        )
        return options_df[mask].sort_values('time')
    
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
        
        # Compute VIX proxy from morning volatility (no look-ahead)
        vix_proxy = self.compute_morning_vol_for_vix(underlying_df)
        
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
            minute = row.get('minute', 0)
            
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
            
            # Get signal from strategy
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
            
            # Position sizing (PASS STOP LOSS for proper risk calculation)
            num_contracts, _ = self.risk_manager.get_position_size(
                entry_price, 
                ml_confidence=None,
                stop_loss_pct=stop_loss
            )
            
            # Skip if position too small
            if num_contracts == 0:
                continue
            
            # ============================================================
            # SIMULATE EXIT with ADAPTIVE TARGETS + TRAILING STOP
            # ============================================================
            exit_price = None
            exit_reason = None
            bars_held = 0
            
            # Trailing stop tracking
            max_profit_pct = 0.0
            trailing_stop_price = None
            breakeven_activated = False
            
            for bar_idx, (_, bar) in enumerate(future_bars.iterrows()):
                bars_held = bar_idx + 1
                bar_price = bar['close']
                
                pct_change = (bar_price - entry_price) / entry_price
                
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
                
                # 3. STOP LOSS (with quick-exit tightening)
                if pct_change <= -current_stop_loss:
                    exit_price = self.risk_manager.apply_slippage(bar_price, is_entry=False)
                    exit_reason = 'STOP'
                    break
                
                # 4. TIME EXIT
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
                print(f"  {emoji} {current_date} {current_time} | K={kelly_pct:.0%} | "
                      f"RSI={rsi:.0f} | {direction} {strike:.0f} | {exit_reason} | ${net_pnl:+.2f}")
            
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
        
        if verbose:
            print()  # Clear progress line
        print(f"\nTotal signals: {total_signals}")
        print(f"Trades executed: {len(trades)}")
        
        return trades
