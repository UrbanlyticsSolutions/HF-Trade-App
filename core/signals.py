"""
Trading Signal Generation for 0DTE Options

Strategies:
- momentum: RSI > 70 → CALL, RSI < 30 → PUT (trend following)
- mean_reversion: RSI < 30 → CALL, RSI > 70 → PUT (fade extremes)
- bb_breakout: Price > BB upper → CALL, Price < BB lower → PUT
- vwap_reversion: Price < VWAP - X% → CALL, Price > VWAP + X% → PUT
- orb: Price breaks opening range high → CALL, breaks low → PUT
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

# Feature columns used
FEATURE_COLS = ['rsi', 'atr_pct', 'bb_position', 'bb_upper', 'bb_lower', 
                'vwap', 'vwap_dev_pct', 'orb_high', 'orb_low', 'momentum', 'vol_ratio']


def compute_features(df: pd.DataFrame, orb_minutes: int = 30) -> pd.DataFrame:
    """Compute all technical features for signal generation."""
    df = df.copy()
    
    # ===== RSI (14-period) =====
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ===== ATR as percentage =====
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100
    
    # ===== Bollinger Bands (20, 2) =====
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_middle'] = sma20
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ===== VWAP =====
    # Handle different column names for timestamp
    timestamp_col = None
    if 'bar_start' in df.columns:
        timestamp_col = 'bar_start'
    elif 'timestamp' in df.columns:
        timestamp_col = 'timestamp'
    
    if timestamp_col:
        df['_ts'] = pd.to_datetime(df[timestamp_col])
        if 'date' not in df.columns:
            df['date'] = df['_ts'].dt.strftime('%Y-%m-%d')
    
    # Calculate cumulative VWAP per day
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_volume'] = df['typical_price'] * df['volume']
    
    if 'date' in df.columns:
        df['cum_tp_vol'] = df.groupby('date')['tp_volume'].cumsum()
        df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    else:
        df['cum_tp_vol'] = df['tp_volume'].cumsum()
        df['cum_vol'] = df['volume'].cumsum()
    
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol'].replace(0, np.nan)
    df['vwap_dev_pct'] = (df['close'] - df['vwap']) / df['vwap'] * 100
    
    # ===== Opening Range (ORB) =====
    # Calculate the high/low of first N minutes each day
    if '_ts' in df.columns and 'date' in df.columns:
        df['minutes_from_open'] = (df['_ts'].dt.hour - 9) * 60 + df['_ts'].dt.minute - 30
        
        # Get ORB (first orb_minutes of day)
        orb_mask = df['minutes_from_open'] < orb_minutes
        orb_data = df[orb_mask].groupby('date').agg({
            'high': 'max',
            'low': 'min'
        }).rename(columns={'high': 'orb_high', 'low': 'orb_low'})
        
        # Merge ORB data back
        df = df.merge(orb_data, on='date', how='left', suffixes=('', '_orb'))
    else:
        df['orb_high'] = np.nan
        df['orb_low'] = np.nan
        df['minutes_from_open'] = 0
    
    # ===== Momentum =====
    df['momentum'] = df['close'].pct_change(5) * 100
    
    # ===== Volume ratio =====
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    return df


def compute_rolling_volatility(df: pd.DataFrame, window: int = 20, lookback_days: int = None) -> pd.Series:
    """Compute rolling volatility."""
    if lookback_days:
        window = lookback_days
    returns = df['close'].pct_change()
    return returns.rolling(window).std() * np.sqrt(252)


def get_basic_signal(
    features: Dict,
    rsi_call_threshold: float = 70,
    rsi_put_threshold: float = 30,
    strategy: str = "momentum",
    bb_buffer_pct: float = 0.0,
    vwap_dev_threshold: float = 0.3,
    orb_buffer_pct: float = 0.1,
) -> Optional[str]:
    """
    Generate trading signal based on strategy type.
    
    Strategies:
    - momentum: RSI > 70 → CALL, RSI < 30 → PUT
    - mean_reversion: RSI < 30 → CALL, RSI > 70 → PUT
    - bb_breakout: Price breaks Bollinger Band → trade breakout direction
    - vwap_reversion: Price deviates from VWAP → fade the move
    - orb: Price breaks opening range → trade breakout direction
    """
    price = features.get('close', 0)
    rsi = features.get('rsi', 50)
    
    # ===== MOMENTUM =====
    if strategy == "momentum":
        if rsi > rsi_call_threshold:
            return 'CALL'
        elif rsi < rsi_put_threshold:
            return 'PUT'
    
    # ===== MEAN REVERSION =====
    elif strategy == "mean_reversion":
        if rsi < rsi_call_threshold:  # Oversold → expect bounce
            return 'CALL'
        elif rsi > rsi_put_threshold:  # Overbought → expect drop
            return 'PUT'
    
    # ===== BOLLINGER BAND BREAKOUT =====
    elif strategy == "bb_breakout":
        bb_upper = features.get('bb_upper', 0)
        bb_lower = features.get('bb_lower', 0)
        
        if bb_upper and bb_lower and not np.isnan(bb_upper) and not np.isnan(bb_lower):
            bb_range = bb_upper - bb_lower
            buffer = bb_range * bb_buffer_pct
            
            if price > bb_upper + buffer:  # Breakout above
                return 'CALL'
            elif price < bb_lower - buffer:  # Breakout below
                return 'PUT'
    
    # ===== VWAP REVERSION =====
    elif strategy == "vwap_reversion":
        vwap_dev = features.get('vwap_dev_pct', 0)
        
        if vwap_dev is not None and not np.isnan(vwap_dev):
            if vwap_dev < -vwap_dev_threshold:  # Price below VWAP → expect bounce up
                return 'CALL'
            elif vwap_dev > vwap_dev_threshold:  # Price above VWAP → expect drop
                return 'PUT'
    
    # ===== OPENING RANGE BREAKOUT =====
    elif strategy == "orb":
        orb_high = features.get('orb_high', 0)
        orb_low = features.get('orb_low', 0)
        minutes_from_open = features.get('minutes_from_open', 0)
        
        # Only trade ORB after the opening range period
        if minutes_from_open >= 30:  # After ORB period
            if orb_high and orb_low and not np.isnan(orb_high) and not np.isnan(orb_low):
                orb_range = orb_high - orb_low
                buffer = orb_range * orb_buffer_pct
                
                if price > orb_high + buffer:  # Breakout above ORB
                    return 'CALL'
                elif price < orb_low - buffer:  # Breakout below ORB
                    return 'PUT'
    
    return None


@dataclass
class StrategyConfig:
    """Configuration for each strategy type."""
    name: str
    description: str
    default_params: Dict
    

STRATEGIES = {
    'momentum': StrategyConfig(
        name='RSI Momentum',
        description='Trade with momentum: RSI > 70 → CALL, RSI < 30 → PUT',
        default_params={'rsi_call_threshold': 70, 'rsi_put_threshold': 30}
    ),
    'mean_reversion': StrategyConfig(
        name='RSI Mean Reversion',
        description='Fade extremes: RSI < 30 → CALL, RSI > 70 → PUT',
        default_params={'rsi_call_threshold': 30, 'rsi_put_threshold': 70}
    ),
    'bb_breakout': StrategyConfig(
        name='Bollinger Band Breakout',
        description='Trade breakouts: Price > BB upper → CALL, < BB lower → PUT',
        default_params={'bb_buffer_pct': 0.0}
    ),
    'vwap_reversion': StrategyConfig(
        name='VWAP Reversion',
        description='Fade VWAP: Price < VWAP-0.3% → CALL, > VWAP+0.3% → PUT',
        default_params={'vwap_dev_threshold': 0.3}
    ),
    'orb': StrategyConfig(
        name='Opening Range Breakout',
        description='Trade ORB breakouts (first 30 min high/low)',
        default_params={'orb_buffer_pct': 0.1, 'orb_minutes': 30}
    ),
}


def list_strategies():
    """Print available strategies."""
    print("\nAvailable Strategies:")
    print("-" * 60)
    for key, cfg in STRATEGIES.items():
        print(f"  {key:20s} - {cfg.description}")
    print()


# ============================================================
# STUB ML CLASSES (for backward compatibility - not used)
# ============================================================

class TradingMLModel:
    """Stub ML model - not used in simplified backtesting."""
    
    def __init__(self):
        self.model = None
        self.kelly_pct = 0.07
        self.is_trained = False
    
    def train(self, data):
        self.is_trained = True
    
    def predict(self, features, entry_price, direction):
        return 0.5
    
    def predict_proba(self, features: Dict) -> float:
        return 0.5
    
    def save(self, path: str, kelly_pct: float = None):
        pass
    
    @classmethod
    def load(cls, path: str):
        return cls(), 0.07


class DayFilterModel:
    """Stub day filter - not used in simplified backtesting."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.is_trained = False
    
    def train(self, day_features, trade_results):
        self.is_trained = True
    
    def predict(self, features: Dict) -> bool:
        return True
    
    def should_trade_today(self, features: Dict):
        return True, 0.7
    
    def save(self, path: str):
        pass
    
    @classmethod
    def load(cls, path: str):
        return cls()
