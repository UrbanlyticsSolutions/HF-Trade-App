"""
Intraday ML-Enhanced Trend Following Strategy
Uses moving averages for trend direction and ML for entry/exit optimization
"""
import numpy as np
from collections import deque
from datetime import datetime
import logging
import pandas as pd

from strategy.risk_manager import RiskLimits
from price_prediction_model import PricePredictionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MLTrendFollowing')

class MLTrendFollowingStrategy:
    """
    Intraday trend-following strategy with ML-optimized entry/exit
    """
    
    def __init__(self, symbol, risk_limits, ml_entry_threshold=0.65, ml_exit_threshold=0.60):
        self.symbol = symbol
        self.risk_limits = risk_limits
        
        # Trend parameters (intraday)
        self.fast_ma_period = 20  # ~20 minutes
        self.slow_ma_period = 50  # ~50 minutes
        
        # ML thresholds (kept for compatibility, though MLTradeClassifier is removed)
        self.ml_entry_threshold = ml_entry_threshold
        self.ml_exit_threshold = ml_exit_threshold
        
        # Data storage
        self.prices = deque(maxlen=100)
        self.volumes = deque(maxlen=100)
        self.highs = deque(maxlen=100)
        self.lows = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)
        
        # Price Prediction Model
        try:
            self.price_predictor = PricePredictionModel()
            self.price_predictor.load_model()
            logger.info("Loaded Price Prediction model")
        except:
            logger.warning("Price Prediction model not found, skipping price confirmation")
            self.price_predictor = None
        
        # Position tracking
        self.position = None
        self.entry_price = None
        self.entry_time = None
        
        # Performance tracking
        self.trades = []
        self.current_time = None
        
    def calculate_ma(self, period):
        """Calculate moving average"""
        if len(self.prices) < period:
            return None
        return np.mean(list(self.prices)[-period:])
    
    def get_trend_direction(self):
        """
        Determine trend direction using MAs
        Returns: 1 (bullish), -1 (bearish), 0 (neutral)
        """
        fast_ma = self.calculate_ma(self.fast_ma_period)
        slow_ma = self.calculate_ma(self.slow_ma_period)
        
        if fast_ma is None or slow_ma is None:
            return 0
        
        # Trend strength
        price = self.prices[-1]
        
        # Bullish: price > fast_ma > slow_ma
        if price > fast_ma and fast_ma > slow_ma:
            return 1
        # Bearish: price < fast_ma < slow_ma
        elif price < fast_ma and fast_ma < slow_ma:
            return -1
        else:
            return 0
    
    def calculate_rsi(self, period=14):
        """Calculate RSI"""
        if len(self.prices) < period + 1:
            return 50.0
        
        deltas = np.diff(list(self.prices)[-(period+1):])
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down if down != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, period=14):
        """Calculate ATR"""
        if len(self.prices) < period + 1:
            return None
        
        prices = list(self.prices)[-(period+1):]
        tr_list = []
        for i in range(1, len(prices)):
            high_low = abs(prices[i] - prices[i-1])
            tr_list.append(high_low)
        
        return np.mean(tr_list) if tr_list else None
    
    def calculate_adx(self, period=14):
        """Calculate ADX"""
        if len(self.prices) < period * 2:
            return 0
        
        highs = list(self.highs)
        lows = list(self.lows)
        prices = list(self.prices)
        
        tr_list = []
        plus_dm_list = []
        minus_dm_list = []
        
        for i in range(1, len(prices)):
            h = highs[i]
            l = lows[i]
            prev_h = highs[i-1]
            prev_l = lows[i-1]
            prev_c = prices[i-1]
            
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            tr_list.append(tr)
            
            plus_dm = h - prev_h
            minus_dm = prev_l - l
            
            if plus_dm > minus_dm and plus_dm > 0:
                plus_dm_list.append(plus_dm)
            else:
                plus_dm_list.append(0)
                
            if minus_dm > plus_dm and minus_dm > 0:
                minus_dm_list.append(minus_dm)
            else:
                minus_dm_list.append(0)
        
        if not tr_list:
            return 0
            
        tr_smooth = np.mean(tr_list[-period:])
        plus_dm_smooth = np.mean(plus_dm_list[-period:])
        minus_dm_smooth = np.mean(minus_dm_list[-period:])
        
        if tr_smooth == 0:
            return 0
            
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        
        return dx

    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        if len(self.prices) < slow + signal:
            return 0, 0, 0
            
        prices = list(self.prices)
        # Simple EMA implementation
        def ema(data, span):
            return pd.Series(data).ewm(span=span, adjust=False).mean().values
            
        emas_fast = ema(prices, fast)
        emas_slow = ema(prices, slow)
        macd_line = emas_fast - emas_slow
        signal_line = ema(macd_line, signal)
        hist = macd_line - signal_line
        
        return macd_line[-1], signal_line[-1], hist[-1]

    def get_price_prediction_features(self):
        """Extract features for Price Prediction Model"""
        if len(self.prices) < 20:
            return None
            
        price = self.prices[-1]
        
        # Calculate basic features needed for the model
        # Note: This is a simplified extraction matching the model's expectations
        features = {
            'price_change_1': (price - list(self.prices)[-2]) if len(self.prices) >= 2 else 0,
            'price_change_3': (price - list(self.prices)[-4]) if len(self.prices) >= 4 else 0,
            'price_change_5': (price - list(self.prices)[-6]) if len(self.prices) >= 6 else 0,
            'price_volatility': np.std(list(self.prices)[-10:]) if len(self.prices) >= 10 else 0,
            'ma_5': self.calculate_ma(5) or price,
            'ma_10': self.calculate_ma(10) or price,
            'ma_20': self.calculate_ma(20) or price,
            'price_to_ma5': price / (self.calculate_ma(5) or price),
            'price_to_ma20': price / (self.calculate_ma(20) or price),
            'rsi': self.calculate_rsi(),
            'macd': self.calculate_macd()[0],
            'macd_hist': self.calculate_macd()[2],
            'volume_ratio': self.volumes[-1] / np.mean(list(self.volumes)[-20:]) if len(self.volumes) >= 20 else 1.0,
            'volume_change': (self.volumes[-1] - list(self.volumes)[-2]) if len(self.volumes) >= 2 else 0,
            'adx': self.calculate_adx(),
            'trend_strength': abs(self.calculate_ma(5) - self.calculate_ma(20)) if self.calculate_ma(20) else 0,
            'hour': self.current_time.hour if self.current_time else 12,
            'minute': self.current_time.minute if self.current_time else 0
        }
        
        # Convert to list as expected by the model
        return [features.get(k, 0) for k in self.price_predictor.feature_names]
    
    def should_enter(self, trend_direction):
        """
        Determine if we should enter a position
        Uses ONLY Price Predictor to filter entries (Pure Trend/Price Focus)
        """
        # Check Price Prediction Model (Direction Confirmation)
        if self.price_predictor is not None:
            pp_features = self.get_price_prediction_features()
            if pp_features is not None:
                predicted_direction = self.price_predictor.predict(pp_features)
                # 0: DOWN, 1: NEUTRAL, 2: UP
                
                if trend_direction == 1:  # Bullish Trend
                    # Require UP (2) or NEUTRAL (1), reject DOWN (0)
                    if predicted_direction == 0:
                        logger.info("Entry rejected: Price Predictor sees DOWN trend")
                        return False
                        
                elif trend_direction == -1:  # Bearish Trend
                    # Require DOWN (0) or NEUTRAL (1), reject UP (2)
                    if predicted_direction == 2:
                        logger.info("Entry rejected: Price Predictor sees UP trend")
                        return False
        
        return True
    
    def should_exit(self):
        """
        Determine if we should exit current position
        Uses ML model + risk management
        """
        if self.position is None:
            return False
        
        price = self.prices[-1]
        atr = self.calculate_atr()
        
        # Hard stop-loss
        if self.position == 'LONG':
            pnl = price - self.entry_price
            if pnl < -1.5 * atr:  # Stop loss
                return True
        else:  # SHORT
            pnl = self.entry_price - price
            if pnl < -1.5 * atr:  # Stop loss
                return True
        
        # ML-based exit (optional, can be enhanced)
        # For now, use simple profit target
        if abs(pnl) > 2.0 * atr:  # Take profit
            return True
        
        return False
    
    def on_tick(self, tick):
        """Process new tick data"""
        self.current_time = tick['timestamp']
        self.prices.append(tick['price'])
        self.volumes.append(tick['volume'])
        # Handle cases where tick might not have high/low (e.g. simple trade tick)
        # If it's a candle, it has them. If it's a trade, high=low=price
        self.highs.append(tick.get('high', tick['price']))
        self.lows.append(tick.get('low', tick['price']))
        self.timestamps.append(tick['timestamp'])
        
        # Need minimum data
        if len(self.prices) < self.slow_ma_period:
            return
        
        # Check for exit first
        if self.position is not None:
            if self.should_exit():
                self.close_position(tick['price'])
                return
        
        # Check for entry
        if self.position is None:
            trend = self.get_trend_direction()
            
            if trend != 0:  # Trend exists
                if self.should_enter(trend):
                    if trend == 1:
                        self.open_position('LONG', tick['price'])
                    elif trend == -1:
                        self.open_position('SHORT', tick['price'])
    
    def open_position(self, direction, price):
        """Open a new position"""
        self.position = direction
        self.entry_price = price
        self.entry_time = self.current_time
        logger.info(f"OPEN {direction} at {price:.2f} | Time: {self.current_time}")
    
    def close_position(self, price):
        """Close current position"""
        if self.position == 'LONG':
            pnl = price - self.entry_price
        else:  # SHORT
            pnl = self.entry_price - price
        
        trade = {
            'entry_time': self.entry_time,
            'exit_time': self.current_time,
            'direction': self.position,
            'entry_price': self.entry_price,
            'exit_price': price,
            'pnl': pnl,
            'profitable': 1 if pnl > 0 else 0
        }
        
        self.trades.append(trade)
        logger.info(f"CLOSE {self.position} at {price:.2f} | PnL: ${pnl:.2f}")
        
        self.position = None
        self.entry_price = None
        self.entry_time = None
    
    def get_performance_stats(self):
        """Calculate performance statistics"""
        if not self.trades:
            return {}
        
        wins = [t for t in self.trades if t['profitable'] == 1]
        losses = [t for t in self.trades if t['profitable'] == 0]
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t['pnl']) for t in losses]) if losses else 0
        
        profit_factor = (sum(t['pnl'] for t in wins) / sum(abs(t['pnl']) for t in losses)) if losses else 0
        
        return {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
