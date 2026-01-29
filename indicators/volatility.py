"""
Volatility Indicators: ATR, Bollinger Bands
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class ATRResult:
    """ATR result"""
    values: np.ndarray
    current: float
    current_pct: float  # ATR as % of price
    volatility_level: str  # low, normal, high, extreme


@dataclass
class BollingerResult:
    """Bollinger Bands result"""
    upper: np.ndarray
    middle: np.ndarray
    lower: np.ndarray
    current_upper: float
    current_middle: float
    current_lower: float
    bandwidth: float
    percent_b: float  # Position within bands (0-1)
    squeeze: bool  # Low volatility squeeze


class ATR:
    """Average True Range"""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> ATRResult:
        """Calculate ATR"""
        if len(closes) < self.period + 1:
            raise ValueError(f"Need at least {self.period + 1} data points")
        
        # True Range
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]
        
        for i in range(1, len(closes)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
        
        # ATR (Wilder's smoothing)
        atr = np.zeros(len(closes))
        atr[:self.period] = np.nan
        atr[self.period-1] = np.mean(tr[:self.period])
        
        for i in range(self.period, len(closes)):
            atr[i] = (atr[i-1] * (self.period - 1) + tr[i]) / self.period
        
        current = atr[-1]
        current_pct = (current / closes[-1]) * 100 if closes[-1] > 0 else 0
        
        # Volatility level
        avg_atr_pct = np.mean([atr[i] / closes[i] for i in range(-20, 0) if not np.isnan(atr[i])]) * 100
        
        if current_pct < avg_atr_pct * 0.7:
            level = "low"
        elif current_pct < avg_atr_pct * 1.3:
            level = "normal"
        elif current_pct < avg_atr_pct * 2.0:
            level = "high"
        else:
            level = "extreme"
        
        return ATRResult(
            values=atr,
            current=current,
            current_pct=current_pct,
            volatility_level=level,
        )


class BollingerBands:
    """Bollinger Bands"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
    
    def calculate(self, closes: np.ndarray) -> BollingerResult:
        """Calculate Bollinger Bands"""
        if len(closes) < self.period:
            raise ValueError(f"Need at least {self.period} data points")
        
        n = len(closes)
        
        middle = np.zeros(n)
        upper = np.zeros(n)
        lower = np.zeros(n)
        
        middle[:self.period-1] = np.nan
        upper[:self.period-1] = np.nan
        lower[:self.period-1] = np.nan
        
        for i in range(self.period - 1, n):
            window = closes[i - self.period + 1:i + 1]
            sma = np.mean(window)
            std = np.std(window)
            
            middle[i] = sma
            upper[i] = sma + (self.std_dev * std)
            lower[i] = sma - (self.std_dev * std)
        
        current_price = closes[-1]
        current_upper = upper[-1]
        current_middle = middle[-1]
        current_lower = lower[-1]
        
        # Bandwidth
        bandwidth = (current_upper - current_lower) / current_middle if current_middle > 0 else 0
        
        # %B (position within bands)
        band_range = current_upper - current_lower
        percent_b = (current_price - current_lower) / band_range if band_range > 0 else 0.5
        
        # Squeeze detection (low bandwidth)
        recent_bandwidths = []
        for i in range(-20, 0):
            if not np.isnan(upper[i]) and not np.isnan(lower[i]) and middle[i] > 0:
                bw = (upper[i] - lower[i]) / middle[i]
                recent_bandwidths.append(bw)
        
        avg_bandwidth = np.mean(recent_bandwidths) if recent_bandwidths else bandwidth
        squeeze = bandwidth < avg_bandwidth * 0.7
        
        return BollingerResult(
            upper=upper,
            middle=middle,
            lower=lower,
            current_upper=current_upper,
            current_middle=current_middle,
            current_lower=current_lower,
            bandwidth=bandwidth,
            percent_b=percent_b,
            squeeze=squeeze,
        )
