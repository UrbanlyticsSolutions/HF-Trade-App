"""
Trend Indicators: SMA, EMA, ADX
"""
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class MAResult:
    """Moving Average result"""
    values: np.ndarray
    current: float
    price_above: bool
    price_below: bool
    slope: float  # Positive = uptrend, Negative = downtrend


@dataclass
class ADXResult:
    """ADX result"""
    adx: np.ndarray
    plus_di: np.ndarray
    minus_di: np.ndarray
    current_adx: float
    trend_strength: str  # weak, moderate, strong, very_strong
    trend_direction: str  # up, down, sideways


class SMA:
    """Simple Moving Average"""
    
    def __init__(self, period: int = 20):
        self.period = period
    
    def calculate(self, closes: np.ndarray) -> MAResult:
        """Calculate SMA"""
        if len(closes) < self.period:
            raise ValueError(f"Need at least {self.period} data points")
        
        sma = np.zeros(len(closes))
        sma[:self.period-1] = np.nan
        
        for i in range(self.period - 1, len(closes)):
            sma[i] = np.mean(closes[i - self.period + 1:i + 1])
        
        current = sma[-1]
        current_price = closes[-1]
        
        # Calculate slope
        slope = (sma[-1] - sma[-5]) / 5 if len(sma) >= 5 else 0
        
        return MAResult(
            values=sma,
            current=current,
            price_above=current_price > current,
            price_below=current_price < current,
            slope=slope,
        )


class EMA:
    """Exponential Moving Average"""
    
    def __init__(self, period: int = 20):
        self.period = period
    
    def calculate(self, closes: np.ndarray) -> MAResult:
        """Calculate EMA"""
        if len(closes) < self.period:
            raise ValueError(f"Need at least {self.period} data points")
        
        alpha = 2 / (self.period + 1)
        ema = np.zeros(len(closes))
        ema[0] = closes[0]
        
        for i in range(1, len(closes)):
            ema[i] = alpha * closes[i] + (1 - alpha) * ema[i-1]
        
        current = ema[-1]
        current_price = closes[-1]
        
        # Calculate slope
        slope = (ema[-1] - ema[-5]) / 5 if len(ema) >= 5 else 0
        
        return MAResult(
            values=ema,
            current=current,
            price_above=current_price > current,
            price_below=current_price < current,
            slope=slope,
        )


class ADX:
    """Average Directional Index - Trend strength indicator"""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> ADXResult:
        """Calculate ADX, +DI, -DI"""
        if len(closes) < self.period * 2:
            raise ValueError(f"Need at least {self.period * 2} data points")
        
        n = len(closes)
        
        # True Range
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
        
        # Directional Movement
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        # Smoothed values
        atr = self._smooth(tr, self.period)
        smooth_plus_dm = self._smooth(plus_dm, self.period)
        smooth_minus_dm = self._smooth(minus_dm, self.period)
        
        # +DI and -DI
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        
        for i in range(n):
            if atr[i] != 0:
                plus_di[i] = (smooth_plus_dm[i] / atr[i]) * 100
                minus_di[i] = (smooth_minus_dm[i] / atr[i]) * 100
        
        # DX
        dx = np.zeros(n)
        for i in range(n):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
        
        # ADX (smoothed DX)
        adx = self._smooth(dx, self.period)
        
        current_adx = adx[-1]
        
        # Trend strength
        if current_adx < 20:
            strength = "weak"
        elif current_adx < 40:
            strength = "moderate"
        elif current_adx < 60:
            strength = "strong"
        else:
            strength = "very_strong"
        
        # Trend direction
        if plus_di[-1] > minus_di[-1]:
            direction = "up"
        elif minus_di[-1] > plus_di[-1]:
            direction = "down"
        else:
            direction = "sideways"
        
        return ADXResult(
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            current_adx=current_adx,
            trend_strength=strength,
            trend_direction=direction,
        )
    
    def _smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing"""
        smoothed = np.zeros(len(data))
        smoothed[:period] = np.nan
        
        if len(data) >= period:
            smoothed[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                smoothed[i] = (smoothed[i-1] * (period - 1) + data[i]) / period
        
        return smoothed
