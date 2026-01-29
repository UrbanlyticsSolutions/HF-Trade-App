"""
Volume Indicators: Volume Profile, VWAP, OBV
"""
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class VolumeResult:
    """Volume analysis result"""
    current_volume: float
    avg_volume: float
    volume_ratio: float  # Current / Average
    is_high_volume: bool  # > 2x average
    is_low_volume: bool  # < 0.5x average
    volume_trend: str  # increasing, decreasing, stable


@dataclass
class VWAPResult:
    """VWAP result"""
    values: np.ndarray
    current: float
    price_above: bool
    price_below: bool
    deviation: float  # % from VWAP


@dataclass
class OBVResult:
    """On Balance Volume result"""
    values: np.ndarray
    current: float
    trend: str  # up, down, sideways
    divergence: str  # bullish, bearish, or None


class VolumeProfile:
    """Volume analysis"""
    
    def __init__(self, period: int = 20):
        self.period = period
    
    def calculate(self, volumes: np.ndarray) -> VolumeResult:
        """Analyze volume"""
        if len(volumes) < self.period:
            raise ValueError(f"Need at least {self.period} data points")
        
        current = volumes[-1]
        avg = np.mean(volumes[-self.period:])
        ratio = current / avg if avg > 0 else 1
        
        # Volume trend
        recent_avg = np.mean(volumes[-5:])
        older_avg = np.mean(volumes[-self.period:-5]) if len(volumes) > 5 else recent_avg
        
        if recent_avg > older_avg * 1.2:
            trend = "increasing"
        elif recent_avg < older_avg * 0.8:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return VolumeResult(
            current_volume=current,
            avg_volume=avg,
            volume_ratio=ratio,
            is_high_volume=ratio > 2.0,
            is_low_volume=ratio < 0.5,
            volume_trend=trend,
        )
    
    def is_volume_spike(self, volumes: np.ndarray, threshold: float = 2.0) -> bool:
        """Check if current bar has volume spike"""
        if len(volumes) < self.period:
            return False
        
        avg = np.mean(volumes[-self.period:-1])
        return volumes[-1] > avg * threshold


class VWAP:
    """Volume Weighted Average Price"""
    
    def calculate(self, highs: np.ndarray, lows: np.ndarray, 
                  closes: np.ndarray, volumes: np.ndarray) -> VWAPResult:
        """Calculate VWAP"""
        typical_price = (highs + lows + closes) / 3
        
        cumulative_tp_vol = np.cumsum(typical_price * volumes)
        cumulative_vol = np.cumsum(volumes)
        
        vwap = np.zeros(len(closes))
        for i in range(len(closes)):
            if cumulative_vol[i] > 0:
                vwap[i] = cumulative_tp_vol[i] / cumulative_vol[i]
            else:
                vwap[i] = closes[i]
        
        current = vwap[-1]
        current_price = closes[-1]
        
        deviation = (current_price - current) / current * 100 if current > 0 else 0
        
        return VWAPResult(
            values=vwap,
            current=current,
            price_above=current_price > current,
            price_below=current_price < current,
            deviation=deviation,
        )


class OBV:
    """On Balance Volume"""
    
    def calculate(self, closes: np.ndarray, volumes: np.ndarray) -> OBVResult:
        """Calculate OBV"""
        obv = np.zeros(len(closes))
        obv[0] = volumes[0]
        
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
        
        # OBV trend
        recent = obv[-10:] if len(obv) >= 10 else obv
        slope = (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0
        
        if slope > 0:
            trend = "up"
        elif slope < 0:
            trend = "down"
        else:
            trend = "sideways"
        
        # Check divergence
        divergence = self._check_divergence(closes, obv)
        
        return OBVResult(
            values=obv,
            current=obv[-1],
            trend=trend,
            divergence=divergence,
        )
    
    def _check_divergence(self, prices: np.ndarray, obv: np.ndarray, lookback: int = 20) -> str:
        """Check for OBV divergence"""
        if len(prices) < lookback:
            return None
        
        recent_prices = prices[-lookback:]
        recent_obv = obv[-lookback:]
        
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        obv_change = (recent_obv[-1] - recent_obv[0]) / abs(recent_obv[0]) if recent_obv[0] != 0 else 0
        
        # Price down, OBV up = bullish divergence
        if price_change < -0.01 and obv_change > 0.1:
            return "bullish"
        # Price up, OBV down = bearish divergence
        elif price_change > 0.01 and obv_change < -0.1:
            return "bearish"
        
        return None
