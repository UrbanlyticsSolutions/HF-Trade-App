"""
Momentum Indicators: RSI, MACD, Stochastic
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class RSIResult:
    """RSI calculation result"""
    values: np.ndarray
    current: float
    is_oversold: bool  # < 30
    is_overbought: bool  # > 70
    divergence: Optional[str]  # bullish, bearish, or None


@dataclass
class MACDResult:
    """MACD calculation result"""
    macd_line: np.ndarray
    signal_line: np.ndarray
    histogram: np.ndarray
    current_macd: float
    current_signal: float
    current_hist: float
    crossover: Optional[str]  # bullish, bearish, or None


@dataclass
class StochasticResult:
    """Stochastic calculation result"""
    k_line: np.ndarray
    d_line: np.ndarray
    current_k: float
    current_d: float
    is_oversold: bool  # < 20
    is_overbought: bool  # > 80
    crossover: Optional[str]


class RSI:
    """Relative Strength Index"""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, closes: np.ndarray) -> RSIResult:
        """Calculate RSI"""
        if len(closes) < self.period + 1:
            raise ValueError(f"Need at least {self.period + 1} data points")
        
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate RSI for each point
        rsi_values = np.zeros(len(closes) - 1)
        
        # First RSI
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])
        
        for i in range(self.period, len(deltas)):
            avg_gain = (avg_gain * (self.period - 1) + gains[i]) / self.period
            avg_loss = (avg_loss * (self.period - 1) + losses[i]) / self.period
            
            if avg_loss == 0:
                rsi_values[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100 - (100 / (1 + rs))
        
        current = rsi_values[-1]
        
        # Check for divergence
        divergence = self._check_divergence(closes, rsi_values)
        
        return RSIResult(
            values=rsi_values,
            current=current,
            is_oversold=current < 30,
            is_overbought=current > 70,
            divergence=divergence,
        )
    
    def _check_divergence(self, prices: np.ndarray, rsi: np.ndarray, lookback: int = 20) -> Optional[str]:
        """Check for RSI divergence"""
        if len(prices) < lookback or len(rsi) < lookback:
            return None
        
        recent_prices = prices[-lookback:]
        recent_rsi = rsi[-lookback:]
        
        # Price making lower lows but RSI making higher lows = bullish divergence
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        rsi_trend = recent_rsi[-1] - recent_rsi[0]
        
        if price_trend < -0.01 and rsi_trend > 5:
            return "bullish"
        elif price_trend > 0.01 and rsi_trend < -5:
            return "bearish"
        
        return None


class MACD:
    """Moving Average Convergence Divergence"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def calculate(self, closes: np.ndarray) -> MACDResult:
        """Calculate MACD"""
        if len(closes) < self.slow + self.signal:
            raise ValueError(f"Need at least {self.slow + self.signal} data points")
        
        # Calculate EMAs
        ema_fast = self._ema(closes, self.fast)
        ema_slow = self._ema(closes, self.slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        signal_line = self._ema(macd_line[~np.isnan(macd_line)], self.signal)
        
        # Pad signal line to match length
        padded_signal = np.full(len(macd_line), np.nan)
        padded_signal[-len(signal_line):] = signal_line
        
        # Histogram
        histogram = macd_line - padded_signal
        
        # Check crossover
        crossover = None
        if len(histogram) >= 2:
            if histogram[-2] < 0 and histogram[-1] > 0:
                crossover = "bullish"
            elif histogram[-2] > 0 and histogram[-1] < 0:
                crossover = "bearish"
        
        return MACDResult(
            macd_line=macd_line,
            signal_line=padded_signal,
            histogram=histogram,
            current_macd=macd_line[-1],
            current_signal=padded_signal[-1],
            current_hist=histogram[-1] if not np.isnan(histogram[-1]) else 0,
            crossover=crossover,
        )
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA"""
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema


class Stochastic:
    """Stochastic Oscillator"""
    
    def __init__(self, k_period: int = 14, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
    
    def calculate(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> StochasticResult:
        """Calculate Stochastic %K and %D"""
        if len(closes) < self.k_period + self.d_period:
            raise ValueError(f"Need at least {self.k_period + self.d_period} data points")
        
        k_values = np.zeros(len(closes))
        
        for i in range(self.k_period - 1, len(closes)):
            highest = np.max(highs[i - self.k_period + 1:i + 1])
            lowest = np.min(lows[i - self.k_period + 1:i + 1])
            
            if highest == lowest:
                k_values[i] = 50
            else:
                k_values[i] = ((closes[i] - lowest) / (highest - lowest)) * 100
        
        # %D is SMA of %K
        d_values = np.zeros(len(closes))
        for i in range(self.k_period + self.d_period - 2, len(closes)):
            d_values[i] = np.mean(k_values[i - self.d_period + 1:i + 1])
        
        current_k = k_values[-1]
        current_d = d_values[-1]
        
        # Check crossover
        crossover = None
        if len(k_values) >= 2 and len(d_values) >= 2:
            if k_values[-2] < d_values[-2] and k_values[-1] > d_values[-1]:
                crossover = "bullish"
            elif k_values[-2] > d_values[-2] and k_values[-1] < d_values[-1]:
                crossover = "bearish"
        
        return StochasticResult(
            k_line=k_values,
            d_line=d_values,
            current_k=current_k,
            current_d=current_d,
            is_oversold=current_k < 20 and current_d < 20,
            is_overbought=current_k > 80 and current_d > 80,
            crossover=crossover,
        )
