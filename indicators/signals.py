"""
Signal Generator - Combines indicators for trading signals
"""
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from .momentum import RSI, MACD, Stochastic
from .trend import SMA, EMA, ADX
from .volume import VolumeProfile, VWAP, OBV
from .volatility import ATR, BollingerBands


class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradingSignal:
    """Complete trading signal with analysis"""
    signal: SignalType
    confidence: float  # 0-1
    reasons: List[str]
    
    # Entry/Exit levels
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    
    # Indicators
    rsi: float
    macd_hist: float
    volume_ratio: float
    trend: str
    volatility: str


class SignalGenerator:
    """
    Rule-based signal generator using multiple technical indicators
    
    BUY CONDITIONS (Oversold Reversal):
    - RSI < 30 (oversold)
    - Volume > 2x average (capitulation)
    - Red bar (close < open)
    - Price near support (Bollinger lower or VWAP)
    - MACD histogram turning up
    
    SELL CONDITIONS (Overbought Reversal):
    - RSI > 70 (overbought)
    - Volume > 2x average (climax)
    - Green bar (close > open)
    - Price near resistance (Bollinger upper)
    - MACD histogram turning down
    """
    
    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        volume_threshold: float = 2.0,
        atr_multiplier_sl: float = 1.5,
        atr_multiplier_tp: float = 2.5,
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_threshold = volume_threshold
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        
        # Initialize indicators
        self.rsi = RSI(14)
        self.macd = MACD(12, 26, 9)
        self.stoch = Stochastic(14, 3)
        self.sma_20 = SMA(20)
        self.sma_50 = SMA(50)
        self.ema_9 = EMA(9)
        self.adx = ADX(14)
        self.volume = VolumeProfile(20)
        self.vwap = VWAP()
        self.obv = OBV()
        self.atr = ATR(14)
        self.bollinger = BollingerBands(20, 2.0)
    
    def generate_signal(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> TradingSignal:
        """Generate trading signal from OHLCV data"""
        
        # Calculate all indicators
        rsi_result = self.rsi.calculate(closes)
        macd_result = self.macd.calculate(closes)
        stoch_result = self.stoch.calculate(highs, lows, closes)
        adx_result = self.adx.calculate(highs, lows, closes)
        volume_result = self.volume.calculate(volumes)
        vwap_result = self.vwap.calculate(highs, lows, closes, volumes)
        obv_result = self.obv.calculate(closes, volumes)
        atr_result = self.atr.calculate(highs, lows, closes)
        bb_result = self.bollinger.calculate(closes)
        
        current_price = closes[-1]
        current_open = opens[-1]
        is_red_bar = closes[-1] < opens[-1]
        is_green_bar = closes[-1] > opens[-1]
        
        # Score system
        buy_score = 0
        sell_score = 0
        reasons = []
        
        # ========== BUY CONDITIONS ==========
        
        # 1. RSI oversold
        if rsi_result.is_oversold:
            buy_score += 2
            reasons.append(f"RSI oversold ({rsi_result.current:.1f})")
        elif rsi_result.current < 40:
            buy_score += 1
            reasons.append(f"RSI low ({rsi_result.current:.1f})")
        
        # 2. High volume on down move (capitulation)
        if volume_result.is_high_volume and is_red_bar:
            buy_score += 2
            reasons.append(f"High volume red bar ({volume_result.volume_ratio:.1f}x avg)")
        
        # 3. Stochastic oversold with crossover
        if stoch_result.is_oversold:
            buy_score += 1
            reasons.append("Stochastic oversold")
        if stoch_result.crossover == "bullish":
            buy_score += 1
            reasons.append("Stochastic bullish crossover")
        
        # 4. Price at Bollinger lower band
        if bb_result.percent_b < 0.1:
            buy_score += 2
            reasons.append("Price at Bollinger lower band")
        elif bb_result.percent_b < 0.2:
            buy_score += 1
            reasons.append("Price near Bollinger lower band")
        
        # 5. Price below VWAP (value area)
        if vwap_result.price_below:
            buy_score += 1
            reasons.append("Price below VWAP")
        
        # 6. MACD bullish crossover or histogram turning up
        if macd_result.crossover == "bullish":
            buy_score += 2
            reasons.append("MACD bullish crossover")
        elif macd_result.current_hist > macd_result.histogram[-2] and macd_result.current_hist < 0:
            buy_score += 1
            reasons.append("MACD histogram improving")
        
        # 7. RSI bullish divergence
        if rsi_result.divergence == "bullish":
            buy_score += 2
            reasons.append("RSI bullish divergence")
        
        # 8. OBV bullish divergence
        if obv_result.divergence == "bullish":
            buy_score += 1
            reasons.append("OBV bullish divergence")
        
        # ========== SELL CONDITIONS ==========
        
        # 1. RSI overbought
        if rsi_result.is_overbought:
            sell_score += 2
            reasons.append(f"RSI overbought ({rsi_result.current:.1f})")
        elif rsi_result.current > 60:
            sell_score += 1
            reasons.append(f"RSI high ({rsi_result.current:.1f})")
        
        # 2. High volume on up move (climax)
        if volume_result.is_high_volume and is_green_bar:
            sell_score += 2
            reasons.append(f"High volume green bar ({volume_result.volume_ratio:.1f}x avg)")
        
        # 3. Stochastic overbought with crossover
        if stoch_result.is_overbought:
            sell_score += 1
            reasons.append("Stochastic overbought")
        if stoch_result.crossover == "bearish":
            sell_score += 1
            reasons.append("Stochastic bearish crossover")
        
        # 4. Price at Bollinger upper band
        if bb_result.percent_b > 0.9:
            sell_score += 2
            reasons.append("Price at Bollinger upper band")
        elif bb_result.percent_b > 0.8:
            sell_score += 1
            reasons.append("Price near Bollinger upper band")
        
        # 5. Price above VWAP (extended)
        if vwap_result.price_above and vwap_result.deviation > 0.5:
            sell_score += 1
            reasons.append(f"Price extended above VWAP ({vwap_result.deviation:.2f}%)")
        
        # 6. MACD bearish crossover
        if macd_result.crossover == "bearish":
            sell_score += 2
            reasons.append("MACD bearish crossover")
        elif macd_result.current_hist < macd_result.histogram[-2] and macd_result.current_hist > 0:
            sell_score += 1
            reasons.append("MACD histogram weakening")
        
        # 7. RSI bearish divergence
        if rsi_result.divergence == "bearish":
            sell_score += 2
            reasons.append("RSI bearish divergence")
        
        # 8. OBV bearish divergence
        if obv_result.divergence == "bearish":
            sell_score += 1
            reasons.append("OBV bearish divergence")
        
        # ========== DETERMINE SIGNAL ==========
        
        net_score = buy_score - sell_score
        max_score = max(buy_score, sell_score)
        
        if net_score >= 5:
            signal = SignalType.STRONG_BUY
        elif net_score >= 3:
            signal = SignalType.BUY
        elif net_score <= -5:
            signal = SignalType.STRONG_SELL
        elif net_score <= -3:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD
        
        # Confidence
        confidence = min(max_score / 10, 1.0)
        
        # Entry/Exit levels
        atr_val = atr_result.current
        
        if signal in [SignalType.BUY, SignalType.STRONG_BUY]:
            entry = current_price
            stop_loss = entry - (atr_val * self.atr_multiplier_sl)
            take_profit = entry + (atr_val * self.atr_multiplier_tp)
        elif signal in [SignalType.SELL, SignalType.STRONG_SELL]:
            entry = current_price
            stop_loss = entry + (atr_val * self.atr_multiplier_sl)
            take_profit = entry - (atr_val * self.atr_multiplier_tp)
        else:
            entry = current_price
            stop_loss = current_price
            take_profit = current_price
        
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        risk_reward = reward / risk if risk > 0 else 0
        
        return TradingSignal(
            signal=signal,
            confidence=confidence,
            reasons=reasons,
            entry_price=round(entry, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            risk_reward=round(risk_reward, 2),
            rsi=round(rsi_result.current, 1),
            macd_hist=round(macd_result.current_hist, 4),
            volume_ratio=round(volume_result.volume_ratio, 2),
            trend=adx_result.trend_direction,
            volatility=atr_result.volatility_level,
        )
