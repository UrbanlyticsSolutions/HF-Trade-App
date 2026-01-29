"""
Settings Configuration
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class APISettings:
    """API configuration"""
    fmp_api_key: str = ""
    fmp_base_url: str = "https://financialmodelingprep.com/stable"


@dataclass
class TradingSettings:
    """Trading parameters"""
    # Symbols to trade
    watchlist: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "AMZN", "AMD"
    ])
    
    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio per trade
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_concurrent_positions: int = 5
    max_daily_trades: int = 10
    min_risk_reward: float = 1.5
    
    # Position sizing
    use_fixed_size: bool = False
    fixed_position_size: int = 100
    
    # Trailing stops
    use_trailing_stop: bool = True
    trailing_atr_multiplier: float = 2.0


@dataclass
class IndicatorSettings:
    """Indicator parameters"""
    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Stochastic
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    
    # Moving averages
    sma_short: int = 20
    sma_long: int = 50
    ema_fast: int = 9
    ema_slow: int = 21
    
    # ATR
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5
    atr_target_multiplier: float = 2.5
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # Volume
    volume_period: int = 20
    volume_spike_threshold: float = 2.0
    
    # ADX
    adx_period: int = 14
    adx_trend_threshold: float = 25


@dataclass
class TimeSettings:
    """Trading time settings"""
    # Market hours (Eastern Time)
    trading_start_hour: int = 9
    trading_start_minute: int = 30
    trading_end_hour: int = 15
    trading_end_minute: int = 45
    
    # Avoid volatility at open/close
    avoid_first_minutes: int = 30
    avoid_last_minutes: int = 15
    
    # Data settings
    intraday_interval: str = "5min"
    lookback_bars: int = 100


@dataclass
class Settings:
    """Main settings container"""
    api: APISettings = field(default_factory=APISettings)
    trading: TradingSettings = field(default_factory=TradingSettings)
    indicators: IndicatorSettings = field(default_factory=IndicatorSettings)
    time: TimeSettings = field(default_factory=TimeSettings)


def load_settings(env_file: Optional[str] = None) -> Settings:
    """
    Load settings from environment and .env file
    """
    # Load .env file
    if env_file:
        load_dotenv(env_file)
    else:
        # Try to find .env in current and parent directories
        for path in [Path('.'), Path('..'), Path('../..')]:
            env_path = path / '.env'
            if env_path.exists():
                load_dotenv(env_path)
                break
    
    settings = Settings()
    
    # Load API keys from environment
    settings.api.fmp_api_key = os.getenv('FMP_API_KEY', '')
    
    # Override from environment variables if present
    if os.getenv('MAX_POSITION_SIZE'):
        settings.trading.max_position_size = float(os.getenv('MAX_POSITION_SIZE'))
    
    if os.getenv('RISK_PER_TRADE'):
        settings.trading.risk_per_trade = float(os.getenv('RISK_PER_TRADE'))
    
    if os.getenv('MAX_DAILY_LOSS'):
        settings.trading.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS'))
    
    return settings


def get_api_key(key_name: str = 'FMP_API_KEY') -> str:
    """Get API key from environment"""
    load_dotenv()
    key = os.getenv(key_name, '')
    if not key:
        raise ValueError(f"API key {key_name} not found in environment")
    return key
