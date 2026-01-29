"""
Helper Utilities
"""
import numpy as np
from typing import List, Optional
from datetime import datetime


def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency"""
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage"""
    return f"{value:.{decimals}f}%"


def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate percentage returns"""
    returns = np.diff(prices) / prices[:-1] * 100
    return returns


def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate log returns"""
    return np.diff(np.log(prices))


def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Array of returns (as decimals, not percentages)
        risk_free_rate: Annual risk-free rate (default 0)
        periods: Annualization factor (252 for daily, 52 for weekly)
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods)
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return / std_return) * np.sqrt(periods)
    return sharpe


def calculate_sortino(returns: np.ndarray, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """
    Calculate Sortino ratio (only considers downside volatility)
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods)
    mean_return = np.mean(excess_returns)
    
    # Downside deviation
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return float('inf')
    
    downside_std = np.std(negative_returns, ddof=1)
    
    if downside_std == 0:
        return 0.0
    
    sortino = (mean_return / downside_std) * np.sqrt(periods)
    return sortino


def calculate_max_drawdown(prices: np.ndarray) -> tuple:
    """
    Calculate maximum drawdown
    
    Returns:
        (max_drawdown_pct, peak_idx, trough_idx)
    """
    peak = prices[0]
    peak_idx = 0
    max_dd = 0
    max_dd_peak_idx = 0
    max_dd_trough_idx = 0
    
    for i, price in enumerate(prices):
        if price > peak:
            peak = price
            peak_idx = i
        
        dd = (peak - price) / peak * 100
        if dd > max_dd:
            max_dd = dd
            max_dd_peak_idx = peak_idx
            max_dd_trough_idx = i
    
    return max_dd, max_dd_peak_idx, max_dd_trough_idx


def calculate_win_rate(pnls: List[float]) -> float:
    """Calculate win rate"""
    if not pnls:
        return 0.0
    
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls)


def calculate_profit_factor(pnls: List[float]) -> float:
    """Calculate profit factor (gross profit / gross loss)"""
    wins = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    
    if losses == 0:
        return float('inf') if wins > 0 else 0.0
    
    return wins / losses


def calculate_expectancy(pnls: List[float]) -> float:
    """Calculate average expected P&L per trade"""
    if not pnls:
        return 0.0
    return sum(pnls) / len(pnls)


def is_market_hours(dt: Optional[datetime] = None) -> bool:
    """Check if current time is during regular market hours (9:30 AM - 4:00 PM ET)"""
    if dt is None:
        dt = datetime.now()
    
    # Simple check (doesn't account for holidays or timezone)
    if dt.weekday() >= 5:  # Weekend
        return False
    
    market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= dt <= market_close


def is_trading_hours(dt: Optional[datetime] = None) -> bool:
    """Check if within preferred trading hours (avoid open/close volatility)"""
    if dt is None:
        dt = datetime.now()
    
    if not is_market_hours(dt):
        return False
    
    # Avoid first 30 min and last 15 min
    trading_start = dt.replace(hour=10, minute=0, second=0, microsecond=0)
    trading_end = dt.replace(hour=15, minute=45, second=0, microsecond=0)
    
    return trading_start <= dt <= trading_end


def round_to_tick(price: float, tick_size: float = 0.01) -> float:
    """Round price to tick size"""
    return round(price / tick_size) * tick_size


def calculate_position_size(
    account_value: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
) -> int:
    """
    Calculate position size based on risk
    
    Args:
        account_value: Total account value
        risk_pct: Percentage of account to risk (e.g., 0.02 for 2%)
        entry_price: Entry price
        stop_loss: Stop loss price
        
    Returns:
        Number of shares
    """
    risk_amount = account_value * risk_pct
    risk_per_share = abs(entry_price - stop_loss)
    
    if risk_per_share == 0:
        return 0
    
    shares = int(risk_amount / risk_per_share)
    return max(0, shares)
