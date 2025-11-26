"""
Risk Management Module for HFT Trading
Provides position sizing, stop-loss, profit targets, and portfolio risk controls
"""
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger('RiskManager')

@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position_size: int = 1000  # Maximum shares per position
    max_portfolio_value: float = 100000.0  # Maximum total portfolio value
    max_position_pct: float = 0.20  # Max 20% of portfolio in single position
    max_drawdown_pct: float = 0.10  # Circuit breaker at 10% drawdown
    stop_loss_pct: float = 0.02  # 2% stop loss per trade
    profit_target_pct: float = 0.03  # 3% profit target
    trailing_stop_pct: float = 0.015  # 1.5% trailing stop
    max_daily_loss: float = 5000.0  # Maximum daily loss limit
    

@dataclass
class TradeRisk:
    """Risk assessment for a potential trade"""
    approved: bool
    position_size: int
    stop_loss_price: float
    profit_target_price: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    reason: str = ""


class RiskManager:
    """
    Manages trading risk including position sizing, stop-loss, and portfolio limits
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Initialize risk manager with specified limits
        
        Args:
            limits: RiskLimits object, uses defaults if None
        """
        self.limits = limits or RiskLimits()
        
        # Portfolio state
        self.initial_capital = self.limits.max_portfolio_value
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.now().date()
        
        # Position tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking for Kelly Criterion
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        
        logger.info(f"Risk Manager initialized with capital: ${self.initial_capital:,.2f}")
        
    def set_initial_equity(self, equity: float):
        """
        Set initial equity/capital
        
        Args:
            equity: Initial capital amount
        """
        self.initial_capital = equity
        self.current_capital = equity
        self.peak_capital = equity
        logger.info(f"Initial equity set to: ${equity:,.2f}")
        
    def reset_daily_limits(self):
        """Reset daily tracking metrics"""
        current_date = datetime.now().date()
        if current_date > self.daily_reset_time:
            logger.info(f"Resetting daily limits. Previous day P&L: ${self.daily_pnl:,.2f}")
            self.daily_pnl = 0.0
            self.daily_reset_time = current_date
            
    def check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker should be triggered
        
        Returns:
            True if trading should be halted, False otherwise
        """
        self.reset_daily_limits()
        
        # Check drawdown from peak
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown >= self.limits.max_drawdown_pct:
            logger.warning(f"CIRCUIT BREAKER: Drawdown {drawdown:.2%} exceeds limit {self.limits.max_drawdown_pct:.2%}")
            return True
            
        # Check daily loss limit
        if self.daily_pnl <= -self.limits.max_daily_loss:
            logger.warning(f"CIRCUIT BREAKER: Daily loss ${abs(self.daily_pnl):,.2f} exceeds limit ${self.limits.max_daily_loss:,.2f}")
            return True
            
        return False
        
    def calculate_position_size(self, 
                                symbol: str, 
                                current_price: float,
                                signal_strength: float = 1.0,
                                use_kelly: bool = False) -> int:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            signal_strength: Signal confidence (0.0 to 1.0)
            use_kelly: Whether to use Kelly Criterion for sizing
            
        Returns:
            Position size in shares
        """
        if current_price <= 0:
            return 0
            
        # Base position size from portfolio percentage
        max_position_value = self.current_capital * self.limits.max_position_pct
        base_size = int(max_position_value / current_price)
        
        # Apply hard limit
        base_size = min(base_size, self.limits.max_position_size)
        
        # Adjust for signal strength
        adjusted_size = int(base_size * signal_strength)
        
        # Kelly Criterion adjustment (if enabled and we have enough trade history)
        if use_kelly and self.total_trades >= 20:
            kelly_fraction = self._calculate_kelly_fraction()
            kelly_size = int(base_size * kelly_fraction)
            adjusted_size = min(adjusted_size, kelly_size)
            logger.debug(f"Kelly fraction: {kelly_fraction:.3f}, Kelly size: {kelly_size}")
            
        return max(1, adjusted_size)  # At least 1 share
        
    def _calculate_kelly_fraction(self) -> float:
        """
        Calculate Kelly Criterion fraction
        
        Kelly% = W - [(1-W) / R]
        Where:
            W = Win rate
            R = Average win / Average loss
        """
        if self.total_trades == 0 or self.winning_trades == 0:
            return 0.5  # Conservative default
            
        win_rate = self.winning_trades / self.total_trades
        
        avg_win = self.total_profit / max(1, self.winning_trades)
        losing_trades = self.total_trades - self.winning_trades
        avg_loss = abs(self.total_loss) / max(1, losing_trades)
        
        if avg_loss == 0:
            return 0.5
            
        risk_reward = avg_win / avg_loss
        
        kelly = win_rate - ((1 - win_rate) / risk_reward)
        
        # Use fractional Kelly (25% of full Kelly) for safety
        fractional_kelly = max(0.0, min(0.25, kelly * 0.25))
        
        return fractional_kelly
        
    def assess_trade(self,
                     symbol: str,
                     action: str,
                     current_price: float,
                     signal_strength: float = 1.0) -> TradeRisk:
        """
        Assess if a trade should be approved and calculate risk parameters
        
        Args:
            symbol: Trading symbol
            action: 'BUY' or 'SELL'
            current_price: Current market price
            signal_strength: Signal confidence (0.0 to 1.0)
            
        Returns:
            TradeRisk object with approval decision and parameters
        """
        # Check circuit breaker
        if self.check_circuit_breaker():
            return TradeRisk(
                approved=False,
                position_size=0,
                stop_loss_price=0.0,
                profit_target_price=0.0,
                risk_amount=0.0,
                reward_amount=0.0,
                risk_reward_ratio=0.0,
                reason="Circuit breaker triggered"
            )
            
        # Calculate position size
        position_size = self.calculate_position_size(symbol, current_price, signal_strength)
        
        if position_size == 0:
            return TradeRisk(
                approved=False,
                position_size=0,
                stop_loss_price=0.0,
                profit_target_price=0.0,
                risk_amount=0.0,
                reward_amount=0.0,
                risk_reward_ratio=0.0,
                reason="Position size calculated as 0"
            )
            
        # Calculate stop loss and profit target
        if action == 'BUY':
            stop_loss_price = current_price * (1 - self.limits.stop_loss_pct)
            profit_target_price = current_price * (1 + self.limits.profit_target_pct)
        else:  # SELL
            stop_loss_price = current_price * (1 + self.limits.stop_loss_pct)
            profit_target_price = current_price * (1 - self.limits.profit_target_pct)
            
        # Calculate risk and reward amounts
        risk_amount = abs(current_price - stop_loss_price) * position_size
        reward_amount = abs(profit_target_price - current_price) * position_size
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        # Check if we have enough capital
        trade_value = current_price * position_size
        if trade_value > self.current_capital:
            return TradeRisk(
                approved=False,
                position_size=position_size,
                stop_loss_price=stop_loss_price,
                profit_target_price=profit_target_price,
                risk_amount=risk_amount,
                reward_amount=reward_amount,
                risk_reward_ratio=risk_reward_ratio,
                reason=f"Insufficient capital: need ${trade_value:,.2f}, have ${self.current_capital:,.2f}"
            )
            
        return TradeRisk(
            approved=True,
            position_size=position_size,
            stop_loss_price=stop_loss_price,
            profit_target_price=profit_target_price,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_reward_ratio=risk_reward_ratio,
            reason="Trade approved"
        )
        
    def open_position(self, symbol: str, action: str, quantity: int, entry_price: float, 
                     stop_loss: float, profit_target: float):
        """
        Record a new position opening
        
        Args:
            symbol: Trading symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            entry_price: Entry price
            stop_loss: Stop loss price
            profit_target: Profit target price
        """
        self.positions[symbol] = {
            'action': action,
            'quantity': quantity,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'profit_target': profit_target,
            'trailing_stop': stop_loss,
            'peak_price': entry_price,
            'entry_time': datetime.now()
        }
        
        # Update capital
        trade_value = quantity * entry_price
        self.current_capital -= trade_value
        
        logger.info(f"Position opened: {action} {quantity} {symbol} @ ${entry_price:.2f}, "
                   f"SL: ${stop_loss:.2f}, PT: ${profit_target:.2f}")
        
    def update_position(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Update position and check for stop loss or profit target
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            'STOP_LOSS', 'PROFIT_TARGET', 'TRAILING_STOP', or None
        """
        if symbol not in self.positions:
            return None
            
        pos = self.positions[symbol]
        action = pos['action']
        
        # Update trailing stop
        if action == 'BUY':
            if current_price > pos['peak_price']:
                pos['peak_price'] = current_price
                new_trailing = current_price * (1 - self.limits.trailing_stop_pct)
                pos['trailing_stop'] = max(pos['trailing_stop'], new_trailing)
                
            # Check exit conditions
            if current_price <= pos['stop_loss']:
                return 'STOP_LOSS'
            elif current_price <= pos['trailing_stop']:
                return 'TRAILING_STOP'
            elif current_price >= pos['profit_target']:
                return 'PROFIT_TARGET'
                
        else:  # SELL
            if current_price < pos['peak_price']:
                pos['peak_price'] = current_price
                new_trailing = current_price * (1 + self.limits.trailing_stop_pct)
                pos['trailing_stop'] = min(pos['trailing_stop'], new_trailing)
                
            # Check exit conditions
            if current_price >= pos['stop_loss']:
                return 'STOP_LOSS'
            elif current_price >= pos['trailing_stop']:
                return 'TRAILING_STOP'
            elif current_price <= pos['profit_target']:
                return 'PROFIT_TARGET'
                
        return None
        
    def close_position(self, symbol: str, exit_price: float, reason: str = "Manual"):
        """
        Close a position and update metrics
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Reason for closing
        """
        if symbol not in self.positions:
            logger.warning(f"Attempted to close non-existent position: {symbol}")
            return
            
        pos = self.positions[symbol]
        action = pos['action']
        quantity = pos['quantity']
        entry_price = pos['entry_price']
        
        # Calculate P&L
        if action == 'BUY':
            pnl = (exit_price - entry_price) * quantity
        else:  # SELL
            pnl = (entry_price - exit_price) * quantity
            
        # Update capital
        trade_value = quantity * exit_price
        self.current_capital += trade_value
        
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
            
        # Update daily P&L
        self.daily_pnl += pnl
        
        # Update performance metrics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        else:
            self.total_loss += pnl
            
        logger.info(f"Position closed: {symbol} @ ${exit_price:.2f}, P&L: ${pnl:,.2f}, "
                   f"Reason: {reason}, Total Capital: ${self.current_capital:,.2f}")
        
        # Remove position
        del self.positions[symbol]
        
    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Get current portfolio status
        
        Returns:
            Dictionary with portfolio metrics
        """
        total_position_value = sum(
            pos['quantity'] * pos['entry_price'] 
            for pos in self.positions.values()
        )
        
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        return {
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'drawdown_pct': drawdown,
            'daily_pnl': self.daily_pnl,
            'total_position_value': total_position_value,
            'available_capital': self.current_capital,
            'open_positions': len(self.positions),
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'total_pnl': self.current_capital - self.initial_capital
        }
