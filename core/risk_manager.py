"""
Risk Management Module for 0DTE Trading
Handles Kelly Criterion position sizing and risk controls.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class RiskConfig:
    """Risk management configuration"""
    # ============================================================
    # CAPITAL-BASED RISK CONTROLS (THE KEY FIX!)
    # ============================================================
    max_risk_per_trade_pct: float = 0.02  # Max 2% of capital risked per trade
    max_position_pct: float = 0.07        # Max 7% of capital in single position
    
    # Kelly Criterion
    kelly_fraction: float = 0.20     # Fractional Kelly (20% - reduced for lower DD)
    min_kelly_pct: float = 0.02      # Minimum position size (2% of capital)
    max_kelly_pct: float = 0.20      # Maximum position size (20% cap - reduced)
    default_position_pct: float = 0.07  # Before enough trades (reduced)
    min_trades_for_kelly: int = 10   # Minimum samples for Kelly
    
    # Position limits
    max_position_value: float = 5000  # Absolute dollar cap (increased for larger capital)
    max_contracts: int = 50           # Maximum contracts per trade
    
    # Transition zone optimization (medium trades have lower WR)
    use_transition_filters: bool = False  # Apply stricter filters when 10-49 contracts
    min_contracts_for_full_risk: int = 50  # Min contracts for full-risk trading
    max_contracts_for_learning: int = 9    # Max contracts for early learning phase
    
    # Daily risk controls
    max_trades_per_day: int = 999     # No limit on trades per day
    stop_after_first_loss: bool = False  # Disabled - allow all trades
    max_daily_loss_pct: float = 0.99    # Effectively disabled
    
    # Portfolio risk
    max_drawdown_pct: float = 0.99   # Effectively disabled
    
    # Drawdown-based position reduction
    reduce_size_at_dd_pct: float = 0.99  # Effectively disabled
    max_dd_reduction: float = 0.50       # Reduce size by up to 50%
    
    # Consecutive loss protection
    max_consecutive_losses: int = 999    # Effectively disabled
    consec_loss_reduction: float = 0.50  # Reduce by 50% after streak
    wins_to_reset_streak: int = 2        # Need 2 wins to reset streak
    
    # Trade costs
    slippage_pct: float = 0.005      # 0.5% slippage
    commission_per_contract: float = 0.65


# ============================================================
# KELLY CRITERION CALCULATOR
# ============================================================

class KellyCalculator:
    """
    Kelly Criterion position sizing calculator.
    
    Kelly Formula: f* = (bp - q) / b
    Where:
        f* = fraction of capital to bet
        b = odds received on the bet (avg_win / avg_loss)
        p = probability of winning
        q = probability of losing (1 - p)
    """
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.kelly_pct = self.config.default_position_pct
        self.stats = {}
    
    def calculate_from_trades(self, trades_df: pd.DataFrame) -> Tuple[float, dict]:
        """
        Calculate Kelly from historical trade data.
        
        Args:
            trades_df: DataFrame with 'win' (0/1) and 'pnl_pct' columns
            
        Returns:
            Tuple of (kelly_pct, stats_dict)
        """
        if len(trades_df) < self.config.min_trades_for_kelly:
            print(f"  Warning: {len(trades_df)} trades < {self.config.min_trades_for_kelly} minimum, using default")
            return self.config.default_position_pct, {}
        
        # Calculate win rate and average win/loss
        wins = trades_df[trades_df['win'] == 1]
        losses = trades_df[trades_df['win'] == 0]
        
        win_rate = len(wins) / len(trades_df)
        
        # Use pnl_pct for win/loss ratio (more stable)
        avg_win_pct = wins['pnl_pct'].mean() if len(wins) > 0 else 0.25
        avg_loss_pct = abs(losses['pnl_pct'].mean()) if len(losses) > 0 else 0.35
        
        # Calculate Kelly
        p = win_rate
        q = 1 - p
        b = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 1
        kelly_raw = (b * p - q) / b if b > 0 else 0
        
        # Apply fractional Kelly
        kelly_fractional = kelly_raw * self.config.kelly_fraction
        
        # Bound the result
        if kelly_fractional <= 0:
            kelly_pct = self.config.min_kelly_pct
        else:
            kelly_pct = max(self.config.min_kelly_pct, 
                          min(kelly_fractional, self.config.max_kelly_pct))
        
        self.kelly_pct = kelly_pct
        self.stats = {
            'win_rate': win_rate,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'b_ratio': b,
            'kelly_raw': kelly_raw,
            'kelly_fractional': kelly_fractional,
            'kelly_final': kelly_pct,
            'samples': len(trades_df)
        }
        
        return kelly_pct, self.stats
    
    def calculate_from_winrate(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly from win rate and average win/loss.
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive number)
            
        Returns:
            Kelly fraction (position size as fraction of capital)
        """
        if avg_loss == 0 or win_rate == 0:
            return self.config.default_position_pct
        
        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss  # Win/loss ratio (the odds)
        
        # Kelly formula
        kelly = (b * p - q) / b
        
        # Apply fractional Kelly
        kelly_fractional = kelly * self.config.kelly_fraction
        
        # Bound the result
        if kelly_fractional <= 0:
            return self.config.min_kelly_pct
        
        return max(self.config.min_kelly_pct, 
                  min(kelly_fractional, self.config.max_kelly_pct))
    
    def print_stats(self):
        """Print Kelly calculation statistics"""
        if not self.stats:
            print("  No Kelly stats available (run calculate first)")
            return
        
        print(f"\n  Kelly Calculation:")
        print(f"    Win Rate: {self.stats['win_rate']:.1%}")
        print(f"    Avg Win %: {self.stats['avg_win_pct']*100:.1f}%")
        print(f"    Avg Loss %: {self.stats['avg_loss_pct']*100:.1f}%")
        print(f"    W/L Ratio (b): {self.stats['b_ratio']:.2f}")
        print(f"    Raw Kelly: {self.stats['kelly_raw']*100:.1f}%")
        print(f"    Fractional ({self.config.kelly_fraction:.0%}): {self.stats['kelly_fractional']*100:.1f}%")
        print(f"    Final (capped): {self.stats['kelly_final']*100:.1f}%")


# ============================================================
# POSITION SIZER
# ============================================================

class PositionSizer:
    """Calculate position sizes with risk controls"""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.kelly_pct = self.config.default_position_pct
    
    def set_kelly(self, kelly_pct: float):
        """Set Kelly percentage from calculator"""
        self.kelly_pct = kelly_pct
    
    def get_position_size(self, capital: float, ml_confidence: float = None) -> float:
        """
        Calculate position value based on capital and confidence.
        
        Args:
            capital: Current capital
            ml_confidence: ML model confidence (0-1), optional
            
        Returns:
            Position value in dollars
        """
        # Base size from Kelly
        position_value = capital * self.kelly_pct
        
        # Optional: boost for high ML confidence
        if ml_confidence is not None and ml_confidence > 0.75:
            position_value *= 1.1  # 10% boost
        
        # Apply absolute cap
        position_value = min(position_value, self.config.max_position_value)
        
        return position_value
    
    def get_num_contracts(self, capital: float, option_price: float, 
                         ml_confidence: float = None) -> int:
        """
        Calculate number of contracts to trade.
        
        Args:
            capital: Current capital
            option_price: Option price per share (will multiply by 100)
            ml_confidence: ML confidence (0-1), optional
            
        Returns:
            Number of contracts (minimum 1)
        """
        position_value = self.get_position_size(capital, ml_confidence)
        contract_cost = option_price * 100
        
        if contract_cost <= 0:
            return 0
        
        num_contracts = int(position_value / contract_cost)
        
        # Apply limits
        num_contracts = max(1, num_contracts)
        num_contracts = min(num_contracts, self.config.max_contracts)
        
        return num_contracts


# ============================================================
# RISK MANAGER (COMBINES ALL RISK CONTROLS)
# ============================================================

class RiskManager:
    """
    Central risk management class.
    Tracks drawdowns, daily limits, and provides trade authorization.
    """
    
    def __init__(self, initial_capital: float, config: RiskConfig = None):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.config = config or RiskConfig()
        
        # Kelly and position sizing
        self.kelly_calculator = KellyCalculator(self.config)
        self.position_sizer = PositionSizer(self.config)
        
        # Daily tracking
        self.daily_trades: Dict[str, int] = {}
        self.daily_pnl: Dict[str, float] = {}
        self.daily_had_loss: Dict[str, bool] = {}
        
        # Portfolio tracking
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Consecutive loss tracking
        self.consecutive_losses = 0
        self.consecutive_wins = 0  # Track wins to reset streak
        self.in_reduced_mode = False  # Stay in reduced mode until enough wins
    
    def setup_kelly(self, training_data: pd.DataFrame) -> Tuple[float, dict]:
        """
        Calculate Kelly from training data.
        
        Args:
            training_data: DataFrame with 'win' and 'pnl_pct' columns
            
        Returns:
            Tuple of (kelly_pct, stats)
        """
        kelly_pct, stats = self.kelly_calculator.calculate_from_trades(training_data)
        self.position_sizer.set_kelly(kelly_pct)
        return kelly_pct, stats
    
    def set_kelly(self, kelly_pct: float):
        """Set Kelly percentage directly"""
        self.kelly_calculator.kelly_pct = kelly_pct
        self.position_sizer.set_kelly(kelly_pct)
    
    def can_trade(self, date: str) -> Tuple[bool, str]:
        """
        Check if we can take a trade based on risk limits.
        
        Args:
            date: Current date string
            
        Returns:
            Tuple of (can_trade, reason)
        """
        # Check max drawdown
        if self.peak_capital > 0:
            self.current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
            if self.current_drawdown >= self.config.max_drawdown_pct:
                return False, f"Max DD {self.current_drawdown:.1%} >= {self.config.max_drawdown_pct:.1%}"
        
        # Initialize daily tracking
        if date not in self.daily_trades:
            self.daily_trades[date] = 0
            self.daily_pnl[date] = 0.0
            self.daily_had_loss[date] = False
        
        # Check daily trade limit
        if self.daily_trades[date] >= self.config.max_trades_per_day:
            return False, f"Max trades {self.config.max_trades_per_day} reached"
        
        # Check stop after first loss
        if self.config.stop_after_first_loss and self.daily_had_loss.get(date, False):
            return False, "Stopped after first loss"
        
        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl.get(date, 0)) / self.capital
        if self.daily_pnl.get(date, 0) < 0 and daily_loss_pct >= self.config.max_daily_loss_pct:
            return False, f"Daily loss {daily_loss_pct:.1%} >= {self.config.max_daily_loss_pct:.1%}"
        
        return True, "OK"
    
    def get_position_size(self, option_price: float, ml_confidence: float = None,
                          stop_loss_pct: float = 0.28) -> Tuple[int, float]:
        """
        Get position size for a trade.
        Uses KELLY-BASED sizing with RISK CAPS for protection.
        
        Args:
            option_price: Option price per share
            ml_confidence: ML confidence (0-1)
            stop_loss_pct: Stop loss percentage (for risk calculation)
            
        Returns:
            Tuple of (num_contracts, position_value)
        """
        # ============================================================
        # KELLY + RISK-BASED POSITION SIZING
        # ============================================================
        
        # 1. START with Kelly-based position (the optimal size)
        kelly_pct = self.position_sizer.kelly_pct
        kelly_position = self.capital * kelly_pct
        
        # 2. Calculate RISK CAP: max position based on max allowed loss
        #    If stop loss = 28%, and we can lose 2% of capital max:
        #    position × stop_loss = max_loss
        #    position = max_loss / stop_loss = (capital × 2%) / 28% = 7.1% of capital
        max_loss_dollars = self.capital * self.config.max_risk_per_trade_pct
        if stop_loss_pct > 0:
            risk_capped_position = max_loss_dollars / stop_loss_pct
        else:
            risk_capped_position = self.capital * self.config.max_position_pct
        
        # 3. Calculate PERCENTAGE CAP (max % of capital)
        pct_capped_position = self.capital * self.config.max_position_pct
        
        # 4. Calculate DYNAMIC DOLLAR CAP (scales with capital for early protection)
        #    At $10K: min($5000, 10K × 5%) = $500
        #    At $100K: min($5000, 100K × 5%) = $5000
        dynamic_dollar_cap = min(
            self.config.max_position_value,
            self.capital * 0.05  # Max 5% of capital as dollar cap
        )
        
        # 5. Use MINIMUM of all caps (most conservative)
        position_value = min(
            kelly_position,        # Kelly optimal
            risk_capped_position,  # Risk-based cap
            pct_capped_position,   # Percentage cap
            dynamic_dollar_cap     # Dynamic dollar cap
        )
        
        # 6. EARLY-STAGE PROTECTION: DISABLED - follow backtest exactly
        # if self.capital < self.initial_capital * 1.5:
        #     position_value *= 0.50
        
        # 7. Apply drawdown-based reduction (start reducing at 5% DD)
        if self.current_drawdown > self.config.reduce_size_at_dd_pct:
            dd_excess = self.current_drawdown - self.config.reduce_size_at_dd_pct
            dd_factor = min(dd_excess / 0.10, 1.0)  # Scale over 10% DD range
            reduction = dd_factor * self.config.max_dd_reduction
            position_value = position_value * (1 - reduction)
        
        # 8. Apply consecutive loss reduction
        if self.in_reduced_mode:
            position_value = position_value * (1 - self.config.consec_loss_reduction)
        
        # 9. Calculate contracts
        contract_cost = option_price * 100
        if contract_cost <= 0:
            return 0, 0
        
        num_contracts = int(position_value / contract_cost)
        
        # 10. SKIP trade if position too small (don't force minimum 1)
        if num_contracts < 1:
            return 0, 0  # Skip - capital too small for safe position
        
        num_contracts = min(num_contracts, self.config.max_contracts)
        
        actual_position = num_contracts * contract_cost
        return num_contracts, actual_position
    
    def record_trade(self, date: str, pnl: float):
        """
        Record a completed trade.
        
        Args:
            date: Trade date
            pnl: Net P&L (positive or negative)
        """
        self.capital += pnl
        
        # Update peak
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        # Update drawdown
        self.current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown
        
        # Daily tracking
        if date not in self.daily_trades:
            self.daily_trades[date] = 0
            self.daily_pnl[date] = 0.0
            self.daily_had_loss[date] = False
        
        self.daily_trades[date] += 1
        self.daily_pnl[date] += pnl
        
        if pnl < 0:
            self.daily_had_loss[date] = True
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            # Enter reduced mode after max consecutive losses
            if self.consecutive_losses >= self.config.max_consecutive_losses:
                self.in_reduced_mode = True
        else:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            # Exit reduced mode only after enough consecutive wins
            if self.consecutive_wins >= self.config.wins_to_reset_streak:
                self.in_reduced_mode = False
    
    def calculate_trade_pnl(self, entry_price: float, exit_price: float, 
                           num_contracts: int) -> Tuple[float, float, float]:
        """
        Calculate P&L for a trade including costs.
        
        Args:
            entry_price: Entry price per share
            exit_price: Exit price per share
            num_contracts: Number of contracts
            
        Returns:
            Tuple of (gross_pnl, commission, net_pnl)
        """
        gross_pnl = num_contracts * 100 * (exit_price - entry_price)
        commission = self.config.commission_per_contract * num_contracts * 2
        net_pnl = gross_pnl - commission
        return gross_pnl, commission, net_pnl
    
    def apply_slippage(self, price: float, is_entry: bool) -> float:
        """
        Apply slippage to a price.
        
        Args:
            price: Raw price
            is_entry: True for entry, False for exit
            
        Returns:
            Price after slippage
        """
        if is_entry:
            return price * (1 + self.config.slippage_pct)
        else:
            return price * (1 - self.config.slippage_pct)
    
    def get_summary(self) -> dict:
        """Get risk management summary"""
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.capital,
            'peak_capital': self.peak_capital,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'kelly_pct': self.position_sizer.kelly_pct,
            'total_trades': sum(self.daily_trades.values()),
            'profitable_days': sum(1 for pnl in self.daily_pnl.values() if pnl > 0),
            'losing_days': sum(1 for pnl in self.daily_pnl.values() if pnl < 0),
        }
    
    def print_summary(self):
        """Print risk management summary"""
        summary = self.get_summary()
        print("\n" + "=" * 50)
        print("RISK MANAGEMENT SUMMARY")
        print("=" * 50)
        print(f"  Initial Capital: ${summary['initial_capital']:,.2f}")
        print(f"  Current Capital: ${summary['current_capital']:,.2f}")
        print(f"  Peak Capital: ${summary['peak_capital']:,.2f}")
        print(f"  Max Drawdown: {summary['max_drawdown']:.1%}")
        print(f"  Kelly Size: {summary['kelly_pct']:.1%}")
        print(f"  Total Trades: {summary['total_trades']}")
        print(f"  Profitable Days: {summary['profitable_days']}")
        print(f"  Losing Days: {summary['losing_days']}")
