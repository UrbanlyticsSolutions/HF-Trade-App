"""
Performance Tracking Module for HFT Trading
Tracks and analyzes trading performance metrics in real-time
"""
import logging
import csv
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import deque
import statistics

logger = logging.getLogger('PerformanceTracker')


class PerformanceTracker:
    """
    Tracks trading performance metrics including returns, risk, and execution quality
    """
    
    def __init__(self, symbol: str, output_dir: str = "output"):
        """
        Initialize performance tracker
        
        Args:
            symbol: Trading symbol
            output_dir: Directory for output files
        """
        self.symbol = symbol
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Trade history
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        
        # Running metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        
        # Position tracking
        self.current_position_size = 0
        self.current_entry_price = 0.0
        
        # Equity tracking
        self.initial_equity = 0.0
        self.current_equity = 0.0
        self.peak_equity = 0.0
        self.max_drawdown = 0.0
        
        # Rolling windows for metrics
        self.returns_window = deque(maxlen=100)  # Last 100 trades
        
        # Setup CSV logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trades_file = os.path.join(output_dir, f"trades_{symbol}_{timestamp}.csv")
        self.metrics_file = os.path.join(output_dir, f"metrics_{symbol}_{timestamp}.json")
        
        # Initialize trades CSV
        with open(self.trades_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Symbol', 'Action', 'Quantity', 'Price', 
                'PnL', 'Cumulative_PnL', 'Equity', 'Exit_Reason'
            ])
            
        logger.info(f"Performance tracker initialized for {symbol}")
        
    def set_initial_equity(self, equity: float):
        """Set initial equity/capital"""
        self.initial_equity = equity
        self.current_equity = equity
        self.peak_equity = equity
        
    def record_trade(self, 
                    action: str,
                    quantity: int,
                    price: float,
                    pnl: float = 0.0,
                    exit_reason: str = ""):
        """
        Record a trade execution
        
        Args:
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            price: Execution price
            pnl: Profit/Loss for this trade (for closing trades)
            exit_reason: Reason for exit (if applicable)
        """
        timestamp = datetime.now()
        
        # Update equity
        if pnl != 0:
            self.current_equity += pnl
            
            # Update peak and drawdown
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
                
            current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
                
            # Update trade statistics
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
                self.total_profit += pnl
                self.gross_profit += pnl
            else:
                self.losing_trades += 1
                self.total_loss += pnl
                self.gross_loss += abs(pnl)
                
            # Calculate return percentage
            if self.current_entry_price > 0:
                return_pct = pnl / (self.current_entry_price * quantity)
                self.returns_window.append(return_pct)
        
        # Record trade
        trade_record = {
            'timestamp': timestamp.isoformat(),
            'symbol': self.symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'pnl': pnl,
            'cumulative_pnl': self.current_equity - self.initial_equity,
            'equity': self.current_equity,
            'exit_reason': exit_reason
        }
        
        self.trades.append(trade_record)
        
        # Log to CSV
        with open(self.trades_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp.isoformat(),
                self.symbol,
                action,
                quantity,
                price,
                f"{pnl:.2f}",
                f"{self.current_equity - self.initial_equity:.2f}",
                f"{self.current_equity:.2f}",
                exit_reason
            ])
            
        # Update position tracking
        if action == 'BUY':
            self.current_entry_price = price
            self.current_position_size = quantity
        elif action == 'SELL' and exit_reason:
            self.current_position_size = 0
            self.current_entry_price = 0.0
            
    def record_equity_snapshot(self, equity: float):
        """
        Record equity at a point in time for equity curve
        
        Args:
            equity: Current equity value
        """
        self.equity_curve.append({
            'timestamp': datetime.now().isoformat(),
            'equity': equity
        })
        
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio from returns
        
        Args:
            risk_free_rate: Annual risk-free rate (default 0%)
            
        Returns:
            Sharpe ratio
        """
        if len(self.returns_window) < 2:
            return 0.0
            
        returns = list(self.returns_window)
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        if std_return == 0:
            return 0.0
            
        # Annualize (assuming daily returns, 252 trading days)
        sharpe = (avg_return - risk_free_rate) / std_return * (252 ** 0.5)
        
        return sharpe
        
    def calculate_profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss)
        
        Returns:
            Profit factor
        """
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0.0
            
        return self.gross_profit / self.gross_loss
        
    def calculate_win_rate(self) -> float:
        """
        Calculate win rate percentage
        
        Returns:
            Win rate (0.0 to 1.0)
        """
        if self.total_trades == 0:
            return 0.0
            
        return self.winning_trades / self.total_trades
        
    def calculate_average_win_loss(self) -> tuple:
        """
        Calculate average win and average loss
        
        Returns:
            Tuple of (average_win, average_loss)
        """
        avg_win = self.total_profit / self.winning_trades if self.winning_trades > 0 else 0.0
        avg_loss = self.total_loss / self.losing_trades if self.losing_trades > 0 else 0.0
        
        return (avg_win, abs(avg_loss))
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary
        
        Returns:
            Dictionary of performance metrics
        """
        total_return = self.current_equity - self.initial_equity
        total_return_pct = (total_return / self.initial_equity) if self.initial_equity > 0 else 0.0
        
        avg_win, avg_loss = self.calculate_average_win_loss()
        
        metrics = {
            # Equity metrics
            'initial_equity': self.initial_equity,
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': self.max_drawdown,
            
            # Trade metrics
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.calculate_win_rate(),
            
            # Profit metrics
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'profit_factor': self.calculate_profit_factor(),
            'average_win': avg_win,
            'average_loss': avg_loss,
            
            # Risk metrics
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            
            # Current position
            'current_position_size': self.current_position_size,
            'current_entry_price': self.current_entry_price
        }
        
        return metrics
        
    def print_summary(self):
        """Print performance summary to console"""
        metrics = self.get_metrics_summary()
        
        print("\n" + "="*60)
        print(f"PERFORMANCE SUMMARY - {self.symbol}")
        print("="*60)
        
        print(f"\nEquity:")
        print(f"  Initial:        ${metrics['initial_equity']:,.2f}")
        print(f"  Current:        ${metrics['current_equity']:,.2f}")
        print(f"  Peak:           ${metrics['peak_equity']:,.2f}")
        print(f"  Total Return:   ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2%})")
        print(f"  Max Drawdown:   {metrics['max_drawdown_pct']:.2%}")
        
        print(f"\nTrades:")
        print(f"  Total:          {metrics['total_trades']}")
        print(f"  Winners:        {metrics['winning_trades']}")
        print(f"  Losers:         {metrics['losing_trades']}")
        print(f"  Win Rate:       {metrics['win_rate']:.2%}")
        
        print(f"\nProfit/Loss:")
        print(f"  Gross Profit:   ${metrics['gross_profit']:,.2f}")
        print(f"  Gross Loss:     ${metrics['gross_loss']:,.2f}")
        print(f"  Profit Factor:  {metrics['profit_factor']:.2f}")
        print(f"  Avg Win:        ${metrics['average_win']:,.2f}")
        print(f"  Avg Loss:       ${metrics['average_loss']:,.2f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")
        
        print("="*60 + "\n")
        
    def save_metrics(self):
        """Save metrics to JSON file"""
        metrics = self.get_metrics_summary()
        
        # Add timestamp
        metrics['generated_at'] = datetime.now().isoformat()
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Metrics saved to {self.metrics_file}")
        
    def export_equity_curve(self, filename: Optional[str] = None) -> str:
        """
        Export equity curve to CSV
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"equity_curve_{self.symbol}_{timestamp}.csv"
            
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Equity'])
            
            for snapshot in self.equity_curve:
                writer.writerow([snapshot['timestamp'], snapshot['equity']])
                
        logger.info(f"Equity curve exported to {filepath}")
        return filepath
