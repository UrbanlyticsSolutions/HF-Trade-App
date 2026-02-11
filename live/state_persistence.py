"""
State Persistence for Live Trading

Saves and restores:
- Account capital tracking
- Daily P&L history
- Strategy state
- Position history

IMPORTANT: State is reconciled with DB on startup to ensure consistency.
"""
import json
import os
import sqlite3
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DailyRecord:
    """Record for a single trading day"""
    date: str
    starting_capital: float
    ending_capital: float
    trades: int
    wins: int
    losses: int
    pnl: float
    max_drawdown: float = 0.0
    notes: str = ""


@dataclass
class TradingState:
    """Persistent trading state"""
    # Capital tracking
    initial_capital: float = 10000.0
    current_capital: float = 10000.0
    high_water_mark: float = 10000.0
    
    # Cumulative stats
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    
    # Equity curve for charts
    equity_curve: List[Dict] = field(default_factory=list)
    
    # Daily history
    daily_records: List[Dict] = field(default_factory=list)
    
    # Last update
    last_updated: str = ""
    last_trade_date: str = ""
    
    # Engine status
    engine_status: str = "unknown"
    
    # Strategy-specific state
    strategy_state: Dict = field(default_factory=dict)


class StatePersistence:
    """
    Manages persistent state for live trading.
    
    Saves state to JSON file that can be reloaded on restart.
    Reconciles with database on startup to ensure consistency.
    """
    
    def __init__(self, state_file: str = "trading_state.json", db_path: str = None):
        """
        Initialize state persistence.
        
        Args:
            state_file: Path to state JSON file
            db_path: Path to trades database (for reconciliation)
        """
        self.state_file = state_file
        self.db_path = db_path
        self.state = TradingState()
        self._load_state()
        
        # Auto-reconcile with DB on startup if DB path provided
        if db_path and os.path.exists(db_path):
            self.reconcile_with_db()
    
    def reconcile_with_db(self, db_path: str = None) -> Dict[str, Any]:
        """
        Reconcile state with database - DB is source of truth.
        
        This recalculates all stats from closed trades in the database
        and updates the JSON state to match.
        
        Args:
            db_path: Optional path to database (uses self.db_path if not provided)
            
        Returns:
            Dict with reconciliation results
        """
        db_path = db_path or self.db_path
        if not db_path or not os.path.exists(db_path):
            logger.warning(f"Cannot reconcile - database not found: {db_path}")
            return {"status": "error", "message": "Database not found"}
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get stats from CLOSED trades only (DB is source of truth)
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins,
                    COUNT(CASE WHEN pnl <= 0 THEN 1 END) as losses,
                    COALESCE(SUM(pnl), 0) as total_pnl
                FROM trades 
                WHERE status = 'closed' AND pnl IS NOT NULL
            """)
            row = cursor.fetchone()
            
            db_total_trades = row[0] or 0
            db_wins = row[1] or 0
            db_losses = row[2] or 0
            db_total_pnl = row[3] or 0.0
            
            # Check for discrepancies
            old_state = {
                "total_trades": self.state.total_trades,
                "total_wins": self.state.total_wins,
                "total_losses": self.state.total_losses,
                "total_pnl": self.state.total_pnl,
                "current_capital": self.state.current_capital
            }
            
            discrepancies = []
            
            if self.state.total_trades != db_total_trades:
                discrepancies.append(f"total_trades: JSON={self.state.total_trades}, DB={db_total_trades}")
            if self.state.total_wins != db_wins:
                discrepancies.append(f"total_wins: JSON={self.state.total_wins}, DB={db_wins}")
            if self.state.total_losses != db_losses:
                discrepancies.append(f"total_losses: JSON={self.state.total_losses}, DB={db_losses}")
            if abs(self.state.total_pnl - db_total_pnl) > 0.01:
                discrepancies.append(f"total_pnl: JSON=${self.state.total_pnl:.2f}, DB=${db_total_pnl:.2f}")
            
            # Calculate correct capital
            db_current_capital = self.state.initial_capital + db_total_pnl
            
            if abs(self.state.current_capital - db_current_capital) > 0.01:
                discrepancies.append(f"current_capital: JSON=${self.state.current_capital:.2f}, calculated=${db_current_capital:.2f}")
            
            # Update state from DB
            self.state.total_trades = db_total_trades
            self.state.total_wins = db_wins
            self.state.total_losses = db_losses
            self.state.total_pnl = db_total_pnl
            self.state.current_capital = db_current_capital
            
            # Update high water mark if needed
            if self.state.current_capital > self.state.high_water_mark:
                self.state.high_water_mark = self.state.current_capital
            
            # Recalculate max drawdown from trade history
            cursor.execute("""
                SELECT pnl, exit_time 
                FROM trades 
                WHERE status = 'closed' AND pnl IS NOT NULL 
                ORDER BY exit_time
            """)
            trades = cursor.fetchall()
            
            if trades:
                running_capital = self.state.initial_capital
                running_hwm = self.state.initial_capital
                max_dd = 0.0
                
                for pnl, _ in trades:
                    running_capital += pnl
                    if running_capital > running_hwm:
                        running_hwm = running_capital
                    dd = (running_hwm - running_capital) / running_hwm if running_hwm > 0 else 0
                    if dd > max_dd:
                        max_dd = dd
                
                self.state.max_drawdown = max_dd
                self.state.high_water_mark = running_hwm
            
            # Get last trade date
            cursor.execute("""
                SELECT MAX(DATE(exit_time)) FROM trades WHERE status = 'closed'
            """)
            last_date = cursor.fetchone()[0]
            if last_date:
                self.state.last_trade_date = last_date
            
            conn.close()
            
            # Save reconciled state
            self.save_state()
            
            if discrepancies:
                logger.warning(f"Reconciled {len(discrepancies)} discrepancies with DB:")
                for d in discrepancies:
                    logger.warning(f"  - {d}")
            else:
                logger.info("State reconciled with DB - no discrepancies found")
            
            return {
                "status": "success",
                "discrepancies": discrepancies,
                "old_state": old_state,
                "new_state": {
                    "total_trades": self.state.total_trades,
                    "total_wins": self.state.total_wins,
                    "total_losses": self.state.total_losses,
                    "total_pnl": self.state.total_pnl,
                    "current_capital": self.state.current_capital,
                    "max_drawdown": self.state.max_drawdown
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to reconcile with DB: {e}")
            return {"status": "error", "message": str(e)}
    
    def _load_state(self):
        """Load state from file if exists"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                self.state = TradingState(
                    initial_capital=data.get('initial_capital', 10000),
                    current_capital=data.get('current_capital', 10000),
                    high_water_mark=data.get('high_water_mark', 10000),
                    total_trades=data.get('total_trades', 0),
                    total_wins=data.get('total_wins', 0),
                    total_losses=data.get('total_losses', 0),
                    total_pnl=data.get('total_pnl', 0),
                    max_drawdown=data.get('max_drawdown', 0),
                    equity_curve=data.get('equity_curve', []),
                    daily_records=data.get('daily_records', []),
                    last_updated=data.get('last_updated', ''),
                    last_trade_date=data.get('last_trade_date', ''),
                    engine_status=data.get('engine_status', 'unknown'),
                    strategy_state=data.get('strategy_state', {})
                )
                
                logger.info(f"Loaded state from {self.state_file}")
                logger.info(f"  Capital: ${self.state.current_capital:,.2f}")
                logger.info(f"  Total P&L: ${self.state.total_pnl:,.2f}")
                logger.info(f"  Total Trades: {self.state.total_trades}")
                
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                self.state = TradingState()
        else:
            logger.info(f"No existing state file, starting fresh")
    
    def save_state(self):
        """Save current state to file"""
        self.state.last_updated = datetime.now().isoformat()
        
        try:
            data = asdict(self.state)
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def set_initial_capital(self, capital: float):
        """Set initial capital (only if not already set)"""
        if self.state.total_trades == 0:
            self.state.initial_capital = capital
            self.state.current_capital = capital
            self.state.high_water_mark = capital
            self.save_state()
    
    def record_trade(self, pnl: float, is_win: bool, trade_id: int = None, option_type: str = None):
        """
        Record a completed trade.
        
        Args:
            pnl: Trade P&L in dollars
            is_win: Whether trade was profitable
            trade_id: Optional trade ID for equity curve
            option_type: Optional option type (PUT/CALL) for equity curve
        """
        self.state.total_trades += 1
        self.state.total_pnl += pnl
        self.state.current_capital += pnl
        
        if is_win:
            self.state.total_wins += 1
        else:
            self.state.total_losses += 1
        
        # Update high water mark
        if self.state.current_capital > self.state.high_water_mark:
            self.state.high_water_mark = self.state.current_capital
        
        # Update max drawdown
        drawdown = (self.state.high_water_mark - self.state.current_capital) / self.state.high_water_mark
        if drawdown > self.state.max_drawdown:
            self.state.max_drawdown = drawdown
        
        # Update equity curve
        if trade_id is not None:
            # Initialize equity curve if empty
            if not self.state.equity_curve:
                self.state.equity_curve = [
                    {"trade_id": 0, "type": "-", "equity": self.state.initial_capital, "pnl": 0}
                ]
            
            self.state.equity_curve.append({
                "trade_id": trade_id,
                "type": (option_type or "").upper(),
                "equity": self.state.current_capital,
                "pnl": pnl,
                "time": datetime.now().isoformat()
            })
        
        self.state.last_trade_date = datetime.now().strftime("%Y-%m-%d")
        self.save_state()
    
    def record_daily_summary(
        self,
        trades: int,
        wins: int,
        losses: int,
        pnl: float,
        notes: str = ""
    ):
        """
        Record end-of-day summary.
        
        Args:
            trades: Number of trades today
            wins: Number of winning trades
            losses: Number of losing trades
            pnl: Total P&L for the day
            notes: Optional notes
        """
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Calculate starting capital (current - today's pnl)
        starting = self.state.current_capital - pnl
        
        record = DailyRecord(
            date=today,
            starting_capital=starting,
            ending_capital=self.state.current_capital,
            trades=trades,
            wins=wins,
            losses=losses,
            pnl=pnl,
            max_drawdown=self.state.max_drawdown,
            notes=notes
        )
        
        # Update or append daily record
        existing = next((r for r in self.state.daily_records if r.get('date') == today), None)
        if existing:
            idx = self.state.daily_records.index(existing)
            self.state.daily_records[idx] = asdict(record)
        else:
            self.state.daily_records.append(asdict(record))
        
        self.save_state()
    
    def save_strategy_state(self, strategy_name: str, state: Dict):
        """Save strategy-specific state"""
        self.state.strategy_state[strategy_name] = state
        self.save_state()
    
    def get_strategy_state(self, strategy_name: str) -> Dict:
        """Get strategy-specific state"""
        return self.state.strategy_state.get(strategy_name, {})
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of trading state"""
        win_rate = self.state.total_wins / self.state.total_trades if self.state.total_trades > 0 else 0
        
        return {
            "initial_capital": self.state.initial_capital,
            "current_capital": self.state.current_capital,
            "total_pnl": self.state.total_pnl,
            "total_pnl_pct": (self.state.total_pnl / self.state.initial_capital * 100) if self.state.initial_capital > 0 else 0,
            "total_trades": self.state.total_trades,
            "total_wins": self.state.total_wins,
            "total_losses": self.state.total_losses,
            "win_rate": win_rate,
            "high_water_mark": self.state.high_water_mark,
            "max_drawdown": self.state.max_drawdown,
            "trading_days": len(self.state.daily_records),
            "last_trade_date": self.state.last_trade_date
        }
    
    def print_summary(self):
        """Print trading summary"""
        s = self.get_summary()
        
        print("\n" + "=" * 60)
        print("TRADING STATE SUMMARY")
        print("=" * 60)
        print(f"Initial Capital:    ${s['initial_capital']:>12,.2f}")
        print(f"Current Capital:    ${s['current_capital']:>12,.2f}")
        print(f"Total P&L:          ${s['total_pnl']:>12,.2f} ({s['total_pnl_pct']:+.1f}%)")
        print("-" * 60)
        print(f"Total Trades:       {s['total_trades']:>12}")
        print(f"Wins / Losses:      {s['total_wins']:>5} / {s['total_losses']:<5}")
        print(f"Win Rate:           {s['win_rate']:>12.1%}")
        print(f"Max Drawdown:       {s['max_drawdown']:>12.1%}")
        print("-" * 60)
        print(f"Trading Days:       {s['trading_days']:>12}")
        last_trade = s['last_trade_date'] or 'N/A'
        print(f"Last Trade:         {last_trade:>12}")
        print("=" * 60)
    
    def get_daily_history(self, days: int = 30) -> List[Dict]:
        """Get recent daily history"""
        return self.state.daily_records[-days:]
    
    def reset(self, initial_capital: float = 10000):
        """Reset all state (use with caution!)"""
        self.state = TradingState(
            initial_capital=initial_capital,
            current_capital=initial_capital,
            high_water_mark=initial_capital
        )
        self.save_state()
        logger.info(f"State reset with ${initial_capital:,.2f} capital")


# Singleton instance for easy access
_default_persistence: Optional[StatePersistence] = None


def get_persistence(state_file: str = "trading_state.json", db_path: str = None) -> StatePersistence:
    """Get or create the default persistence instance"""
    global _default_persistence
    if _default_persistence is None or _default_persistence.state_file != state_file:
        _default_persistence = StatePersistence(state_file, db_path=db_path)
    return _default_persistence
