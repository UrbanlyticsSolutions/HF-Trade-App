"""
Trade Database - SQLite storage for live trades

Stores all trade information including:
- Order details (entry, exit)
- Position tracking
- P&L calculation
- Option Greeks at time of trade
- Real-time quote snapshots
"""
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trade record"""
    id: Optional[int] = None
    symbol: str = ""
    underlying: str = ""
    trade_type: str = ""  # option, stock
    option_type: Optional[str] = None  # call, put
    strike: Optional[float] = None
    expiration: Optional[str] = None
    action: str = ""  # buy, sell
    quantity: int = 0
    entry_price: float = 0.0
    entry_time: str = ""
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    status: str = "open"  # open, closed, cancelled
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    commission: float = 0.0
    # Greeks at entry
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    iv: Optional[float] = None
    # Underlying price at entry/exit
    underlying_price_entry: Optional[float] = None
    underlying_price_exit: Optional[float] = None
    # Order IDs
    entry_order_id: Optional[int] = None
    exit_order_id: Optional[int] = None
    # Strategy info
    strategy_name: Optional[str] = None
    strategy_params: Optional[str] = None  # JSON string
    notes: Optional[str] = None
    account_id: Optional[str] = None


@dataclass
class QuoteSnapshot:
    """Quote snapshot at a point in time"""
    id: Optional[int] = None
    symbol: str = ""
    timestamp: str = ""
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    last_price: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    iv: Optional[float] = None
    # Underlying
    underlying_price: Optional[float] = None


class TradeDatabase:
    """
    SQLite database for storing live trades and quote snapshots.
    """
    
    def __init__(self, db_path: str = "live_trades.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()
        logger.info(f"Trade database initialized: {self.db_path}")
    
    def _init_tables(self):
        """Initialize database tables"""
        cursor = self.conn.cursor()
        
        # Main trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                underlying TEXT,
                trade_type TEXT NOT NULL,
                option_type TEXT,
                strike REAL,
                expiration TEXT,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_price REAL,
                exit_time TEXT,
                status TEXT DEFAULT 'open',
                pnl REAL,
                pnl_percent REAL,
                commission REAL DEFAULT 0,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                iv REAL,
                underlying_price_entry REAL,
                underlying_price_exit REAL,
                entry_order_id INTEGER,
                exit_order_id INTEGER,
                strategy_name TEXT,
                strategy_params TEXT,
                notes TEXT,
                account_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Quote snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quote_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                bid_price REAL,
                ask_price REAL,
                last_price REAL,
                bid_size INTEGER,
                ask_size INTEGER,
                volume INTEGER,
                open_interest INTEGER,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                iv REAL,
                underlying_price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Orders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER,
                symbol TEXT NOT NULL,
                account_id TEXT,
                action TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                limit_price REAL,
                stop_price REAL,
                filled_quantity INTEGER DEFAULT 0,
                avg_fill_price REAL,
                status TEXT DEFAULT 'pending',
                submitted_at TEXT,
                filled_at TEXT,
                cancelled_at TEXT,
                error_message TEXT,
                trade_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        ''')
        
        # Daily P&L summary
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_pnl (
                date TEXT PRIMARY KEY,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                trades_opened INTEGER DEFAULT 0,
                trades_closed INTEGER DEFAULT 0,
                win_count INTEGER DEFAULT 0,
                loss_count INTEGER DEFAULT 0,
                commission_total REAL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Position history for tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_cost REAL,
                market_value REAL,
                unrealized_pnl REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_quotes_symbol ON quote_snapshots(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)')
        
        self.conn.commit()
    
    # ==================== TRADE OPERATIONS ====================
    
    def insert_trade(self, trade: Trade) -> int:
        """Insert a new trade and return its ID"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (
                symbol, underlying, trade_type, option_type, strike, expiration,
                action, quantity, entry_price, entry_time, exit_price, exit_time,
                status, pnl, pnl_percent, commission, delta, gamma, theta, vega, iv,
                underlying_price_entry, underlying_price_exit, entry_order_id, exit_order_id,
                strategy_name, strategy_params, notes, account_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.symbol, trade.underlying, trade.trade_type, trade.option_type,
            trade.strike, trade.expiration, trade.action, trade.quantity,
            trade.entry_price, trade.entry_time, trade.exit_price, trade.exit_time,
            trade.status, trade.pnl, trade.pnl_percent, trade.commission,
            trade.delta, trade.gamma, trade.theta, trade.vega, trade.iv,
            trade.underlying_price_entry, trade.underlying_price_exit,
            trade.entry_order_id, trade.exit_order_id,
            trade.strategy_name, trade.strategy_params, trade.notes, trade.account_id
        ))
        
        self.conn.commit()
        trade_id = cursor.lastrowid
        logger.info(f"Inserted trade {trade_id}: {trade.action} {trade.quantity} {trade.symbol} @ {trade.entry_price}")
        return trade_id
    
    def update_trade(self, trade_id: int, **updates) -> bool:
        """Update trade fields"""
        if not updates:
            return False
        
        updates['updated_at'] = datetime.now().isoformat()
        
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [trade_id]
        
        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE trades SET {set_clause} WHERE id = ?", values)
        self.conn.commit()
        
        return cursor.rowcount > 0
    
    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_time: Optional[str] = None,
        exit_order_id: Optional[int] = None,
        underlying_price_exit: Optional[float] = None
    ) -> Optional[Trade]:
        """Close a trade and calculate P&L"""
        trade = self.get_trade(trade_id)
        if not trade:
            return None
        
        exit_time = exit_time or datetime.now().isoformat()
        
        # Calculate P&L
        if trade['action'].lower() == 'buy':
            pnl = (exit_price - trade['entry_price']) * trade['quantity'] * 100  # Options multiplier
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['quantity'] * 100
        
        pnl -= trade['commission']
        pnl_percent = (pnl / (trade['entry_price'] * trade['quantity'] * 100)) * 100
        
        self.update_trade(
            trade_id,
            exit_price=exit_price,
            exit_time=exit_time,
            exit_order_id=exit_order_id,
            underlying_price_exit=underlying_price_exit,
            status='closed',
            pnl=pnl,
            pnl_percent=pnl_percent
        )
        
        # Update daily P&L
        self._update_daily_pnl(exit_time[:10], pnl, is_win=(pnl > 0))
        
        logger.info(f"Closed trade {trade_id}: P&L ${pnl:.2f} ({pnl_percent:.2f}%)")
        return self.get_trade(trade_id)
    
    def get_trade(self, trade_id: int) -> Optional[Dict]:
        """Get trade by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_open_trades(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open trades"""
        cursor = self.conn.cursor()
        
        if symbol:
            cursor.execute("SELECT * FROM trades WHERE status = 'open' AND symbol = ?", (symbol,))
        else:
            cursor.execute("SELECT * FROM trades WHERE status = 'open'")
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_trades_by_date(self, date: str) -> List[Dict]:
        """Get all trades for a specific date"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM trades WHERE DATE(entry_time) = ? ORDER BY entry_time",
            (date,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_trades_by_strategy(self, strategy_name: str) -> List[Dict]:
        """Get all trades for a strategy"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM trades WHERE strategy_name = ? ORDER BY entry_time DESC",
            (strategy_name,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Get most recent trades"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?",
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    # ==================== QUOTE SNAPSHOTS ====================
    
    def insert_quote_snapshot(self, snapshot: QuoteSnapshot) -> int:
        """Insert a quote snapshot"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO quote_snapshots (
                symbol, timestamp, bid_price, ask_price, last_price,
                bid_size, ask_size, volume, open_interest,
                delta, gamma, theta, vega, iv, underlying_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            snapshot.symbol, snapshot.timestamp, snapshot.bid_price,
            snapshot.ask_price, snapshot.last_price, snapshot.bid_size,
            snapshot.ask_size, snapshot.volume, snapshot.open_interest,
            snapshot.delta, snapshot.gamma, snapshot.theta, snapshot.vega,
            snapshot.iv, snapshot.underlying_price
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_quote_snapshots_bulk(self, snapshots: List[QuoteSnapshot]) -> int:
        """Bulk insert quote snapshots"""
        cursor = self.conn.cursor()
        
        data = [
            (s.symbol, s.timestamp, s.bid_price, s.ask_price, s.last_price,
             s.bid_size, s.ask_size, s.volume, s.open_interest,
             s.delta, s.gamma, s.theta, s.vega, s.iv, s.underlying_price)
            for s in snapshots
        ]
        
        cursor.executemany('''
            INSERT INTO quote_snapshots (
                symbol, timestamp, bid_price, ask_price, last_price,
                bid_size, ask_size, volume, open_interest,
                delta, gamma, theta, vega, iv, underlying_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data)
        
        self.conn.commit()
        return len(data)
    
    def get_quote_history(
        self, 
        symbol: str, 
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Get quote history for a symbol"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM quote_snapshots WHERE symbol = ?"
        params = [symbol]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    # ==================== ORDER TRACKING ====================
    
    def insert_order(
        self,
        order_id: int,
        symbol: str,
        account_id: str,
        action: str,
        order_type: str,
        quantity: int,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trade_id: Optional[int] = None
    ) -> int:
        """Insert an order record"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO orders (
                order_id, symbol, account_id, action, order_type, quantity,
                limit_price, stop_price, status, submitted_at, trade_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'submitted', ?, ?)
        ''', (
            order_id, symbol, account_id, action, order_type, quantity,
            limit_price, stop_price, datetime.now().isoformat(), trade_id
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def update_order_status(
        self,
        order_id: int,
        status: str,
        filled_quantity: Optional[int] = None,
        avg_fill_price: Optional[float] = None,
        error_message: Optional[str] = None
    ):
        """Update order status"""
        cursor = self.conn.cursor()
        
        updates = {"status": status}
        if filled_quantity is not None:
            updates["filled_quantity"] = filled_quantity
        if avg_fill_price is not None:
            updates["avg_fill_price"] = avg_fill_price
        if error_message:
            updates["error_message"] = error_message
        
        if status == 'filled':
            updates["filled_at"] = datetime.now().isoformat()
        elif status == 'cancelled':
            updates["cancelled_at"] = datetime.now().isoformat()
        
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [order_id]
        
        cursor.execute(f"UPDATE orders SET {set_clause} WHERE order_id = ?", values)
        self.conn.commit()
    
    def get_pending_orders(self) -> List[Dict]:
        """Get all pending orders"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM orders WHERE status IN ('submitted', 'pending', 'partial') ORDER BY submitted_at"
        )
        return [dict(row) for row in cursor.fetchall()]
    
    # ==================== P&L TRACKING ====================
    
    def _update_daily_pnl(self, date: str, pnl: float, is_win: bool):
        """Update daily P&L summary"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT * FROM daily_pnl WHERE date = ?", (date,))
        row = cursor.fetchone()
        
        if row:
            cursor.execute('''
                UPDATE daily_pnl SET
                    realized_pnl = realized_pnl + ?,
                    total_pnl = realized_pnl + unrealized_pnl,
                    trades_closed = trades_closed + 1,
                    win_count = win_count + ?,
                    loss_count = loss_count + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE date = ?
            ''', (pnl, 1 if is_win else 0, 0 if is_win else 1, date))
        else:
            cursor.execute('''
                INSERT INTO daily_pnl (date, realized_pnl, total_pnl, trades_closed, win_count, loss_count)
                VALUES (?, ?, ?, 1, ?, ?)
            ''', (date, pnl, pnl, 1 if is_win else 0, 0 if is_win else 1))
        
        self.conn.commit()
    
    def update_unrealized_pnl(self, date: str, unrealized_pnl: float):
        """Update unrealized P&L for a date"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT * FROM daily_pnl WHERE date = ?", (date,))
        row = cursor.fetchone()
        
        if row:
            cursor.execute('''
                UPDATE daily_pnl SET
                    unrealized_pnl = ?,
                    total_pnl = realized_pnl + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE date = ?
            ''', (unrealized_pnl, unrealized_pnl, date))
        else:
            cursor.execute('''
                INSERT INTO daily_pnl (date, unrealized_pnl, total_pnl)
                VALUES (?, ?, ?)
            ''', (date, unrealized_pnl, unrealized_pnl))
        
        self.conn.commit()
    
    def get_daily_pnl(self, date: str) -> Optional[Dict]:
        """Get P&L summary for a date"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM daily_pnl WHERE date = ?", (date,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_pnl_history(self, days: int = 30) -> List[Dict]:
        """Get P&L history for last N days"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM daily_pnl ORDER BY date DESC LIMIT ?",
            (days,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    # ==================== STATISTICS ====================
    
    def get_trade_statistics(self, strategy_name: Optional[str] = None) -> Dict:
        """Get comprehensive trade statistics"""
        cursor = self.conn.cursor()
        
        where_clause = "WHERE status = 'closed'"
        params = []
        
        if strategy_name:
            where_clause += " AND strategy_name = ?"
            params.append(strategy_name)
        
        # Total trades
        cursor.execute(f"SELECT COUNT(*) FROM trades {where_clause}", params)
        total_trades = cursor.fetchone()[0]
        
        if total_trades == 0:
            return {"total_trades": 0, "message": "No closed trades"}
        
        # Win/Loss
        cursor.execute(f"SELECT COUNT(*) FROM trades {where_clause} AND pnl > 0", params)
        wins = cursor.fetchone()[0]
        
        cursor.execute(f"SELECT COUNT(*) FROM trades {where_clause} AND pnl <= 0", params)
        losses = cursor.fetchone()[0]
        
        # P&L stats
        cursor.execute(f"SELECT SUM(pnl), AVG(pnl), MAX(pnl), MIN(pnl) FROM trades {where_clause}", params)
        row = cursor.fetchone()
        total_pnl, avg_pnl, max_pnl, min_pnl = row
        
        # Average win/loss
        cursor.execute(f"SELECT AVG(pnl) FROM trades {where_clause} AND pnl > 0", params)
        avg_win = cursor.fetchone()[0] or 0
        
        cursor.execute(f"SELECT AVG(pnl) FROM trades {where_clause} AND pnl <= 0", params)
        avg_loss = cursor.fetchone()[0] or 0
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else float('inf')
        
        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl or 0, 2),
            "avg_pnl": round(avg_pnl or 0, 2),
            "max_pnl": round(max_pnl or 0, 2),
            "min_pnl": round(min_pnl or 0, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "∞"
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Trade database closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
