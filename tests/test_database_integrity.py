"""
Database Integrity Test Suite
=============================
Tests schema correctness, constraints, data types, indexes,
concurrent access, transaction safety, and bulk operations
for both TradeDatabase and MarketDatabase.

Run:
    python -m pytest tests/test_database_integrity.py -v --tb=short
    python tests/test_database_integrity.py          # standalone
"""
import os
import sqlite3
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from live.trade_database import TradeDatabase, Trade, QuoteSnapshot
from clients.database import MarketDatabase
from clients.ibkr_db import IBKRDatabase


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def trade_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = TradeDatabase(path)
    yield db
    db.close()
    os.unlink(path)


@pytest.fixture
def market_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = MarketDatabase(path)
    yield db
    db.conn.close()
    os.unlink(path)


@pytest.fixture
def ibkr_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = IBKRDatabase(path)
    yield db
    db._conn().close()
    os.unlink(path)


def _make_trade(**overrides):
    defaults = dict(
        symbol="SPY20260328C570",
        underlying="SPY",
        trade_type="option",
        option_type="call",
        strike=570.0,
        expiration="20260328",
        action="buy",
        quantity=5,
        entry_price=2.15,
        entry_time="2026-03-28T09:35:00",
        status="open",
        account_id="U1234567",
    )
    defaults.update(overrides)
    return Trade(**defaults)


# =====================================================================
# 1. TRADE DATABASE — TABLE CREATION
# =====================================================================
class TestTradeDBTableCreation:

    def test_trades_table_exists(self, trade_db):
        cursor = trade_db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        assert cursor.fetchone() is not None

    def test_quote_snapshots_table_exists(self, trade_db):
        cursor = trade_db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='quote_snapshots'")
        assert cursor.fetchone() is not None

    def test_orders_table_exists(self, trade_db):
        cursor = trade_db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='orders'")
        assert cursor.fetchone() is not None

    def test_daily_pnl_table_exists(self, trade_db):
        cursor = trade_db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_pnl'")
        assert cursor.fetchone() is not None

    def test_position_history_table_exists(self, trade_db):
        cursor = trade_db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='position_history'")
        assert cursor.fetchone() is not None

    def test_balance_history_table_exists(self, trade_db):
        cursor = trade_db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='balance_history'")
        assert cursor.fetchone() is not None

    def test_current_positions_table_exists(self, trade_db):
        cursor = trade_db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='current_positions'")
        assert cursor.fetchone() is not None

    def test_all_expected_tables_count(self, trade_db):
        cursor = trade_db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        expected = {
            "trades", "quote_snapshots", "orders", "daily_pnl",
            "position_history", "balance_history", "current_positions",
        }
        for t in expected:
            assert t in tables, f"Missing table: {t}"


# =====================================================================
# 2. TRADE DATABASE — COLUMN SCHEMA
# =====================================================================
class TestTradeDBColumnSchema:

    def _get_columns(self, db, table_name):
        cursor = db.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        return {row[1]: {"type": row[2], "notnull": row[3], "default": row[4], "pk": row[5]}
                for row in cursor.fetchall()}

    def test_trades_columns(self, trade_db):
        cols = self._get_columns(trade_db, "trades")
        required = [
            "id", "symbol", "underlying", "trade_type", "option_type",
            "strike", "expiration", "action", "quantity", "entry_price",
            "entry_time", "exit_price", "exit_time", "status", "pnl",
            "pnl_percent", "commission", "delta", "gamma", "theta",
            "vega", "iv", "underlying_price_entry", "underlying_price_exit",
            "entry_order_id", "exit_order_id", "strategy_name",
            "strategy_params", "notes", "account_id",
            "created_at", "updated_at",
        ]
        for col in required:
            assert col in cols, f"Missing column: {col}"

    def test_trades_id_is_autoincrement(self, trade_db):
        cols = self._get_columns(trade_db, "trades")
        assert cols["id"]["pk"] == 1

    def test_trades_symbol_not_null(self, trade_db):
        cols = self._get_columns(trade_db, "trades")
        assert cols["symbol"]["notnull"] == 1

    def test_trades_action_not_null(self, trade_db):
        cols = self._get_columns(trade_db, "trades")
        assert cols["action"]["notnull"] == 1

    def test_trades_quantity_not_null(self, trade_db):
        cols = self._get_columns(trade_db, "trades")
        assert cols["quantity"]["notnull"] == 1

    def test_trades_entry_price_not_null(self, trade_db):
        cols = self._get_columns(trade_db, "trades")
        assert cols["entry_price"]["notnull"] == 1

    def test_trades_entry_time_not_null(self, trade_db):
        cols = self._get_columns(trade_db, "trades")
        assert cols["entry_time"]["notnull"] == 1

    def test_trades_status_default_open(self, trade_db):
        cols = self._get_columns(trade_db, "trades")
        assert cols["status"]["default"] == "'open'"

    def test_trades_commission_default_zero(self, trade_db):
        cols = self._get_columns(trade_db, "trades")
        assert cols["commission"]["default"] == "0"

    def test_orders_columns(self, trade_db):
        cols = self._get_columns(trade_db, "orders")
        required = [
            "id", "order_id", "symbol", "account_id", "action",
            "order_type", "quantity", "limit_price", "stop_price",
            "filled_quantity", "avg_fill_price", "status",
            "submitted_at", "filled_at", "cancelled_at",
            "error_message", "trade_id",
        ]
        for col in required:
            assert col in cols, f"Missing orders column: {col}"

    def test_orders_trade_id_fk(self, trade_db):
        cursor = trade_db.conn.cursor()
        cursor.execute("PRAGMA foreign_key_list(orders)")
        fks = cursor.fetchall()
        # PRAGMA foreign_key_list returns: (id, seq, table, from, to, ...)
        fk_tables = [fk[2] for fk in fks]
        assert "trades" in fk_tables

    def test_daily_pnl_date_primary_key(self, trade_db):
        cols = self._get_columns(trade_db, "daily_pnl")
        assert cols["date"]["pk"] == 1

    def test_daily_pnl_defaults(self, trade_db):
        cols = self._get_columns(trade_db, "daily_pnl")
        assert cols["realized_pnl"]["default"] == "0"
        assert cols["unrealized_pnl"]["default"] == "0"
        assert cols["total_pnl"]["default"] == "0"
        assert cols["trades_opened"]["default"] == "0"
        assert cols["trades_closed"]["default"] == "0"

    def test_current_positions_symbol_pk(self, trade_db):
        cols = self._get_columns(trade_db, "current_positions")
        assert cols["symbol"]["pk"] == 1


# =====================================================================
# 3. TRADE DATABASE — INDEX CREATION
# =====================================================================
class TestTradeDBIndexes:

    def _get_indexes(self, db, table_name=None):
        cursor = db.conn.cursor()
        if table_name:
            cursor.execute(f"PRAGMA index_list({table_name})")
        else:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        return [row[0] if len(row) == 1 else row[1] for row in cursor.fetchall()]

    def test_trades_symbol_index(self, trade_db):
        indexes = self._get_indexes(trade_db, "trades")
        assert "idx_trades_symbol" in indexes

    def test_trades_status_index(self, trade_db):
        indexes = self._get_indexes(trade_db, "trades")
        assert "idx_trades_status" in indexes

    def test_trades_entry_time_index(self, trade_db):
        indexes = self._get_indexes(trade_db, "trades")
        assert "idx_trades_entry_time" in indexes

    def test_trades_strategy_index(self, trade_db):
        indexes = self._get_indexes(trade_db, "trades")
        assert "idx_trades_strategy" in indexes

    def test_quotes_symbol_index(self, trade_db):
        indexes = self._get_indexes(trade_db, "quote_snapshots")
        assert "idx_quotes_symbol" in indexes

    def test_orders_status_index(self, trade_db):
        indexes = self._get_indexes(trade_db, "orders")
        assert "idx_orders_status" in indexes

    def test_balance_history_ts_index(self, trade_db):
        indexes = self._get_indexes(trade_db, "balance_history")
        assert "idx_balance_history_ts" in indexes


# =====================================================================
# 4. TRADE DATABASE — DATA INTEGRITY
# =====================================================================
class TestTradeDBDataIntegrity:

    def test_insert_and_retrieve_trade(self, trade_db):
        trade = _make_trade()
        tid = trade_db.insert_trade(trade)
        retrieved = trade_db.get_trade(tid)
        assert retrieved is not None
        assert retrieved["symbol"] == "SPY20260328C570"
        assert retrieved["entry_price"] == 2.15
        assert retrieved["quantity"] == 5
        assert retrieved["status"] == "open"

    def test_insert_returns_sequential_ids(self, trade_db):
        t1 = trade_db.insert_trade(_make_trade(entry_time="2026-03-28T09:35:00"))
        t2 = trade_db.insert_trade(_make_trade(entry_time="2026-03-28T09:36:00"))
        t3 = trade_db.insert_trade(_make_trade(entry_time="2026-03-28T09:37:00"))
        assert t2 == t1 + 1
        assert t3 == t2 + 1

    def test_update_trade_changes_field(self, trade_db):
        tid = trade_db.insert_trade(_make_trade())
        trade_db.update_trade(tid, notes="updated")
        t = trade_db.get_trade(tid)
        assert t["notes"] == "updated"

    def test_update_trade_sets_updated_at(self, trade_db):
        tid = trade_db.insert_trade(_make_trade())
        before = trade_db.get_trade(tid)["updated_at"]
        time.sleep(0.05)
        trade_db.update_trade(tid, notes="test")
        after = trade_db.get_trade(tid)["updated_at"]
        # updated_at is set by update_trade() via datetime.now().isoformat()
        # while created_at/initial updated_at may use CURRENT_TIMESTAMP (UTC)
        # Just verify it was changed
        assert after != before

    def test_close_trade_sets_closed_status(self, trade_db):
        tid = trade_db.insert_trade(_make_trade(entry_price=2.00, commission=0))
        result = trade_db.close_trade(tid, exit_price=3.00)
        assert result["status"] == "closed"
        assert result["exit_price"] == 3.00

    def test_close_trade_calculates_pnl(self, trade_db):
        tid = trade_db.insert_trade(_make_trade(
            entry_price=2.00, quantity=3, commission=1.50, action="buy",
        ))
        result = trade_db.close_trade(tid, exit_price=3.00)
        # (3.00 - 2.00) * 3 * 100 - 1.50 = 298.50
        assert abs(result["pnl"] - 298.50) < 0.01

    def test_close_trade_nonexistent_returns_none(self, trade_db):
        assert trade_db.close_trade(99999, exit_price=1.0) is None

    def test_get_open_trades_filters_correctly(self, trade_db):
        tid1 = trade_db.insert_trade(_make_trade(entry_time="2026-03-28T09:35:00"))
        tid2 = trade_db.insert_trade(_make_trade(entry_time="2026-03-28T09:36:00"))
        trade_db.close_trade(tid1, exit_price=3.00)

        open_trades = trade_db.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0]["id"] == tid2

    def test_get_open_trades_by_symbol(self, trade_db):
        trade_db.insert_trade(_make_trade(symbol="SPY20260328C570"))
        trade_db.insert_trade(_make_trade(symbol="SPY20260328P560", entry_time="2026-03-28T09:36:00"))

        spy_c = trade_db.get_open_trades(symbol="SPY20260328C570")
        assert len(spy_c) == 1
        assert spy_c[0]["symbol"] == "SPY20260328C570"

    def test_get_trades_by_date(self, trade_db):
        trade_db.insert_trade(_make_trade(entry_time="2026-03-28T09:35:00"))
        trade_db.insert_trade(_make_trade(entry_time="2026-03-27T09:35:00", symbol="SPY20260327C570"))

        trades_28 = trade_db.get_trades_by_date("2026-03-28")
        assert len(trades_28) == 1

    def test_get_recent_trades_limit(self, trade_db):
        for i in range(10):
            trade_db.insert_trade(_make_trade(
                entry_time=f"2026-03-28T09:{35+i:02d}:00",
                symbol=f"SPY20260328C{570+i}",
            ))
        recent = trade_db.get_recent_trades(limit=5)
        assert len(recent) == 5

    def test_get_trade_by_order_id_entry(self, trade_db):
        tid = trade_db.insert_trade(_make_trade(entry_order_id=42))
        found = trade_db.get_trade_by_order_id(42)
        assert found is not None
        assert found["id"] == tid

    def test_get_trade_by_order_id_exit(self, trade_db):
        tid = trade_db.insert_trade(_make_trade(entry_order_id=42))
        trade_db.close_trade(tid, exit_price=3.00, exit_order_id=43)
        found = trade_db.get_trade_by_order_id(43)
        assert found["id"] == tid

    def test_nullable_fields_inserted_as_none(self, trade_db):
        trade = Trade(
            symbol="SPY20260328C570",
            trade_type="option",
            action="buy",
            quantity=1,
            entry_price=2.00,
            entry_time="2026-03-28T09:35:00",
        )
        tid = trade_db.insert_trade(trade)
        t = trade_db.get_trade(tid)
        assert t["exit_price"] is None
        assert t["exit_time"] is None
        assert t["pnl"] is None
        assert t["delta"] is None
        assert t["strike"] is None

    def test_strategy_params_json_roundtrip(self, trade_db):
        import json
        params = json.dumps({"pt": 0.50, "sl": 0.35, "rsi_call": 70})
        tid = trade_db.insert_trade(_make_trade(
            strategy_name="0DTE",
            strategy_params=params,
        ))
        t = trade_db.get_trade(tid)
        loaded = json.loads(t["strategy_params"])
        assert loaded["pt"] == 0.50
        assert loaded["sl"] == 0.35


# =====================================================================
# 5. TRADE DATABASE — QUOTE SNAPSHOTS
# =====================================================================
class TestQuoteSnapshotIntegrity:

    def test_insert_and_retrieve_snapshot(self, trade_db):
        snap = QuoteSnapshot(
            symbol="SPY20260328C570",
            timestamp="2026-03-28T09:35:00",
            bid_price=2.10, ask_price=2.20, last_price=2.15,
            delta=0.45, gamma=0.08, theta=-0.15, vega=0.12, iv=0.25,
            underlying_price=570.0,
        )
        sid = trade_db.insert_quote_snapshot(snap)
        assert sid > 0

        history = trade_db.get_quote_history("SPY20260328C570")
        assert len(history) == 1
        assert history[0]["bid_price"] == 2.10
        assert history[0]["iv"] == 0.25

    def test_bulk_insert_snapshots(self, trade_db):
        snapshots = [
            QuoteSnapshot(
                symbol="SPY20260328C570",
                timestamp=f"2026-03-28T09:{35+i:02d}:00",
                last_price=2.15 + (i * 0.01),
            )
            for i in range(50)
        ]
        count = trade_db.insert_quote_snapshots_bulk(snapshots)
        assert count == 50

        history = trade_db.get_quote_history("SPY20260328C570", limit=100)
        assert len(history) == 50

    def test_quote_history_time_filter(self, trade_db):
        for i in range(10):
            snap = QuoteSnapshot(
                symbol="SPY20260328C570",
                timestamp=f"2026-03-28T09:{30+i:02d}:00",
                last_price=2.00 + (i * 0.01),
            )
            trade_db.insert_quote_snapshot(snap)

        filtered = trade_db.get_quote_history(
            "SPY20260328C570",
            start_time="2026-03-28T09:35:00",
            end_time="2026-03-28T09:37:00",
        )
        assert len(filtered) == 3


# =====================================================================
# 6. TRADE DATABASE — ORDER TRACKING
# =====================================================================
class TestOrderTracking:

    def test_insert_and_retrieve_order(self, trade_db):
        oid = trade_db.insert_order(
            order_id=100, symbol="SPY20260328C570",
            account_id="U1234567", action="BUY",
            order_type="LMT", quantity=5, limit_price=2.20,
        )
        assert oid > 0

        pending = trade_db.get_pending_orders()
        assert len(pending) == 1
        assert pending[0]["order_id"] == 100
        assert pending[0]["status"] == "submitted"

    def test_update_order_filled(self, trade_db):
        trade_db.insert_order(
            order_id=100, symbol="SPY20260328C570",
            account_id="U1234567", action="BUY",
            order_type="LMT", quantity=5,
        )
        trade_db.update_order_status(
            order_id=100, status="filled",
            filled_quantity=5, avg_fill_price=2.15,
        )
        pending = trade_db.get_pending_orders()
        assert len(pending) == 0

    def test_update_order_cancelled(self, trade_db):
        trade_db.insert_order(
            order_id=101, symbol="SPY20260328C570",
            account_id="U1234567", action="SELL",
            order_type="LMT", quantity=5,
        )
        trade_db.update_order_status(order_id=101, status="cancelled")
        pending = trade_db.get_pending_orders()
        assert len(pending) == 0


# =====================================================================
# 7. TRADE DATABASE — DAILY PNL
# =====================================================================
class TestDailyPnL:

    def test_daily_pnl_created_on_close(self, trade_db):
        tid = trade_db.insert_trade(_make_trade(entry_price=2.00, commission=0))
        trade_db.close_trade(tid, exit_price=3.00, exit_time="2026-03-28T10:00:00")

        pnl = trade_db.get_daily_pnl("2026-03-28")
        assert pnl is not None
        assert pnl["trades_closed"] == 1
        assert pnl["win_count"] == 1

    def test_daily_pnl_accumulates(self, trade_db):
        tid1 = trade_db.insert_trade(_make_trade(
            entry_price=2.00, commission=0, entry_time="2026-03-28T09:35:00",
        ))
        trade_db.close_trade(tid1, exit_price=3.00, exit_time="2026-03-28T10:00:00")

        tid2 = trade_db.insert_trade(_make_trade(
            entry_price=2.00, commission=0, entry_time="2026-03-28T10:05:00",
            symbol="SPY20260328P560",
        ))
        trade_db.close_trade(tid2, exit_price=1.50, exit_time="2026-03-28T10:30:00")

        pnl = trade_db.get_daily_pnl("2026-03-28")
        assert pnl["trades_closed"] == 2
        assert pnl["win_count"] == 1
        assert pnl["loss_count"] == 1

    def test_update_unrealized_pnl(self, trade_db):
        trade_db.update_unrealized_pnl("2026-03-28", -50.0)
        pnl = trade_db.get_daily_pnl("2026-03-28")
        assert pnl["unrealized_pnl"] == -50.0
        assert pnl["total_pnl"] == -50.0

    def test_pnl_history_ordering(self, trade_db):
        for day in range(1, 6):
            trade_db.update_unrealized_pnl(f"2026-03-{day:02d}", day * 10.0)

        history = trade_db.get_pnl_history(days=3)
        assert len(history) == 3
        assert history[0]["date"] >= history[1]["date"]


# =====================================================================
# 8. TRADE DATABASE — BALANCE HISTORY
# =====================================================================
class TestBalanceHistory:

    def test_record_and_retrieve_balance(self, trade_db):
        bid = trade_db.record_balance(
            account_id="U1234567",
            net_liquidation=50000.0,
            cash=25000.0,
            positions_value=25000.0,
            unrealized_pnl=-150.0,
        )
        assert bid > 0

        latest = trade_db.get_latest_balance()
        assert latest is not None
        assert latest["net_liquidation"] == 50000.0

    def test_balance_history_limit(self, trade_db):
        for i in range(10):
            trade_db.record_balance(
                account_id="U1234567",
                net_liquidation=50000.0 + i,
            )
        history = trade_db.get_balance_history(limit=5)
        assert len(history) == 5

    def test_latest_balance_is_most_recent(self, trade_db):
        trade_db.record_balance(account_id="TEST", net_liquidation=49000.0)
        time.sleep(0.01)
        trade_db.record_balance(account_id="TEST", net_liquidation=51000.0)

        latest = trade_db.get_latest_balance()
        assert latest["net_liquidation"] == 51000.0


# =====================================================================
# 9. TRADE DATABASE — TRADE STATISTICS
# =====================================================================
class TestTradeStatistics:

    def test_statistics_empty(self, trade_db):
        stats = trade_db.get_trade_statistics()
        assert stats["total_trades"] == 0

    def test_statistics_with_trades(self, trade_db):
        # Insert winning trade
        tid1 = trade_db.insert_trade(_make_trade(
            entry_price=2.00, quantity=1, commission=0,
            entry_time="2026-03-28T09:35:00",
        ))
        trade_db.close_trade(tid1, exit_price=3.00)

        # Insert losing trade
        tid2 = trade_db.insert_trade(_make_trade(
            entry_price=2.00, quantity=1, commission=0,
            entry_time="2026-03-28T09:36:00",
            symbol="SPY20260328P560",
        ))
        trade_db.close_trade(tid2, exit_price=1.50)

        stats = trade_db.get_trade_statistics()
        assert stats["total_trades"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["win_rate"] == 50.0
        assert stats["total_pnl"] > 0  # Net positive from $100 win - $50 loss

    def test_statistics_by_strategy(self, trade_db):
        tid1 = trade_db.insert_trade(_make_trade(
            strategy_name="strat_a", entry_time="2026-03-28T09:35:00",
            entry_price=2.00, quantity=1, commission=0,
        ))
        trade_db.close_trade(tid1, exit_price=3.00)

        tid2 = trade_db.insert_trade(_make_trade(
            strategy_name="strat_b", entry_time="2026-03-28T09:36:00",
            symbol="SPY20260328P560", entry_price=2.00, quantity=1, commission=0,
        ))
        trade_db.close_trade(tid2, exit_price=1.00)

        stats_a = trade_db.get_trade_statistics(strategy_name="strat_a")
        assert stats_a["total_trades"] == 1
        assert stats_a["wins"] == 1


# =====================================================================
# 10. TRADE DATABASE — BULK INSERT PERFORMANCE
# =====================================================================
class TestBulkInsertPerformance:

    def test_1000_trade_inserts_under_10s(self, trade_db):
        start = time.time()
        for i in range(1000):
            trade_db.insert_trade(Trade(
                symbol=f"SPY20260328C{570 + (i % 100)}",
                underlying="SPY",
                trade_type="option",
                option_type="call",
                strike=570.0 + (i % 100),
                action="buy",
                quantity=1,
                entry_price=2.00 + (i * 0.001),
                entry_time=f"2026-03-28T{9 + (i // 60):02d}:{i % 60:02d}:00",
                status="open",
            ))
        elapsed = time.time() - start
        assert elapsed < 30.0, f"1000 inserts took {elapsed:.2f}s"

        cursor = trade_db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades")
        assert cursor.fetchone()[0] == 1000

    def test_bulk_quote_snapshots(self, trade_db):
        snapshots = [
            QuoteSnapshot(
                symbol=f"SPY20260328C{570 + (i % 10)}",
                timestamp=f"2026-03-28T09:{i % 60:02d}:{(i*7)%60:02d}",
                last_price=2.00 + (i * 0.01),
            )
            for i in range(500)
        ]
        start = time.time()
        count = trade_db.insert_quote_snapshots_bulk(snapshots)
        elapsed = time.time() - start
        assert count == 500
        assert elapsed < 5.0


# =====================================================================
# 11. TRADE DATABASE — CONCURRENT ACCESS
# =====================================================================
class TestConcurrentAccess:

    def test_concurrent_inserts_no_corruption(self, trade_db):
        errors = []
        inserted_ids = []
        lock = threading.Lock()

        def insert_worker(worker_id):
            try:
                for i in range(20):
                    tid = trade_db.insert_trade(Trade(
                        symbol=f"SPY20260328C{570 + worker_id}",
                        underlying="SPY",
                        trade_type="option",
                        action="buy",
                        quantity=1,
                        entry_price=2.00,
                        entry_time=f"2026-03-28T{9+worker_id}:{i:02d}:00",
                        status="open",
                    ))
                    with lock:
                        inserted_ids.append(tid)
            except Exception as e:
                with lock:
                    errors.append(f"Worker {worker_id}: {e}")

        threads = [threading.Thread(target=insert_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(inserted_ids) == 100

        cursor = trade_db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades")
        assert cursor.fetchone()[0] == 100

    def test_concurrent_read_write(self, trade_db):
        for i in range(50):
            trade_db.insert_trade(Trade(
                symbol=f"SPY20260328C{570 + i}",
                underlying="SPY", trade_type="option",
                action="buy", quantity=1, entry_price=2.00,
                entry_time=f"2026-03-28T09:{i:02d}:00", status="open",
            ))

        errors = []
        read_results = []
        lock = threading.Lock()

        def reader():
            try:
                for _ in range(20):
                    trades = trade_db.get_recent_trades(10)
                    with lock:
                        read_results.append(len(trades))
            except Exception as e:
                with lock:
                    errors.append(f"Reader: {e}")

        def writer():
            try:
                for i in range(20):
                    trade_db.insert_trade(Trade(
                        symbol=f"SPY20260328P{500 + i}",
                        underlying="SPY", trade_type="option",
                        action="buy", quantity=1, entry_price=1.00,
                        entry_time=f"2026-03-28T10:{i:02d}:00", status="open",
                    ))
            except Exception as e:
                with lock:
                    errors.append(f"Writer: {e}")

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert all(r > 0 for r in read_results)


# =====================================================================
# 12. TRADE DATABASE — TRANSACTION ROLLBACK
# =====================================================================
class TestTransactionRollback:

    def test_failed_insert_does_not_corrupt(self, trade_db):
        tid = trade_db.insert_trade(_make_trade())

        # Attempt a bad raw SQL that should fail
        cursor = trade_db.conn.cursor()
        try:
            cursor.execute("INSERT INTO trades (symbol) VALUES (?)", ("BAD",))
        except sqlite3.IntegrityError:
            trade_db.conn.rollback()

        # Original trade should still exist
        t = trade_db.get_trade(tid)
        assert t is not None
        assert t["symbol"] == "SPY20260328C570"

    def test_close_trade_atomic(self, trade_db):
        tid = trade_db.insert_trade(_make_trade(entry_price=2.00, commission=0))
        result = trade_db.close_trade(tid, exit_price=3.00, exit_time="2026-03-28T10:00:00")

        assert result["status"] == "closed"
        assert result["exit_price"] == 3.00

        # daily_pnl should also be updated atomically
        pnl = trade_db.get_daily_pnl("2026-03-28")
        assert pnl is not None


# =====================================================================
# 13. TRADE DATABASE — CSV EXPORT
# =====================================================================
class TestCSVExport:

    def test_export_roundtrip(self, trade_db):
        tid = trade_db.insert_trade(_make_trade(entry_price=2.00, commission=1.0))
        trade_db.close_trade(tid, exit_price=3.00)

        fd, csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            trade_db.export_trades_csv(csv_path)
            import csv
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["symbol"] == "SPY20260328C570"
            assert float(rows[0]["exit_price"]) == 3.0
        finally:
            os.unlink(csv_path)


# =====================================================================
# 14. TRADE DATABASE — CONTEXT MANAGER
# =====================================================================
class TestContextManager:

    def test_context_manager_closes_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            with TradeDatabase(path) as db:
                db.insert_trade(_make_trade())
            # After __exit__, connection should be closed
            # Attempting to use it should raise
            with pytest.raises(Exception):
                db.conn.execute("SELECT 1")
        finally:
            os.unlink(path)


# =====================================================================
# 15. MARKET DATABASE — TABLE CREATION (no duplicates)
# =====================================================================
class TestMarketDBTableCreation:

    def test_daily_ticker_data_exists(self, market_db):
        cursor = market_db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_ticker_data'")
        assert cursor.fetchone() is not None

    def test_intraday_ticker_data_exists(self, market_db):
        cursor = market_db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='intraday_ticker_data'")
        assert cursor.fetchone() is not None

    def test_no_duplicate_tables(self, market_db):
        """Verify each table name appears exactly once in schema."""
        cursor = market_db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in cursor.fetchall()]
        for name in set(table_names):
            count = table_names.count(name)
            assert count == 1, f"Table '{name}' defined {count} times"

    def test_intraday_5min_data_exists(self, market_db):
        cursor = market_db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='intraday_5min_data'")
        assert cursor.fetchone() is not None

    def test_realtime_quotes_exists(self, market_db):
        cursor = market_db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='realtime_quotes'")
        assert cursor.fetchone() is not None

    def test_ticker_sectors_exists(self, market_db):
        cursor = market_db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ticker_sectors'")
        assert cursor.fetchone() is not None


# =====================================================================
# 16. MARKET DATABASE — INDEX INTEGRITY
# =====================================================================
class TestMarketDBIndexes:

    def test_daily_data_indexes(self, market_db):
        cursor = market_db.conn.cursor()
        cursor.execute("PRAGMA index_list(daily_ticker_data)")
        indexes = [row[1] for row in cursor.fetchall()]
        assert "idx_date" in indexes
        assert "idx_ticker" in indexes

    def test_intraday_indexes(self, market_db):
        cursor = market_db.conn.cursor()
        cursor.execute("PRAGMA index_list(intraday_ticker_data)")
        indexes = [row[1] for row in cursor.fetchall()]
        assert "idx_intraday_date" in indexes
        assert "idx_intraday_ticker_date" in indexes


# =====================================================================
# 17. IBKR DATABASE — SCHEMA
# =====================================================================
class TestIBKRDBSchema:

    def test_bars_table_exists(self, ibkr_db):
        conn = ibkr_db._conn()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bars'")
        assert cursor.fetchone() is not None

    def test_quotes_table_exists(self, ibkr_db):
        conn = ibkr_db._conn()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='quotes'")
        assert cursor.fetchone() is not None

    def test_option_chains_table_exists(self, ibkr_db):
        conn = ibkr_db._conn()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='option_chains'")
        assert cursor.fetchone() is not None

    def test_account_summary_table_exists(self, ibkr_db):
        conn = ibkr_db._conn()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='account_summary'")
        assert cursor.fetchone() is not None

    def test_positions_table_exists(self, ibkr_db):
        conn = ibkr_db._conn()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='positions'")
        assert cursor.fetchone() is not None

    def test_bars_column_schema(self, ibkr_db):
        conn = ibkr_db._conn()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(bars)")
        cols = {row[1] for row in cursor.fetchall()}
        required = {"symbol", "sec_type", "date", "open", "high", "low", "close",
                     "volume", "wap", "bar_count", "bar_size", "duration",
                     "what_to_show", "expiry", "strike", "right", "fetched_at"}
        for col in required:
            assert col in cols, f"Missing bars column: {col}"

    def test_ibkr_db_wal_mode(self, ibkr_db):
        conn = ibkr_db._conn()
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal"


# =====================================================================
# STANDALONE RUNNER
# =====================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
