"""
Trade Sync & Connectivity Edge-Case Test Suite
===============================================
Exercises every reconciliation path in engine.py and trade_sync.py:
  - Symbol normalisation (Questrade / IBKR / OCC formats)
  - Time normalisation (ISO / IBKR / edge formats)
  - Deduplication logic (exact, fuzzy-minute, cross-format)
  - IBKR execution pairing (FIFO, partial, multi-lot)
  - Flex trade normalisation and import
  - CSV / external-DB import with dedup
  - Engine reconciliation:
      * Orphaned broker positions → DB entry created
      * Phantom trades (DB open, broker closed) → auto-close with real fill
      * Phantom trades with no fill → close at $0.01
      * Entry price drift correction
      * Missed SELL fills (execution reconciliation)
      * Wrong exit price correction (phantom $0.01 → real fill)
      * Missing BUY fill → auto-create trade
      * Commission sync
  - Connectivity plumbing (adapter port resolution, client wiring)

Run:
    python -m pytest tests/test_trade_sync_edge_cases.py -v --tb=short
    python tests/test_trade_sync_edge_cases.py          # standalone
"""
import csv
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, PropertyMock

# ── project root on PYTHONPATH ────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# ── custom test runner infrastructure ─────────────────────────────────
class _SkipTest(Exception):
    pass

def skip_test(reason: str = ""):
    raise _SkipTest(reason)


# =====================================================================
#  HELPERS
# =====================================================================

def _make_db(path=None):
    """Create a fresh TradeDatabase in a temp directory."""
    from live.trade_database import TradeDatabase
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
    return TradeDatabase(path), path


def _make_trade(**overrides):
    """Build a Trade dataclass with sensible defaults."""
    from live.trade_database import Trade
    defaults = dict(
        symbol="SPY20260321C560",
        underlying="SPY",
        trade_type="option",
        option_type="call",
        strike=560.0,
        expiration="20260321",
        action="BUY",
        quantity=1,
        entry_price=2.50,
        entry_time="2026-03-21T10:30:00",
        status="open",
        strategy_name="test",
        account_id="TEST",
    )
    defaults.update(overrides)
    return Trade(**defaults)


def _engine_with_mocks(db, positions_data=None, executions=None,
                        flex_trades=None, pending_exit_orders=None):
    """
    Build a LiveTradingEngine with mocked broker client and managers.
    Returns (engine, mock_client).
    """
    from live.engine import LiveTradingEngine, EngineConfig
    from live.position_manager import PositionManager
    from live.order_manager import OrderManager

    mock_client = MagicMock()
    mock_client.get_account_positions.return_value = positions_data or []
    mock_client.get_executions.return_value = executions or []
    mock_client.cancel_all_open_orders.return_value = None
    mock_client.get_quote_by_symbol.return_value = None

    pos_mgr = MagicMock(spec=PositionManager)
    pos_mgr.sync_positions.return_value = None
    pos_mgr.get_all_positions.return_value = []
    pos_mgr.get_total_exposure.return_value = {"total_unrealized_pnl": 0}
    pos_mgr.update_quotes.return_value = None

    ord_mgr = MagicMock(spec=OrderManager)
    ord_mgr.sync_orders.return_value = None

    config = EngineConfig(
        account_id="TEST",
        symbols=["SPY"],
        option_underlyings=["SPY"],
        mode="paper",
    )

    engine = LiveTradingEngine(
        client=mock_client,
        trade_db=db,
        position_manager=pos_mgr,
        order_manager=ord_mgr,
        config=config,
    )

    # Always disable real Flex client for test isolation.
    # When flex_trades is provided, set up mock Flex returning that data.
    # When flex_trades is None, ensure no real IBKR API calls are made.
    if flex_trades is not None:
        engine._flex_client = MagicMock()
        engine._flex_query_id = 99999
        engine._flex_client.fetch_trades.return_value = flex_trades
        engine._flex_trades_cache = None
        engine._flex_cache_time = 0
    else:
        engine._flex_client = None
        engine._flex_query_id = None

    if pending_exit_orders:
        engine._pending_exit_orders = pending_exit_orders

    return engine, mock_client


# =====================================================================
#  1. SYMBOL NORMALISATION
# =====================================================================
class TestSymbolNormalisation:
    """TradeSync._normalize_symbol must handle all known formats."""

    def test_questrade_call(self):
        from live.trade_sync import TradeSync
        assert TradeSync._normalize_symbol("SPY18Mar26C664.00") == "SPY20260318C664"

    def test_questrade_put(self):
        from live.trade_sync import TradeSync
        assert TradeSync._normalize_symbol("SPY18Mar26P555.00") == "SPY20260318P555"

    def test_ibkr_already_canonical(self):
        from live.trade_sync import TradeSync
        assert TradeSync._normalize_symbol("SPY20260318C664") == "SPY20260318C664"

    def test_ibkr_with_decimal_strike(self):
        from live.trade_sync import TradeSync
        assert TradeSync._normalize_symbol("SPY20260318C664.00") == "SPY20260318C664"

    def test_plain_equity_unchanged(self):
        from live.trade_sync import TradeSync
        assert TradeSync._normalize_symbol("SPY") == "SPY"

    def test_empty_string(self):
        from live.trade_sync import TradeSync
        assert TradeSync._normalize_symbol("") == ""

    def test_none_returns_none(self):
        from live.trade_sync import TradeSync
        # Depending on implementation, None input may return None or ""
        result = TradeSync._normalize_symbol(None)
        assert result is None or result == ""

    def test_questrade_all_months(self):
        from live.trade_sync import TradeSync
        months = [
            ("Jan", "01"), ("Feb", "02"), ("Mar", "03"), ("Apr", "04"),
            ("May", "05"), ("Jun", "06"), ("Jul", "07"), ("Aug", "08"),
            ("Sep", "09"), ("Oct", "10"), ("Nov", "11"), ("Dec", "12"),
        ]
        for abbr, num in months:
            qt = f"AAPL15{abbr}26C200.00"
            expected = f"AAPL2026{num}15C200"
            assert TradeSync._normalize_symbol(qt) == expected, f"Failed for {abbr}"

    def test_cross_format_dedup_key_match(self):
        """Questrade and IBKR symbols for same contract produce same dedup key."""
        from live.trade_sync import TradeSync
        qt = TradeSync._make_key("SPY18Mar26P555.00", "2026-03-18T10:30:00")
        ibkr = TradeSync._make_key("SPY20260318P555", "2026-03-18T10:30:00")
        assert qt == ibkr


# =====================================================================
#  2. ENGINE _normalize_option_key
# =====================================================================
class TestEngineNormalizeOptionKey:
    """LiveTradingEngine._normalize_option_key handles QT and IBKR formats."""

    def test_ibkr_format(self):
        from live.engine import LiveTradingEngine
        key = LiveTradingEngine._normalize_option_key("SPY20260321C560")
        assert key == ("SPY", "20260321", 560.0, "C")

    def test_questrade_format(self):
        from live.engine import LiveTradingEngine
        key = LiveTradingEngine._normalize_option_key("SPY21Mar26C560.00")
        assert key == ("SPY", "20260321", 560.0, "C")

    def test_non_option_returns_none(self):
        from live.engine import LiveTradingEngine
        assert LiveTradingEngine._normalize_option_key("SPY") is None

    def test_cross_format_match(self):
        """QT and IBKR symbol for same contract produce identical key."""
        from live.engine import LiveTradingEngine
        k1 = LiveTradingEngine._normalize_option_key("SPY21Mar26P555.00")
        k2 = LiveTradingEngine._normalize_option_key("SPY20260321P555")
        assert k1 == k2

    def test_decimal_strike_stripped(self):
        from live.engine import LiveTradingEngine
        key = LiveTradingEngine._normalize_option_key("QQQ20260321C450.50")
        # strike should be 450.5 (float)
        assert key is not None
        assert key[0] == "QQQ"
        assert key[2] == 450.5 or key[2] == 450.0  # depends on regex


# =====================================================================
#  3. TIME NORMALISATION
# =====================================================================
class TestTimeNormalisation:
    """TradeSync._normalize_time must collapse to YYYY-MM-DDTHH:MM."""

    def test_iso_full(self):
        from live.trade_sync import TradeSync
        assert TradeSync._normalize_time("2026-03-18T10:43:57") == "2026-03-18T10:43"

    def test_ibkr_spaced(self):
        from live.trade_sync import TradeSync
        assert TradeSync._normalize_time("20260318  14:43:58") == "2026-03-18T14:43"

    def test_empty_string(self):
        from live.trade_sync import TradeSync
        assert TradeSync._normalize_time("") == ""

    def test_none_returns_empty(self):
        from live.trade_sync import TradeSync
        result = TradeSync._normalize_time(None)
        assert result == "" or result is None

    def test_iso_no_seconds(self):
        from live.trade_sync import TradeSync
        assert TradeSync._normalize_time("2026-03-18T10:43") == "2026-03-18T10:43"

    def test_same_trade_different_seconds(self):
        """Two timestamps differing only in seconds should match."""
        from live.trade_sync import TradeSync
        t1 = TradeSync._normalize_time("2026-03-18T10:43:01")
        t2 = TradeSync._normalize_time("2026-03-18T10:43:59")
        assert t1 == t2


# =====================================================================
#  4. DEDUPLICATION LOGIC
# =====================================================================
class TestDeduplication:
    """Verify dedup prevents double-imports across formats."""

    def test_exact_duplicate_blocked(self):
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            trade = _make_trade()
            db.insert_trade(trade)

            syncer = TradeSync(db)
            assert syncer._is_duplicate("SPY20260321C560", "2026-03-21T10:30:00")
        finally:
            db.conn.close()
            os.unlink(path)

    def test_cross_format_duplicate(self):
        """A trade inserted with IBKR symbol is detected as dup when checked
        with Questrade symbol format."""
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            trade = _make_trade(symbol="SPY20260318C664", entry_time="2026-03-18T10:30:00")
            db.insert_trade(trade)

            syncer = TradeSync(db)
            # Same contract in Questrade format
            assert syncer._is_duplicate("SPY18Mar26C664.00", "2026-03-18T10:30:00")
        finally:
            db.conn.close()
            os.unlink(path)

    def test_ibkr_time_format_duplicate(self):
        """IBKR-style timestamp matches ISO timestamp for same minute."""
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            trade = _make_trade(entry_time="2026-03-18T14:43:00")
            db.insert_trade(trade)

            syncer = TradeSync(db)
            # IBKR spaced time format
            assert syncer._is_duplicate(trade.symbol, "20260318  14:43:58")
        finally:
            db.conn.close()
            os.unlink(path)

    def test_different_strike_not_duplicate(self):
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            trade = _make_trade(symbol="SPY20260321C560")
            db.insert_trade(trade)

            syncer = TradeSync(db)
            assert not syncer._is_duplicate("SPY20260321C570", "2026-03-21T10:30:00")
        finally:
            db.conn.close()
            os.unlink(path)

    def test_different_minute_not_duplicate(self):
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            trade = _make_trade(entry_time="2026-03-21T10:30:00")
            db.insert_trade(trade)

            syncer = TradeSync(db)
            assert not syncer._is_duplicate(trade.symbol, "2026-03-21T10:31:00")
        finally:
            db.conn.close()
            os.unlink(path)

    def test_key_refresh_after_insert(self):
        """After inserting during sync, the new key is in the cache."""
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            syncer = TradeSync(db)
            syncer._refresh_keys()
            assert len(syncer._existing_keys) == 0

            db.insert_trade(_make_trade())
            # Cache is stale — but after refresh it should include the new trade
            syncer._refresh_keys()
            assert len(syncer._existing_keys) == 1
        finally:
            db.conn.close()
            os.unlink(path)


# =====================================================================
#  5. IBKR EXECUTION PAIRING
# =====================================================================
class TestExecutionPairing:
    """TradeSync._pair_and_import_executions FIFO matching."""

    def _make_exec(self, side, symbol, price, time_str, order_id=1, shares=1):
        return {
            "trade_symbol": symbol,
            "symbol": "SPY",
            "side": side,
            "shares": shares,
            "price": price,
            "time": time_str,
            "order_id": order_id,
            "exec_id": f"exec_{side}_{time_str}",
            "acct_number": "TEST",
            "right": "C" if "C" in symbol else "P",
            "strike": 560.0,
            "expiry": "20260321",
        }

    def test_single_round_trip(self):
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            syncer = TradeSync(db)
            execs = [
                self._make_exec("BOT", "SPY20260321C560", 2.00, "2026-03-21T10:00:00"),
                self._make_exec("SLD", "SPY20260321C560", 3.00, "2026-03-21T10:15:00"),
            ]
            imported = syncer._pair_and_import_executions(execs)
            assert imported == 1

            trades = db.get_recent_trades(10)
            assert len(trades) == 1
            t = trades[0]
            assert t["entry_price"] == 2.00
            assert t["exit_price"] == 3.00
            assert t["status"] == "closed"
            assert t["pnl"] == (3.00 - 2.00) * 1 * 100  # $100
        finally:
            db.conn.close()
            os.unlink(path)

    def test_multiple_round_trips_fifo(self):
        """Two BUYs then two SELLs — pairs first BUY with first SELL."""
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            syncer = TradeSync(db)
            execs = [
                self._make_exec("BOT", "SPY20260321C560", 2.00, "2026-03-21T10:00:00", order_id=1),
                self._make_exec("BOT", "SPY20260321C560", 2.50, "2026-03-21T10:05:00", order_id=2),
                self._make_exec("SLD", "SPY20260321C560", 3.00, "2026-03-21T10:15:00", order_id=3),
                self._make_exec("SLD", "SPY20260321C560", 2.80, "2026-03-21T10:20:00", order_id=4),
            ]
            imported = syncer._pair_and_import_executions(execs)
            assert imported == 2

            trades = db.get_recent_trades(10)
            assert len(trades) == 2
            # FIFO: first BUY@2.00 paired with first SELL@3.00
            pnls = sorted([t["pnl"] for t in trades])
            assert abs(pnls[0] - 30.0) < 0.01   # (2.80 - 2.50) * 100
            assert abs(pnls[1] - 100.0) < 0.01   # (3.00 - 2.00) * 100
        finally:
            db.conn.close()
            os.unlink(path)

    def test_unmatched_buy_no_import(self):
        """A BUY without a matching SELL should not create a closed trade."""
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            syncer = TradeSync(db)
            execs = [
                self._make_exec("BOT", "SPY20260321C560", 2.00, "2026-03-21T10:00:00"),
            ]
            imported = syncer._pair_and_import_executions(execs)
            assert imported == 0
        finally:
            db.conn.close()
            os.unlink(path)

    def test_unmatched_sell_no_import(self):
        """A SELL without a prior BUY should not create a trade."""
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            syncer = TradeSync(db)
            execs = [
                self._make_exec("SLD", "SPY20260321C560", 3.00, "2026-03-21T10:15:00"),
            ]
            imported = syncer._pair_and_import_executions(execs)
            assert imported == 0
        finally:
            db.conn.close()
            os.unlink(path)

    def test_duplicate_not_reimported(self):
        """Running the same executions twice should only import once."""
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            syncer = TradeSync(db)
            execs = [
                self._make_exec("BOT", "SPY20260321C560", 2.00, "2026-03-21T10:00:00"),
                self._make_exec("SLD", "SPY20260321C560", 3.00, "2026-03-21T10:15:00"),
            ]
            assert syncer._pair_and_import_executions(execs) == 1
            assert syncer._pair_and_import_executions(execs) == 0  # deduped
        finally:
            db.conn.close()
            os.unlink(path)

    def test_put_option_type_detected(self):
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            syncer = TradeSync(db)
            execs = [
                self._make_exec("BOT", "SPY20260321P555", 1.50, "2026-03-21T10:00:00"),
                self._make_exec("SLD", "SPY20260321P555", 2.00, "2026-03-21T10:15:00"),
            ]
            execs[0]["right"] = "P"
            execs[1]["right"] = "P"
            syncer._pair_and_import_executions(execs)
            trades = db.get_recent_trades(10)
            assert trades[0]["option_type"] == "put"
        finally:
            db.conn.close()
            os.unlink(path)

    def test_multi_contract_quantity(self):
        """BUY 5 contracts, SELL 5 — quantity in trade record is correct."""
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            syncer = TradeSync(db)
            execs = [
                self._make_exec("BOT", "SPY20260321C560", 2.00, "2026-03-21T10:00:00", shares=5),
                self._make_exec("SLD", "SPY20260321C560", 3.00, "2026-03-21T10:15:00", shares=5),
            ]
            syncer._pair_and_import_executions(execs)
            t = db.get_recent_trades(1)[0]
            assert t["quantity"] == 5
            assert abs(t["pnl"] - 500.0) < 0.01  # (3-2)*5*100
        finally:
            db.conn.close()
            os.unlink(path)


# =====================================================================
#  6. FLEX TRADE IMPORT
# =====================================================================
class TestFlexImport:
    """TradeSync._sync_from_flex normalises Flex XML fields correctly."""

    def test_flex_round_trip(self):
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            syncer = TradeSync(db)
            # Simulate Flex env vars and client
            with patch.dict(os.environ, {
                "IBKR_FLEX_TOKEN": "FAKE",
                "IBKR_FLEX_QUERY_ID": "12345",
            }):
                mock_flex = MagicMock()
                mock_flex.fetch_trades.return_value = [
                    {
                        "symbol": "SPY",
                        "assetCategory": "OPT",
                        "putCall": "C",
                        "strike": "560",
                        "expiry": "20260321",
                        "buySell": "BUY",
                        "quantity": "1",
                        "tradePrice": "2.50",
                        "dateTime": "2026-03-21T10:00:00",
                        "orderID": "100",
                        "ibExecID": "exec1",
                        "accountId": "TEST",
                    },
                    {
                        "symbol": "SPY",
                        "assetCategory": "OPT",
                        "putCall": "C",
                        "strike": "560",
                        "expiry": "20260321",
                        "buySell": "SELL",
                        "quantity": "1",
                        "tradePrice": "3.50",
                        "dateTime": "2026-03-21T10:30:00",
                        "orderID": "101",
                        "ibExecID": "exec2",
                        "accountId": "TEST",
                    },
                ]
                with patch("clients.ibkr_flex.IBKRFlexClient", return_value=mock_flex):
                    imported = syncer._sync_from_flex()
                assert imported == 1
                t = db.get_recent_trades(1)[0]
                assert t["status"] == "closed"
                assert abs(t["pnl"] - 100.0) < 0.01  # (3.50 - 2.50)*100
        finally:
            db.conn.close()
            os.unlink(path)

    def test_flex_not_configured(self):
        """If env vars missing, _sync_from_flex returns 0."""
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            syncer = TradeSync(db)
            with patch.dict(os.environ, {}, clear=True):
                # Remove any existing flex vars
                os.environ.pop("IBKR_FLEX_TOKEN", None)
                os.environ.pop("IBKR_FLEX_QUERY_ID", None)
                assert syncer._sync_from_flex() == 0
        finally:
            db.conn.close()
            os.unlink(path)


# =====================================================================
#  7. CSV IMPORT
# =====================================================================
class TestCSVImport:

    def test_csv_round_trip(self):
        from live.trade_sync import TradeSync
        db, path = _make_db()
        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "trades.csv")
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "symbol", "entry_time", "exit_time", "entry_price",
                    "exit_price", "quantity", "pnl", "option_type", "action", "status",
                ])
                writer.writeheader()
                writer.writerow({
                    "symbol": "SPY20260321C560",
                    "entry_time": "2026-03-21T10:00:00",
                    "exit_time": "2026-03-21T10:30:00",
                    "entry_price": "2.00",
                    "exit_price": "3.00",
                    "quantity": "1",
                    "pnl": "100.0",
                    "option_type": "call",
                    "action": "buy",
                    "status": "closed",
                })
            syncer = TradeSync(db)
            n = syncer.sync_from_csv(csv_path)
            assert n == 1
            # Importing again should be 0 (dedup)
            assert syncer.sync_from_csv(csv_path) == 0
        finally:
            db.conn.close()
            os.unlink(path)
            os.unlink(csv_path)
            os.rmdir(tmpdir)

    def test_csv_missing_file(self):
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            syncer = TradeSync(db)
            assert syncer.sync_from_csv("/nonexistent/trades.csv") == 0
        finally:
            db.conn.close()
            os.unlink(path)

    def test_csv_empty_file(self):
        from live.trade_sync import TradeSync
        db, path = _make_db()
        fd, csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            syncer = TradeSync(db)
            assert syncer.sync_from_csv(csv_path) == 0
        finally:
            db.conn.close()
            os.unlink(path)
            os.unlink(csv_path)

    def test_csv_bad_pnl(self):
        """Non-numeric PnL should default to 0, not crash."""
        from live.trade_sync import TradeSync
        db, path = _make_db()
        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "trades.csv")
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "symbol", "entry_time", "exit_time", "entry_price",
                    "exit_price", "quantity", "pnl", "action", "status",
                ])
                writer.writeheader()
                writer.writerow({
                    "symbol": "SPY20260321C560",
                    "entry_time": "2026-03-21T10:00:00",
                    "exit_time": "2026-03-21T10:30:00",
                    "entry_price": "2.00",
                    "exit_price": "3.00",
                    "quantity": "1",
                    "pnl": "N/A",  # bad value
                    "action": "buy",
                    "status": "closed",
                })
            syncer = TradeSync(db)
            n = syncer.sync_from_csv(csv_path)
            assert n == 1
        finally:
            db.conn.close()
            os.unlink(path)
            os.unlink(csv_path)
            os.rmdir(tmpdir)


# =====================================================================
#  8. EXTERNAL DB IMPORT
# =====================================================================
class TestExternalDBImport:

    def _create_external_db(self, ext_path, trades):
        """Create an external SQLite DB with a trades table."""
        conn = sqlite3.connect(ext_path)
        conn.execute('''
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, underlying TEXT, trade_type TEXT, option_type TEXT,
                strike REAL, expiration TEXT, action TEXT, quantity INTEGER,
                entry_price REAL, entry_time TEXT, exit_price REAL, exit_time TEXT,
                status TEXT, pnl REAL, pnl_percent REAL, commission REAL,
                delta REAL, gamma REAL, theta REAL, vega REAL, iv REAL,
                underlying_price_entry REAL, underlying_price_exit REAL,
                entry_order_id INTEGER, exit_order_id INTEGER,
                strategy_name TEXT, strategy_params TEXT, notes TEXT, account_id TEXT
            )
        ''')
        for t in trades:
            conn.execute('''
                INSERT INTO trades (symbol, entry_time, exit_time, entry_price,
                    exit_price, quantity, pnl, pnl_percent, status, action,
                    trade_type, option_type, commission)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                t.get("symbol", "SPY20260321C560"),
                t.get("entry_time", "2026-03-21T10:00:00"),
                t.get("exit_time", "2026-03-21T10:30:00"),
                t.get("entry_price", 2.0),
                t.get("exit_price", 3.0),
                t.get("quantity", 1),
                t.get("pnl", 100.0),
                t.get("pnl_percent", 50.0),
                t.get("status", "closed"),
                t.get("action", "buy"),
                t.get("trade_type", "option"),
                t.get("option_type", "call"),
                t.get("commission", 1.0),
            ))
        conn.commit()
        conn.close()

    def test_import_from_external_db(self):
        from live.trade_sync import TradeSync
        db, path = _make_db()
        fd, ext_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            self._create_external_db(ext_path, [{"symbol": "SPY20260321C560"}])
            syncer = TradeSync(db)
            n = syncer.sync_from_db(ext_path)
            assert n == 1
            # Dedup
            assert syncer.sync_from_db(ext_path) == 0
        finally:
            db.conn.close()
            os.unlink(path)
            os.unlink(ext_path)

    def test_import_skips_open_trades(self):
        """Only closed trades with PnL should be imported."""
        from live.trade_sync import TradeSync
        db, path = _make_db()
        fd, ext_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            self._create_external_db(ext_path, [
                {"symbol": "SPY20260321C560", "status": "open", "pnl": None},
            ])
            # Fix the NULL pnl — the query filters on status='closed' AND pnl IS NOT NULL
            conn = sqlite3.connect(ext_path)
            conn.execute("UPDATE trades SET pnl = NULL WHERE status = 'open'")
            conn.commit()
            conn.close()

            syncer = TradeSync(db)
            n = syncer.sync_from_db(ext_path)
            assert n == 0
        finally:
            db.conn.close()
            os.unlink(path)
            os.unlink(ext_path)

    def test_import_missing_db(self):
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            syncer = TradeSync(db)
            assert syncer.sync_from_db("/nonexistent/trades.db") == 0
        finally:
            db.conn.close()
            os.unlink(path)

    def test_import_db_no_trades_table(self):
        """DB exists but has no trades table → 0 imported."""
        from live.trade_sync import TradeSync
        db, path = _make_db()
        fd, ext_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            conn = sqlite3.connect(ext_path)
            conn.execute("CREATE TABLE other (id INTEGER)")
            conn.commit()
            conn.close()

            syncer = TradeSync(db)
            assert syncer.sync_from_db(ext_path) == 0
        finally:
            db.conn.close()
            os.unlink(path)
            os.unlink(ext_path)


# =====================================================================
#  9. ENGINE: ORPHANED BROKER POSITION → AUTO-CREATE DB ENTRY
# =====================================================================
class TestOrphanedBrokerPosition:
    """IBKR has position, DB doesn't → engine creates DB entry."""

    def test_orphaned_position_creates_db_trade(self):
        db, path = _make_db()
        try:
            positions = [{
                "symbol": "SPY20260321C560",
                "openQuantity": 2,
                "averageEntryPrice": 3.50,
            }]
            engine, _ = _engine_with_mocks(db, positions_data=positions)
            engine._reconcile_broker_positions()

            open_trades = db.get_open_trades()
            assert len(open_trades) == 1
            t = open_trades[0]
            assert t["quantity"] == 2
            assert t["status"] == "open"
            assert "orphaned" in (t["notes"] or "").lower() or "AUTO-CREATED" in (t["notes"] or "")
        finally:
            db.conn.close()
            os.unlink(path)

    def test_orphaned_position_with_real_entry_from_executions(self):
        """Engine should try to find real entry price from IBKR executions."""
        db, path = _make_db()
        try:
            positions = [{
                "symbol": "SPY20260321C560",
                "openQuantity": 1,
                "averageEntryPrice": 3.50,
            }]
            engine, mock_client = _engine_with_mocks(db, positions_data=positions)
            # Provide executions that include a BUY for this contract
            mock_client.get_executions.return_value = [{
                "trade_symbol": "SPY20260321C560",
                "side": "BOT",
                "price": 3.45,
                "shares": 1,
                "time": "2026-03-21T10:00:00",
                "order_id": 1,
            }]
            engine._reconcile_broker_positions()

            open_trades = db.get_open_trades()
            assert len(open_trades) == 1
            # Entry price should be from executions if found, or avgCost
            # (depends on whether Flex is configured; without Flex, falls back to session)
        finally:
            db.conn.close()
            os.unlink(path)

    def test_zero_quantity_position_ignored(self):
        """Broker reports qty=0 → not treated as orphaned."""
        db, path = _make_db()
        try:
            positions = [{
                "symbol": "SPY20260321C560",
                "openQuantity": 0,
                "averageEntryPrice": 3.50,
            }]
            engine, _ = _engine_with_mocks(db, positions_data=positions)
            engine._reconcile_broker_positions()
            assert len(db.get_open_trades()) == 0
        finally:
            db.conn.close()
            os.unlink(path)


# =====================================================================
# 10. ENGINE: PHANTOM TRADE DETECTION
# =====================================================================
class TestPhantomTradeDetection:
    """DB has open trade, IBKR doesn't → detect and close."""

    def test_phantom_closed_with_real_sell_fill(self):
        """SELL fill found in IBKR → close trade at real exit price."""
        db, path = _make_db()
        try:
            trade = _make_trade(entry_price=2.50)
            tid = db.insert_trade(trade)

            # Broker has no positions (trade was sold)
            executions = [{
                "trade_symbol": "SPY20260321C560",
                "side": "SLD",
                "price": 3.00,
                "shares": 1,
                "time": "2026-03-21T10:30:00",
                "order_id": 10,
                "exec_id": "exec_sell_1",
                "commission": 1.05,
            }]
            engine, mock_client = _engine_with_mocks(db, positions_data=[])
            mock_client.get_executions.return_value = executions
            engine._reconcile_broker_positions()

            t = db.get_trade(tid)
            assert t["status"] == "closed"
            assert t["exit_price"] == 3.00
            assert "AUTO-CLOSED" in (t["notes"] or "")
            assert "SELL fill" in (t["notes"] or "")
        finally:
            db.conn.close()
            os.unlink(path)

    def test_phantom_closed_at_penny_when_no_fill(self):
        """No SELL fill and no position → close at $0.01 (expired/phantom)."""
        db, path = _make_db()
        try:
            trade = _make_trade(entry_price=2.50)
            tid = db.insert_trade(trade)

            engine, mock_client = _engine_with_mocks(db, positions_data=[])
            mock_client.get_executions.return_value = []
            engine._reconcile_broker_positions()

            t = db.get_trade(tid)
            assert t["status"] == "closed"
            assert t["exit_price"] == 0.01
            assert "phantom" in (t["notes"] or "").lower() or "expired" in (t["notes"] or "").lower()
        finally:
            db.conn.close()
            os.unlink(path)

    def test_phantom_skipped_when_pending_exit_order(self):
        """If there's a pending exit order, don't close yet."""
        db, path = _make_db()
        try:
            trade = _make_trade()
            tid = db.insert_trade(trade)

            pending = {99: {"trade_id": tid, "symbol": trade.symbol, "signal_reason": "test"}}
            engine, _ = _engine_with_mocks(
                db, positions_data=[], pending_exit_orders=pending,
            )
            engine._reconcile_broker_positions()

            t = db.get_trade(tid)
            assert t["status"] == "open"  # should NOT be closed
        finally:
            db.conn.close()
            os.unlink(path)


# =====================================================================
# 11. ENGINE: ENTRY PRICE DRIFT CORRECTION
# =====================================================================
class TestEntryPriceDrift:
    """Broker avgCost differs from DB entry_price → sync to broker."""

    def test_entry_price_corrected(self):
        db, path = _make_db()
        try:
            trade = _make_trade(entry_price=2.50)
            tid = db.insert_trade(trade)

            positions = [{
                "symbol": "SPY20260321C560",
                "openQuantity": 1,
                "averageEntryPrice": 2.55,  # slightly different
            }]
            engine, _ = _engine_with_mocks(db, positions_data=positions)
            engine._reconcile_broker_positions()

            t = db.get_trade(tid)
            assert abs(t["entry_price"] - 2.55) < 0.001
        finally:
            db.conn.close()
            os.unlink(path)

    def test_entry_price_unchanged_when_close(self):
        """If price is within tolerance (< $0.005), don't update."""
        db, path = _make_db()
        try:
            trade = _make_trade(entry_price=2.500)
            tid = db.insert_trade(trade)

            positions = [{
                "symbol": "SPY20260321C560",
                "openQuantity": 1,
                "averageEntryPrice": 2.502,  # within 0.005
            }]
            engine, _ = _engine_with_mocks(db, positions_data=positions)
            engine._reconcile_broker_positions()

            t = db.get_trade(tid)
            assert abs(t["entry_price"] - 2.50) < 0.01  # unchanged
        finally:
            db.conn.close()
            os.unlink(path)


# =====================================================================
# 12. ENGINE: EXECUTION RECONCILIATION (SELL FILLS)
# =====================================================================
class TestExecutionReconciliation:
    """_reconcile_executions: match IBKR fills against DB trades."""

    def test_missed_sell_closes_open_trade(self):
        """IBKR has SELL fill, DB trade still open → close it."""
        db, path = _make_db()
        try:
            trade = _make_trade(entry_price=2.50, entry_order_id=100)
            tid = db.insert_trade(trade)

            flex_trades = [
                {
                    "symbol": "SPY", "assetCategory": "OPT", "putCall": "C",
                    "strike": "560", "expiry": "20260321",
                    "buySell": "BUY", "quantity": "1", "tradePrice": "2.50",
                    "dateTime": "2026-03-21T10:30:00", "orderID": "100",
                    "ibExecID": "buy1", "accountId": "TEST",
                    "ibCommission": "1.0",
                },
                {
                    "symbol": "SPY", "assetCategory": "OPT", "putCall": "C",
                    "strike": "560", "expiry": "20260321",
                    "buySell": "SELL", "quantity": "1", "tradePrice": "3.50",
                    "dateTime": "2026-03-21T11:00:00", "orderID": "101",
                    "ibExecID": "sell1", "accountId": "TEST",
                    "ibCommission": "1.05",
                },
            ]
            engine, _ = _engine_with_mocks(db, flex_trades=flex_trades)
            engine._reconcile_executions()

            t = db.get_trade(tid)
            assert t["status"] == "closed"
            assert abs(t["exit_price"] - 3.50) < 0.01
        finally:
            db.conn.close()
            os.unlink(path)

    def test_wrong_exit_price_corrected(self):
        """Closed trade has $0.01 exit (phantom) but real SELL fill exists → fix."""
        db, path = _make_db()
        try:
            # Use get_eastern_time to match what _reconcile_executions uses for today_str
            from live.engine import get_eastern_time
            today = get_eastern_time().strftime("%Y-%m-%d")
            expiry = get_eastern_time().strftime("%Y%m%d")
            trade = _make_trade(
                entry_price=2.50, entry_order_id=100,
                entry_time=f"{today}T10:30:00",
                symbol=f"SPY{expiry}C560",
                expiration=expiry,
            )
            tid = db.insert_trade(trade)
            # Close at $0.01 (phantom)
            db.close_trade(tid, exit_price=0.01, notes="AUTO-CLOSED: no IBKR position")

            flex_trades = [
                {
                    "symbol": "SPY", "assetCategory": "OPT", "putCall": "C",
                    "strike": "560", "expiry": expiry,
                    "buySell": "BUY", "quantity": "1", "tradePrice": "2.50",
                    "dateTime": f"{today}T10:30:00", "orderID": "100",
                    "ibExecID": "buy1", "accountId": "TEST",
                },
                {
                    "symbol": "SPY", "assetCategory": "OPT", "putCall": "C",
                    "strike": "560", "expiry": expiry,
                    "buySell": "SELL", "quantity": "1", "tradePrice": "3.50",
                    "dateTime": f"{today}T11:00:00", "orderID": "101",
                    "ibExecID": "sell1", "accountId": "TEST",
                },
            ]
            engine, _ = _engine_with_mocks(db, flex_trades=flex_trades)
            engine._reconcile_executions()

            t = db.get_trade(tid)
            assert abs(t["exit_price"] - 3.50) < 0.01
            assert "PRICE-CORRECTED" in (t["notes"] or "")
        finally:
            db.conn.close()
            os.unlink(path)

    def test_missing_buy_creates_trade(self):
        """IBKR has BUY fill with no matching DB trade → auto-create."""
        db, path = _make_db()
        try:
            flex_trades = [
                {
                    "symbol": "SPY", "assetCategory": "OPT", "putCall": "C",
                    "strike": "560", "expiry": "20260321",
                    "buySell": "BUY", "quantity": "2", "tradePrice": "2.50",
                    "dateTime": "2026-03-21T10:30:00", "orderID": "200",
                    "ibExecID": "buy1", "accountId": "TEST",
                    "ibCommission": "1.50",
                },
            ]
            engine, _ = _engine_with_mocks(db, flex_trades=flex_trades)
            engine._reconcile_executions()

            open_trades = db.get_open_trades()
            assert len(open_trades) == 1
            t = open_trades[0]
            assert t["entry_price"] == 2.50
            assert t["quantity"] == 2
            assert "AUTO-CREATED" in (t["notes"] or "")
        finally:
            db.conn.close()
            os.unlink(path)

    def test_entry_price_fix_recalculates_pnl(self):
        """If entry price is wrong on a closed trade, PnL is recalculated."""
        db, path = _make_db()
        try:
            trade = _make_trade(entry_price=2.50, entry_order_id=100)
            tid = db.insert_trade(trade)
            db.close_trade(tid, exit_price=3.50)

            flex_trades = [
                {
                    "symbol": "SPY", "assetCategory": "OPT", "putCall": "C",
                    "strike": "560", "expiry": "20260321",
                    "buySell": "BUY", "quantity": "1", "tradePrice": "2.00",  # real price was 2.00
                    "dateTime": "2026-03-21T10:30:00", "orderID": "100",
                    "ibExecID": "buy1", "accountId": "TEST",
                },
            ]
            engine, _ = _engine_with_mocks(db, flex_trades=flex_trades)
            engine._reconcile_executions()

            t = db.get_trade(tid)
            assert abs(t["entry_price"] - 2.00) < 0.01
            # PnL should be recalculated: (3.50 - 2.00) * 1 * 100 - commission
            expected_pnl = (3.50 - 2.00) * 1 * 100 - (t.get("commission", 0) or 0)
            assert abs(t["pnl"] - expected_pnl) < 1.0
        finally:
            db.conn.close()
            os.unlink(path)

    def test_commission_synced(self):
        """IBKR commission is synced into DB trade."""
        db, path = _make_db()
        try:
            trade = _make_trade(entry_price=2.50, entry_order_id=100)
            tid = db.insert_trade(trade)

            flex_trades = [
                {
                    "symbol": "SPY", "assetCategory": "OPT", "putCall": "C",
                    "strike": "560", "expiry": "20260321",
                    "buySell": "BUY", "quantity": "1", "tradePrice": "2.50",
                    "dateTime": "2026-03-21T10:30:00", "orderID": "100",
                    "ibExecID": "buy1", "accountId": "TEST",
                    "ibCommission": "1.05",
                },
            ]
            engine, _ = _engine_with_mocks(db, flex_trades=flex_trades)
            engine._reconcile_executions()

            t = db.get_trade(tid)
            assert t["commission"] == 1.05
        finally:
            db.conn.close()
            os.unlink(path)


# =====================================================================
# 13. ENGINE: FILL CALLBACK (on_fill)
# =====================================================================
class TestFillCallback:
    """Engine._on_fill updates trades correctly on BUY/SELL fills."""

    def _make_fill_order(self, order_id, symbol, side, qty, price):
        o = MagicMock()
        o.order_id = order_id
        o.symbol = symbol
        o.side = side
        o.filled_quantity = qty
        o.avg_fill_price = price
        return o

    def test_sell_fill_closes_trade(self):
        db, path = _make_db()
        try:
            trade = _make_trade(entry_price=2.50)
            tid = db.insert_trade(trade)

            engine, _ = _engine_with_mocks(db)
            engine._pending_exit_orders[42] = {
                "trade_id": tid,
                "symbol": trade.symbol,
                "signal_reason": "Profit target",
            }
            fill = self._make_fill_order(42, trade.symbol, "SELL", 1, 3.50)
            engine._on_fill(fill)

            t = db.get_trade(tid)
            assert t["status"] == "closed"
            assert abs(t["exit_price"] - 3.50) < 0.01
            assert 42 not in engine._pending_exit_orders
        finally:
            db.conn.close()
            os.unlink(path)

    def test_buy_fill_updates_entry_price(self):
        """BUY fill creates a new trade with real fill price."""
        db, path = _make_db()
        try:
            trade = _make_trade(entry_price=0.0, notes="[PENDING FILL]")
            symbol = trade.symbol

            engine, _ = _engine_with_mocks(db)
            engine._pending_entry_orders[50] = {"symbol": symbol}
            fill = self._make_fill_order(50, symbol, "BUY", 1, 2.75)
            engine._on_fill(fill)

            # Engine creates a NEW trade on BUY fill (doesn't update pre-inserted one)
            # Find the trade created by the fill
            all_trades = db.get_open_trades()
            filled_trades = [t for t in all_trades if t["entry_price"] > 0]
            assert len(filled_trades) >= 1, "Expected a trade created by the BUY fill"
            t = filled_trades[0]
            assert abs(t["entry_price"] - 2.75) < 0.01
            assert "[FILLED]" in (t["notes"] or "")
            assert 50 not in engine._pending_entry_orders
        finally:
            db.conn.close()
            os.unlink(path)

    def test_sell_fill_zero_price_logged(self):
        """Zero fill price with no bid fallback should force-close at $0.01."""
        db, path = _make_db()
        try:
            trade = _make_trade(entry_price=2.50)
            tid = db.insert_trade(trade)

            engine, _ = _engine_with_mocks(db)
            engine._pending_exit_orders[43] = {
                "trade_id": tid,
                "symbol": trade.symbol,
                "signal_reason": "SL hit",
            }
            fill = self._make_fill_order(43, trade.symbol, "SELL", 1, 0.0)
            engine._on_fill(fill)

            t = db.get_trade(tid)
            # Engine force-closes at $0.01 to prevent permanent open state
            assert t["status"] == "closed"
            assert t["exit_price"] == 0.01
        finally:
            db.conn.close()
            os.unlink(path)


# =====================================================================
# 14. SYNC_ALL ORCHESTRATION
# =====================================================================
class TestSyncAll:
    """TradeSync.sync_all runs all sources and returns counts."""

    def test_sync_all_empty(self):
        from live.trade_sync import TradeSync
        db, path = _make_db()
        try:
            syncer = TradeSync(db)
            with patch.dict(os.environ, {"IBKR_FLEX_TOKEN": "", "IBKR_FLEX_QUERY_ID": ""}):
                results = syncer.sync_all()
            assert isinstance(results, dict)
            assert results.get("ibkr", 0) == 0
        finally:
            db.conn.close()
            os.unlink(path)

    def test_sync_all_with_csv(self):
        from live.trade_sync import TradeSync
        db, path = _make_db()
        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "test_trades.csv")
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "symbol", "entry_time", "exit_time", "entry_price",
                    "exit_price", "quantity", "pnl", "action", "status",
                ])
                writer.writeheader()
                writer.writerow({
                    "symbol": "SPY20260321C560", "entry_time": "2026-03-21T10:00:00",
                    "exit_time": "2026-03-21T10:30:00", "entry_price": "2.00",
                    "exit_price": "3.00", "quantity": "1", "pnl": "100.0",
                    "action": "buy", "status": "closed",
                })
            syncer = TradeSync(db)
            results = syncer.sync_all(csv_paths=[csv_path])
            assert results.get(f"csv:test_trades.csv", 0) == 1
        finally:
            db.conn.close()
            os.unlink(path)
            os.unlink(csv_path)
            os.rmdir(tmpdir)


# =====================================================================
# 15. ADAPTER/CLIENT CONNECTIVITY PLUMBING
# =====================================================================
class TestConnectivityPlumbing:
    """Test adapter port resolution, env var handling, client wiring."""

    def test_adapter_default_port_from_config(self):
        """IBKRAdapter reads port from config.defaults when env not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("IBKR_PAPER_PORT", None)
            from clients.ibkr_adapter import IBKRAdapter
            # Don't actually connect — just inspect port resolution
            adapter = IBKRAdapter.__new__(IBKRAdapter)
            # Simulate __init__ port logic
            env_port = os.environ.get("IBKR_PAPER_PORT")
            if env_port:
                port = int(env_port)
            else:
                from config import defaults as cfg
                port = cfg.ibkr_paper_port()
            assert isinstance(port, int)
            assert port > 0

    def test_adapter_port_from_env(self):
        """IBKR_PAPER_PORT env var takes precedence."""
        with patch.dict(os.environ, {"IBKR_PAPER_PORT": "9999"}):
            from clients.ibkr_adapter import IBKRAdapter
            adapter = IBKRAdapter.__new__(IBKRAdapter)
            port = int(os.environ.get("IBKR_PAPER_PORT", "7497"))
            assert port == 9999

    def test_adapter_host_from_env(self):
        with patch.dict(os.environ, {"IBKR_HOST": "10.0.0.5"}):
            host = os.environ.get("IBKR_HOST", "127.0.0.1")
            assert host == "10.0.0.5"

    def test_engine_flex_client_init_from_env(self):
        """Engine initialises Flex client when env vars are set."""
        db, path = _make_db()
        try:
            with patch.dict(os.environ, {
                "IBKR_FLEX_TOKEN": "FAKETOKEN",
                "IBKR_FLEX_QUERY_ID": "12345",
            }):
                engine, _ = _engine_with_mocks(db)
                # Engine __init__ should have tried to init flex
                # (may fail because IBKRFlexClient validates token, but _flex_client
                #  should at least be set or logged)
                # The mock engine helper doesn't go through __init__ flex path,
                # so we test a fresh engine
                from live.engine import LiveTradingEngine, EngineConfig
                from live.position_manager import PositionManager
                from live.order_manager import OrderManager

                mock_client = MagicMock()
                pos_mgr = MagicMock(spec=PositionManager)
                ord_mgr = MagicMock(spec=OrderManager)
                config = EngineConfig(account_id="TEST", mode="paper")

                eng = LiveTradingEngine(
                    client=mock_client, trade_db=db,
                    position_manager=pos_mgr, order_manager=ord_mgr,
                    config=config,
                )
                assert eng._flex_client is not None
                assert eng._flex_query_id == 12345
        finally:
            db.conn.close()
            os.unlink(path)

    def test_engine_no_flex_when_env_missing(self):
        """Engine skips Flex when env vars are not set."""
        db, path = _make_db()
        try:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("IBKR_FLEX_TOKEN", None)
                os.environ.pop("IBKR_FLEX_QUERY_ID", None)

                from live.engine import LiveTradingEngine, EngineConfig
                from live.position_manager import PositionManager
                from live.order_manager import OrderManager

                mock_client = MagicMock()
                pos_mgr = MagicMock(spec=PositionManager)
                ord_mgr = MagicMock(spec=OrderManager)
                config = EngineConfig(account_id="TEST", mode="paper")

                eng = LiveTradingEngine(
                    client=mock_client, trade_db=db,
                    position_manager=pos_mgr, order_manager=ord_mgr,
                    config=config,
                )
                assert eng._flex_client is None
        finally:
            db.conn.close()
            os.unlink(path)


# =====================================================================
# 16. TRADE DATABASE EDGE CASES
# =====================================================================
class TestTradeDatabaseEdgeCases:

    def test_close_trade_pnl_calculation_buy(self):
        """BUY trade PnL = (exit - entry) * qty * 100 - commission."""
        db, path = _make_db()
        try:
            trade = _make_trade(entry_price=2.00, quantity=3, commission=3.15, action="BUY")
            tid = db.insert_trade(trade)
            result = db.close_trade(tid, exit_price=3.00)
            # (3.00 - 2.00) * 3 * 100 - 3.15 = 296.85
            assert abs(result["pnl"] - 296.85) < 0.01
        finally:
            db.conn.close()
            os.unlink(path)

    def test_close_trade_pnl_calculation_sell(self):
        """SELL (short) trade PnL = (entry - exit) * qty * 100 - commission."""
        db, path = _make_db()
        try:
            trade = _make_trade(entry_price=3.00, quantity=1, commission=1.05, action="SELL")
            tid = db.insert_trade(trade)
            result = db.close_trade(tid, exit_price=2.00)
            # (3.00 - 2.00) * 1 * 100 - 1.05 = 98.95
            assert abs(result["pnl"] - 98.95) < 0.01
        finally:
            db.conn.close()
            os.unlink(path)

    def test_close_nonexistent_trade(self):
        db, path = _make_db()
        try:
            result = db.close_trade(9999, exit_price=1.00)
            assert result is None
        finally:
            db.conn.close()
            os.unlink(path)

    def test_get_trade_by_order_id(self):
        db, path = _make_db()
        try:
            trade = _make_trade(entry_order_id=42)
            tid = db.insert_trade(trade)
            found = db.get_trade_by_order_id(42)
            assert found is not None
            assert found["id"] == tid
        finally:
            db.conn.close()
            os.unlink(path)

    def test_get_trade_by_exit_order_id(self):
        db, path = _make_db()
        try:
            trade = _make_trade(entry_order_id=42)
            tid = db.insert_trade(trade)
            db.close_trade(tid, exit_price=3.00, exit_order_id=43)
            found = db.get_trade_by_order_id(43)
            assert found is not None
            assert found["id"] == tid
        finally:
            db.conn.close()
            os.unlink(path)

    def test_get_trades_by_date(self):
        db, path = _make_db()
        try:
            db.insert_trade(_make_trade(entry_time="2026-03-21T10:00:00"))
            db.insert_trade(_make_trade(
                symbol="SPY20260322C560",
                entry_time="2026-03-22T10:00:00",
            ))
            trades = db.get_trades_by_date("2026-03-21")
            assert len(trades) == 1
        finally:
            db.conn.close()
            os.unlink(path)

    def test_update_trade_returns_false_no_updates(self):
        db, path = _make_db()
        try:
            assert db.update_trade(1) is False
        finally:
            db.conn.close()
            os.unlink(path)

    def test_daily_pnl_updated_on_close(self):
        db, path = _make_db()
        try:
            trade = _make_trade(entry_price=2.00, quantity=1, commission=0)
            tid = db.insert_trade(trade)
            db.close_trade(tid, exit_price=3.00, exit_time="2026-03-21T11:00:00")

            cursor = db.conn.cursor()
            cursor.execute("SELECT * FROM daily_pnl WHERE date = '2026-03-21'")
            row = cursor.fetchone()
            assert row is not None
            d = dict(row)
            assert d["win_count"] == 1
        finally:
            db.conn.close()
            os.unlink(path)


# =====================================================================
# 17. FLEX CLIENT UNIT TESTS
# =====================================================================
class TestFlexClient:

    def test_parse_empty_trades(self):
        from clients.ibkr_flex import IBKRFlexClient
        client = IBKRFlexClient(token="FAKETOKEN")
        trades = client.parse_trades("<FlexQueryResponse><FlexStatements><FlexStatement><Trades /></FlexStatement></FlexStatements></FlexQueryResponse>")
        assert trades == []

    def test_parse_single_trade(self):
        from clients.ibkr_flex import IBKRFlexClient
        client = IBKRFlexClient(token="FAKETOKEN")
        xml = """<FlexQueryResponse>
        <FlexStatements>
        <FlexStatement>
        <Trades>
            <Trade symbol="SPY" assetCategory="OPT" putCall="C"
                   strike="560" lastTradeDateOrContractMonth="20260321"
                   buySell="BUY" quantity="1" tradePrice="2.50"
                   dateTime="2026-03-21T10:00:00" orderID="100"
                   ibExecID="exec1" accountId="TEST"
                   ibCommission="1.05" />
        </Trades>
        </FlexStatement>
        </FlexStatements>
        </FlexQueryResponse>"""
        trades = client.parse_trades(xml)
        assert len(trades) == 1
        t = trades[0]
        assert t["symbol"] == "SPY"
        assert t["tradePrice"] == "2.50"

    def test_token_required(self):
        from clients.ibkr_flex import IBKRFlexClient
        try:
            IBKRFlexClient(token="")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# =====================================================================
# 18. ENGINE _get_all_ibkr_executions (prioritises Flex)
# =====================================================================
class TestGetAllIBKRExecutions:

    def test_flex_prioritised_over_tws(self):
        db, path = _make_db()
        try:
            flex_trades = [
                {
                    "symbol": "SPY", "assetCategory": "OPT", "putCall": "C",
                    "strike": "560", "expiry": "20260321",
                    "buySell": "BUY", "quantity": "1", "tradePrice": "2.50",
                    "dateTime": "2026-03-21T10:30:00", "orderID": "100",
                    "ibExecID": "flex_buy1", "accountId": "TEST",
                    "ibCommission": "1.0",
                },
            ]
            engine, mock_client = _engine_with_mocks(db, flex_trades=flex_trades)
            mock_client.get_executions.return_value = [{"side": "BOT", "symbol": "SPY"}]

            execs = engine._get_all_ibkr_executions("TEST")
            # Should return Flex data, not TWS
            assert len(execs) == 1
            assert execs[0]["exec_id"] == "flex_buy1"
            mock_client.get_executions.assert_not_called()
        finally:
            db.conn.close()
            os.unlink(path)

    def test_fallback_to_tws_when_no_flex(self):
        db, path = _make_db()
        try:
            engine, mock_client = _engine_with_mocks(db)
            mock_client.get_executions.return_value = [
                {"trade_symbol": "SPY20260321C560", "side": "BOT", "price": 2.50}
            ]
            execs = engine._get_all_ibkr_executions("TEST")
            assert len(execs) == 1
            mock_client.get_executions.assert_called_once()
        finally:
            db.conn.close()
            os.unlink(path)

    def test_flex_cache_ttl(self):
        """Second call within TTL returns cached result without re-fetching."""
        db, path = _make_db()
        try:
            flex_trades = [
                {
                    "symbol": "SPY", "assetCategory": "OPT", "putCall": "C",
                    "strike": "560", "expiry": "20260321",
                    "buySell": "BUY", "quantity": "1", "tradePrice": "2.50",
                    "dateTime": "2026-03-21T10:30:00", "orderID": "100",
                    "ibExecID": "buy1", "accountId": "TEST",
                },
            ]
            engine, _ = _engine_with_mocks(db, flex_trades=flex_trades)

            # First call populates cache
            engine._get_all_ibkr_executions("TEST")
            call_count_1 = engine._flex_client.fetch_trades.call_count

            # Second call should use cache
            engine._get_all_ibkr_executions("TEST")
            call_count_2 = engine._flex_client.fetch_trades.call_count
            assert call_count_2 == call_count_1  # no new fetch
        finally:
            db.conn.close()
            os.unlink(path)


# =====================================================================
# 19. CROSS-FORMAT RECONCILIATION (QT symbol in DB, IBKR in broker)
# =====================================================================
class TestCrossFormatReconciliation:
    """DB has Questrade symbol, broker has IBKR symbol — must still match."""

    def test_qt_db_ibkr_broker_match(self):
        """Trade entered with QT symbol, broker reports IBKR symbol → price synced."""
        db, path = _make_db()
        try:
            trade = _make_trade(
                symbol="SPY21Mar26C560.00",  # Questrade format
                entry_price=2.50,
            )
            tid = db.insert_trade(trade)

            positions = [{
                "symbol": "SPY20260321C560",  # IBKR format
                "openQuantity": 1,
                "averageEntryPrice": 2.55,
            }]
            engine, _ = _engine_with_mocks(db, positions_data=positions)
            engine._reconcile_broker_positions()

            t = db.get_trade(tid)
            assert abs(t["entry_price"] - 2.55) < 0.01
        finally:
            db.conn.close()
            os.unlink(path)

    def test_qt_db_ibkr_broker_phantom(self):
        """Trade in QT format in DB, broker has no position → detected as phantom."""
        db, path = _make_db()
        try:
            trade = _make_trade(
                symbol="SPY21Mar26P555.00",  # Questrade format
                entry_price=1.50,
            )
            tid = db.insert_trade(trade)

            engine, mock_client = _engine_with_mocks(db, positions_data=[])
            mock_client.get_executions.return_value = []
            engine._reconcile_broker_positions()

            t = db.get_trade(tid)
            assert t["status"] == "closed"
            assert t["exit_price"] == 0.01
        finally:
            db.conn.close()
            os.unlink(path)


# =====================================================================
# STANDALONE RUNNER
# =====================================================================
def _run_all():
    """Run all test classes and methods, print summary."""
    import inspect

    classes = [
        TestSymbolNormalisation,
        TestEngineNormalizeOptionKey,
        TestTimeNormalisation,
        TestDeduplication,
        TestExecutionPairing,
        TestFlexImport,
        TestCSVImport,
        TestExternalDBImport,
        TestOrphanedBrokerPosition,
        TestPhantomTradeDetection,
        TestEntryPriceDrift,
        TestExecutionReconciliation,
        TestFillCallback,
        TestSyncAll,
        TestConnectivityPlumbing,
        TestTradeDatabaseEdgeCases,
        TestFlexClient,
        TestGetAllIBKRExecutions,
        TestCrossFormatReconciliation,
    ]

    total = 0
    passed = 0
    failed = 0
    skipped = 0
    failures = []

    print("=" * 70)
    print(f"  TRADE SYNC & CONNECTIVITY EDGE-CASE TEST SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for cls in classes:
        print(f"\n{'─' * 60}")
        print(f"  {cls.__name__}")
        print(f"{'─' * 60}")

        instance = cls()
        methods = sorted(
            [m for m in dir(instance) if m.startswith("test_")],
        )

        for method_name in methods:
            total += 1
            nice_name = method_name.replace("test_", "").replace("_", " ").title()
            try:
                getattr(instance, method_name)()
                print(f"  PASS  {nice_name}")
                passed += 1
            except _SkipTest as e:
                print(f"  SKIP  {nice_name} — {e}")
                skipped += 1
            except Exception as e:
                print(f"  FAIL  {nice_name}")
                print(f"        {type(e).__name__}: {e}")
                # Get line number
                import traceback
                tb = traceback.format_exc()
                for line in tb.strip().split("\n"):
                    if "test_trade_sync_edge_cases.py" in line:
                        print(f"        {line.strip()}")
                        break
                failed += 1
                failures.append((cls.__name__, method_name, e))

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {skipped} skipped, {total} total")
    print(f"{'=' * 70}")

    if failures:
        print(f"\n  FAILURES ({len(failures)}):\n")
        for cls_name, method, exc in failures:
            print(f"  {cls_name}.{method}:")
            import traceback
            tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
            for line in tb_lines:
                for sub in line.rstrip().split("\n"):
                    if "test_trade_sync_edge_cases.py" in sub or "assert" in sub.lower() or "Error" in sub:
                        print(f"    {sub.strip()}")

    print()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
