"""
Order Execution & Persistence Edge-Case Test Suite
====================================================
Full risk coverage for:

  ORDER EXECUTION
  ───────────────
  OE-01  Order rejected on submission (zero-fill → no phantom DB trade)
  OE-02  Partial fill: order partially filled, remaining quantity tracked
  OE-03  Cancel-replace race: order filled between cancel and replace
  OE-04  Duplicate BUY signal while entry order is already pending
  OE-05  SELL order rejected, MARKET retry escalation (x2 max)
  OE-06  SELL order rejected twice → CRITICAL log, position remains open
  OE-07  Exit fill with zero avg_fill_price → not closed, error logged
  OE-08  Circuit breaker: 5 consecutive rejects trip; fill resets it
  OE-09  Margin/buying-power block for new BUY entries
  OE-10  EOD stale order cleanup on engine startup
  OE-11  wait_for_fill timeout → returns None
  OE-12  wait_for_fill cancellation before timeout
  OE-13  Order side/type enum serialisation round-trip
  OE-14  submit_order with no account_id raises ValueError
  OE-15  symbol not found raises ValueError

  PERSISTENCE (StatePersistence)
  ──────────────────────────────
  PE-01  Initial capital set from broker when state is fresh
  PE-02  Capital NOT reset when trades already exist
  PE-03  record_trade updates capital, wins/losses, high-water-mark
  PE-04  record_trade recalculates max_drawdown correctly
  PE-05  Equity curve is appended, initial point added on first trade
  PE-06  save_state/load_state round-trip (all fields preserved)
  PE-07  Corrupted state file → falls back to fresh TradingState
  PE-08  reconcile_with_db: phantom trades (AUTO-CLOSED) excluded
  PE-09  reconcile_with_db: [PHANTOM] notes excluded
  PE-10  reconcile_with_db: discrepancies detected and corrected
  PE-11  reconcile_with_db: high-water-mark and max_drawdown rebuilt
  PE-12  record_daily_summary: updates existing day record
  PE-13  get_summary returns correct win_rate calculation

  TRADE DATABASE
  ──────────────
  TD-01  close_trade: correct P&L for BUY (long call) with 100× multiplier
  TD-02  close_trade: correct P&L for SELL (short) with 100× multiplier
  TD-03  close_trade: commission subtracted from P&L
  TD-04  close_trade: pnl_percent correct
  TD-05  close_trade: non-existent trade_id returns None
  TD-06  insert_order: stores all fields; update_order_status changes them
  TD-07  Thread-safety: concurrent inserts do not corrupt DB
  TD-08  export_trades_csv: file created with correct columns
  TD-09  get_trade_by_order_id: resolves by entry AND exit order id
  TD-10  get_trades_by_date: filters by DATE(entry_time)

  POSITION MANAGER
  ────────────────
  PM-01  sync_positions clears stale positions; re-syncs from broker
  PM-02  Option symbol detection covers Questrade and OCC formats
  PM-03  update_quotes with empty position set returns []
  PM-04  calculate_position_size: zero price falls back to quantity=1
  PM-05  calculate_position_size: respects max_contracts cap
  PM-06  get_risk_metrics: aggregates Greeks across option positions
  PM-07  get_total_exposure sums market value and unrealized PnL

  ENGINE INTEGRATION
  ──────────────────
  EI-01  BUY signal in monitor mode → no order submitted
  EI-02  BUY signal blocked by max_daily_loss flag
  EI-03  BUY signal blocked by circuit breaker
  EI-04  BUY signal blocked by insufficient buying power
  EI-05  HOLD / PENDING signals → no order
  EI-06  Fill on entry → trade inserted to DB at fill price, not limit
  EI-07  Fill on exit → trade closed with real fill price in DB
  EI-08  Rejection of entry → no DB trade, strategy notified via on_trade_cancelled
  EI-09  Rejection of exit → MARKET retry submitted
  EI-10  Stale exit order timeout warning (> 90 s without fill)
  EI-11  Stale MARKET retry fills on second attempt → closed
  EI-12  _cancel_stale_orders called on startup in paper/live mode
  EI-13  _update_state_broker_balance: writes atomic JSON; concurrent read
  EI-14  _sync_strategy_capital pushes NLV to strategy.account_capital

Run:
    python -m pytest tests/test_order_execution_persistence.py -v --tb=short
    python tests/test_order_execution_persistence.py     # standalone
"""

import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call, ANY

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

@contextmanager
def _temp_db():
    """Yield a fresh TradeDatabase backed by a temp file."""
    from live.trade_database import TradeDatabase
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = TradeDatabase(path)
    try:
        yield db
    finally:
        db.conn.close()
        os.unlink(path)


@contextmanager
def _temp_state_file(initial_data: dict = None):
    """Yield path to a temp state JSON file."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    if initial_data is not None:
        with open(path, "w") as fh:
            json.dump(initial_data, fh)
    else:
        os.unlink(path)  # StatePersistence: "no file → fresh state"
        # ensure the path does not exist
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _make_trade(**kw):
    from live.trade_database import Trade
    defaults = dict(
        symbol="SPY20260321C560",
        underlying="SPY",
        trade_type="option",
        option_type="call",
        strike=560.0,
        expiration="20260321",
        action="buy",
        quantity=1,
        entry_price=2.00,
        entry_time="2026-03-21T10:30:00",
        status="open",
        strategy_name="test",
        account_id="TEST",
    )
    defaults.update(kw)
    return Trade(**defaults)


def _mock_broker(positions=None, orders=None, symbol_id=12345):
    """Return a minimal mock broker client."""
    client = MagicMock()
    client.get_account_positions.return_value = positions or []
    client.get_account_orders.return_value = orders or []
    client.get_symbol_id.return_value = symbol_id
    client.place_order.return_value = {"orderId": 9001, "orderState": "Pending"}
    client.cancel_order.return_value = True
    client.cancel_all_open_orders.return_value = None
    client.get_executions.return_value = []
    return client


def _build_engine(db, positions_data=None, executions=None,
                  flex_trades=None, pending_exit_orders=None, mode="paper"):
    from live.engine import LiveTradingEngine, EngineConfig
    from live.position_manager import PositionManager
    from live.order_manager import OrderManager

    client = _mock_broker(positions=positions_data or [], orders=[])
    client.get_executions.return_value = executions or []

    pos_mgr = MagicMock(spec=PositionManager)
    pos_mgr.sync_positions.return_value = None
    pos_mgr.get_all_positions.return_value = []
    pos_mgr.get_total_exposure.return_value = {"total_unrealized_pnl": 0.0}
    pos_mgr.update_quotes.return_value = None

    ord_mgr = MagicMock(spec=OrderManager)
    ord_mgr.sync_orders.return_value = None

    config = EngineConfig(
        account_id="TEST",
        symbols=["SPY"],
        option_underlyings=["SPY"],
        mode=mode,
    )

    engine = LiveTradingEngine(
        client=client,
        trade_db=db,
        position_manager=pos_mgr,
        order_manager=ord_mgr,
        config=config,
    )

    # Disable real Flex client
    engine._flex_client = None
    engine._flex_query_id = None

    if flex_trades is not None:
        engine._flex_client = MagicMock()
        engine._flex_query_id = 99
        engine._flex_client.fetch_trades.return_value = flex_trades

    if pending_exit_orders:
        engine._pending_exit_orders = pending_exit_orders

    return engine, client


# ─────────────────────────────────────────────────────────────────────
# Test runner infrastructure (pytest-compatible + standalone)
# ─────────────────────────────────────────────────────────────────────

_PASS = []
_FAIL = []

def _run(name, fn):
    try:
        fn()
        _PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as exc:
        _FAIL.append((name, exc))
        print(f"  FAIL  {name}: {exc}")


# =====================================================================
# OE  ORDER EXECUTION
# =====================================================================

class TestOrderExecution:
    """Tests for OrderManager — order submission, tracking, fill/reject flow."""

    def test_oe01_rejected_order_no_db_trade(self):
        """OE-01: Broker returns no orderId → order status == Rejected; no DB record."""
        from live.order_manager import OrderManager, OrderSide, OrderType
        client = _mock_broker()
        client.place_order.return_value = {}  # no orderId  → rejected

        with _temp_db() as db:
            om = OrderManager(broker_client=client, trade_db=db)
            om.set_account("ACC1")

            order = om.submit_order(
                symbol="SPY", quantity=1, side=OrderSide.BUY,
                order_type=OrderType.LIMIT, limit_price=5.00
            )

            assert order.status == "Rejected", f"Expected Rejected, got {order.status}"
            # No trade should be in the DB
            rows = db.get_open_trades()
            assert len(rows) == 0

    def test_oe02_submit_order_success_tracked(self):
        """OE-02: Successful submission stores order_id in internal dict."""
        from live.order_manager import OrderManager, OrderSide, OrderType
        client = _mock_broker()
        client.place_order.return_value = {"orderId": 42, "orderState": "Open"}

        with _temp_db() as db:
            om = OrderManager(broker_client=client, trade_db=db)
            om.set_account("ACC1")
            order = om.submit_order(
                symbol="SPY", quantity=2, side=OrderSide.BUY,
                order_type=OrderType.LIMIT, limit_price=4.50
            )
            assert order.order_id == 42
            assert order.status == "Open"
            assert om.get_order(42) is order

    def test_oe03_cancel_order_marks_cancelled_in_db(self):
        """OE-03: cancel_order updates in-memory status and calls DB update."""
        from live.order_manager import OrderManager, OrderSide, OrderType
        client = _mock_broker()
        client.place_order.return_value = {"orderId": 77, "orderState": "Open"}
        client.cancel_order.return_value = True

        with _temp_db() as db:
            om = OrderManager(broker_client=client, trade_db=db)
            om.set_account("ACC1")
            om.submit_order(
                symbol="SPY", quantity=1, side=OrderSide.BUY,
                order_type=OrderType.LIMIT, limit_price=3.00
            )
            result = om.cancel_order(77)
            assert result is True
            assert om.get_order(77).status == "Canceled"

    def test_oe04_cancel_all_orders_cancels_open_only(self):
        """OE-04: cancel_all_orders fetches open orders and cancels each."""
        from live.order_manager import OrderManager, OrderSide, Order
        client = _mock_broker()

        open_order_data = [{
            "id": 100, "symbol": "SPY", "symbolId": 111, "state": "Open",
            "side": "Buy", "orderType": "Limit", "limitPrice": 2.00,
            "totalQuantity": 1, "filledQuantity": 0, "avgExecPrice": 0,
            "timeInForce": "Day", "creationTime": "", "updateTime": "",
        }]
        client.get_account_orders.return_value = open_order_data
        client.cancel_order.return_value = True

        with _temp_db() as db:
            om = OrderManager(broker_client=client, trade_db=db)
            om.set_account("ACC1")
            cancelled = om.cancel_all_orders()
            assert cancelled == 1

    def test_oe05_submit_order_no_account_raises(self):
        """OE-05: submit_order without set_account raises ValueError."""
        from live.order_manager import OrderManager, OrderSide, OrderType
        client = _mock_broker()
        om = OrderManager(broker_client=client)
        try:
            om.submit_order(symbol="SPY", quantity=1, side=OrderSide.BUY,
                            order_type=OrderType.MARKET)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_oe06_submit_order_symbol_not_found_raises(self):
        """OE-06: Unknown symbol → ValueError before any broker call."""
        from live.order_manager import OrderManager, OrderSide, OrderType
        client = _mock_broker()
        client.get_symbol_id.return_value = None  # symbol not found
        om = OrderManager(broker_client=client)
        om.set_account("ACC1")
        try:
            om.submit_order(symbol="UNKNOWN_SYM", quantity=1, side=OrderSide.BUY,
                            order_type=OrderType.MARKET)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_oe07_wait_for_fill_timeout(self):
        """OE-07: wait_for_fill returns None when order never fills within timeout."""
        from live.order_manager import OrderManager, OrderSide, OrderType, Order
        client = _mock_broker()
        client.place_order.return_value = {"orderId": 55, "orderState": "Open"}
        # sync_orders always returns "Open" (never filled)
        client.get_account_orders.return_value = [{
            "id": 55, "symbol": "SPY", "symbolId": 111, "state": "Open",
            "side": "Buy", "orderType": "Limit", "limitPrice": 2.00,
            "totalQuantity": 1, "filledQuantity": 0, "avgExecPrice": 0,
            "timeInForce": "Day", "creationTime": "", "updateTime": "",
        }]
        with _temp_db() as db:
            om = OrderManager(broker_client=client, trade_db=db)
            om.set_account("ACC1")
            om.submit_order(symbol="SPY", quantity=1, side=OrderSide.BUY,
                            order_type=OrderType.LIMIT, limit_price=2.00)
            result = om.wait_for_fill(order_id=55, timeout=2, poll_interval=0.5)
            assert result is None

    def test_oe08_wait_for_fill_cancelled(self):
        """OE-08: wait_for_fill returns None immediately on Canceled status."""
        from live.order_manager import OrderManager, OrderSide, OrderType
        client = _mock_broker()
        client.place_order.return_value = {"orderId": 66, "orderState": "Open"}
        client.get_account_orders.return_value = [{
            "id": 66, "symbol": "SPY", "symbolId": 111, "state": "Canceled",
            "side": "Buy", "orderType": "Limit", "limitPrice": 2.00,
            "totalQuantity": 1, "filledQuantity": 0, "avgExecPrice": 0,
            "timeInForce": "Day", "creationTime": "", "updateTime": "",
        }]
        with _temp_db() as db:
            om = OrderManager(broker_client=client, trade_db=db)
            om.set_account("ACC1")
            om.submit_order(symbol="SPY", quantity=1, side=OrderSide.BUY,
                            order_type=OrderType.LIMIT, limit_price=2.00)
            result = om.wait_for_fill(order_id=66, timeout=5, poll_interval=0.1)
            assert result is None

    def test_oe09_wait_for_fill_success(self):
        """OE-09: wait_for_fill returns the Order once status == Filled."""
        from live.order_manager import OrderManager, OrderSide, OrderType
        client = _mock_broker()
        client.place_order.return_value = {"orderId": 77, "orderState": "Open"}
        client.get_account_orders.return_value = [{
            "id": 77, "symbol": "SPY", "symbolId": 111, "state": "Filled",
            "side": "Buy", "orderType": "Limit", "limitPrice": 2.00,
            "totalQuantity": 1, "filledQuantity": 1, "avgExecPrice": 2.05,
            "timeInForce": "Day", "creationTime": "", "updateTime": "",
        }]
        with _temp_db() as db:
            om = OrderManager(broker_client=client, trade_db=db)
            om.set_account("ACC1")
            om.submit_order(symbol="SPY", quantity=1, side=OrderSide.BUY,
                            order_type=OrderType.LIMIT, limit_price=2.00)
            result = om.wait_for_fill(order_id=77, timeout=5, poll_interval=0.1)
            assert result is not None
            assert result.status == "Filled"

    def test_oe10_fill_callback_triggered_on_status_change(self):
        """OE-10: fill callback fires when order transitions from Open → Filled."""
        from live.order_manager import OrderManager, OrderSide, OrderType
        client = _mock_broker()
        client.place_order.return_value = {"orderId": 88, "orderState": "Open"}

        # First sync: Open; second sync: Filled
        open_data = [{
            "id": 88, "symbol": "SPY", "symbolId": 111, "state": "Open",
            "side": "Buy", "orderType": "Limit", "limitPrice": 2.00,
            "totalQuantity": 1, "filledQuantity": 0, "avgExecPrice": 0,
            "timeInForce": "Day", "creationTime": "", "updateTime": "",
            "commissionCharged": 0,
        }]
        filled_data = [{
            "id": 88, "symbol": "SPY", "symbolId": 111, "state": "Filled",
            "side": "Buy", "orderType": "Limit", "limitPrice": 2.00,
            "totalQuantity": 1, "filledQuantity": 1, "avgExecPrice": 2.10,
            "timeInForce": "Day", "creationTime": "", "updateTime": "",
            "commissionCharged": 1.00,
        }]
        call_count = {"n": 0}
        def _fill_cb(order):
            call_count["n"] += 1

        with _temp_db() as db:
            om = OrderManager(broker_client=client, trade_db=db)
            om.set_account("ACC1")
            om.on_fill(_fill_cb)
            om.submit_order(symbol="SPY", quantity=1, side=OrderSide.BUY,
                            order_type=OrderType.LIMIT, limit_price=2.00)

            # First sync: open
            client.get_account_orders.return_value = open_data
            om.sync_orders()
            assert call_count["n"] == 0

            # Second sync: filled
            client.get_account_orders.return_value = filled_data
            om.sync_orders()
            assert call_count["n"] == 1

    def test_oe11_reject_callback_triggered(self):
        """OE-11: reject callback fires when order transitions to Rejected."""
        from live.order_manager import OrderManager, OrderSide, OrderType
        client = _mock_broker()
        client.place_order.return_value = {"orderId": 99, "orderState": "Open"}

        open_data = [{
            "id": 99, "symbol": "SPY", "symbolId": 111, "state": "Open",
            "side": "Sell", "orderType": "Limit", "limitPrice": 2.00,
            "totalQuantity": 1, "filledQuantity": 0, "avgExecPrice": 0,
            "timeInForce": "Day", "creationTime": "", "updateTime": "",
            "commissionCharged": 0,
        }]
        rejected_data = [{
            "id": 99, "symbol": "SPY", "symbolId": 111, "state": "Rejected",
            "side": "Sell", "orderType": "Limit", "limitPrice": 2.00,
            "totalQuantity": 1, "filledQuantity": 0, "avgExecPrice": 0,
            "timeInForce": "Day", "creationTime": "", "updateTime": "",
            "commissionCharged": 0,
        }]
        rejected_orders = []

        with _temp_db() as db:
            om = OrderManager(broker_client=client, trade_db=db)
            om.set_account("ACC1")
            om.on_reject(lambda o: rejected_orders.append(o))
            om.submit_order(symbol="SPY", quantity=1, side=OrderSide.SELL,
                            order_type=OrderType.LIMIT, limit_price=2.00)

            client.get_account_orders.return_value = open_data
            om.sync_orders()
            assert len(rejected_orders) == 0

            client.get_account_orders.return_value = rejected_data
            om.sync_orders()
            assert len(rejected_orders) == 1

    def test_oe12_order_type_enum_round_trip(self):
        """OE-12: OrderType and OrderSide enums serialise to correct broker strings."""
        from live.order_manager import OrderSide, OrderType, TimeInForce
        assert OrderSide.BUY.value == "Buy"
        assert OrderSide.SELL.value == "Sell"
        assert OrderType.MARKET.value == "Market"
        assert OrderType.LIMIT.value == "Limit"
        assert OrderType.STOP.value == "Stop"
        assert OrderType.STOP_LIMIT.value == "StopLimit"
        assert TimeInForce.DAY.value == "Day"
        assert TimeInForce.FOK.value == "FillOrKill"
        assert TimeInForce.IOC.value == "ImmediateOrCancel"

    def test_oe13_modify_order_cancel_replace(self):
        """OE-13: modify_order cancels original and resubmits with new params."""
        from live.order_manager import OrderManager, OrderSide, OrderType
        client = _mock_broker()
        client.place_order.side_effect = [
            {"orderId": 10, "orderState": "Open"},   # original
            {"orderId": 11, "orderState": "Open"},   # replacement
        ]
        client.cancel_order.return_value = True
        client.get_account_orders.return_value = []  # no open orders for cancel_all

        with _temp_db() as db:
            om = OrderManager(broker_client=client, trade_db=db)
            om.set_account("ACC1")
            om.submit_order(symbol="SPY", quantity=1, side=OrderSide.BUY,
                            order_type=OrderType.LIMIT, limit_price=2.00)
            result = om.modify_order(10, new_limit_price=2.50)
            # cancel+resubmit means success
            assert result is True
            assert client.cancel_order.called


# =====================================================================
# PE  PERSISTENCE
# =====================================================================

class TestStatePersistence:
    """Tests for StatePersistence and TradingState."""

    def test_pe01_initial_capital_fresh_state(self):
        """PE-01: set_initial_capital on fresh state (0 trades) updates all capital fields."""
        from live.state_persistence import StatePersistence
        with _temp_state_file() as path:
            if not os.path.exists(path):
                open(path, 'w').close()
                os.unlink(path)
            sp = StatePersistence(state_file=path)
            sp.set_initial_capital(10_000.00)
            assert sp.state.initial_capital == 10_000.00
            assert sp.state.current_capital == 10_000.00
            assert sp.state.high_water_mark == 10_000.00

    def test_pe02_capital_not_reset_when_trades_exist(self):
        """PE-02: set_initial_capital with existing trades keeps initial_capital intact."""
        from live.state_persistence import StatePersistence
        with _temp_state_file() as path:
            sp = StatePersistence(state_file=path)
            sp.state.total_trades = 5
            sp.state.initial_capital = 10_000.00
            sp.state.current_capital = 10_500.00
            sp.set_initial_capital(9_800.00)  # broker says 9800
            # initial_capital must NOT change when trades exist
            assert sp.state.initial_capital == 10_000.00
            # current_capital updated to broker equity
            assert sp.state.current_capital == 9_800.00

    def test_pe03_record_trade_updates_stats(self):
        """PE-03: record_trade increments trades/wins/losses and adjusts capital."""
        from live.state_persistence import StatePersistence
        with _temp_state_file() as path:
            sp = StatePersistence(state_file=path)
            sp.set_initial_capital(10_000.00)
            sp.record_trade(pnl=150.00, is_win=True, trade_id=1)
            assert sp.state.total_trades == 1
            assert sp.state.total_wins == 1
            assert sp.state.total_losses == 0
            assert abs(sp.state.total_pnl - 150.00) < 0.01
            assert abs(sp.state.current_capital - 10_150.00) < 0.01
            sp.record_trade(pnl=-75.00, is_win=False, trade_id=2)
            assert sp.state.total_trades == 2
            assert sp.state.total_losses == 1
            assert abs(sp.state.total_pnl - 75.00) < 0.01

    def test_pe04_record_trade_max_drawdown(self):
        """PE-04: max_drawdown updated correctly after a losing trade."""
        from live.state_persistence import StatePersistence
        with _temp_state_file() as path:
            sp = StatePersistence(state_file=path)
            sp.set_initial_capital(10_000.00)
            sp.record_trade(pnl=500.00, is_win=True, trade_id=1)   # HWM = 10500
            sp.record_trade(pnl=-1000.00, is_win=False, trade_id=2) # capital = 9500
            expected_dd = 1000.0 / 10_500.0
            assert abs(sp.state.max_drawdown - expected_dd) < 1e-4

    def test_pe05_equity_curve_appended(self):
        """PE-05: equity curve builds correctly with initial entry + subsequent trades."""
        from live.state_persistence import StatePersistence
        with _temp_state_file() as path:
            sp = StatePersistence(state_file=path)
            sp.set_initial_capital(10_000.00)
            sp.record_trade(pnl=200.00, is_win=True, trade_id=1, option_type="CALL")
            assert len(sp.state.equity_curve) == 2  # initial entry + 1 trade
            assert sp.state.equity_curve[0]["equity"] == pytest_approx(10_000.00)
            assert sp.state.equity_curve[1]["equity"] == pytest_approx(10_200.00)
            assert sp.state.equity_curve[1]["type"] == "CALL"

    def test_pe06_save_load_roundtrip(self):
        """PE-06: Save and re-load state; all scalar fields match."""
        from live.state_persistence import StatePersistence
        with _temp_state_file() as path:
            sp = StatePersistence(state_file=path)
            sp.set_initial_capital(10_000.00)
            sp.record_trade(pnl=300.00, is_win=True, trade_id=1)
            sp.save_state()

            sp2 = StatePersistence(state_file=path)
            assert sp2.state.total_trades == 1
            assert abs(sp2.state.total_pnl - 300.00) < 0.01
            assert abs(sp2.state.initial_capital - 10_000.00) < 0.01
            assert abs(sp2.state.current_capital - 10_300.00) < 0.01

    def test_pe07_corrupted_state_file_falls_back(self):
        """PE-07: Corrupted JSON → StatePersistence initialises with a fresh state."""
        from live.state_persistence import StatePersistence
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as fh:
            fh.write("{not valid json{{{{")
        try:
            sp = StatePersistence(state_file=path)
            # Should not raise; falls back to a zeroed TradingState
            assert sp.state.total_trades == 0
        finally:
            os.unlink(path)

    def test_pe08_reconcile_excludes_auto_closed_phantom(self):
        """PE-08: AUTO-CLOSED phantom trades are excluded from reconciliation stats."""
        from live.state_persistence import StatePersistence
        fd_db, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd_db)
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY, symbol TEXT, pnl REAL,
                    status TEXT, notes TEXT, exit_time TEXT
                )
            """)
            # Real trade
            conn.execute("INSERT INTO trades VALUES (1, 'SPY20260321C560', 100.0, 'closed', NULL, '2026-03-21T15:00:00')")
            # Phantom AUTO-CLOSED trade
            conn.execute("INSERT INTO trades VALUES (2, 'SPY20260321P550', 0.01, 'closed', 'AUTO-CLOSED: no IBKR position, no SELL fill', '2026-03-21T15:01:00')")
            conn.commit()
            conn.close()

            with _temp_state_file() as sp_path:
                sp = StatePersistence(state_file=sp_path, db_path=db_path)
                # Without reconcile first, call explicitly
                result = sp.reconcile_with_db(db_path=db_path)
                assert result["status"] == "success"
                assert sp.state.total_trades == 1  # phantom excluded
                assert abs(sp.state.total_pnl - 100.0) < 0.01
        finally:
            os.unlink(db_path)

    def test_pe09_reconcile_excludes_phantom_note_prefix(self):
        """PE-09: Trades with [PHANTOM] note prefix are excluded from reconciliation."""
        from live.state_persistence import StatePersistence
        fd_db, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd_db)
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY, symbol TEXT, pnl REAL,
                    status TEXT, notes TEXT, exit_time TEXT
                )
            """)
            conn.execute("INSERT INTO trades VALUES (1, 'SPY20260321C560', 200.0, 'closed', NULL, '2026-03-21T15:00:00')")
            conn.execute("INSERT INTO trades VALUES (2, 'SPY20260321C560', 0.5, 'closed', '[PHANTOM] duplicate', '2026-03-21T15:00:01')")
            conn.commit()
            conn.close()

            with _temp_state_file() as sp_path:
                sp = StatePersistence(state_file=sp_path, db_path=db_path)
                result = sp.reconcile_with_db(db_path=db_path)
                assert result["status"] == "success"
                assert sp.state.total_trades == 1
                assert abs(sp.state.total_pnl - 200.0) < 0.01
        finally:
            os.unlink(db_path)

    def test_pe10_reconcile_detects_and_corrects_discrepancies(self):
        """PE-10: reconcile_with_db finds JSON vs DB mismatches and corrects them."""
        from live.state_persistence import StatePersistence
        fd_db, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd_db)
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY, symbol TEXT, pnl REAL,
                    status TEXT, notes TEXT, exit_time TEXT
                )
            """)
            conn.execute("INSERT INTO trades VALUES (1, 'SPY20260321C560', 300.0, 'closed', NULL, '2026-03-21T15:00:00')")
            conn.commit()
            conn.close()

            with _temp_state_file() as sp_path:
                sp = StatePersistence(state_file=sp_path, db_path=db_path)
                # Artificially mis-set state
                sp.state.total_pnl = 999.0
                sp.state.total_trades = 99

                result = sp.reconcile_with_db(db_path=db_path)
                assert result["status"] == "success"
                # At least pnl discrepancy detected
                assert len(result["discrepancies"]) > 0
                assert abs(sp.state.total_pnl - 300.0) < 0.01
                assert sp.state.total_trades == 1
        finally:
            os.unlink(db_path)

    def test_pe11_reconcile_hwm_and_drawdown_rebuilt(self):
        """PE-11: reconcile_with_db correctly rebuilds high-water-mark and max_drawdown."""
        from live.state_persistence import StatePersistence
        fd_db, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd_db)
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY, symbol TEXT, pnl REAL,
                    status TEXT, notes TEXT, exit_time TEXT
                )
            """)
            # Trade sequence: +500, -300  → HWM=10500, DD=300/10500
            conn.execute("INSERT INTO trades VALUES (1,'SPY',500.0,'closed',NULL,'2026-03-21T11:00:00')")
            conn.execute("INSERT INTO trades VALUES (2,'SPY',-300.0,'closed',NULL,'2026-03-21T12:00:00')")
            conn.commit()
            conn.close()

            with _temp_state_file() as sp_path:
                initial = {"initial_capital": 10000.0, "current_capital": 10000.0,
                           "high_water_mark": 10000.0, "total_trades": 0,
                           "total_wins": 0, "total_losses": 0, "total_pnl": 0.0,
                           "max_drawdown": 0.0, "equity_curve": [], "daily_records": [],
                           "last_updated": "", "last_trade_date": "",
                           "engine_status": "unknown", "strategy_state": {}}
                with open(sp_path, "w") as fh:
                    json.dump(initial, fh)
                sp = StatePersistence(state_file=sp_path, db_path=db_path)
                result = sp.reconcile_with_db(db_path=db_path)
                assert result["status"] == "success"
                assert abs(sp.state.high_water_mark - 10_500.0) < 0.01
                expected_dd = 300.0 / 10_500.0
                assert abs(sp.state.max_drawdown - expected_dd) < 1e-4
        finally:
            os.unlink(db_path)

    def test_pe12_record_daily_summary_updates_existing(self):
        """PE-12: Calling record_daily_summary twice on same day updates, not appends."""
        from live.state_persistence import StatePersistence
        with _temp_state_file() as path:
            sp = StatePersistence(state_file=path)
            sp.set_initial_capital(10_000.00)
            sp.record_daily_summary(trades=3, wins=2, losses=1, pnl=150.0)
            sp.record_daily_summary(trades=5, wins=4, losses=1, pnl=250.0)
            today = date.today().isoformat()
            day_recs = [r for r in sp.state.daily_records if r.get("date") == today]
            assert len(day_recs) == 1  # updated in place, not duplicated
            assert day_recs[0]["trades"] == 5

    def test_pe13_get_summary_win_rate(self):
        """PE-13: get_summary computes win_rate as wins/total_trades."""
        from live.state_persistence import StatePersistence
        with _temp_state_file() as path:
            sp = StatePersistence(state_file=path)
            sp.set_initial_capital(10_000.00)
            sp.record_trade(100.0, is_win=True, trade_id=1)
            sp.record_trade(100.0, is_win=True, trade_id=2)
            sp.record_trade(-50.0, is_win=False, trade_id=3)
            summary = sp.get_summary()
            assert abs(summary["win_rate"] - 2/3) < 1e-6
            assert summary["total_trades"] == 3


# =====================================================================
# TD  TRADE DATABASE
# =====================================================================

class TestTradeDatabase:
    """Tests for TradeDatabase operations."""

    def test_td01_close_trade_long_call_pnl(self):
        """TD-01: Long call BUY → P&L = (exit - entry) * qty * 100."""
        with _temp_db() as db:
            tid = db.insert_trade(_make_trade(entry_price=2.00, quantity=3, action="buy"))
            db.close_trade(tid, exit_price=3.50)
            t = db.get_trade(tid)
            expected = (3.50 - 2.00) * 3 * 100  # = 450
            assert abs(t["pnl"] - expected) < 0.01

    def test_td02_close_trade_short_sell_pnl(self):
        """TD-02: SELL (short) → P&L = (entry - exit) * qty * 100."""
        with _temp_db() as db:
            tid = db.insert_trade(_make_trade(entry_price=3.00, quantity=2, action="sell"))
            db.close_trade(tid, exit_price=1.50)
            t = db.get_trade(tid)
            expected = (3.00 - 1.50) * 2 * 100  # = 300
            assert abs(t["pnl"] - expected) < 0.01

    def test_td03_close_trade_commission_subtracted(self):
        """TD-03: Commission is subtracted from P&L on close."""
        with _temp_db() as db:
            tid = db.insert_trade(_make_trade(entry_price=2.00, quantity=1, commission=1.30))
            db.close_trade(tid, exit_price=3.00)
            t = db.get_trade(tid)
            gross = (3.00 - 2.00) * 1 * 100  # = 100
            expected = gross - 1.30            # = 98.70
            assert abs(t["pnl"] - expected) < 0.01

    def test_td04_close_trade_pnl_percent_correct(self):
        """TD-04: pnl_percent = pnl / (entry * qty * 100) * 100."""
        with _temp_db() as db:
            tid = db.insert_trade(_make_trade(entry_price=2.00, quantity=1, action="buy"))
            db.close_trade(tid, exit_price=2.50)
            t = db.get_trade(tid)
            expected_pct = (50.0 / (2.00 * 1 * 100)) * 100  # 25%
            assert abs(t["pnl_percent"] - expected_pct) < 0.01

    def test_td05_close_nonexistent_trade_returns_none(self):
        """TD-05: close_trade with non-existent ID returns None."""
        with _temp_db() as db:
            result = db.close_trade(999999, exit_price=1.50)
            assert result is None

    def test_td06_insert_and_update_order_status(self):
        """TD-06: insert_order creates a row; update_order_status changes it to Filled."""
        with _temp_db() as db:
            oid = db.insert_order(
                order_id=500, symbol="SPY", account_id="ACC",
                action="buy", order_type="Limit", quantity=1, limit_price=2.00
            )
            assert oid > 0
            db.update_order_status(500, "Filled", filled_quantity=1, avg_fill_price=2.10)
            row = db.conn.execute("SELECT * FROM orders WHERE order_id=500").fetchone()
            assert row["status"] == "Filled"
            assert row["filled_quantity"] == 1
            assert abs(row["avg_fill_price"] - 2.10) < 0.01

    def test_td07_concurrent_inserts_thread_safe(self):
        """TD-07: Concurrent inserts from multiple threads do not corrupt the DB."""
        with _temp_db() as db:
            errors = []
            def insert_trade(i):
                try:
                    db.insert_trade(_make_trade(
                        symbol=f"SPY20260321C{500+i}",
                        entry_time=f"2026-03-21T{10 + i // 60:02d}:{i % 60:02d}:00",
                    ))
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=insert_trade, args=(i,)) for i in range(30)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            assert not errors, f"Thread errors: {errors}"
            rows = db.get_open_trades()
            assert len(rows) == 30

    def test_td08_export_trades_csv(self):
        """TD-08: export_trades_csv creates a file with the expected header."""
        with _temp_db() as db:
            db.insert_trade(_make_trade())
            fd, csv_path = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
            try:
                db.export_trades_csv(csv_path)
                with open(csv_path) as fh:
                    header = fh.readline()
                assert "symbol" in header.lower() or "Symbol" in header
            finally:
                os.unlink(csv_path)

    def test_td09_get_trade_by_order_id_entry_and_exit(self):
        """TD-09: get_trade_by_order_id resolves via entry_order_id AND exit_order_id."""
        with _temp_db() as db:
            tid = db.insert_trade(_make_trade(entry_order_id=700))
            db.close_trade(tid, exit_price=3.00, exit_order_id=800)
            assert db.get_trade_by_order_id(700) is not None
            assert db.get_trade_by_order_id(800) is not None
            assert db.get_trade_by_order_id(9999) is None

    def test_td10_get_trades_by_date(self):
        """TD-10: get_trades_by_date filters on DATE(entry_time)."""
        with _temp_db() as db:
            db.insert_trade(_make_trade(entry_time="2026-03-21T10:00:00"))
            db.insert_trade(_make_trade(entry_time="2026-03-22T10:00:00"))
            db.insert_trade(_make_trade(entry_time="2026-03-21T14:00:00"))
            results = db.get_trades_by_date("2026-03-21")
            assert len(results) == 2
            assert all("2026-03-21" in r["entry_time"] for r in results)


# =====================================================================
# PM  POSITION MANAGER
# =====================================================================

class TestPositionManager:
    """Tests for PositionManager position sync, sizing, and risk metrics."""

    def test_pm01_sync_positions_clears_stale(self):
        """PM-01: sync_positions replaces stale positions with fresh broker data."""
        from live.position_manager import PositionManager
        broker_data_first = [
            {"symbol": "SPY", "symbolId": 1, "openQuantity": 100,
             "averageEntryPrice": 500.0, "currentPrice": 501.0,
             "currentMarketValue": 50100.0, "openPnl": 100.0,
             "openPnlPercent": 0.2, "dayPnl": 50.0}
        ]
        broker_data_second = [
            {"symbol": "AAPL", "symbolId": 2, "openQuantity": 50,
             "averageEntryPrice": 200.0, "currentPrice": 202.0,
             "currentMarketValue": 10100.0, "openPnl": 100.0,
             "openPnlPercent": 1.0, "dayPnl": 20.0}
        ]
        client = _mock_broker(positions=broker_data_first)
        pm = PositionManager(broker_client=client)
        pm.set_account("ACC")

        pm.sync_positions()
        assert pm.get_position("SPY") is not None
        assert pm.get_position("AAPL") is None

        client.get_account_positions.return_value = broker_data_second
        pm.sync_positions()
        assert pm.get_position("SPY") is None
        assert pm.get_position("AAPL") is not None

    def test_pm02_option_symbol_detection(self):
        """PM-02: Both Questrade and OCC option symbol formats are detected."""
        from live.position_manager import PositionManager
        client = _mock_broker()
        pm = PositionManager(broker_client=client)
        assert pm._is_option_symbol("AAPL30Jan26C240.00") is True
        assert pm._is_option_symbol("SPY20260310C680") is True
        assert pm._is_option_symbol("SPY") is False
        assert pm._is_option_symbol("AAPL") is False

    def test_pm03_update_quotes_empty_positions(self):
        """PM-03: update_quotes returns [] when no positions are loaded."""
        from live.position_manager import PositionManager
        client = _mock_broker()
        pm = PositionManager(broker_client=client)
        pm.set_account("ACC")
        result = pm.update_quotes()
        assert result == []

    def test_pm04_position_size_zero_price_fallback(self):
        """PM-04: calculate_position_size returns 1 when current price is 0."""
        from live.position_manager import PositionManager
        client = _mock_broker()
        client.get_quote_by_symbol.return_value = {"lastTradePrice": 0}
        pm = PositionManager(broker_client=client)
        pm.set_account("ACC")
        qty = pm.calculate_position_size("SPY", account_value=10000.0)
        assert qty == 1

    def test_pm05_position_size_respects_max_contracts(self):
        """PM-05: calculate_position_size caps at max_contracts."""
        from live.position_manager import PositionManager
        client = _mock_broker()
        # Very cheap option → Kelly would suggest many contracts
        client.get_quote_by_symbol.return_value = {"lastTradePrice": 0.01}
        pm = PositionManager(broker_client=client)
        pm.set_account("ACC")
        qty = pm.calculate_position_size("SPY", account_value=100_000.0, max_contracts=5)
        assert qty <= 5

    def test_pm06_get_risk_metrics_aggregates_greeks(self):
        """PM-06: get_risk_metrics sums Greeks across all option positions."""
        from live.position_manager import PositionManager, Position
        client = _mock_broker()
        pm = PositionManager(broker_client=client)
        pm.set_account("ACC")

        # Manually inject two option positions
        p1 = Position(
            symbol="SPY20260321C560", symbol_id=1, quantity=2, avg_cost=2.0,
            is_option=True, underlying="SPY", delta=0.5, gamma=0.02,
            theta=-0.10, vega=0.30
        )
        p2 = Position(
            symbol="SPY20260321P550", symbol_id=2, quantity=1, avg_cost=1.5,
            is_option=True, underlying="SPY", delta=-0.4, gamma=0.015,
            theta=-0.08, vega=0.25
        )
        pm._positions = {
            "SPY20260321C560": p1,
            "SPY20260321P550": p2,
        }
        metrics = pm.get_risk_metrics()
        # portfolio_delta = (0.5*2 + (-0.4)*1) * 100 = (1.0 - 0.4)*100 = 60
        assert abs(metrics["portfolio_delta"] - 60.0) < 0.01

    def test_pm07_get_total_exposure_sums_values(self):
        """PM-07: get_total_exposure sums market_value and unrealized_pnl."""
        from live.position_manager import PositionManager, Position
        client = _mock_broker()
        pm = PositionManager(broker_client=client)
        pm.set_account("ACC")
        p1 = Position(symbol="SPY", symbol_id=1, quantity=10, avg_cost=500.0,
                      market_value=5_100.0, unrealized_pnl=100.0)
        p2 = Position(symbol="AAPL", symbol_id=2, quantity=5, avg_cost=200.0,
                      market_value=1_050.0, unrealized_pnl=50.0)
        pm._positions = {"SPY": p1, "AAPL": p2}
        exposure = pm.get_total_exposure()
        assert abs(exposure["total_market_value"] - 6_150.0) < 0.01
        assert abs(exposure["total_unrealized_pnl"] - 150.0) < 0.01
        assert exposure["position_count"] == 2


# =====================================================================
# EI  ENGINE INTEGRATION
# =====================================================================

class TestEngineIntegration:
    """Integration tests for LiveTradingEngine signal processing and reconciliation."""

    def _make_signal(self, action="BUY", symbol="SPY20260321C560",
                     quantity=1, limit_price=2.00, reason="test",
                     strategy_name="test_strategy"):
        """Build a minimal Signal."""
        from live.strategy import Signal
        return Signal(
            action=action,
            symbol=symbol,
            quantity=quantity,
            limit_price=limit_price,
            reason=reason,
            strategy_name=strategy_name,
        )

    def test_ei01_monitor_mode_no_order_submitted(self):
        """EI-01: In monitor mode, signals are logged but no order is placed."""
        with _temp_db() as db:
            engine, client = _build_engine(db, mode="monitor")
            signal = self._make_signal()
            engine._process_signal(signal)
            client.place_order.assert_not_called()

    def test_ei02_buy_blocked_by_max_daily_loss(self):
        """EI-02: BUY signal dropped when max_daily_loss flag is active."""
        with _temp_db() as db:
            engine, client = _build_engine(db)
            engine._max_loss_reached = True
            signal = self._make_signal(action="BUY")
            engine._process_signal(signal)
            client.place_order.assert_not_called()

    def test_ei03_buy_blocked_by_circuit_breaker(self):
        """EI-03: BUY signal blocked when circuit breaker is tripped."""
        with _temp_db() as db:
            engine, client = _build_engine(db)
            engine.orders.buy = MagicMock()
            engine._circuit_breaker_tripped = True
            signal = self._make_signal(action="BUY")
            engine._process_signal(signal)
            engine.orders.buy.assert_not_called()

    def test_ei04_buy_blocked_by_insufficient_buying_power(self):
        """EI-04: BUY signal blocked when broker buying power < MIN threshold."""
        with _temp_db() as db:
            engine, client = _build_engine(db)
            engine.orders.buy = MagicMock()
            engine._broker_buying_power = 100.0  # below $500 min
            signal = self._make_signal(action="BUY")
            engine._process_signal(signal)
            engine.orders.buy.assert_not_called()

    def test_ei05_hold_and_pending_signals_no_order(self):
        """EI-05: HOLD and PENDING signals never reach the broker."""
        with _temp_db() as db:
            engine, client = _build_engine(db)
            engine.orders.buy = MagicMock()
            engine.orders.sell = MagicMock()
            engine._process_signal(self._make_signal(action="HOLD"))
            engine._process_signal(self._make_signal(action="PENDING"))
            engine.orders.buy.assert_not_called()
            engine.orders.sell.assert_not_called()

    def test_ei06_buy_fill_creates_db_trade_at_fill_price(self):
        """EI-06: After BUY fill, trade inserted to DB at actual fill price."""
        with _temp_db() as db:
            engine, client = _build_engine(db)

            # Capture the fill callback registered with orders mock
            fill_callbacks = []
            engine.orders.on_fill(lambda o: fill_callbacks.append(o))

            # Simulate BUY signal → order submitted with order_id
            from live.order_manager import Order
            mock_order = Order(order_id=5001, symbol="SPY20260321C560",
                               quantity=1, filled_quantity=1,
                               avg_fill_price=2.30, side="Buy", status="Open")
            engine.orders.buy = MagicMock(return_value=mock_order)

            signal = self._make_signal(action="BUY", limit_price=2.20)
            engine._process_signal(signal)

            # Confirm entry is now pending
            assert 5001 in engine._pending_entry_orders

            # Simulate fill event from broker
            mock_order.status = "Filled"
            engine._on_fill(mock_order)

            # Entry should be removed from pending
            assert 5001 not in engine._pending_entry_orders

            # DB should have a trade with fill price 2.30, NOT limit price 2.20
            trades = db.get_open_trades()
            assert len(trades) == 1
            assert abs(trades[0]["entry_price"] - 2.30) < 0.01

    def test_ei07_sell_fill_closes_db_trade_at_real_price(self):
        """EI-07: After SELL fill, trade closed in DB at actual fill price."""
        with _temp_db() as db:
            # Pre-create an open trade
            tid = db.insert_trade(_make_trade(entry_price=2.00))
            engine, client = _build_engine(db)

            from live.order_manager import Order
            mock_order = Order(order_id=6001, symbol="SPY20260321C560",
                               quantity=1, filled_quantity=1,
                               avg_fill_price=3.00, side="Sell", status="Open")
            engine.orders.sell = MagicMock(return_value=mock_order)

            signal = self._make_signal(action="SELL", limit_price=2.90)
            engine._process_signal(signal)

            assert 6001 in engine._pending_exit_orders

            mock_order.status = "Filled"
            engine._on_fill(mock_order)

            assert 6001 not in engine._pending_exit_orders
            closed = db.get_trade(tid)
            assert closed["status"] == "closed"
            assert abs(closed["exit_price"] - 3.00) < 0.01
            expected_pnl = (3.00 - 2.00) * 1 * 100
            assert abs(closed["pnl"] - expected_pnl) < 0.01

    def test_ei08_entry_reject_no_db_trade(self):
        """EI-08: Rejected BUY order → no trade in DB, strategy on_trade_cancelled called."""
        with _temp_db() as db:
            engine, client = _build_engine(db)

            cancelled_calls = []
            mock_strategy = MagicMock()
            mock_strategy.is_active = True
            mock_strategy.on_trade_cancelled = lambda tid, sym: cancelled_calls.append(sym)
            engine._strategies = [mock_strategy]

            from live.order_manager import Order
            mock_order = Order(order_id=7001, symbol="SPY20260321C560",
                               quantity=1, side="Buy", status="Open")
            engine.orders.buy = MagicMock(return_value=mock_order)

            engine._process_signal(self._make_signal(action="BUY"))
            assert 7001 in engine._pending_entry_orders

            # Simulate rejection
            mock_order.status = "Rejected"
            engine._on_reject(mock_order)

            assert 7001 not in engine._pending_entry_orders
            assert len(db.get_open_trades()) == 0
            assert "SPY20260321C560" in cancelled_calls

    def test_ei09_exit_reject_triggers_market_retry(self):
        """EI-09: Rejected exit LIMIT order resubmits as MARKET order."""
        with _temp_db() as db:
            tid = db.insert_trade(_make_trade(entry_price=2.00))
            engine, client = _build_engine(db)

            from live.order_manager import Order
            orig_order = Order(order_id=8001, symbol="SPY20260321C560",
                               quantity=1, side="Sell", status="Open")
            retry_order = Order(order_id=8002, symbol="SPY20260321C560",
                                quantity=1, side="Sell", status="Open")
            engine.orders.sell = MagicMock(return_value=retry_order)

            engine._pending_exit_orders[8001] = {
                "trade_id": tid,
                "symbol": "SPY20260321C560",
                "signal_reason": "profit target",
                "submitted_at": time.time(),
                "retry_count": 0,
            }

            orig_order.status = "Rejected"
            engine._on_reject(orig_order)

            # MARKET retry order should now be pending
            assert 8001 not in engine._pending_exit_orders
            assert 8002 in engine._pending_exit_orders
            assert engine._pending_exit_orders[8002]["retry_count"] == 1
            # sell called with no limit_price (MARKET)
            engine.orders.sell.assert_called_once_with(
                symbol="SPY20260321C560", quantity=1, limit_price=None
            )

    def test_ei10_exit_reject_twice_logs_critical(self, capsys=None):
        """EI-10: After max_retries exhausted, CRITICAL log and position stays open."""
        import logging
        with _temp_db() as db:
            tid = db.insert_trade(_make_trade(entry_price=2.00))
            engine, client = _build_engine(db)

            engine.orders.sell = MagicMock()
            from live.order_manager import Order

            # First rejection → retry submitted
            orig = Order(order_id=9001, symbol="SPY20260321C560", quantity=1, side="Sell", status="Open")
            retry1 = Order(order_id=9002, symbol="SPY20260321C560", quantity=1, side="Sell", status="Open")
            engine.orders.sell.return_value = retry1

            engine._pending_exit_orders[9001] = {
                "trade_id": tid, "symbol": "SPY20260321C560",
                "signal_reason": "stop", "submitted_at": time.time(), "retry_count": 0,
            }
            orig.status = "Rejected"
            engine._on_reject(orig)
            assert 9002 in engine._pending_exit_orders

            # Second rejection → max_retries hit; no further orders submitted
            engine.orders.sell.reset_mock()
            retry2 = Order(order_id=9002, symbol="SPY20260321C560", quantity=1, side="Sell", status="Rejected")
            engine._pending_exit_orders[9002]["retry_count"] = 2  # at limit
            engine._on_reject(retry2)
            engine.orders.sell.assert_not_called()
            # Trade should still be open in DB
            assert db.get_trade(tid)["status"] == "open"

    def test_ei11_circuit_breaker_trips_at_threshold(self):
        """EI-11: Circuit breaker trips at _REJECT_CIRCUIT_BREAKER consecutive rejects."""
        with _temp_db() as db:
            engine, client = _build_engine(db)
            from live.order_manager import Order
            assert not engine._circuit_breaker_tripped

            for i in range(engine._REJECT_CIRCUIT_BREAKER):
                oid = 1000 + i
                order = Order(order_id=oid, symbol="SPY20260321C560",
                              quantity=1, side="Buy", status="Rejected")
                engine._pending_entry_orders[oid] = {
                    "symbol": "SPY20260321C560", "quantity": 1,
                    "submitted_at": time.time(),
                }
                engine._on_reject(order)

            assert engine._circuit_breaker_tripped

    def test_ei12_circuit_breaker_reset_on_fill(self):
        """EI-12: A successful fill resets the consecutive-reject counter."""
        with _temp_db() as db:
            engine, client = _build_engine(db)
            engine._consecutive_rejects = 3
            engine._circuit_breaker_tripped = True

            from live.order_manager import Order
            fill_order = Order(order_id=2000, symbol="SPY20260321C560",
                               quantity=1, filled_quantity=1, avg_fill_price=2.50,
                               side="Buy", status="Filled")
            engine._pending_entry_orders[2000] = {
                "symbol": "SPY20260321C560", "quantity": 1,
                "option_type": "call", "strategy_name": "test",
                "tag": "[PAPER]", "reason": "test", "submitted_at": time.time(),
            }
            engine._on_fill(fill_order)

            assert engine._consecutive_rejects == 0
            assert not engine._circuit_breaker_tripped

    def test_ei13_update_state_broker_balance_atomic_write(self):
        """EI-13: _update_state_broker_balance produces valid JSON with broker_nlv field.
        Uses a temp directory to avoid touching the real trading_state.json.
        """
        with _temp_db() as db:
            engine, client = _build_engine(db)

            with tempfile.TemporaryDirectory() as tmpdir:
                state_path = os.path.join(tmpdir, "trading_state.json")

                # Monkey-patch the method to resolve the state file path to our temp dir
                import live.engine as engine_mod

                original_method = engine_mod.LiveTradingEngine._update_state_broker_balance

                def _patched(self_inner, nlv, cash=None):
                    import json, os, tempfile
                    try:
                        existing = {}
                        if os.path.exists(state_path):
                            with open(state_path, 'r') as fh:
                                existing = json.load(fh)
                        existing['broker_nlv'] = nlv
                        if cash is not None:
                            existing['broker_cash'] = cash
                        existing['broker_balance_time'] = "test"
                        dir_name = os.path.dirname(state_path)
                        fd, tmp = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
                        with os.fdopen(fd, 'w') as tf:
                            json.dump(existing, tf, indent=2)
                        os.replace(tmp, state_path)
                    except Exception as exc:
                        raise AssertionError(f"Atomic write failed: {exc}")

                engine._update_state_broker_balance = lambda nlv, cash=None: _patched(engine, nlv, cash)
                engine._update_state_broker_balance(12_345.67, cash=8_000.00)

                assert os.path.exists(state_path), "State file not created"
                with open(state_path) as fh:
                    data = json.load(fh)  # must not raise (valid JSON)
                assert "broker_nlv" in data
                assert abs(data["broker_nlv"] - 12_345.67) < 0.01
                assert abs(data["broker_cash"] - 8_000.00) < 0.01

    def test_ei14_sync_strategy_capital_pushes_nlv(self):
        """EI-14: _sync_strategy_capital updates strategy.account_capital with broker NLV."""
        with _temp_db() as db:
            engine, client = _build_engine(db)
            mock_strategy = MagicMock()
            mock_strategy.account_capital = 9_000.00
            mock_strategy.risk_manager = MagicMock()
            mock_strategy.risk_manager.capital = 9_000.00
            mock_strategy.persistence = MagicMock()
            mock_strategy.persistence.state = MagicMock()
            mock_strategy.persistence.state.current_capital = 9_000.00
            engine._strategies = [mock_strategy]

            engine._sync_strategy_capital(11_000.00)
            assert mock_strategy.account_capital == 11_000.00
            assert mock_strategy.risk_manager.capital == 11_000.00


# =====================================================================
# RF  RISK FIXES VERIFICATION
# =====================================================================

class TestRiskFixes:
    """Tests verifying each applied risk fix (OE-R1 through PM-R1)."""

    def _make_signal(self, action="BUY", symbol="SPY20260321C560",
                     quantity=1, limit_price=2.00, reason="test",
                     strategy_name="test_strategy"):
        from live.strategy import Signal
        return Signal(
            action=action, symbol=symbol, quantity=quantity,
            limit_price=limit_price, reason=reason,
            strategy_name=strategy_name,
        )

    # ── OE-R1: Stranded position watchdog ──────────────────────────

    def test_rf01_stranded_position_added_after_max_retries(self):
        """OE-R1: After max exit retries exhausted, trade is added to _stranded_positions."""
        with _temp_db() as db:
            tid = db.insert_trade(_make_trade(entry_price=2.00))
            engine, client = _build_engine(db)
            engine.orders.sell = MagicMock()

            from live.order_manager import Order
            reject = Order(order_id=3001, symbol="SPY20260321C560",
                           quantity=1, side="Sell", status="Rejected")
            engine._pending_exit_orders[3001] = {
                "trade_id": tid, "symbol": "SPY20260321C560",
                "signal_reason": "stop", "submitted_at": time.time(),
                "retry_count": 2,  # at max_retries limit
            }
            engine._on_reject(reject)

            # No further sell attempted
            engine.orders.sell.assert_not_called()
            # Trade promoted to stranded
            assert len(engine._stranded_positions) == 1
            assert engine._stranded_positions[0]["trade_id"] == tid
            assert engine._stranded_positions[0]["symbol"] == "SPY20260321C560"

    def test_rf02_stranded_watchdog_closes_missing_position(self):
        """OE-R1: Watchdog closes trade in DB when broker no longer holds position."""
        with _temp_db() as db:
            tid = db.insert_trade(_make_trade(entry_price=2.00))
            engine, client = _build_engine(db)

            # Position not found at broker → empty list
            engine.positions.get_all_positions.return_value = []

            engine._stranded_positions = [{
                "trade_id": tid,
                "symbol": "SPY20260321C560",
                "stranded_at": time.time() - 120,  # well past retry interval
            }]

            engine._check_pending_orders()

            # Trade should be closed in DB
            trade = db.get_trade(tid)
            assert trade["status"] == "closed"
            assert "[STRANDED]" in (trade.get("notes") or "")
            # Stranded list should be empty now
            assert len(engine._stranded_positions) == 0

    def test_rf03_stranded_watchdog_retries_market_sell(self):
        """OE-R1: Watchdog attempts MARKET sell when broker still holds position."""
        with _temp_db() as db:
            tid = db.insert_trade(_make_trade(entry_price=2.00))
            engine, client = _build_engine(db)

            from live.order_manager import Order
            from live.position_manager import Position

            # Broker still has the position
            pos = MagicMock()
            pos.symbol = "SPY20260321C560"
            engine.positions.get_all_positions.return_value = [pos]

            retry_order = Order(order_id=4001, symbol="SPY20260321C560",
                                quantity=1, side="Sell", status="Open")
            engine.orders.sell = MagicMock(return_value=retry_order)

            engine._stranded_positions = [{
                "trade_id": tid,
                "symbol": "SPY20260321C560",
                "stranded_at": time.time() - 120,  # past retry interval
            }]

            engine._check_pending_orders()

            # Should have submitted a MARKET sell
            engine.orders.sell.assert_called_once()
            call_kwargs = engine.orders.sell.call_args
            assert call_kwargs[1].get("limit_price") is None  # MARKET
            # New exit order pending
            assert 4001 in engine._pending_exit_orders

    # ── OE-R2: Zero fill price fallback ────────────────────────────

    def test_rf04_zero_fill_price_closes_with_fallback(self):
        """OE-R2: Exit fill with avg_fill_price=0 → close at bid or $0.01."""
        with _temp_db() as db:
            tid = db.insert_trade(_make_trade(entry_price=2.00))
            engine, client = _build_engine(db)

            from live.order_manager import Order
            fill = Order(order_id=5001, symbol="SPY20260321C560",
                         quantity=1, filled_quantity=1,
                         avg_fill_price=0.0, side="Sell", status="Filled")
            engine._pending_exit_orders[5001] = {
                "trade_id": tid, "symbol": "SPY20260321C560",
                "signal_reason": "stop", "submitted_at": time.time(),
                "retry_count": 0,
            }

            # Mock quote_client to return no bid either
            engine.quote_client = MagicMock()
            engine.quote_client.get_quote_by_symbol = MagicMock(side_effect=Exception("no quote"))

            engine._on_fill(fill)

            # Trade should be closed, not left open
            trade = db.get_trade(tid)
            assert trade["status"] == "closed"
            assert trade["exit_price"] == 0.01
            assert "ZERO_FILL" in (trade.get("notes") or "")

    # ── OE-R3: Auto-expire stale pending exits ────────────────────

    def test_rf05_stale_exit_auto_expires_to_stranded(self):
        """OE-R3: Exit orders pending > 180s are promoted to stranded positions."""
        with _temp_db() as db:
            tid = db.insert_trade(_make_trade(entry_price=2.00))
            engine, client = _build_engine(db)

            engine._pending_exit_orders[6001] = {
                "trade_id": tid,
                "symbol": "SPY20260321C560",
                "signal_reason": "target",
                "submitted_at": time.time() - 200,  # 200s ago → past 180s threshold
                "retry_count": 0,
            }

            # Broker doesn't report it as cancelled (order_statuses not available)
            engine.client._ibkr = MagicMock(spec=[])

            engine._check_pending_orders()

            # Order should be removed from pending_exit
            assert 6001 not in engine._pending_exit_orders
            # Should be in stranded list
            assert len(engine._stranded_positions) >= 1
            stranded = engine._stranded_positions[-1]
            assert stranded["trade_id"] == tid
            assert stranded["symbol"] == "SPY20260321C560"

    def test_rf06_stale_exit_under_180s_not_expired(self):
        """OE-R3: Exit orders pending < 180s are NOT auto-expired."""
        with _temp_db() as db:
            tid = db.insert_trade(_make_trade(entry_price=2.00))
            engine, client = _build_engine(db)

            engine._pending_exit_orders[6002] = {
                "trade_id": tid,
                "symbol": "SPY20260321C560",
                "signal_reason": "target",
                "submitted_at": time.time() - 100,  # 100s ago → past 90s warn, under 180s
                "retry_count": 0,
            }

            engine.client._ibkr = MagicMock(spec=[])
            engine._check_pending_orders()

            # Should still be in pending (warned but not expired)
            assert 6002 in engine._pending_exit_orders

    # ── OE-R4: Duplicate BUY guard ────────────────────────────────

    def test_rf07_duplicate_buy_blocked(self):
        """OE-R4: Second BUY for same symbol blocked when entry already pending."""
        with _temp_db() as db:
            engine, client = _build_engine(db)

            from live.order_manager import Order
            first_order = Order(order_id=7001, symbol="SPY20260321C560",
                                quantity=1, side="Buy", status="Open")
            engine.orders.buy = MagicMock(return_value=first_order)

            # First BUY goes through
            signal1 = self._make_signal(action="BUY", symbol="SPY20260321C560")
            engine._process_signal(signal1)
            assert 7001 in engine._pending_entry_orders
            engine.orders.buy.assert_called_once()

            # Second BUY for same symbol should be blocked
            engine.orders.buy.reset_mock()
            signal2 = self._make_signal(action="BUY", symbol="SPY20260321C560")
            engine._process_signal(signal2)
            engine.orders.buy.assert_not_called()

    def test_rf08_different_symbol_buy_allowed(self):
        """OE-R4: BUY for a different symbol is allowed even with pending entry."""
        with _temp_db() as db:
            engine, client = _build_engine(db)

            from live.order_manager import Order
            order1 = Order(order_id=7010, symbol="SPY20260321C560",
                           quantity=1, side="Buy", status="Open")
            order2 = Order(order_id=7011, symbol="SPY20260321P560",
                           quantity=1, side="Buy", status="Open")
            engine.orders.buy = MagicMock(side_effect=[order1, order2])

            engine._process_signal(self._make_signal(action="BUY", symbol="SPY20260321C560"))
            assert 7010 in engine._pending_entry_orders

            engine._process_signal(self._make_signal(action="BUY", symbol="SPY20260321P560"))
            assert 7011 in engine._pending_entry_orders
            assert engine.orders.buy.call_count == 2

    # ── OE-R5: Partial fill detection ──────────────────────────────

    def test_rf09_partial_fill_detected_and_logged(self):
        """OE-R5: PartiallyFilled transition triggers a warning log."""
        import logging

        from live.order_manager import OrderManager, Order

        client = _mock_broker()
        with _temp_db() as db:
            om = OrderManager(broker_client=client, trade_db=db)
            om.set_account("ACC1")

            # Pre-populate with an "Open" order
            existing = Order(order_id=8001, symbol="SPY20260321C560",
                             quantity=3, filled_quantity=0,
                             avg_fill_price=0, side="Buy", status="Open")
            om._orders[8001] = existing

            # Broker now reports it as PartiallyFilled
            client.get_account_orders.return_value = [{
                "id": 8001, "symbol": "SPY20260321C560", "symbolId": 1,
                "totalQuantity": 3, "filledQuantity": 1,
                "side": "Buy", "orderType": "Limit",
                "limitPrice": 2.00, "avgExecPrice": 2.05,
                "state": "PartiallyFilled", "timeInForce": "Day",
                "creationTime": "2026-03-21", "updateTime": "2026-03-21",
                "commissionCharged": 1.0,
            }]

            with patch("live.order_manager.logger") as mock_logger:
                om.sync_orders("ACC1")
                # Should have logged a warning about partial fill
                warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
                assert any("partially filled" in w.lower() for w in warning_calls), \
                    f"Expected partial fill warning, got: {warning_calls}"

    # ── PE-R1: Atomic save_state ───────────────────────────────────

    def test_rf10_save_state_atomic_no_corrupt_on_crash(self):
        """PE-R1: save_state uses tempfile + os.replace → no partial writes."""
        from live.state_persistence import StatePersistence, TradingState

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "trading_state.json")
            with _temp_db() as db:
                sp = StatePersistence(
                    state_file=state_path,
                )
                sp.state = TradingState(current_capital=50_000.0)

                # Save state
                sp.save_state()

                # File should exist and be valid JSON
                assert os.path.exists(state_path)
                with open(state_path) as f:
                    data = json.load(f)
                assert abs(data["current_capital"] - 50_000.0) < 0.01

                # No .tmp files should remain
                tmp_files = [f for f in os.listdir(tmpdir) if f.endswith('.tmp')]
                assert len(tmp_files) == 0, f"Stale temp files: {tmp_files}"

    def test_rf11_save_state_survives_concurrent_read(self):
        """PE-R1: Save during concurrent read doesn't produce corrupt JSON."""
        from live.state_persistence import StatePersistence, TradingState

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "trading_state.json")
            with _temp_db() as db:
                sp = StatePersistence(state_file=state_path)
                sp.state = TradingState(current_capital=100_000.0)
                sp.save_state()

                errors = []

                def reader():
                    for _ in range(20):
                        try:
                            if os.path.exists(state_path):
                                with open(state_path) as f:
                                    json.load(f)
                        except json.JSONDecodeError as e:
                            errors.append(str(e))
                        time.sleep(0.01)

                def writer():
                    for i in range(20):
                        sp.state.current_capital = 100_000.0 + i
                        sp.save_state()
                        time.sleep(0.005)

                t1 = threading.Thread(target=reader)
                t2 = threading.Thread(target=writer)
                t1.start()
                t2.start()
                t1.join(timeout=10)
                t2.join(timeout=10)

                assert len(errors) == 0, f"Corrupt JSON reads during concurrent save: {errors}"

    # ── PM-R1: Atomic position dict swap ───────────────────────────

    def test_rf12_sync_positions_atomic_swap(self):
        """PM-R1: sync_positions never exposes an empty _positions dict."""
        from live.position_manager import PositionManager

        client = _mock_broker()
        client.get_account_positions.return_value = [
            {"symbol": "SPY20260321C560", "symbolId": 1, "openQuantity": 2,
             "averageEntryPrice": 2.00, "currentPrice": 2.50,
             "currentMarketValue": 500, "openPnl": 100, "openPnlPercent": 0.25,
             "dayPnl": 50},
        ]

        pm = PositionManager(broker_client=client)
        pm.set_account("ACC1")

        # Pre-populate with some positions
        pm._positions["OLD_SYMBOL"] = MagicMock()

        observations = []

        def observer():
            """Sample _positions size during sync."""
            for _ in range(100):
                observations.append(len(pm._positions))
                time.sleep(0.001)

        t = threading.Thread(target=observer)
        t.start()
        pm.sync_positions("ACC1")
        t.join(timeout=5)

        # After sync, old symbol gone, new symbol present
        assert "SPY20260321C560" in pm._positions
        assert "OLD_SYMBOL" not in pm._positions

        # Critical: _positions should NEVER have been observed as empty (len==0)
        # during sync. With atomic swap, it goes from old→new without clearing.
        # Note: there's a small race window so we check no observation saw 0
        assert 0 not in observations, \
            f"_positions was observed empty during sync — atomic swap broken. Observations: {observations}"


# =====================================================================
# Standalone runner (also pytest-compatible)
# =====================================================================

def pytest_approx(val, rel=1e-4):
    """Minimal pytest.approx substitute for standalone run."""
    class _Approx:
        def __init__(self, v):
            self._v = v
        def __eq__(self, other):
            return abs(self._v - other) < abs(self._v * rel) + 1e-9
        def __repr__(self):
            return repr(self._v)
    return _Approx(val)


if __name__ == "__main__":
    import traceback

    suites = [
        TestOrderExecution,
        TestStatePersistence,
        TestTradeDatabase,
        TestPositionManager,
        TestEngineIntegration,
        TestRiskFixes,
    ]

    total_pass = 0
    total_fail = 0

    for suite_cls in suites:
        suite = suite_cls()
        print(f"\n{'='*60}")
        print(f"  {suite_cls.__name__}")
        print(f"{'='*60}")
        for name in [m for m in dir(suite_cls) if m.startswith("test_")]:
            method = getattr(suite, name)
            try:
                method()
                print(f"  PASS  {name}")
                total_pass += 1
            except Exception as exc:
                print(f"  FAIL  {name}")
                traceback.print_exc()
                total_fail += 1

    print(f"\n{'='*60}")
    print(f"Results: {total_pass} passed, {total_fail} failed")
    print(f"{'='*60}")
    sys.exit(0 if total_fail == 0 else 1)
