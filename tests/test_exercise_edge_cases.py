"""
Exercise & EOD Edge-Case Test Suite
====================================
Tests every fix from the Mar 27 root cause analysis:

  Bug 1 (max_hold): MAX HOLD was under `elif` smart_exit — dead code when smart exit enabled
  Bug 2 (SFL gate): Stop-After-First-Loss only checked in on_quote signal generation,
                     not in on_option_quote entry path — pending direction bypassed SFL
  Bug 3 (time exit): Exit signals required option quotes; no quote = no exit
  Bug 4 (EOD LIMIT): EOD forced liquidation used entry_price * 0.01 — wrong for ITM options
  Bug 5 (exercise):  Stock positions from option exercise invisible to reconciliation

Run:
    python -m pytest tests/test_exercise_edge_cases.py -v --tb=short
    python tests/test_exercise_edge_cases.py
"""
import os
import re
import sqlite3
import sys
import tempfile
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, PropertyMock, call

# ── project root on PYTHONPATH ────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from live.strategy import OptionQuote, Signal
from live.strategy_0dte import Live0DTEStrategy, TradeState, DayState, get_eastern_time
from live.trade_database import TradeDatabase, Trade


# =====================================================================
#  HELPERS
# =====================================================================

def _make_db(path=None):
    """Create a fresh TradeDatabase in a temp directory."""
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
    return TradeDatabase(path), path


def _make_strategy(**overrides):
    """Build a Live0DTEStrategy with test-friendly defaults.

    Passes explicit params so tests don't drift when strategy.json changes.
    """
    defaults = dict(
        strategy="momentum",
        profit_target_pct=0.50,
        stop_loss_pct=0.35,
        min_option_price=0.50,
        max_option_price=2.00,
        trade_start_hour=10,
        trade_start_minute=0,
        trade_end_hour=11,
        trade_end_minute=0,
        exit_hour=15,
        max_hold_minutes=80,
        max_contracts=50,
        max_daily_losses=1,  # SFL = 1
        account_capital=10000.0,
        risk_per_trade_pct=0.05,
    )
    defaults.update(overrides)
    s = Live0DTEStrategy(**defaults)
    # Disable regime detection for clean tests
    s._use_regime_detection = False
    return s


def _make_option_quote(symbol="SPY20260327P639", option_type="put", bid=1.50,
                        ask=1.60, last=1.55, delta=-0.45, underlying_price=640.0,
                        strike=639.0, expiration=None):
    """Build an OptionQuote for tests."""
    if expiration is None:
        expiration = get_eastern_time().strftime("%Y-%m-%d")
    return OptionQuote(
        symbol=symbol,
        underlying="SPY",
        underlying_price=underlying_price,
        strike=strike,
        expiration=expiration,
        option_type=option_type,
        bid=bid,
        ask=ask,
        last=last,
        volume=1000,
        open_interest=5000,
        delta=delta,
        gamma=0.05,
        theta=-0.10,
        vega=0.08,
        iv=0.25,
        timestamp=get_eastern_time().isoformat(),
    )


def _cleanup_engine(db, db_path):
    """Close DB and remove temp file (Windows needs conn closed first)."""
    try:
        db.conn.close()
    except Exception:
        pass
    try:
        os.unlink(db_path)
    except OSError:
        pass


def _mock_engine(option_underlyings=None, mode="paper"):
    """Build a minimal LiveTradingEngine with mocked broker components.

    Bypasses the complex parts of the constructor (Flex client, threading)
    while keeping the real reconciliation / EOD / time-exit methods.
    """
    from live.engine import LiveTradingEngine, EngineConfig

    client = MagicMock()
    client.get_account_positions.return_value = []
    client.get_executions.return_value = []
    client.get_quote_by_symbol.return_value = None

    db, db_path = _make_db()

    config = EngineConfig(
        account_id="DU12345",
        symbols=["SPY"],
        option_underlyings=option_underlyings or ["SPY"],
        mode=mode,
    )

    position_mgr = MagicMock()
    order_mgr = MagicMock()
    # order_mgr.sell / buy return a mock Order with order_id
    mock_order = MagicMock()
    mock_order.order_id = 999
    order_mgr.sell.return_value = mock_order
    order_mgr.buy.return_value = mock_order

    with patch.dict(os.environ, {"IBKR_FLEX_TOKEN": "", "IBKR_FLEX_QUERY_ID": ""}):
        engine = LiveTradingEngine(
            client=client,
            trade_db=db,
            position_manager=position_mgr,
            order_manager=order_mgr,
            config=config,
            quote_client=client,
        )

    engine._db_path = db_path  # for cleanup
    return engine, db, db_path


# =====================================================================
#  BUG 1: MAX HOLD FIRES EVEN WHEN SMART EXIT IS ENABLED
# =====================================================================

class TestMaxHoldNotDeadCode:
    """MAX HOLD must fire independently of the smart exit elif chain.

    Before the fix, max hold was ``elif self.trade_state.entry_time:``
    in the PT → SL → smart → max_hold chain.  When smart_exit was
    evaluated (even without producing a reason) the elif was dead code.
    After the fix, max hold is ``if not exit_reason and ...`` — a
    standalone block.
    """

    def test_max_hold_fires_with_smart_exit_enabled(self):
        """Position held past max_hold should exit even if smart exit is active
        and produces no exit_reason."""
        s = _make_strategy(max_hold_minutes=80)
        s._smart_exit_enabled = True

        now = get_eastern_time()
        entry_time = (now - timedelta(minutes=90)).isoformat()
        s.trade_state = TradeState(
            in_trade=True,
            symbol="SPY20260327P639",
            direction="PUT",
            entry_price=1.57,
            entry_time=entry_time,
            quantity=32,
            highest_price=1.57,
            lowest_price=1.40,
        )
        s.day_state = DayState(date=now.strftime("%Y-%m-%d"))
        # Push EOD exit far into the future so only max hold can fire
        s.exit_time = dt_time(23, 59)

        # PnL = (1.40 - 1.57) / 1.57 = -10.8% — won't trigger PT or SL (-35%)
        quote = _make_option_quote(
            symbol="SPY20260327P639",
            bid=1.35, ask=1.45, last=1.40,
            delta=-0.45, strike=639.0,
        )

        signal = s._check_exit(quote, now)
        assert signal is not None, "MAX HOLD should have triggered an exit signal"
        assert signal.action == "SELL"
        assert "MAX HOLD" in signal.reason

    def test_max_hold_does_not_fire_within_limit(self):
        """Position within max_hold should NOT be force-closed."""
        s = _make_strategy(max_hold_minutes=80)
        s._smart_exit_enabled = True

        now = get_eastern_time()
        entry_time = (now - timedelta(minutes=30)).isoformat()
        s.trade_state = TradeState(
            in_trade=True,
            symbol="SPY20260327P639",
            direction="PUT",
            entry_price=1.57,
            entry_time=entry_time,
            quantity=32,
            highest_price=1.57,
            lowest_price=1.40,
        )
        s.day_state = DayState(date=now.strftime("%Y-%m-%d"))
        # Push EOD exit far into the future so it doesn't interfere
        s.exit_time = dt_time(23, 59)

        quote = _make_option_quote(
            symbol="SPY20260327P639",
            bid=1.35, ask=1.45, last=1.40,
        )

        signal = s._check_exit(quote, now)
        assert signal is None, "Should NOT exit within max hold time"

    def test_profit_target_takes_priority_over_max_hold(self):
        """Profit target fires before max hold check."""
        s = _make_strategy(max_hold_minutes=80, profit_target_pct=0.50)
        s._smart_exit_enabled = True

        now = get_eastern_time()
        entry_time = (now - timedelta(minutes=90)).isoformat()
        s.trade_state = TradeState(
            in_trade=True,
            symbol="SPY20260327P639",
            direction="PUT",
            entry_price=1.00,
            entry_time=entry_time,
            quantity=32,
            highest_price=1.60,
            lowest_price=1.00,
        )
        s.day_state = DayState(date=now.strftime("%Y-%m-%d"))
        s.exit_time = dt_time(23, 59)

        # PnL = (1.60 - 1.00) / 1.00 = +60% > 50% PT
        quote = _make_option_quote(
            symbol="SPY20260327P639",
            bid=1.55, ask=1.65, last=1.60,
        )

        signal = s._check_exit(quote, now)
        assert signal is not None
        assert "PROFIT TARGET" in signal.reason

    def test_stop_loss_takes_priority_over_max_hold(self):
        """Stop loss fires before max hold check."""
        s = _make_strategy(max_hold_minutes=80, stop_loss_pct=0.35)
        s._smart_exit_enabled = True

        now = get_eastern_time()
        entry_time = (now - timedelta(minutes=90)).isoformat()
        s.trade_state = TradeState(
            in_trade=True,
            symbol="SPY20260327P639",
            direction="PUT",
            entry_price=1.57,
            entry_time=entry_time,
            quantity=32,
            highest_price=1.57,
            lowest_price=0.80,
        )
        s.day_state = DayState(date=now.strftime("%Y-%m-%d"))
        s.exit_time = dt_time(23, 59)

        # PnL = (0.80 - 1.57) / 1.57 = -49% > -35% SL
        quote = _make_option_quote(
            symbol="SPY20260327P639",
            bid=0.75, ask=0.85, last=0.80,
        )

        signal = s._check_exit(quote, now)
        assert signal is not None
        assert "STOP LOSS" in signal.reason


# =====================================================================
#  BUG 2: SFL HARD GATE IN on_option_quote
# =====================================================================

class TestSFLHardGate:
    """Stop After First Loss must block entries in on_option_quote,
    even if a pending direction was already set before the loss."""

    def test_sfl_blocks_pending_entry_after_loss(self):
        """After first loss, pending direction should be cleared and entry blocked."""
        s = _make_strategy(max_daily_losses=1)
        s._smart_exit_enabled = False

        now = get_eastern_time()
        today = now.strftime("%Y-%m-%d")
        s.day_state = DayState(
            date=today,
            losses_today=1,  # Already had one loss
        )
        s.trade_state = TradeState()  # Not in trade

        # Set a pending direction (as if signal was generated before the loss)
        s._pending_direction = "PUT"
        s._pending_direction_time = now

        quote = _make_option_quote(
            symbol="SPY20260327P639",
            option_type="put",
            bid=1.50, ask=1.60, last=1.55,
            delta=-0.45,
        )

        signal = s.on_option_quote(quote)
        assert signal is None, "SFL should block entry after first loss"
        assert s._pending_direction is None, "Pending direction should be cleared by SFL gate"

    def test_sfl_allows_entry_before_loss(self):
        """Before any loss, pending direction should work normally."""
        s = _make_strategy(max_daily_losses=1)
        s._smart_exit_enabled = False

        now = get_eastern_time()
        today = now.strftime("%Y-%m-%d")
        s.day_state = DayState(date=today, losses_today=0)
        s.trade_state = TradeState()

        # ORB must be calculated
        s.orb_state.orb_calculated = True
        s.orb_state.orb_high = 645.0
        s.orb_state.orb_low = 635.0

        s._pending_direction = "PUT"
        s._pending_direction_time = now
        s._pending_expiry_seconds = 300

        quote = _make_option_quote(
            symbol="SPY20260327P639",
            option_type="put",
            bid=1.50, ask=1.60, last=1.55,
            delta=-0.45,
            strike=639.0,
        )

        signal = s.on_option_quote(quote)
        assert signal is not None, "Entry should be allowed before any loss"
        assert signal.action == "BUY"

    def test_sfl_blocks_with_max_daily_losses_2(self):
        """With max_daily_losses=2, entry is blocked after 2 losses."""
        s = _make_strategy(max_daily_losses=2)

        now = get_eastern_time()
        today = now.strftime("%Y-%m-%d")
        s.day_state = DayState(date=today, losses_today=2)
        s.trade_state = TradeState()

        s._pending_direction = "CALL"
        s._pending_direction_time = now

        quote = _make_option_quote(
            symbol="SPY20260327C645",
            option_type="call",
            delta=0.50,
        )

        signal = s.on_option_quote(quote)
        assert signal is None

    def test_sfl_still_allows_exit_when_in_trade(self):
        """SFL should not block exit checks for existing positions."""
        s = _make_strategy(max_daily_losses=1)
        s._smart_exit_enabled = False

        now = get_eastern_time()
        today = now.strftime("%Y-%m-%d")
        s.day_state = DayState(date=today, losses_today=1)

        entry_time = (now - timedelta(minutes=5)).isoformat()
        s.trade_state = TradeState(
            in_trade=True,
            symbol="SPY20260327P639",
            direction="PUT",
            entry_price=1.57,
            entry_time=entry_time,
            quantity=32,
            highest_price=1.57,
            lowest_price=0.80,
        )

        # Quote that triggers stop loss: pnl = (0.80-1.57)/1.57 = -49%
        quote = _make_option_quote(
            symbol="SPY20260327P639",
            bid=0.75, ask=0.85, last=0.80,
        )

        signal = s.on_option_quote(quote)
        # Exit should happen (stop loss), SFL doesn't block exits
        assert signal is not None
        assert signal.action == "SELL"


# =====================================================================
#  BUG 3: QUOTE-INDEPENDENT TIME EXIT (engine._check_time_based_exits)
# =====================================================================

class TestQuoteIndependentTimeExit:
    """Engine must force-exit positions past max_hold or EOD exit time
    even when no option quotes are flowing."""

    def test_time_exit_fires_past_max_hold(self):
        """Position past max_hold gets MARKET exit from position loop."""
        engine, db, db_path = _mock_engine()
        try:
            strategy = _make_strategy(max_hold_minutes=80)
            now = get_eastern_time()
            entry_time = (now - timedelta(minutes=100)).isoformat()
            strategy.trade_state = TradeState(
                in_trade=True,
                symbol="SPY20260327P639",
                direction="PUT",
                entry_price=1.57,
                entry_time=entry_time,
                quantity=32,
            )
            strategy.exit_time = dt_time(15, 0)
            engine._strategies = [strategy]

            trade = Trade(
                symbol="SPY20260327P639",
                action="BUY",
                quantity=32,
                entry_price=1.57,
                entry_time=entry_time,
                status="open",
            )
            db.insert_trade(trade)

            engine._check_time_based_exits()

            engine.orders.sell.assert_called_once_with(
                symbol="SPY20260327P639",
                quantity=32,
                limit_price=None,
            )
        finally:
            _cleanup_engine(db, db_path)

    def test_time_exit_fires_past_eod(self):
        """Position past EOD exit_time gets MARKET exit."""
        engine, db, db_path = _mock_engine()
        try:
            strategy = _make_strategy()
            now = get_eastern_time()
            entry_time = (now - timedelta(minutes=30)).isoformat()
            strategy.trade_state = TradeState(
                in_trade=True,
                symbol="SPY20260327P639",
                direction="PUT",
                entry_price=1.57,
                entry_time=entry_time,
                quantity=32,
            )
            # Set exit_time to the past so it triggers
            strategy.exit_time = dt_time(0, 0)  # midnight — always past
            engine._strategies = [strategy]

            engine._check_time_based_exits()

            engine.orders.sell.assert_called_once()
        finally:
            _cleanup_engine(db, db_path)

    def test_time_exit_skips_if_pending_exit_exists(self):
        """Don't double-submit if an exit order is already pending."""
        engine, db, db_path = _mock_engine()
        try:
            strategy = _make_strategy(max_hold_minutes=80)
            now = get_eastern_time()
            entry_time = (now - timedelta(minutes=100)).isoformat()
            strategy.trade_state = TradeState(
                in_trade=True,
                symbol="SPY20260327P639",
                direction="PUT",
                entry_price=1.57,
                entry_time=entry_time,
                quantity=32,
            )
            engine._strategies = [strategy]

            # Simulate an already-pending exit
            engine._pending_exit_orders[888] = {
                "symbol": "SPY20260327P639",
                "trade_id": 1,
                "signal_reason": "previous exit",
                "submitted_at": _time.time(),
            }

            engine._check_time_based_exits()

            engine.orders.sell.assert_not_called()
        finally:
            _cleanup_engine(db, db_path)

    def test_time_exit_does_not_fire_within_limits(self):
        """Position within max_hold and before exit_time should be left alone."""
        engine, db, db_path = _mock_engine()
        try:
            strategy = _make_strategy(max_hold_minutes=80)
            now = get_eastern_time()
            entry_time = (now - timedelta(minutes=10)).isoformat()
            strategy.trade_state = TradeState(
                in_trade=True,
                symbol="SPY20260327P639",
                direction="PUT",
                entry_price=1.57,
                entry_time=entry_time,
                quantity=32,
            )
            strategy.exit_time = dt_time(23, 59)  # Far in the future
            engine._strategies = [strategy]

            engine._check_time_based_exits()

            engine.orders.sell.assert_not_called()
        finally:
            _cleanup_engine(db, db_path)


# =====================================================================
#  BUG 4: EOD LIMIT USES LIVE BID (not entry * 0.01)
# =====================================================================

class TestEODLimitPrice:
    """EOD Phase 1 should fetch live bid for LIMIT price, not use entry * 0.01."""

    def test_eod_limit_uses_broker_bid(self):
        """EOD LIMIT phase should fetch live bid and use bid * 0.95."""
        engine, db, db_path = _mock_engine()
        try:
            trade = Trade(
                symbol="SPY20260327P639",
                action="BUY",
                quantity=32,
                entry_price=1.57,
                entry_time=get_eastern_time().isoformat(),
                status="open",
            )
            db.insert_trade(trade)

            # Mock quote_client to return a bid of $3.00 (ITM put)
            engine.quote_client.get_quote_by_symbol.return_value = {
                'bidPrice': 3.00,
                'askPrice': 3.20,
                'lastTradePrice': 3.10,
            }

            engine._eod_force_close_positions(use_market=False)

            engine.orders.sell.assert_called_once()
            call_args = engine.orders.sell.call_args
            limit_price = call_args.kwargs.get('limit_price') or call_args[1].get('limit_price')
            assert limit_price == 2.85, f"Expected $2.85, got ${limit_price}"
        finally:
            _cleanup_engine(db, db_path)

    def test_eod_limit_falls_back_to_minimum_on_quote_failure(self):
        """If quote fetch fails, EOD LIMIT falls back to $0.01."""
        engine, db, db_path = _mock_engine()
        try:
            trade = Trade(
                symbol="SPY20260327P639",
                action="BUY",
                quantity=32,
                entry_price=1.57,
                entry_time=get_eastern_time().isoformat(),
                status="open",
            )
            db.insert_trade(trade)

            engine.quote_client.get_quote_by_symbol.side_effect = Exception("connection error")

            engine._eod_force_close_positions(use_market=False)

            engine.orders.sell.assert_called_once()
            call_args = engine.orders.sell.call_args
            limit_price = call_args.kwargs.get('limit_price') or call_args[1].get('limit_price')
            assert limit_price == 0.01, f"Expected $0.01 fallback, got ${limit_price}"
        finally:
            _cleanup_engine(db, db_path)

    def test_eod_market_phase_sends_no_limit(self):
        """EOD Phase 2 (market) should send limit_price=None."""
        engine, db, db_path = _mock_engine()
        try:
            trade = Trade(
                symbol="SPY20260327P639",
                action="BUY",
                quantity=32,
                entry_price=1.57,
                entry_time=get_eastern_time().isoformat(),
                status="open",
            )
            db.insert_trade(trade)

            engine._eod_force_close_positions(use_market=True)

            engine.orders.sell.assert_called_once()
            call_args = engine.orders.sell.call_args
            limit_price = call_args.kwargs.get('limit_price') or call_args[1].get('limit_price')
            assert limit_price is None, f"Market order should have no limit, got ${limit_price}"
        finally:
            _cleanup_engine(db, db_path)


# =====================================================================
#  BUG 5: STOCK POSITION FROM EXERCISE DETECTED AND LIQUIDATED
# =====================================================================

class TestExerciseStockDetection:
    """Stock positions in option underlyings should be detected and auto-liquidated."""

    def test_short_stock_from_put_exercise_gets_covered(self):
        """Short SPY stock (-3200 shares) should trigger BUY cover order."""
        engine, db, db_path = _mock_engine(option_underlyings=["SPY"])
        try:
            engine.client.get_account_positions.return_value = [
                {
                    'symbol': 'SPY',
                    'openQuantity': -3200,
                    'averageEntryPrice': 639.00,
                    'assetCategory': 'STK',
                    'secType': 'STK',
                },
            ]

            engine._reconcile_broker_positions()

            engine.orders.buy.assert_called_once_with(
                symbol="SPY",
                quantity=3200,
                limit_price=None,
            )
        finally:
            _cleanup_engine(db, db_path)

    def test_long_stock_from_call_exercise_gets_sold(self):
        """Long SPY stock (+100 shares) from call exercise should trigger SELL."""
        engine, db, db_path = _mock_engine(option_underlyings=["SPY"])
        try:
            engine.client.get_account_positions.return_value = [
                {
                    'symbol': 'SPY',
                    'openQuantity': 100,
                    'averageEntryPrice': 645.00,
                    'assetCategory': 'STK',
                },
            ]

            engine._reconcile_broker_positions()

            engine.orders.sell.assert_called_once_with(
                symbol="SPY",
                quantity=100,
                limit_price=None,
            )
        finally:
            _cleanup_engine(db, db_path)

    def test_stock_position_not_in_option_underlyings_ignored(self):
        """Stock positions for symbols NOT in option_underlyings are left alone."""
        engine, db, db_path = _mock_engine(option_underlyings=["SPY"])
        try:
            engine.client.get_account_positions.return_value = [
                {
                    'symbol': 'AAPL',
                    'openQuantity': 100,
                    'averageEntryPrice': 180.00,
                    'assetCategory': 'STK',
                },
            ]

            engine._reconcile_broker_positions()

            engine.orders.sell.assert_not_called()
            engine.orders.buy.assert_not_called()
        finally:
            _cleanup_engine(db, db_path)

    def test_option_position_not_treated_as_stock(self):
        """Option positions should be handled normally, not as exercise stock."""
        engine, db, db_path = _mock_engine(option_underlyings=["SPY"])
        try:
            engine.client.get_account_positions.return_value = [
                {
                    'symbol': 'SPY20260327P639',
                    'openQuantity': 32,
                    'averageEntryPrice': 1.57,
                    'assetCategory': 'OPT',
                },
            ]

            engine._reconcile_broker_positions()

            # Should NOT trigger stock liquidation.
            # Verify no BUY/SELL for plain "SPY" (stock) was issued.
            for c in engine.orders.buy.call_args_list:
                sym = c.kwargs.get('symbol') or (c.args[0] if c.args else '')
                assert sym != 'SPY', "Should not submit stock order for an option position"

        finally:
            _cleanup_engine(db, db_path)

    def test_exercise_stock_skips_if_pending_exit_exists(self):
        """If an exit order is already pending for SPY stock, don't double-submit."""
        engine, db, db_path = _mock_engine(option_underlyings=["SPY"])
        try:
            engine.client.get_account_positions.return_value = [
                {
                    'symbol': 'SPY',
                    'openQuantity': -3200,
                    'averageEntryPrice': 639.00,
                },
            ]
            engine._pending_exit_orders[777] = {
                "symbol": "SPY",
                "trade_id": None,
                "signal_reason": "previous cover",
                "submitted_at": _time.time(),
            }

            engine._reconcile_broker_positions()

            engine.orders.buy.assert_not_called()
        finally:
            _cleanup_engine(db, db_path)

    def test_zero_quantity_stock_ignored(self):
        """Stock positions with qty=0 are ignored."""
        engine, db, db_path = _mock_engine(option_underlyings=["SPY"])
        try:
            engine.client.get_account_positions.return_value = [
                {
                    'symbol': 'SPY',
                    'openQuantity': 0,
                    'averageEntryPrice': 639.00,
                },
            ]

            engine._reconcile_broker_positions()

            engine.orders.buy.assert_not_called()
            engine.orders.sell.assert_not_called()
        finally:
            _cleanup_engine(db, db_path)


# =====================================================================
#  INTEGRATION: FULL MAR 27 SCENARIO REPLAY
# =====================================================================

class TestMar27ScenarioReplay:
    """Replay the exact Mar 27 failure chain and verify each fix prevents it."""

    def test_sfl_prevents_second_trade_after_637p_loss(self):
        """After losing on $637P, the $639P entry must be blocked by SFL."""
        s = _make_strategy(max_daily_losses=1)

        now = get_eastern_time()
        today = now.strftime("%Y-%m-%d")

        # Simulate the first trade: $637P lost (losses_today = 1)
        s.day_state = DayState(
            date=today,
            trades_today=1,
            losses_today=1,
            pnl_today=-1834.0,
        )
        s.trade_state = TradeState()  # Not in trade (first trade closed)

        # Pending direction was set by on_quote before the loss was recorded
        s._pending_direction = "PUT"
        s._pending_direction_time = now

        # The $639P option comes through on_option_quote
        quote = _make_option_quote(
            symbol="SPY20260327P639",
            option_type="put",
            bid=1.50, ask=1.60, last=1.57,
            delta=-0.50, strike=639.0,
        )

        signal = s.on_option_quote(quote)
        assert signal is None, "SFL should have blocked the $639P entry"
        assert s._pending_direction is None, "Pending direction should be cleared"

    def test_max_hold_exits_639p_before_exercise(self):
        """If $639P was somehow entered, max_hold should close it before 4 PM."""
        s = _make_strategy(max_hold_minutes=80)
        s._smart_exit_enabled = True
        # Override direction-specific hold to match generic (strategy.json sets put_max_hold_bars=18 → 90min)
        s.put_max_hold_minutes = 80

        now = get_eastern_time()
        today = now.strftime("%Y-%m-%d")
        entry_time = (now - timedelta(minutes=88)).isoformat()

        s.trade_state = TradeState(
            in_trade=True,
            symbol="SPY20260327P639",
            direction="PUT",
            entry_price=1.57,
            entry_time=entry_time,
            quantity=32,
            highest_price=2.50,
            lowest_price=1.57,
        )
        s.day_state = DayState(date=today)
        # Push EOD exit far into the future so only max hold can fire
        s.exit_time = dt_time(23, 59)

        # Price between entry and PT (not triggering PT or SL)
        quote = _make_option_quote(
            symbol="SPY20260327P639",
            bid=1.80, ask=1.90, last=1.85,
        )

        signal = s._check_exit(quote, now)
        assert signal is not None, "MAX HOLD should exit after 88 min (limit 80)"
        assert signal.action == "SELL"
        assert "MAX HOLD" in signal.reason

    def test_engine_detects_short_spy_from_exercise(self):
        """Engine reconciliation should detect and cover -3200 SPY short."""
        engine, db, db_path = _mock_engine(option_underlyings=["SPY"])
        try:
            engine.client.get_account_positions.return_value = [
                {
                    'symbol': 'SPY',
                    'openQuantity': -3200,
                    'averageEntryPrice': 637.42,
                    'assetCategory': 'STK',
                },
            ]

            engine._reconcile_broker_positions()

            engine.orders.buy.assert_called_once_with(
                symbol="SPY",
                quantity=3200,
                limit_price=None,
            )
        finally:
            _cleanup_engine(db, db_path)


# =====================================================================
#  _normalize_option_key REGRESSION
# =====================================================================

class TestNormalizeOptionKey:
    """Verify _normalize_option_key returns None for stock symbols."""

    def test_stock_symbol_returns_none(self):
        from live.engine import LiveTradingEngine
        assert LiveTradingEngine._normalize_option_key("SPY") is None
        assert LiveTradingEngine._normalize_option_key("AAPL") is None
        assert LiveTradingEngine._normalize_option_key("USD.CAD") is None

    def test_ibkr_option_symbol_parses(self):
        from live.engine import LiveTradingEngine
        key = LiveTradingEngine._normalize_option_key("SPY20260327P639")
        assert key is not None
        underlying, expiry, strike, right = key
        assert underlying == "SPY"
        assert expiry == "20260327"
        assert strike == 639.0
        assert right == "P"

    def test_questrade_option_symbol_parses(self):
        from live.engine import LiveTradingEngine
        key = LiveTradingEngine._normalize_option_key("SPY27Mar26P639.00")
        assert key is not None
        underlying, expiry, strike, right = key
        assert underlying == "SPY"
        assert expiry == "20260327"
        assert strike == 639.0
        assert right == "P"


# =====================================================================
#  STANDALONE TEST RUNNER
# =====================================================================

def _run_all():
    """Run all test classes and methods, report results."""
    import traceback

    test_classes = [
        TestMaxHoldNotDeadCode,
        TestSFLHardGate,
        TestQuoteIndependentTimeExit,
        TestEODLimitPrice,
        TestExerciseStockDetection,
        TestMar27ScenarioReplay,
        TestNormalizeOptionKey,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            full_name = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS  {full_name}")
            except AssertionError as e:
                failed += 1
                errors.append((full_name, str(e), traceback.format_exc()))
                print(f"  FAIL  {full_name}: {e}")
            except Exception as e:
                failed += 1
                errors.append((full_name, str(e), traceback.format_exc()))
                print(f"  ERROR {full_name}: {type(e).__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print(f"\nFAILURES:")
        for name, msg, tb in errors:
            print(f"\n  {name}:")
            print(f"    {tb.strip().split(chr(10))[-1]}")
    print(f"{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
