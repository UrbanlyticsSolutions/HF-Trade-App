"""
ibkr_client.py  —  Complete self-contained IBKR TWS API client.

Covers
------
  Connection   : EWrapper + EClient in one file, threaded message loop
  Market data  : delayed quotes (no subscription needed), real-time if subscribed
  Historical   : reqHistoricalData for equities and options
  Options      : option-chain discovery (reqSecDefOptParams), contract building,
                 historical option bars, option snapshot ticks
  Account      : account summary, positions, P&L
  Orders       : market / limit / stop / bracket — place, cancel, status
  Diagnostics  : comprehensive error capture, reconnect guard

Usage
-----
  python ibkr_client.py                          # AAPL hist + options
  python ibkr_client.py --symbols AAPL MSFT SPY  # multiple equities
  python ibkr_client.py --option-symbol AAPL     # option chain + option hist
  python ibkr_client.py --port 7496              # live TWS

Output: ibkr_snapshot.md

TWS prerequisites
  File > Global Config > API > Settings
    ✅ Enable ActiveX and Socket Clients
    ✅ Trusted IPs: 127.0.0.1          (avoids the accept-connection dialog)
    Socket port: 7497 (paper) / 7496 (live)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Optional

from ibapi.client import EClient
from ibapi.common import BarData, TickAttrib, TickerId
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.order_state import OrderState
from ibapi.wrapper import EWrapper

try:
    from .ibkr_db import IBKRDatabase
except ImportError:
    from ibkr_db import IBKRDatabase

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BAR_SIZE = {
    "1s": "1 secs", "5s": "5 secs", "15s": "15 secs", "30s": "30 secs",
    "1m": "1 min",  "2m": "2 mins", "3m": "3 mins",  "5m": "5 mins",
    "15m": "15 mins", "30m": "30 mins",
    "1h": "1 hour",  "2h": "2 hours", "4h": "4 hours",
    "1d": "1 day",   "1w": "1 week",  "1M": "1 month",
}
DURATION = {
    "1d": "1 D",  "5d": "5 D",  "1w": "1 W",  "2w": "2 W",
    "1m": "1 M",  "3m": "3 M",  "6m": "6 M",  "1y": "1 Y",
}

# Delayed tick-type offsets (TWS adds 65 to each live type when in delayed mode)
# price_66 = delayed bid, price_67 = delayed ask, price_68 = delayed last
# price_72 = delayed high, price_73 = delayed low, price_75 = delayed close
_DELAYED_PRICE_KEYS = ("price_66", "price_67", "price_68",
                       "price_72", "price_73", "price_75")
_TICK_LABELS = {
    "price_1": "Bid",        "price_2": "Ask",        "price_4": "Last",
    "price_6": "High",       "price_7": "Low",        "price_9": "Close(prev)",
    "price_14": "Open",
    "price_66": "Bid(dly)",  "price_67": "Ask(dly)",  "price_68": "Last(dly)",
    "price_72": "High(dly)", "price_73": "Low(dly)",  "price_75": "Close(dly)",
    "size_0": "Bid Size",    "size_3": "Ask Size",    "size_5": "Last Size",
    "size_8": "Volume",
    "generic_22": "Opt Volume", "generic_23": "Hist Vol",
    "generic_24": "Impl Vol",   "generic_31": "Put/Call ratio",
}

OUTPUT_FILE = "ibkr_snapshot.md"


# ===========================================================================
# EWrapper — all TWS callbacks
# ===========================================================================

class _Wrapper(EWrapper):

    def __init__(self) -> None:
        EWrapper.__init__(self)
        self._lock = threading.Lock()

        # connection state
        self.connected: bool = False
        self._connection_lost: bool = False
        self._connection_restored_event = threading.Event()
        self.next_order_id: Optional[int] = None
        self._connect_event = threading.Event()
        self._order_id_event = threading.Event()

        # market & historical data  {reqId: {field: value}}
        self.tick_data: dict[int, dict] = defaultdict(dict)

        # option chain params  {reqId: {"expirations": set, "strikes": set, ...}}
        self.opt_params: dict[int, list] = defaultdict(list)
        self._opt_params_events: dict[int, threading.Event] = {}

        # contract details  {reqId: [Contract, ...]}
        self._contract_details: dict[int, list] = defaultdict(list)
        self._contract_details_events: dict[int, threading.Event] = {}

        # symbol search  {reqId: [ContractDescription, ...]}
        self._symbol_samples: dict[int, list] = defaultdict(list)
        self._symbol_samples_events: dict[int, threading.Event] = {}

        # account
        self.account_values: dict[str, dict] = defaultdict(dict)
        self.account_summary: dict[str, dict] = defaultdict(dict)
        self.positions: dict[str, dict] = {}
        self.pnl_summary: dict[int, dict] = {}
        self.pnl_single: dict[int, dict] = {}

        # orders / executions
        self.open_orders: dict[int, dict] = {}
        self.order_statuses: dict[int, dict] = {}
        self.executions: list[dict] = []
        self._exec_events: dict[int, threading.Event] = {}

        # reqCompletedOrders (synthetic fills — same shape as executions)
        self.completed_orders: list[dict] = []
        self._completed_orders_event: Optional[threading.Event] = None

        # errors
        self.errors: list[dict] = []

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connectAck(self) -> None:
        with self._lock:
            self.connected = True
            self._connection_lost = False
        self._connect_event.set()
        logger.info("connectAck — connection acknowledged")

    def connectionClosed(self) -> None:
        with self._lock:
            self.connected = False
            self._connection_lost = True
        # Unblock any pending waits so connect() fails fast
        self._connect_event.set()
        self._order_id_event.set()
        logger.warning("connectionClosed")

    def nextValidId(self, orderId: int) -> None:
        with self._lock:
            self.next_order_id = orderId
        self._order_id_event.set()
        logger.info("nextValidId: %d", orderId)

    def managedAccounts(self, accountsList: str) -> None:
        logger.info("managedAccounts: %s", accountsList)
        self.managed_accounts = [a.strip() for a in accountsList.split(',') if a.strip()]

    # ------------------------------------------------------------------
    # Errors / info
    # ------------------------------------------------------------------
    def error(self, reqId: TickerId, errorCode: int, errorString: str,
              advancedOrderRejectJson: str = "") -> None:
        entry = {"reqId": reqId, "code": errorCode, "message": errorString}
        if errorCode in (502, 503, 504):
            logger.error("TWS conn error %d: %s", errorCode, errorString)
            with self._lock:
                self.errors.append(entry)
        elif errorCode == 1100:
            # Connectivity lost — TWS/Gateway disconnected
            logger.warning("TWS connectivity LOST: %s", errorString)
            with self._lock:
                self.connected = False
                self._connection_lost = True
                self._connection_restored_event.clear()
        elif errorCode in (1101, 1102):
            # 1101 = connectivity restored (data lost), 1102 = restored (data maintained)
            logger.info("TWS connectivity RESTORED (code %d): %s", errorCode, errorString)
            with self._lock:
                self.connected = True
                self._connection_lost = False
                self._connection_restored_event.set()
        elif errorCode >= 2000 or errorCode in (
            1300, 2103, 2104, 2105, 2106, 2107, 2108,
            2158, 2176,
        ):
            logger.info("TWS info [reqId=%s] %d: %s", reqId, errorCode, errorString)
        else:
            logger.error("TWS error [reqId=%s] %d: %s", reqId, errorCode, errorString)
            with self._lock:
                self.errors.append(entry)

    # ------------------------------------------------------------------
    # Tick data (real-time + delayed)
    # ------------------------------------------------------------------
    def tickPrice(self, reqId: TickerId, tickType: int, price: float,
                  attrib: TickAttrib) -> None:
        with self._lock:
            self.tick_data[reqId][f"price_{tickType}"] = price

    def tickSize(self, reqId: TickerId, tickType: int, size: int) -> None:
        with self._lock:
            self.tick_data[reqId][f"size_{tickType}"] = size

    def tickString(self, reqId: TickerId, tickType: int, value: str) -> None:
        with self._lock:
            self.tick_data[reqId][f"str_{tickType}"] = value

    def tickGeneric(self, reqId: TickerId, tickType: int, value: float) -> None:
        with self._lock:
            self.tick_data[reqId][f"generic_{tickType}"] = value

    def tickOptionComputation(
        self, reqId: TickerId, tickType: int, tickAttrib: int,
        impliedVol: float, delta: float, optPrice: float, pvDividend: float,
        gamma: float, vega: float, theta: float, undPrice: float,
    ) -> None:
        with self._lock:
            self.tick_data[reqId][f"opt_{tickType}"] = {
                "impliedVol": impliedVol, "delta": delta, "optPrice": optPrice,
                "gamma": gamma, "vega": vega, "theta": theta,
                "undPrice": undPrice, "pvDividend": pvDividend,
            }

    def tickSnapshotEnd(self, reqId: int) -> None:
        with self._lock:
            self.tick_data[reqId]["_snapshot_done"] = True

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------
    def historicalData(self, reqId: int, bar: BarData) -> None:
        # Use getattr for fields that disappeared in older API builds
        with self._lock:
            self.tick_data[reqId].setdefault("bars", []).append({
                "date":     bar.date,
                "open":     bar.open,
                "high":     bar.high,
                "low":      bar.low,
                "close":    bar.close,
                "volume":   bar.volume,
                "wap":      getattr(bar, "wap", 0.0),
                "barCount": getattr(bar, "barCount", 0),
            })

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        with self._lock:
            self.tick_data[reqId]["_hist_done"] = True
        logger.info("historicalDataEnd reqId=%d  [%s → %s]", reqId, start, end)

    def historicalDataUpdate(self, reqId: int, bar: BarData) -> None:
        self.historicalData(reqId, bar)

    # ------------------------------------------------------------------
    # Option chain parameters
    # ------------------------------------------------------------------
    def securityDefinitionOptionParameter(
        self, reqId: int, exchange: str, underlyingConId: int,
        tradingClass: str, multiplier: str,
        expirations: set, strikes: set,
    ) -> None:
        with self._lock:
            self.opt_params[reqId].append({
                "exchange":        exchange,
                "underlyingConId": underlyingConId,
                "tradingClass":    tradingClass,
                "multiplier":      multiplier,
                "expirations":     sorted(expirations),
                "strikes":         sorted(float(s) for s in strikes),
            })

    def securityDefinitionOptionParameterEnd(self, reqId: int) -> None:
        ev = self._opt_params_events.get(reqId)
        if ev:
            ev.set()
        logger.info("securityDefinitionOptionParameterEnd reqId=%d", reqId)

    # ------------------------------------------------------------------
    # Contract details
    # ------------------------------------------------------------------
    def contractDetails(self, reqId: int, contractDetails) -> None:
        with self._lock:
            self._contract_details[reqId].append(contractDetails.contract)

    def contractDetailsEnd(self, reqId: int) -> None:
        ev = self._contract_details_events.get(reqId)
        if ev:
            ev.set()
        logger.info("contractDetailsEnd reqId=%d  count=%d",
                    reqId, len(self._contract_details.get(reqId, [])))

    def symbolSamples(self, reqId: int, contractDescriptions: list) -> None:
        with self._lock:
            self._symbol_samples[reqId] = list(contractDescriptions)
        ev = self._symbol_samples_events.get(reqId)
        if ev:
            ev.set()
        logger.info("symbolSamples reqId=%d  count=%d", reqId, len(contractDescriptions))

    # ------------------------------------------------------------------
    # Account / portfolio
    # ------------------------------------------------------------------
    def updateAccountValue(self, key: str, val: str, currency: str,
                           accountName: str) -> None:
        with self._lock:
            self.account_values[key][currency] = val

    def updatePortfolio(self, contract: Contract, position: float,
                        marketPrice: float, marketValue: float,
                        averageCost: float, unrealizedPNL: float,
                        realizedPNL: float, accountName: str) -> None:
        with self._lock:
            # Use localSymbol for options (e.g. "SPY   260310C00680000") to avoid
            # multiple option positions overwriting each other under one "SPY" key.
            key = contract.localSymbol or contract.symbol
            self.positions[key] = {
                "contract": contract, "position": position,
                "market_price": marketPrice, "market_value": marketValue,
                "average_cost": averageCost, "unrealized_pnl": unrealizedPNL,
                "realized_pnl": realizedPNL, "account": accountName,
            }

    def accountSummary(self, reqId: int, account: str, tag: str,
                       value: str, currency: str) -> None:
        with self._lock:
            self.account_summary[tag][currency] = value

    def accountSummaryEnd(self, reqId: int) -> None:
        logger.info("accountSummaryEnd reqId=%d", reqId)

    def position(self, account: str, contract: Contract, position: float,
                 avgCost: float) -> None:
        with self._lock:
            # Use localSymbol for options so each strike/expiry is tracked separately.
            key = contract.localSymbol or contract.symbol
            self.positions[key] = {
                "account": account, "contract": contract,
                "position": position, "avg_cost": avgCost,
            }

    def positionEnd(self) -> None:
        logger.info("positionEnd — all positions received")

    # ------------------------------------------------------------------
    # P&L
    # ------------------------------------------------------------------
    def pnl(self, reqId: int, dailyPnL: float, unrealizedPnL: float,
            realizedPnL: float) -> None:
        with self._lock:
            self.pnl_summary[reqId] = {
                "daily_pnl": dailyPnL,
                "unrealized_pnl": unrealizedPnL,
                "realized_pnl": realizedPnL,
            }

    def pnlSingle(self, reqId: int, pos: Decimal, dailyPnL: float,
                  unrealizedPnL: float, realizedPnL: float, value: float) -> None:
        with self._lock:
            self.pnl_single[reqId] = {
                "pos": float(pos), "daily_pnl": dailyPnL,
                "unrealized_pnl": unrealizedPnL, "realized_pnl": realizedPnL,
                "value": value,
            }

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def openOrder(self, orderId: int, contract: Contract, order: Order,
                  orderState: OrderState) -> None:
        with self._lock:
            self.open_orders[orderId] = {
                "contract": contract, "order": order, "state": orderState,
            }

    def openOrderEnd(self) -> None:
        logger.info("openOrderEnd")

    def orderStatus(self, orderId: int, status: str, filled: float,
                    remaining: float, avgFillPrice: float, permId: int,
                    parentId: int, lastFillPrice: float, clientId: int,
                    whyHeld: str, mktCapPrice: float) -> None:
        with self._lock:
            self.order_statuses[orderId] = {
                "status": status, "filled": filled, "remaining": remaining,
                "avg_fill_price": avgFillPrice,
            }
        logger.info("orderStatus %d: %s  filled=%.2f", orderId, status, filled)

    def execDetails(self, reqId: int, contract: Contract, execution) -> None:
        with self._lock:
            self.executions.append({
                "symbol": contract.symbol,
                "localSymbol": contract.localSymbol or contract.symbol,
                "secType": contract.secType,
                "right": getattr(contract, 'right', ''),
                "strike": getattr(contract, 'strike', 0),
                "expiry": getattr(contract, 'lastTradeDateOrContractMonth', ''),
                "side": execution.side, "shares": execution.shares,
                "price": execution.price, "time": execution.time,
                "order_id": execution.orderId,
                "exec_id": execution.execId,
                "acct_number": execution.acctNumber,
            })

    def execDetailsEnd(self, reqId: int) -> None:
        ev = self._exec_events.get(reqId)
        if ev:
            ev.set()
        logger.info("execDetailsEnd reqId=%d", reqId)

    def completedOrder(self, contract: Contract, order: Order,
                       orderState: OrderState) -> None:
        """Single filled (completed) order from reqCompletedOrders."""
        filled = float(getattr(orderState, "filled", 0) or 0)
        if filled <= 0:
            return
        status = (getattr(orderState, "status", "") or "").strip().lower()
        if status and status not in ("filled", "partfilled"):
            return

        act = (getattr(order, "action", None) or "").upper()
        if act in ("BUY", "BOT"):
            side = "BOT"
        elif act in ("SELL", "SLD"):
            side = "SLD"
        else:
            return

        avg_price = float(getattr(orderState, "avgFillPrice", 0) or 0)
        completed_time = (
            getattr(orderState, "completedTime", "")
            or getattr(orderState, "lastUpdateTime", "")
            or ""
        )
        oid = int(getattr(order, "orderId", 0) or 0)
        acct = getattr(order, "account", "") or ""
        try:
            comm = float(getattr(orderState, "commission", 0) or 0)
        except (TypeError, ValueError):
            comm = 0.0

        with self._lock:
            self.completed_orders.append({
                "symbol": contract.symbol,
                "localSymbol": contract.localSymbol or contract.symbol,
                "secType": contract.secType,
                "right": getattr(contract, "right", ""),
                "strike": getattr(contract, "strike", 0),
                "expiry": getattr(contract, "lastTradeDateOrContractMonth", ""),
                "side": side,
                "shares": int(filled),
                "price": avg_price,
                "time": completed_time,
                "order_id": oid,
                "exec_id": f"CO{oid}",
                "acct_number": acct,
                "commission": comm,
            })

    def completedOrdersEnd(self) -> None:
        ev = self._completed_orders_event
        if ev:
            ev.set()
        logger.info("completedOrdersEnd count=%d", len(self.completed_orders))


# ===========================================================================
# EClient wrapper — connection + atomic request-ID counter
# ===========================================================================

class IBKRClient:
    """
    Thread-safe TWS client. Contains the EWrapper, EClient, and all
    high-level request methods in a single object.

    Parameters
    ----------
    host, port, client_id : connection settings
    connect_timeout       : seconds to wait for handshake
    """

    _req_counter: int = 100
    _req_lock = threading.Lock()

    def __init__(
        self, host: str = "127.0.0.1", port: int = None,
        client_id: int = 0, connect_timeout: float = 15.0,
    ) -> None:
        if client_id == 0:
            import random
            client_id = random.randint(100, 999)
        if port is None:
            from config import defaults as cfg
            port = cfg.ibkr_paper_port()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connect_timeout = connect_timeout

        self.wrapper = _Wrapper()
        self._ec = EClient(wrapper=self.wrapper)
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        logger.info("Connecting to %s:%d (clientId=%d)", self.host, self.port, self.client_id)
        self._ec.connect(self.host, self.port, self.client_id)
        self._thread = threading.Thread(target=self._ec.run, name="tws-loop", daemon=True)
        self._thread.start()

        if not self.wrapper._connect_event.wait(timeout=self.connect_timeout):
            self._safe_disconnect()
            raise ConnectionError(
                f"TWS did not acknowledge connection within {self.connect_timeout}s.\n"
                "  → Ensure TWS is running with API enabled.\n"
                "  → Add 127.0.0.1 to Trusted IPs to skip the accept dialog."
            )
        if self.wrapper._connection_lost:
            self._safe_disconnect()
            raise ConnectionError("Connection closed by TWS immediately after connect.")
        if not self.wrapper._order_id_event.wait(timeout=self.connect_timeout):
            self._safe_disconnect()
            raise ConnectionError("Did not receive nextValidId from TWS.")
        if self.wrapper._connection_lost:
            self._safe_disconnect()
            raise ConnectionError("Connection lost before nextValidId received.")
        logger.info("Connected. nextValidId=%d", self.wrapper.next_order_id)

    def _safe_disconnect(self) -> None:
        """Disconnect EClient, tolerating already-closed connections."""
        try:
            self._ec.disconnect()
        except (AttributeError, OSError):
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def disconnect(self) -> None:
        self._safe_disconnect()
        logger.info("Disconnected.")

    def reconnect(self, max_attempts: int = 5, base_delay: float = 2.0) -> bool:
        """
        Disconnect and re-establish connection to TWS with exponential backoff.

        Returns True if reconnection succeeded, False otherwise.
        """
        logger.warning("Attempting TWS reconnect (up to %d attempts)...", max_attempts)

        # Tear down existing connection cleanly
        try:
            self._ec.disconnect()
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        for attempt in range(1, max_attempts + 1):
            delay = min(base_delay * (2 ** (attempt - 1)), 60)  # cap at 60s
            logger.info("Reconnect attempt %d/%d (delay %.1fs)...",
                        attempt, max_attempts, delay)
            time.sleep(delay)

            try:
                # Reset wrapper events for fresh handshake
                self.wrapper._connect_event.clear()
                self.wrapper._order_id_event.clear()
                self.wrapper._connection_lost = False

                # Fresh EClient + socket
                self._ec = EClient(wrapper=self.wrapper)
                self._ec.connect(self.host, self.port, self.client_id)
                self._thread = threading.Thread(
                    target=self._ec.run, name="tws-loop", daemon=True
                )
                self._thread.start()

                if not self.wrapper._connect_event.wait(timeout=self.connect_timeout):
                    logger.warning("Reconnect attempt %d: no connectAck", attempt)
                    try:
                        self._ec.disconnect()
                    except Exception:
                        pass
                    continue

                if not self.wrapper._order_id_event.wait(timeout=self.connect_timeout):
                    logger.warning("Reconnect attempt %d: no nextValidId", attempt)
                    try:
                        self._ec.disconnect()
                    except Exception:
                        pass
                    continue

                logger.info("Reconnected on attempt %d. nextValidId=%d",
                            attempt, self.wrapper.next_order_id)
                return True

            except Exception as e:
                logger.warning("Reconnect attempt %d failed: %s", attempt, e)
                try:
                    self._ec.disconnect()
                except Exception:
                    pass

        logger.error("All %d reconnect attempts failed.", max_attempts)
        return False

    @property
    def is_connected(self) -> bool:
        return self._ec.isConnected()

    @property
    def connection_lost(self) -> bool:
        """True if TWS sent error 1100 (connectivity lost) and hasn't restored yet."""
        return self.wrapper._connection_lost

    def _next_id(self) -> int:
        with IBKRClient._req_lock:
            IBKRClient._req_counter += 1
            return IBKRClient._req_counter

    def _next_order_id(self) -> int:
        with IBKRClient._req_lock:
            oid = self.wrapper.next_order_id
            self.wrapper.next_order_id += 1  # type: ignore[operator]
        return oid

    # ------------------------------------------------------------------
    # Market data type (call before any reqMktData)
    # ------------------------------------------------------------------

    def set_market_data_type(self, data_type: int = 3) -> None:
        """
        1 = live (requires subscription)
        2 = frozen (last known live price)
        3 = delayed (~15 min, free — works without a data subscription)
        4 = delayed-frozen
        """
        self._ec.reqMarketDataType(data_type)
        logger.info("reqMarketDataType(%d)", data_type)

    # ------------------------------------------------------------------
    # Contract builders
    # ------------------------------------------------------------------

    @staticmethod
    def stock(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Contract:
        c = Contract()
        c.symbol = symbol; c.secType = "STK"; c.exchange = exchange; c.currency = currency
        return c

    @staticmethod
    def option(
        symbol: str, expiry: str, strike: float, right: str,
        exchange: str = "SMART", currency: str = "USD", multiplier: str = "100",
    ) -> Contract:
        """right = 'C' (call) or 'P' (put). expiry = 'YYYYMMDD'."""
        c = Contract()
        c.symbol = symbol
        c.secType = "OPT"
        c.exchange = exchange
        c.currency = currency
        c.lastTradeDateOrContractMonth = expiry
        c.strike = strike
        c.right = right.upper()
        c.multiplier = multiplier
        return c

    @staticmethod
    def futures(symbol: str, expiry: str, exchange: str,
                currency: str = "USD", multiplier: str = "") -> Contract:
        c = Contract()
        c.symbol = symbol; c.secType = "FUT"; c.exchange = exchange
        c.currency = currency; c.lastTradeDateOrContractMonth = expiry
        if multiplier:
            c.multiplier = multiplier
        return c

    @staticmethod
    def forex(base: str, quote: str = "USD") -> Contract:
        c = Contract()
        c.symbol = base; c.secType = "CASH"; c.exchange = "IDEALPRO"; c.currency = quote
        return c

    # ------------------------------------------------------------------
    # Snapshot / streaming quotes
    # ------------------------------------------------------------------

    def get_quote(self, contract: Contract, timeout: float = 10.0) -> dict:
        """
        Request a delayed snapshot tick.
        Returns the tick dict (price_66/67/68 for delayed bid/ask/last,
        or price_1/2/4 if live subscription is active).
        """
        req_id = self._next_id()
        self.wrapper.tick_data.pop(req_id, None)
        # snapshot=True → TWS sends ticks then raises tickSnapshotEnd
        # Empty genericTickList avoids error 321 in delayed mode
        self._ec.reqMktData(req_id, contract, "", True, False, [])
        logger.info("reqMktData snapshot reqId=%d  %s", req_id, contract.symbol)

        deadline = time.monotonic() + timeout
        # For delayed data, TWS may take a moment to push all fields.
        # We wait for _snapshot_done OR at least 2 seconds of price data
        # being stable to allow all delayed fields to arrive.
        first_price_at: Optional[float] = None
        while time.monotonic() < deadline:
            td = self.wrapper.tick_data.get(req_id, {})
            has_prices = any(k.startswith("price_") for k in td)
            if td.get("_snapshot_done"):
                return dict(td)
            if has_prices and first_price_at is None:
                first_price_at = time.monotonic()
            if first_price_at and (time.monotonic() - first_price_at) > 1.5:
                return dict(td)
            time.sleep(0.1)
        logger.warning("get_quote timeout reqId=%d", req_id)
        return dict(self.wrapper.tick_data.get(req_id, {}))

    def subscribe_quotes(self, contract: Contract,
                         generic_tick_list: str = "") -> int:
        """Start a streaming quote subscription. Returns reqId."""
        req_id = self._next_id()
        self._ec.reqMktData(req_id, contract, generic_tick_list, False, False, [])
        logger.info("reqMktData stream reqId=%d  %s", req_id, contract.symbol)
        return req_id

    def cancel_quotes(self, req_id: int) -> None:
        self._ec.cancelMktData(req_id)

    # ------------------------------------------------------------------
    # Historical bars
    # ------------------------------------------------------------------

    def get_historical_bars(
        self,
        contract: Contract,
        end_datetime: str = "",
        duration: str = "5 D",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: int = 1,
        timeout: float = 30.0,
    ) -> list[dict]:
        """
        Block until all bars arrive (historicalDataEnd sets _hist_done flag)
        or timeout expires.  Returns list of OHLCV dicts.
        """
        req_id = self._next_id()
        self.wrapper.tick_data.pop(req_id, None)

        self._ec.reqHistoricalData(
            req_id, contract, end_datetime, duration,
            bar_size, what_to_show, use_rth, 1, False, [],
        )
        logger.info(
            "reqHistoricalData reqId=%d  %s  %s  %s",
            req_id, contract.symbol, duration, bar_size,
        )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            td = self.wrapper.tick_data.get(req_id, {})
            if td.get("_hist_done"):
                break
            time.sleep(0.2)
        else:
            logger.warning("get_historical_bars timeout reqId=%d", req_id)

        return self.wrapper.tick_data.get(req_id, {}).get("bars", [])

    # ------------------------------------------------------------------
    # Option chain discovery
    # ------------------------------------------------------------------

    def get_option_chain_params(
        self, underlying: Contract, timeout: float = 20.0,
    ) -> list[dict]:
        """
        Calls reqSecDefOptParams and returns a list of dicts:
          exchange, underlyingConId, tradingClass, multiplier,
          expirations (sorted list), strikes (sorted list of floats)

        NOTE: reqSecDefOptParams requires the underlying's conId.
        We resolve it first via reqContractDetails if conId == 0.
        """
        # Resolve conId if the caller didn't set it
        con_id = underlying.conId if underlying.conId else 0
        if con_id == 0:
            resolved = self.resolve_contract(underlying, timeout=timeout)
            if resolved:
                con_id = resolved.conId
                logger.info("Resolved %s conId=%d", underlying.symbol, con_id)
            else:
                logger.warning("Could not resolve conId for %s — "
                               "option chain may fail", underlying.symbol)

        req_id = self._next_id()
        self.wrapper.opt_params.pop(req_id, None)
        ev = threading.Event()
        self.wrapper._opt_params_events[req_id] = ev

        self._ec.reqSecDefOptParams(
            req_id, underlying.symbol, "",
            underlying.secType, con_id,
        )
        logger.info("reqSecDefOptParams reqId=%d  %s  conId=%d",
                    req_id, underlying.symbol, con_id)
        ev.wait(timeout=timeout)
        del self.wrapper._opt_params_events[req_id]

        result = list(self.wrapper.opt_params.get(req_id, []))
        logger.info(
            "Option chain params received: %d exchange entries for %s",
            len(result), underlying.symbol,
        )
        return result

    # ------------------------------------------------------------------
    # Contract resolution (conId lookup)
    # ------------------------------------------------------------------

    def resolve_symbol(self, symbol: str, sec_type: str = "STK",
                       currency: str = "USD",
                       timeout: float = 10.0) -> Optional[int]:
        """Use reqMatchingSymbols to find the conId for a symbol.
        Returns the conId or None."""
        req_id = self._next_id()
        ev = threading.Event()
        self.wrapper._symbol_samples[req_id] = []
        self.wrapper._symbol_samples_events[req_id] = ev

        self._ec.reqMatchingSymbols(req_id, symbol)
        logger.info("reqMatchingSymbols reqId=%d  %s", req_id, symbol)
        ev.wait(timeout=timeout)
        del self.wrapper._symbol_samples_events[req_id]

        for desc in self.wrapper._symbol_samples.get(req_id, []):
            c = desc.contract
            if (c.symbol == symbol
                    and c.secType == sec_type
                    and c.currency == currency
                    and c.conId > 0):
                logger.info("resolve_symbol: %s conId=%d primaryExchange=%s",
                            c.symbol, c.conId, c.primaryExchange)
                return c.conId
        logger.warning("resolve_symbol: no match for %s %s %s", symbol, sec_type, currency)
        return None

    def resolve_contract(self, contract: Contract,
                         timeout: float = 10.0,
                         retries: int = 2) -> Optional[Contract]:
        """Resolve a contract's conId.
        Uses reqMatchingSymbols first (reliable), falls back to
        reqContractDetails if needed."""
        # Try reqMatchingSymbols first (works on all TWS builds)
        con_id = self.resolve_symbol(
            contract.symbol, contract.secType, contract.currency,
            timeout=timeout,
        )
        if con_id:
            contract.conId = con_id
            return contract

        # Fallback: reqContractDetails
        for attempt in range(1, retries + 1):
            req_id = self._next_id()
            ev = threading.Event()
            self.wrapper._contract_details[req_id] = []
            self.wrapper._contract_details_events[req_id] = ev

            self._ec.reqContractDetails(req_id, contract)
            logger.info("reqContractDetails reqId=%d  %s (attempt %d/%d)",
                        req_id, contract.symbol, attempt, retries)
            ev.wait(timeout=timeout)
            del self.wrapper._contract_details_events[req_id]

            results = self.wrapper._contract_details.get(req_id, [])
            if results:
                logger.info("resolve_contract: %s conId=%d",
                            results[0].symbol, results[0].conId)
                return results[0]
            logger.warning("resolve_contract attempt %d: no result for %s",
                           attempt, contract.symbol)
            if attempt < retries:
                time.sleep(2)
        return None

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account_summary(
        self,
        tags: str = (
            "NetLiquidation,TotalCashValue,BuyingPower,AvailableFunds,"
            "ExcessLiquidity,InitMarginReq,MaintMarginReq,GrossPositionValue,"
            "EquityWithLoanValue,UnrealizedPnL,RealizedPnL,Cushion,Leverage,"
            "DayTradesRemaining"
        ),
        timeout: float = 15.0,
    ) -> dict:
        req_id = self._next_id()
        self.wrapper.account_summary.clear()
        done = threading.Event()
        orig = self.wrapper.accountSummaryEnd

        def _end(rid: int):
            if rid == req_id:
                done.set()

        self.wrapper.accountSummaryEnd = _end  # type: ignore[method-assign]
        self._ec.reqAccountSummary(req_id, "All", tags)
        logger.info("reqAccountSummary reqId=%d", req_id)
        done.wait(timeout=timeout)
        self.wrapper.accountSummaryEnd = orig  # type: ignore[method-assign]
        # Cancel subscription to avoid TWS rate-limit error 322
        self._ec.cancelAccountSummary(req_id)
        return dict(self.wrapper.account_summary)

    def subscribe_account_updates(self, account: str = "") -> None:
        self._ec.reqAccountUpdates(True, account)
        logger.info("reqAccountUpdates subscribed account=%s", account or "<default>")

    def unsubscribe_account_updates(self, account: str = "") -> None:
        self._ec.reqAccountUpdates(False, account)

    def get_positions(self, timeout: float = 10.0) -> dict:
        self.wrapper.positions.clear()
        done = threading.Event()
        orig = self.wrapper.positionEnd

        def _end():
            done.set()

        self.wrapper.positionEnd = _end  # type: ignore[method-assign]
        self._ec.reqPositions()
        logger.info("reqPositions requested")
        done.wait(timeout=timeout)
        self.wrapper.positionEnd = orig  # type: ignore[method-assign]
        return dict(self.wrapper.positions)

    def subscribe_pnl(self, account: str, model_code: str = "") -> int:
        req_id = self._next_id()
        self._ec.reqPnL(req_id, account, model_code)
        logger.info("reqPnL reqId=%d account=%s", req_id, account)
        return req_id

    def cancel_pnl(self, req_id: int) -> None:
        self._ec.cancelPnL(req_id)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    @staticmethod
    def _make_order(action: str, qty: float, order_type: str,
                    lmt: float = 0.0, aux: float = 0.0,
                    tif: str = "DAY", account: str = "") -> Order:
        o = Order()
        o.action = action.upper()
        o.totalQuantity = qty
        o.orderType = order_type
        o.lmtPrice = lmt
        o.auxPrice = aux
        o.tif = tif.upper()
        o.eTradeOnly = False
        o.firmQuoteOnly = False
        if account:
            o.account = account
        return o

    def market_order(self, action: str, qty: float, account: str = "") -> Order:
        return self._make_order(action, qty, "MKT", account=account)

    def limit_order(self, action: str, qty: float, price: float,
                    tif: str = "DAY", account: str = "") -> Order:
        return self._make_order(action, qty, "LMT", lmt=price, tif=tif, account=account)

    def stop_order(self, action: str, qty: float, stop_price: float,
                   account: str = "") -> Order:
        return self._make_order(action, qty, "STP", aux=stop_price, account=account)

    def place_order(self, contract: Contract, order: Order,
                    wait_seconds: float = 0.0) -> int:
        oid = self._next_order_id()
        self._ec.placeOrder(oid, contract, order)
        logger.info("placeOrder orderId=%d  %s %s x%.0f",
                    oid, order.action, contract.symbol, order.totalQuantity)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        return oid

    def cancel_order(self, order_id: int) -> None:
        self._ec.cancelOrder(order_id)
        logger.info("cancelOrder orderId=%d", order_id)

    def cancel_all_orders(self) -> None:
        self._ec.reqGlobalCancel()

    def get_order_status(self, order_id: int) -> Optional[dict]:
        return self.wrapper.order_statuses.get(order_id)

    def get_executions(self, client_id: int = -1, acct_code: str = "",
                       time_filter: str = "", symbol: str = "",
                       sec_type: str = "", exchange: str = "",
                       side: str = "", timeout: float = 15.0) -> list[dict]:
        """Request execution reports from TWS for the current session.

        Uses reqExecutions with an ExecutionFilter.  Returns the list of
        execution dicts collected by the execDetails callback.

        Parameters match the IBApi ExecutionFilter fields. ``time_filter``
        should be in TWS format: 'yyyymmdd-hh:mm:ss'.
        """
        from ibapi.execution import ExecutionFilter

        req_id = self._next_id()
        ev = threading.Event()
        self.wrapper._exec_events[req_id] = ev
        # Clear previous executions so we get a clean snapshot
        with self.wrapper._lock:
            self.wrapper.executions.clear()

        ef = ExecutionFilter()
        if client_id >= 0:
            ef.clientId = client_id
        if acct_code:
            ef.acctCode = acct_code
        if time_filter:
            ef.time = time_filter
        if symbol:
            ef.symbol = symbol
        if sec_type:
            ef.secType = sec_type
        if exchange:
            ef.exchange = exchange
        if side:
            ef.side = side

        self._ec.reqExecutions(req_id, ef)
        logger.info("reqExecutions reqId=%d", req_id)
        ev.wait(timeout=timeout)
        self.wrapper._exec_events.pop(req_id, None)

        with self.wrapper._lock:
            return list(self.wrapper.executions)

    def get_completed_orders_fills(
        self,
        api_only: Optional[bool] = None,
        timeout: float = 15.0,
    ) -> list[dict]:
        """Request reqCompletedOrders; return synthetic rows matching get_executions keys.

        When session reqExecutions is empty, completed orders can still list recent fills.
        api_only: True = only API-placed orders. If None, read IBKR_COMPLETED_ORDERS_API_ONLY
        env (default false = all completed orders).
        """
        if not hasattr(self._ec, "reqCompletedOrders"):
            logger.debug("ibapi EClient has no reqCompletedOrders — skipping")
            return []

        if api_only is None:
            v = (os.environ.get("IBKR_COMPLETED_ORDERS_API_ONLY", "") or "").lower()
            api_only = v in ("1", "true", "yes", "on")

        ev = threading.Event()
        with self.wrapper._lock:
            self.wrapper.completed_orders.clear()
            self.wrapper._completed_orders_event = ev

        try:
            self._ec.reqCompletedOrders(bool(api_only))
        except Exception as e:
            logger.warning("reqCompletedOrders failed: %s", e)
            with self.wrapper._lock:
                self.wrapper._completed_orders_event = None
            return []

        ev.wait(timeout=timeout)
        with self.wrapper._lock:
            self.wrapper._completed_orders_event = None
            rows = list(self.wrapper.completed_orders)
        if not ev.is_set():
            logger.warning("completedOrdersEnd not received within %.1fs", timeout)
        return rows

    def get_merged_session_fills(
        self,
        client_id: int = -1,
        acct_code: str = "",
        time_filter: str = "",
        symbol: str = "",
        sec_type: str = "",
        exchange: str = "",
        side: str = "",
        timeout: float = 15.0,
        completed_orders_api_only: Optional[bool] = None,
    ) -> list[dict]:
        """reqExecutions plus reqCompletedOrders rows not already covered by order_id."""
        execs = self.get_executions(
            client_id=client_id,
            acct_code=acct_code,
            time_filter=time_filter,
            symbol=symbol,
            sec_type=sec_type,
            exchange=exchange,
            side=side,
            timeout=timeout,
        )
        covered: set[int] = {
            int(e["order_id"])
            for e in execs
            if e.get("order_id")
        }
        extra = self.get_completed_orders_fills(
            api_only=completed_orders_api_only,
            timeout=timeout,
        )
        acct = (acct_code or "").strip()
        n_exec = len(execs)
        n_added = 0
        for row in extra:
            if acct and (row.get("acct_number") or "").strip() != acct:
                continue
            oid = int(row.get("order_id") or 0)
            if oid and oid in covered:
                continue
            execs.append(row)
            n_added += 1
            if oid:
                covered.add(oid)
        if n_added:
            logger.info(
                "Session fills: reqExecutions=%d + %d from reqCompletedOrders",
                n_exec,
                n_added,
            )
        return execs


# ===========================================================================
# Markdown output helpers
# ===========================================================================

def _h1(t: str) -> str: return f"# {t}\n\n"
def _h2(t: str) -> str: return f"## {t}\n\n"
def _h3(t: str) -> str: return f"### {t}\n\n"


def _table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "> *(no data)*\n\n"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))
    fmt = lambda cells: "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(headers), sep] + [fmt(r) for r in rows]) + "\n\n"


def _f(v, decimals: int = 4) -> str:
    try:
        return f"{float(v):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(v) if v not in ("", None) else "—"


def _bars_md(symbol: str, bars: list[dict]) -> str:
    if not bars:
        return f"> No bars received for **{symbol}**.\n\n"
    rows = []
    for b in bars:
        vol = b.get("volume", "")
        try:
            vol_s = f"{int(float(vol)):,}"
        except (TypeError, ValueError):
            vol_s = str(vol)
        rows.append([
            b.get("date", ""), _f(b.get("open")), _f(b.get("high")),
            _f(b.get("low")), _f(b.get("close")), vol_s,
            _f(b.get("wap")), str(b.get("barCount", "")),
        ])
    return _table(["Date", "Open", "High", "Low", "Close", "Volume", "WAP", "BarCount"], rows)


def _quote_md(symbol: str, tick: dict) -> str:
    if not tick:
        return f"> No quote data for **{symbol}**.\n\n"
    rows = []
    for key, label in _TICK_LABELS.items():
        if key in tick:
            rows.append([label, str(tick[key])])
    # option greeks
    for k, v in sorted(tick.items()):
        if k.startswith("opt_") and isinstance(v, dict):
            tick_type = k.split("_", 1)[1]
            for gk, gv in v.items():
                rows.append([f"option[{tick_type}].{gk}", _f(gv, 6)])
    # leftover unknown fields
    shown = set(_TICK_LABELS.keys()) | {k for k in tick if k.startswith("opt_")}
    for k in sorted(tick):
        if k not in shown and not k.startswith("_"):
            rows.append([k, str(tick[k])])
    if not rows:
        return f"> Tick data present but no recognised fields for **{symbol}**.\n\n"
    return _table(["Field", "Value"], rows)


def _chain_md(symbol: str, params: list[dict]) -> str:
    if not params:
        return f"> No option chain parameters for **{symbol}**.\n\n"
    lines = []
    for p in params:
        lines.append(f"**Exchange:** {p['exchange']}  |  "
                     f"**Multiplier:** {p['multiplier']}  |  "
                     f"**TradingClass:** {p['tradingClass']}\n\n")
        exp_str = ", ".join(p["expirations"][:20])
        if len(p["expirations"]) > 20:
            exp_str += f" … (+{len(p['expirations'])-20} more)"
        lines.append(f"*Expirations ({len(p['expirations'])})* — {exp_str}\n\n")
        stk_str = ", ".join(_f(s, 2) for s in p["strikes"][:30])
        if len(p["strikes"]) > 30:
            stk_str += f" … (+{len(p['strikes'])-30} more)"
        lines.append(f"*Strikes ({len(p['strikes'])})* — {stk_str}\n\n")
    return "".join(lines)


def _summary_md(summary: dict) -> str:
    priority = [
        "NetLiquidation", "TotalCashValue", "BuyingPower", "AvailableFunds",
        "ExcessLiquidity", "InitMarginReq", "MaintMarginReq",
        "GrossPositionValue", "EquityWithLoanValue",
        "UnrealizedPnL", "RealizedPnL", "Cushion", "Leverage", "DayTradesRemaining",
    ]
    rows, shown = [], set()
    for tag in priority:
        if tag in summary:
            for cur, val in summary[tag].items():
                rows.append([tag, cur, val])
            shown.add(tag)
    for tag in sorted(summary):
        if tag not in shown:
            for cur, val in summary[tag].items():
                rows.append([tag, cur, val])
    return _table(["Tag", "Currency", "Value"], rows)


def _positions_md(positions: dict) -> str:
    rows = []
    for sym, info in sorted(positions.items()):
        rows.append([
            sym,
            _f(info.get("position"), 0),
            _f(info.get("avg_cost") or info.get("average_cost")),
            _f(info.get("market_value", "")) or "—",
            _f(info.get("unrealized_pnl", "")) or "—",
            info.get("account", ""),
        ])
    return _table(
        ["Symbol", "Position", "Avg Cost", "Mkt Value", "Unreal P&L", "Account"], rows
    )


# ===========================================================================
# Main workflow
# ===========================================================================

def _pick_smart_exchange(params: list[dict]) -> Optional[dict]:
    """Prefer SMART, then the first listed exchange."""
    for p in params:
        if p["exchange"] == "SMART":
            return p
    return params[0] if params else None


def main() -> None:
    parser = argparse.ArgumentParser(description="IBKR complete API client")
    parser.add_argument("--host",         default="127.0.0.1")
    parser.add_argument("--port",         type=int, default=None,
                        help="7497=TWS paper, 7496=TWS live (default from config)")
    parser.add_argument("--client-id",    type=int, default=20, dest="client_id")
    parser.add_argument("--account",      default="",
                        help="Account string e.g. U1234567 (leave blank for default)")
    parser.add_argument("--symbols",      nargs="+", default=["AAPL", "SPY"],
                        help="Equity symbols for historical + quote")
    parser.add_argument("--bar-size",     default="1d", choices=list(BAR_SIZE),
                        dest="bar_size")
    parser.add_argument("--duration",     default="5d", choices=list(DURATION))
    parser.add_argument("--option-symbol", default="AAPL", dest="opt_symbol",
                        metavar="SYMBOL",
                        help="Symbol for option chain discovery + sample option bars")
    parser.add_argument("--option-expiry", default="",  dest="opt_expiry",
                        help="Specific expiry YYYYMMDD for option historical bars "
                             "(blank = use nearest expiry from chain)")
    parser.add_argument("--no-account",   action="store_true", dest="no_account")
    parser.add_argument("--timeout",      type=float, default=30.0)
    parser.add_argument("--output",       default=OUTPUT_FILE)
    parser.add_argument("--db",           default="ibkr_data.db",
                        help="SQLite database path (default: ibkr_data.db)")
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Connect
    # ---------------------------------------------------------------
    client = IBKRClient(
        host=args.host, port=args.port,
        client_id=args.client_id, connect_timeout=args.timeout,
    )
    try:
        client.connect()
    except ConnectionError as exc:
        logger.error("Connection failed:\n  %s", exc)
        sys.exit(1)

    # Brief pause so TWS data-farm connections settle before requests
    time.sleep(1.0)

    # Switch to delayed data so quotes work without a market data subscription
    client.set_market_data_type(3)

    # Open database
    db = IBKRDatabase(args.db)

    # ---------------------------------------------------------------
    # Pre-resolve option underlying conId (do this first while TWS
    # hasn't been hit by many requests yet — avoids pacing drops)
    # ---------------------------------------------------------------
    opt_sym = args.opt_symbol
    opt_underlying = client.stock(opt_sym)
    resolved_opt = client.resolve_contract(opt_underlying, timeout=15.0, retries=2)
    if resolved_opt:
        opt_underlying.conId = resolved_opt.conId
        logger.info("Pre-resolved %s conId=%d", opt_sym, resolved_opt.conId)
    else:
        logger.warning("Pre-resolve failed for %s — option chain will use conId=0", opt_sym)

    lines: list[str] = []
    lines.append(_h1("IBKR Complete Snapshot"))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"**Generated:** {now}  |  **Host:** {args.host}:{args.port}  |  "
                 f"**clientId:** {args.client_id}\n\n---\n\n")

    # ---------------------------------------------------------------
    # Account summary + positions
    # ---------------------------------------------------------------
    if not args.no_account:
        logger.info("=== Account summary ===")
        lines.append(_h2("Account Summary"))
        summary = client.get_account_summary(timeout=args.timeout)
        lines.append(_summary_md(summary))
        db.save_account_summary(summary, account=args.account)

        logger.info("=== Positions ===")
        client.subscribe_account_updates(args.account)
        time.sleep(2.0)
        lines.append(_h2("Positions"))
        positions = client.get_positions(timeout=args.timeout)
        lines.append(_positions_md(positions))
        db.save_positions(positions)
        client.unsubscribe_account_updates(args.account)

    # ---------------------------------------------------------------
    # Historical bars + delayed quotes for each equity symbol
    # ---------------------------------------------------------------
    lines.append(_h2("Equity — Historical Bars & Delayed Quote"))
    for sym in args.symbols:
        logger.info("=== %s historical bars ===", sym)
        contract = client.stock(sym)
        lines.append(_h3(sym))

        bars = client.get_historical_bars(
            contract,
            duration=DURATION[args.duration],
            bar_size=BAR_SIZE[args.bar_size],
            what_to_show="TRADES",
            use_rth=1,
            timeout=args.timeout,
        )
        lines.append(f"**Historical bars** ({BAR_SIZE[args.bar_size]}, "
                     f"{DURATION[args.duration]}, TRADES, RTH)\n\n")
        lines.append(_bars_md(sym, bars))
        db.save_bars(sym, "STK", bars, bar_size=BAR_SIZE[args.bar_size],
                     duration=DURATION[args.duration], what_to_show="TRADES")

        logger.info("=== %s delayed quote ===", sym)
        tick = client.get_quote(contract, timeout=10.0)
        lines.append("**Delayed Quote**\n\n")
        lines.append(_quote_md(sym, tick))
        db.save_quote(sym, "STK", tick)

    # ---------------------------------------------------------------
    # Option chain discovery
    # ---------------------------------------------------------------
    logger.info("=== Option chain: %s ===", opt_sym)
    lines.append(_h2(f"Option Chain — {opt_sym}"))

    chain_params = client.get_option_chain_params(opt_underlying, timeout=args.timeout)
    lines.append(_chain_md(opt_sym, chain_params))
    db.save_option_chain(opt_sym, chain_params)

    # ---------------------------------------------------------------
    # Option historical bars — pick nearest expiry + ATM strike
    # ---------------------------------------------------------------
    best = _pick_smart_exchange(chain_params)
    if best and best["expirations"]:
        # Determine expiry (skip today — no EOD chart data for same-day expiry)
        today_s = datetime.now().strftime("%Y%m%d")
        future = [e for e in best["expirations"] if e > today_s]
        expiry = args.opt_expiry or (future[0] if future else best["expirations"][-1])

        # Find ATM strike using the last equity close price from historical bars
        # (fall back to centre of strikes if no bar data is available)
        strikes = best["strikes"]
        last_close = None
        # Look for bars we already fetched for the option symbol
        for sym_data_key in args.symbols:
            if sym_data_key == opt_sym:
                # Re-scan the wrapper tick_data for bars that match this symbol
                for td in client.wrapper.tick_data.values():
                    bar_list = td.get("bars", [])
                    if bar_list:
                        last_close = bar_list[-1].get("close")
                        break
        if last_close and strikes:
            atm_strike = min(strikes, key=lambda s: abs(s - last_close))
        elif strikes:
            atm_strike = strikes[len(strikes) // 2]
        else:
            atm_strike = 0.0
        logger.info("Option ATM strike selected: %.2f  (last_close=%s)", atm_strike, last_close)

        lines.append(_h2(f"Option Historical Bars — {opt_sym} {expiry} "
                         f"{atm_strike:.0f} C/P"))

        for right in ("C", "P"):
            logger.info("=== Option hist bars: %s %s %s %s ===",
                        opt_sym, expiry, atm_strike, right)
            opt_contract = client.option(
                symbol=opt_sym,
                expiry=expiry,
                strike=atm_strike,
                right=right,
                exchange=best["exchange"] if best["exchange"] != "SMART" else "SMART",
                multiplier=best["multiplier"],
            )
            opt_bars = client.get_historical_bars(
                opt_contract,
                duration="5 D",
                bar_size="1 day",
                what_to_show="MIDPOINT",
                use_rth=1,
                timeout=args.timeout,
            )
            label = "Call" if right == "C" else "Put"
            lines.append(f"**{label} ({right}) — {expiry} strike {atm_strike:.0f}**\n\n")
            lines.append(_bars_md(f"{opt_sym} {right}", opt_bars))
            db.save_bars(opt_sym, "OPT", opt_bars, bar_size="1 day",
                         duration="5 D", what_to_show="MIDPOINT",
                         expiry=expiry, strike=atm_strike, right=right)

            # Delayed quote for the option
            logger.info("=== Option quote: %s %s %s %s ===",
                        opt_sym, expiry, atm_strike, right)
            opt_tick = client.get_quote(opt_contract, timeout=10.0)
            lines.append("**Option Delayed Quote**\n\n")
            lines.append(_quote_md(f"{opt_sym} {right}", opt_tick))
            db.save_quote(opt_sym, "OPT", opt_tick, expiry=expiry,
                          strike=atm_strike, right=right)
    else:
        lines.append("> Option chain params not received — cannot fetch option bars.\n\n")

    # ---------------------------------------------------------------
    # Disconnect + write output
    # ---------------------------------------------------------------
    client.disconnect()
    db.close()

    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))

    logger.info("Saved -> %s  |  DB -> %s", args.output, args.db)
    print(f"\nDone. Snapshot -> {args.output}  |  DB -> {args.db}")


if __name__ == "__main__":
    main()
