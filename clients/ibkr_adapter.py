"""
ibkr_adapter.py — Drop-in replacement for QuestradeClient using IBKR TWS API.

Exposes the same public interface that the live trading engine, order manager,
position manager, and 0DTE strategy expect from QuestradeClient, but routes
all calls through the IBKRClient.

Prerequisites:
  - TWS or IB Gateway running with API enabled on port 7497 (paper) / 7496 (live)
  - ibapi package installed: pip install ibapi
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from ibapi.contract import Contract

try:
    from .ibkr_client import IBKRClient
except ImportError:
    from ibkr_client import IBKRClient

logger = logging.getLogger(__name__)


class IBKRAdapter:
    """
    Adapter that wraps IBKRClient and presents the same method signatures
    that the live trading system expects from QuestradeClient.

    Parameters
    ----------
    host : str   TWS host (default 127.0.0.1)
    port : int   7497 = paper, 7496 = live
    client_id : int   Unique client ID for this connection
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        client_id: int = 0,
        connect_timeout: float = 15.0,
        account: str = "",
    ) -> None:
        if client_id == 0:
            # Use a deterministic client_id so orders survive across restarts.
            # Random IDs cause previous orders to become orphaned on TWS.
            client_id = 51
        if host is None:
            host = os.environ.get("IBKR_HOST", "127.0.0.1")
        if port is None:
            env_port = os.environ.get("IBKR_PAPER_PORT")
            if env_port:
                port = int(env_port)
            else:
                from config import defaults as cfg
                port = cfg.ibkr_paper_port()
        self._host = host
        self._port = port
        self._client_id = client_id
        self._account = account
        self._ibkr = IBKRClient(
            host=host, port=port,
            client_id=client_id,
            connect_timeout=connect_timeout,
        )
        self._connected = False
        # Cache: symbol -> conId
        self._symbol_cache: Dict[str, int] = {}
        # Cache: symbol -> last quote dict
        self._quote_cache: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to TWS and switch to delayed data."""
        self._ibkr.connect()
        self._connected = True
        time.sleep(0.5)
        self._ibkr.set_market_data_type(3)  # Delayed (free, no subscription)
        logger.info("IBKRAdapter connected to TWS %s:%d", self._host, self._port)

    def disconnect(self) -> None:
        self._ibkr.disconnect()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ibkr.is_connected

    @property
    def connection_lost(self) -> bool:
        """True if TWS signalled connectivity loss (error 1100)."""
        return self._ibkr.connection_lost

    def ensure_connected(self) -> None:
        """
        Check TWS connection health and reconnect if needed.

        Called automatically before every broker API call so that transient
        network blips or TWS restarts are recovered transparently.
        """
        # Fast path: socket is up and no 1100 event
        if self._ibkr.is_connected and not self._ibkr.connection_lost:
            return

        # If TWS sent error 1100 (connectivity lost), wait briefly for
        # automatic restore (1101/1102) before forcing a full reconnect.
        if self._ibkr.connection_lost:
            logger.warning("TWS connectivity lost — waiting up to 15s for auto-restore...")
            restored = self._ibkr.wrapper._connection_restored_event.wait(timeout=15)
            if restored and self._ibkr.is_connected:
                logger.info("TWS auto-restored. Re-subscribing delayed data.")
                self._ibkr.set_market_data_type(3)
                self._connected = True
                return

        # Full reconnect required
        logger.warning("TWS socket down — triggering full reconnect...")
        ok = self._ibkr.reconnect()
        if ok:
            self._connected = True
            self._ibkr.set_market_data_type(3)
            logger.info("IBKRAdapter reconnected to TWS %s:%d", self._host, self._port)
        else:
            self._connected = False
            raise ConnectionError(
                f"Failed to reconnect to TWS at {self._host}:{self._port}. "
                "Check that TWS/IB Gateway is running."
            )

    def close(self) -> None:
        self.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Account — matches QuestradeClient API
    # ------------------------------------------------------------------

    def get_accounts(self) -> List[Dict]:
        """Return list of account dicts (IBKR has one per connection)."""
        self.ensure_connected()
        acct_id = self._account
        if not acct_id:
            # Use managedAccounts from TWS callback
            managed = getattr(self._ibkr.wrapper, 'managed_accounts', [])
            if managed:
                acct_id = managed[0]
                self._account = acct_id
        return [{
            "type": "Margin",
            "number": acct_id or "default",
            "status": "Active",
        }]

    def get_account_positions(self, account_id: str = "") -> List[Dict]:
        """Get positions in Questrade response format.
        
        Positions are now keyed by localSymbol (e.g. "SPY   260310C00680000")
        so each option strike/expiry is returned individually.
        We convert the localSymbol to our internal OCC-style symbol for matching.
        """
        self.ensure_connected()
        positions = self._ibkr.get_positions(timeout=10.0)
        result = []
        for key, info in positions.items():
            contract: Contract = info.get("contract", Contract())
            # Build a symbol name the engine can match against DB trade symbols.
            # DB uses format like "SPY20260310C680".
            symbol = self._contract_to_trade_symbol(contract) or key
            raw_avg_cost = info.get("avg_cost", info.get("average_cost", 0))
            # IBKR's position() callback returns avgCost as total cost per
            # share (i.e. option premium * multiplier).  For options with
            # multiplier=100, a $1.44 premium is reported as 144.0.
            # Divide by the contract multiplier to get per-contract price.
            multiplier = int(contract.multiplier) if contract and contract.multiplier else 1
            avg_entry_price = raw_avg_cost / multiplier if multiplier > 1 else raw_avg_cost
            result.append({
                "symbol": symbol,
                "symbolId": contract.conId if contract else 0,
                "openQuantity": info.get("position", 0),
                "averageEntryPrice": avg_entry_price,
                "currentPrice": info.get("market_price", 0),
                "currentMarketValue": info.get("market_value", 0),
                "openPnl": info.get("unrealized_pnl", 0),
                "openPnlPercent": 0,
                "dayPnl": 0,
            })
        return result

    def get_account_balances(self, account_id: str = "") -> Dict:
        self.ensure_connected()
        summary = self._ibkr.get_account_summary(timeout=10.0)
        return summary

    def get_account_summary(self, timeout: float = 15.0) -> Dict:
        """Direct access to IBKR account summary."""
        self.ensure_connected()
        return self._ibkr.get_account_summary(timeout=timeout)

    # ------------------------------------------------------------------
    # Symbol lookup — matches QuestradeClient API
    # ------------------------------------------------------------------

    def _resolve_con_id(self, symbol: str) -> int:
        """Resolve and cache conId for a symbol."""
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]
        opt_info = self._parse_option_symbol_occ(symbol)
        if opt_info:
            logger.info(f"Resolving option: {symbol} -> {opt_info['underlying']} "
                       f"{opt_info['expiry_yyyymmdd']} {opt_info['strike']} {opt_info['right']}")
            contract = IBKRClient.option(
                symbol=opt_info["underlying"],
                expiry=opt_info["expiry_yyyymmdd"],
                strike=opt_info["strike"],
                right=opt_info["right"],
            )
        else:
            contract = IBKRClient.stock(symbol)
        resolved = self._ibkr.resolve_contract(contract, timeout=10.0)
        if resolved and resolved.conId:
            logger.info(f"Resolved {symbol} -> conId={resolved.conId}")
            self._symbol_cache[symbol] = resolved.conId
            return resolved.conId
        logger.warning(f"Failed to resolve conId for {symbol}")
        return 0

    def search_symbols(self, prefix: str, offset: int = 0) -> List[Dict]:
        """Search symbols — returns list of dicts with 'symbol' and 'symbolId'."""
        con_id = self._resolve_con_id(prefix)
        if con_id:
            return [{"symbol": prefix, "symbolId": con_id}]
        return []

    def get_symbol_id(self, symbol: str) -> Optional[int]:
        """Get internal conId for a symbol."""
        con_id = self._resolve_con_id(symbol)
        return con_id if con_id else None

    def get_symbol_by_name(self, symbol: str) -> Optional[Dict]:
        con_id = self._resolve_con_id(symbol)
        if con_id:
            return {"symbol": symbol, "symbolId": con_id}
        return None

    # ------------------------------------------------------------------
    # Quotes — matches QuestradeClient API
    # ------------------------------------------------------------------

    def _tick_to_questrade_quote(self, symbol: str, tick: dict) -> Dict:
        """Convert IBKR tick dict to Questrade-style quote dict."""
        # Try live prices first, fall back to delayed
        bid = tick.get("price_1", tick.get("price_66", 0))
        ask = tick.get("price_2", tick.get("price_67", 0))
        last = tick.get("price_4", tick.get("price_68", 0))
        high = tick.get("price_6", tick.get("price_72", 0))
        low = tick.get("price_7", tick.get("price_73", 0))
        close = tick.get("price_9", tick.get("price_75", 0))
        volume = tick.get("size_8", 0)
        open_price = tick.get("price_14", 0)

        # Extract greeks from option computation ticks
        greeks = {}
        for key, val in tick.items():
            if key.startswith("opt_") and isinstance(val, dict):
                greeks.update(val)

        quote = {
            "symbol": symbol,
            "symbolId": self._symbol_cache.get(symbol, 0),
            "bidPrice": bid or 0,
            "askPrice": ask or 0,
            "lastTradePrice": last or close or 0,
            "lastTradePriceTrHrs": last or close or 0,
            "openPrice": open_price or 0,
            "highPrice": high or 0,
            "lowPrice": low or 0,
            "prevDayClosePrice": close or 0,
            "volume": volume or 0,
            "lastTradeChange": (last - close) if (last and close) else 0,
            "lastTradeChangePercent": ((last - close) / close * 100) if (last and close and close != 0) else 0,
        }

        if greeks:
            quote.update({
                "delta": greeks.get("delta"),
                "gamma": greeks.get("gamma"),
                "theta": greeks.get("theta"),
                "vega": greeks.get("vega"),
                "volatility": greeks.get("impliedVol"),
                "underlyingPrice": greeks.get("undPrice"),
                "openInterest": 0,
            })

        return quote

    def get_quote_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Get a quote for a symbol by name."""
        self.ensure_connected()
        # Detect if this is an IBKR option symbol by checking for OCC-like pattern
        opt_info = self._parse_option_symbol_occ(symbol)
        if opt_info:
            contract = IBKRClient.option(
                symbol=opt_info["underlying"],
                expiry=opt_info["expiry_yyyymmdd"],
                strike=opt_info["strike"],
                right=opt_info["right"],
            )
        else:
            contract = IBKRClient.stock(symbol)

        tick = self._ibkr.get_quote(contract, timeout=10.0)
        if not tick:
            return None
        quote = self._tick_to_questrade_quote(symbol, tick)
        self._quote_cache[symbol] = quote
        return quote

    def get_quote(self, symbol_id: int) -> Optional[Dict]:
        """Get quote by symbol ID (conId)."""
        # Find symbol from cache
        for sym, cid in self._symbol_cache.items():
            if cid == symbol_id:
                return self.get_quote_by_symbol(sym)
        return None

    def get_quotes(self, ids: List[int]) -> List[Dict]:
        """Get quotes for multiple symbol IDs."""
        results = []
        for sid in ids:
            q = self.get_quote(sid)
            if q:
                results.append(q)
        return results

    # ------------------------------------------------------------------
    # Option chain — matches QuestradeClient API
    # ------------------------------------------------------------------

    def get_option_chain(self, symbol_id: int) -> Dict:
        """Get option chain for a symbol by conId."""
        # Find symbol
        symbol = None
        for sym, cid in self._symbol_cache.items():
            if cid == symbol_id:
                symbol = sym
                break
        if not symbol:
            return {"optionChain": []}
        return self.get_option_chain_by_symbol(symbol)

    def get_option_chain_by_symbol(self, symbol: str) -> Dict:
        """Get option chain expiries and strikes for an underlying."""
        self.ensure_connected()
        contract = IBKRClient.stock(symbol)
        con_id = self._resolve_con_id(symbol)
        if con_id:
            contract.conId = con_id
        params = self._ibkr.get_option_chain_params(contract, timeout=20.0)

        # Convert to Questrade format
        chain = []
        for p in params:
            for exp in p.get("expirations", []):
                # IBKR format: YYYYMMDD -> YYYY-MM-DD
                exp_formatted = f"{exp[:4]}-{exp[4:6]}-{exp[6:8]}" if len(exp) == 8 else exp
                chain.append({
                    "expiryDate": exp_formatted,
                    "strikes": p.get("strikes", []),
                    "exchange": p.get("exchange", ""),
                    "multiplier": p.get("multiplier", "100"),
                })
        return {"optionChain": chain}

    def get_option_expiries(self, symbol: str) -> List[Dict]:
        chain = self.get_option_chain_by_symbol(symbol)
        return chain.get("optionChain", [])

    def get_option_quotes(
        self,
        option_ids: Optional[List[int]] = None,
        filters: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Get option quotes — compatible with Questrade filter format."""
        results = []
        if filters:
            for f in filters:
                underlying_id = f.get("underlyingId", 0)
                # Find underlying symbol
                underlying_sym = None
                for sym, cid in self._symbol_cache.items():
                    if cid == underlying_id:
                        underlying_sym = sym
                        break
                if not underlying_sym:
                    continue

                expiry_raw = f.get("expiryDate", "")
                # Parse the expiry to YYYYMMDD
                expiry = self._parse_expiry_to_yyyymmdd(expiry_raw)

                opt_type = f.get("optionType", "")  # Call or Put
                min_strike = f.get("minstrikePrice")
                max_strike = f.get("maxstrikePrice")

                # Get chain to find available strikes
                contract = IBKRClient.stock(underlying_sym)
                contract.conId = underlying_id
                chain_params = self._ibkr.get_option_chain_params(contract, timeout=15.0)

                strikes = set()
                for p in chain_params:
                    for s in p.get("strikes", []):
                        if min_strike is not None and s < min_strike:
                            continue
                        if max_strike is not None and s > max_strike:
                            continue
                        strikes.add(s)

                rights = []
                if opt_type:
                    rights = ["C" if opt_type.lower() == "call" else "P"]
                else:
                    rights = ["C", "P"]

                for right in rights:
                    for strike in sorted(strikes):
                        opt_contract = IBKRClient.option(
                            symbol=underlying_sym,
                            expiry=expiry,
                            strike=strike,
                            right=right,
                        )
                        tick = self._ibkr.get_quote(opt_contract, timeout=8.0)
                        if tick:
                            q = self._tick_to_questrade_quote(
                                f"{underlying_sym}{expiry}{right}{strike:.0f}",
                                tick,
                            )
                            q["strikePrice"] = strike
                            q["optionType"] = "Call" if right == "C" else "Put"
                            q["expiryDate"] = expiry_raw
                            results.append(q)
        return results

    def get_atm_options(
        self,
        symbol: str,
        expiry_date: Union[str, datetime] = "",
        num_strikes: int = 5,
    ) -> Dict[str, List[Dict]]:
        """Get ATM options in Questrade format."""
        # Get underlying price
        underlying_quote = self.get_quote_by_symbol(symbol)
        if not underlying_quote:
            return {"calls": [], "puts": []}

        current_price = underlying_quote.get("lastTradePrice", 0) or underlying_quote.get("bidPrice", 0)
        if not current_price:
            return {"calls": [], "puts": []}

        # Parse expiry
        if isinstance(expiry_date, datetime):
            expiry_yyyymmdd = expiry_date.strftime("%Y%m%d")
        elif isinstance(expiry_date, str):
            expiry_yyyymmdd = expiry_date.replace("-", "")[:8]
        else:
            expiry_yyyymmdd = datetime.now().strftime("%Y%m%d")

        # Get option chain
        contract = IBKRClient.stock(symbol)
        con_id = self._resolve_con_id(symbol)
        if con_id:
            contract.conId = con_id
        chain_params = self._ibkr.get_option_chain_params(contract, timeout=20.0)

        # Find strikes near current price
        all_strikes: List[float] = []
        for p in chain_params:
            if expiry_yyyymmdd in p.get("expirations", []):
                all_strikes.extend(p.get("strikes", []))
        all_strikes = sorted(set(all_strikes))

        if not all_strikes:
            logger.warning("No strikes found for %s expiry %s", symbol, expiry_yyyymmdd)
            return {"calls": [], "puts": []}

        # Pick strikes near ATM
        atm_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - current_price))
        start = max(0, atm_idx - num_strikes)
        end = min(len(all_strikes), atm_idx + num_strikes + 1)
        selected_strikes = all_strikes[start:end]

        calls: List[Dict] = []
        puts: List[Dict] = []

        for strike in selected_strikes:
            for right, bucket in [("C", calls), ("P", puts)]:
                opt_contract = IBKRClient.option(
                    symbol=symbol,
                    expiry=expiry_yyyymmdd,
                    strike=strike,
                    right=right,
                )
                tick = self._ibkr.get_quote(opt_contract, timeout=8.0)
                if tick:
                    q = self._tick_to_questrade_quote(
                        f"{symbol}{expiry_yyyymmdd}{right}{strike:.0f}",
                        tick,
                    )
                    q["strikePrice"] = strike
                    q["strike"] = strike
                    q["optionType"] = "Call" if right == "C" else "Put"
                    q["expiryDate"] = expiry_yyyymmdd
                    q["underlyingPrice"] = current_price
                    bucket.append(q)

        return {"calls": calls, "puts": puts}

    # ------------------------------------------------------------------
    # Historical data — matches QuestradeClient API
    # ------------------------------------------------------------------

    def get_candles(
        self,
        symbol_id: int,
        start_time: datetime,
        end_time: datetime,
        interval: str = "OneDay",
    ) -> List[Dict]:
        """Get historical candles in Questrade format."""
        # Find symbol from cache
        symbol = None
        for sym, cid in self._symbol_cache.items():
            if cid == symbol_id:
                symbol = sym
                break
        if not symbol:
            return []

        # Map Questrade intervals to IBKR bar sizes
        interval_map = {
            "OneMinute": "1 min", "TwoMinutes": "2 mins",
            "ThreeMinutes": "3 mins", "FiveMinutes": "5 mins",
            "TenMinutes": "10 mins", "FifteenMinutes": "15 mins",
            "TwentyMinutes": "20 mins", "HalfHour": "30 mins",
            "OneHour": "1 hour", "TwoHours": "2 hours",
            "FourHours": "4 hours", "OneDay": "1 day",
            "OneWeek": "1 W", "OneMonth": "1 month",
        }
        bar_size = interval_map.get(interval, "1 day")

        # Calculate duration
        delta = end_time - start_time
        if delta.days <= 1:
            duration = "1 D"
        elif delta.days <= 7:
            duration = f"{delta.days} D"
        elif delta.days <= 30:
            duration = "1 M"
        elif delta.days <= 365:
            duration = f"{delta.days // 30} M"
        else:
            duration = "1 Y"

        contract = IBKRClient.stock(symbol)
        end_dt = end_time.strftime("%Y%m%d-%H:%M:%S")

        bars = self._ibkr.get_historical_bars(
            contract,
            end_datetime=end_dt,
            duration=duration,
            bar_size=bar_size,
            what_to_show="TRADES",
            use_rth=1,
            timeout=30.0,
        )

        # Convert to Questrade candle format
        candles = []
        for b in bars:
            candles.append({
                "start": b.get("date", ""),
                "end": "",
                "open": b.get("open", 0),
                "high": b.get("high", 0),
                "low": b.get("low", 0),
                "close": b.get("close", 0),
                "volume": b.get("volume", 0),
                "VWAP": b.get("wap", 0),
            })
        return candles

    # ------------------------------------------------------------------
    # Orders — matches QuestradeClient API
    # ------------------------------------------------------------------

    def place_order(
        self,
        account_id: str = "",
        symbol_id: int = 0,
        quantity: int = 0,
        is_buy: bool = True,
        order_type: str = "Limit",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "Day",
        is_all_or_none: bool = False,
        **kwargs,
    ) -> Dict:
        """Place an order via IBKR, returning Questrade-compatible response."""
        self.ensure_connected()
        # Find symbol from cache
        symbol = None
        for sym, cid in self._symbol_cache.items():
            if cid == symbol_id:
                symbol = sym
                break

        if not symbol:
            raise ValueError(f"Symbol with conId={symbol_id} not resolved. Call get_symbol_id first.")

        # Detect option vs stock
        opt_info = self._parse_option_symbol_occ(symbol)
        if opt_info:
            contract = IBKRClient.option(
                symbol=opt_info["underlying"],
                expiry=opt_info["expiry_yyyymmdd"],
                strike=opt_info["strike"],
                right=opt_info["right"],
            )
        else:
            contract = IBKRClient.stock(symbol)
            contract.conId = symbol_id

        action = "BUY" if is_buy else "SELL"

        if order_type == "Market":
            order = self._ibkr.market_order(action, quantity)
        elif order_type == "Limit" and limit_price:
            order = self._ibkr.limit_order(action, quantity, limit_price)
        elif order_type == "Stop" and stop_price:
            order = self._ibkr.stop_order(action, quantity, stop_price)
        else:
            order = self._ibkr.market_order(action, quantity)

        oid = self._ibkr.place_order(contract, order, wait_seconds=1.0)
        return {
            "orderId": oid,
            "orderState": "Pending",
        }

    def cancel_order(self, account_id: str, order_id: int) -> Dict:
        """Cancel an order."""
        self.ensure_connected()
        self._ibkr.cancel_order(order_id)
        return {"orderId": order_id, "status": "Canceled"}

    def cancel_all_open_orders(self) -> None:
        """Cancel all open orders on IBKR (global cancel)."""
        self.ensure_connected()
        self._ibkr.cancel_all_orders()
        logger.info("Sent global cancel for all open orders")
        time.sleep(2)  # Give TWS time to process cancellations

    def get_account_orders(self, account_id: str = "") -> List[Dict]:
        """Get orders in Questrade format.
        
        Merges open_orders (active orders) with order_statuses (filled/cancelled
        orders that TWS may have removed from the active list). This ensures
        filled orders are never missed between poll cycles.
        """
        self.ensure_connected()
        orders = self._ibkr.wrapper.open_orders
        statuses = self._ibkr.wrapper.order_statuses
        result = []
        seen_oids = set()
        
        # First pass: orders still in the active list
        for oid, info in orders.items():
            contract: Contract = info.get("contract", Contract())
            order = info.get("order")
            state = info.get("state")
            status_info = statuses.get(oid, {})
            seen_oids.add(oid)

            result.append({
                "id": oid,
                "symbol": contract.symbol if contract else "",
                "symbolId": contract.conId if contract else 0,
                "totalQuantity": order.totalQuantity if order else 0,
                "filledQuantity": status_info.get("filled", 0),
                "side": order.action if order else "",
                "orderType": order.orderType if order else "",
                "limitPrice": order.lmtPrice if order else None,
                "stopPrice": order.auxPrice if order else None,
                "avgExecPrice": status_info.get("avg_fill_price", 0),
                "state": status_info.get("status", state.status if state else "Unknown"),
                "timeInForce": order.tif if order else "DAY",
                "creationTime": "",
                "updateTime": "",
                "commissionCharged": 0,
            })
        
        # Second pass: orders that have a status update (filled/cancelled)
        # but were already removed from open_orders by TWS
        for oid, status_info in statuses.items():
            if oid in seen_oids:
                continue
            status = status_info.get("status", "")
            if status in ("Filled", "Cancelled", "Inactive", "ApiCancelled"):
                result.append({
                    "id": oid,
                    "symbol": status_info.get("symbol", ""),
                    "symbolId": 0,
                    "totalQuantity": status_info.get("filled", 0) + status_info.get("remaining", 0),
                    "filledQuantity": status_info.get("filled", 0),
                    "side": "",
                    "orderType": "",
                    "limitPrice": None,
                    "stopPrice": None,
                    "avgExecPrice": status_info.get("avg_fill_price", 0),
                    "state": status,
                    "timeInForce": "DAY",
                    "creationTime": "",
                    "updateTime": "",
                    "commissionCharged": 0,
                })
        return result

    def get_executions(self, account_id: str = "") -> List[Dict]:
        """Get execution reports from IBKR for the current session.

        Returns a list of dicts, each with:
          symbol, trade_symbol, side, shares, price, time, order_id, exec_id

        ``trade_symbol`` is the engine-internal format (e.g. SPY20260310C680)
        derived from the IBKR contract info.
        """
        self.ensure_connected()
        raw = self._ibkr.get_executions(timeout=15.0)
        result = []
        for ex in raw:
            # Build internal trade symbol from IBKR contract fields
            trade_sym = ex.get("symbol", "")
            if ex.get("secType") == "OPT" and ex.get("expiry"):
                strike = ex.get("strike", 0)
                strike_str = str(int(strike)) if strike == int(strike) else str(strike)
                trade_sym = f"{ex['symbol']}{ex['expiry']}{ex.get('right', '')}{strike_str}"
            result.append({
                "symbol": ex.get("symbol", ""),
                "trade_symbol": trade_sym,
                "side": ex.get("side", ""),
                "shares": ex.get("shares", 0),
                "price": ex.get("price", 0),
                "time": ex.get("time", ""),
                "order_id": ex.get("order_id", 0),
                "exec_id": ex.get("exec_id", ""),
                "acct_number": ex.get("acct_number", ""),
            })
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_authenticated(self) -> bool:
        return self.is_connected

    def test_connection(self) -> bool:
        try:
            return self.is_connected
        except Exception:
            return False

    def get_server_time(self) -> datetime:
        return datetime.now()

    # ------------------------------------------------------------------
    # Option symbol parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_option_symbol_occ(symbol: str) -> Optional[Dict]:
        """
        Parse option symbols. Supports:
         - Questrade format: SPY18Feb26P690.00
         - OCC format: SPY   260218P00690000
        Returns dict with underlying, expiry_yyyymmdd, strike, right or None.
        """
        # Questrade-style: SPY18Feb26P690.00
        m = re.match(
            r'^([A-Z]+)(\d{2})([A-Za-z]{3})(\d{2})([CP])(\d+\.?\d*)$',
            symbol,
        )
        if m:
            underlying, day, mon_str, yr, right, strike = m.groups()
            months = {
                "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
            }
            month = months.get(mon_str, "01")
            return {
                "underlying": underlying,
                "expiry_yyyymmdd": f"20{yr}{month}{day}",
                "strike": float(strike),
                "right": right,
            }

        # Generic fallback: SYMBOL + digits + C/P + digits
        m2 = re.match(r'^([A-Z]+)(\d{8})([CP])(\d+)$', symbol)
        if m2:
            underlying, expiry, right, strike_raw = m2.groups()
            return {
                "underlying": underlying,
                "expiry_yyyymmdd": expiry,
                "strike": float(strike_raw),
                "right": right,
            }

        return None

    @staticmethod
    def _contract_to_trade_symbol(contract: Contract) -> Optional[str]:
        """Convert an IBKR Contract to our internal trade symbol format.
        
        Examples:
            OPT contract with symbol=SPY, lastTradeDateOrContractMonth=20260310,
            right=C, strike=680.0  =>  "SPY20260310C680"
            
            STK contract with symbol=SPY  =>  "SPY"
        """
        if contract.secType == "OPT" and contract.lastTradeDateOrContractMonth:
            strike = int(contract.strike) if contract.strike == int(contract.strike) else contract.strike
            return f"{contract.symbol}{contract.lastTradeDateOrContractMonth}{contract.right}{strike}"
        return contract.symbol or None

    @staticmethod
    def _parse_expiry_to_yyyymmdd(expiry_raw: str) -> str:
        """Convert various expiry formats to YYYYMMDD."""
        if not expiry_raw:
            return datetime.now().strftime("%Y%m%d")
        # Already YYYYMMDD
        clean = expiry_raw.replace("-", "")[:8]
        if len(clean) == 8 and clean.isdigit():
            return clean
        # ISO with T
        if "T" in expiry_raw:
            date_part = expiry_raw.split("T")[0].replace("-", "")
            if len(date_part) == 8:
                return date_part
        return datetime.now().strftime("%Y%m%d")


# ------------------------------------------------------------------
# Factory function — drop-in for create_questrade_client
# ------------------------------------------------------------------

def create_ibkr_client(
    host: str = None,
    port: int = None,
    client_id: int = 0,
    account: str = "",
    **kwargs,
) -> IBKRAdapter:
    if host is None:
        host = os.environ.get("IBKR_HOST", "127.0.0.1")
    if port is None:
        env_port = os.environ.get("IBKR_PAPER_PORT")
        if env_port:
            port = int(env_port)
        else:
            from config import defaults as cfg
            port = cfg.ibkr_paper_port()
    """
    Factory function to create and connect an IBKR adapter.
    Drop-in replacement for create_questrade_client().

    Args:
        host: TWS host address
        port: TWS port (7497=paper, 7496=live)
        client_id: TWS client ID
        account: Account string (optional, auto-detected)

    Returns:
        Connected IBKRAdapter instance
    """
    adapter = IBKRAdapter(
        host=host,
        port=port,
        client_id=client_id,
        account=account,
    )
    adapter.connect()
    return adapter
