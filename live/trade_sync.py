"""
Trade Sync — Merge historical trades from multiple sources into the live DB.

Sources:
    1. IBKR session fills  (reqExecutions + reqCompletedOrders merge on IBKRClient)
    2. External SQLite DBs     (e.g. previous Docker volumes, GCloud DBs)
    3. CSV trade exports       (e.g. gcloud_trades.csv)

Deduplication:
    Trades are matched by (symbol, entry_time) to prevent duplicates.
    If a trade with the same symbol+entry_time already exists, it's skipped.

Usage:
    from live.trade_sync import TradeSync
    syncer = TradeSync(trade_db)
    syncer.sync_all(ibkr_client=client)          # Full sync at startup
    syncer.sync_from_ibkr(client)                # After each trade
"""
import csv
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"


class TradeSync:
    """
    Robust trade synchronization from multiple data sources into the live DB.
    
    Merges IBKR executions, external DBs, and CSV exports into the
    canonical trade database, with deduplication.
    """

    def __init__(self, trade_db):
        """
        Args:
            trade_db: TradeDatabase instance (the target DB to sync into)
        """
        self.trade_db = trade_db
        self._existing_keys: Optional[Set[str]] = None

    # ------------------------------------------------------------------
    # Symbol & time normalization
    # ------------------------------------------------------------------

    _MONTH_MAP = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
        "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
    }

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize an option symbol to canonical OCC-style format.

        Converts both Questrade format (SPY18Mar26P664.00) and
        IBKR format (SPY20260318P664) to a common form:
            SPY20260318P664

        Non-option symbols are returned unchanged.
        """
        if not symbol:
            return symbol

        # Questrade-style: SPY18Mar26P664.00 (format: underlying+DD+Mon+YY+Right+strike)
        # Example: SPY18Mar26P664.00 = SPY + day(18) + month(Mar) + year(26) + P + strike(664)
        m = re.match(
            r'^([A-Z]+)(\d{2})([A-Za-z]{3})(\d{2})([CP])(\d+\.?\d*)$',
            symbol,
        )
        if m:
            underlying, day, mon_str, year, right, strike = m.groups()
            month = TradeSync._MONTH_MAP.get(mon_str, "01")
            strike_clean = str(int(float(strike)))  # "664.00" -> "664"
            return f"{underlying}20{year}{month}{day}{right}{strike_clean}"

        # IBKR / OCC: SPY20260318P664 — already canonical, just strip decimals
        m2 = re.match(r'^([A-Z]+)(\d{8})([CP])(\d+\.?\d*)$', symbol)
        if m2:
            underlying, expiry, right, strike = m2.groups()
            strike_clean = str(int(float(strike)))
            return f"{underlying}{expiry}{right}{strike_clean}"

        return symbol

    @staticmethod
    def _normalize_time(entry_time: str) -> str:
        """Normalize a timestamp to 'YYYY-MM-DDTHH:MM' for dedup.

        Handles:
            ISO:  2026-03-18T10:43:57  ->  2026-03-18T10:43
            IBKR: 20260318  14:43:58   ->  2026-03-18T14:43
        """
        if not entry_time:
            return ""
        # IBKR execution time: "20260318  14:43:58"
        m = re.match(r'^(\d{4})(\d{2})(\d{2})\s+(\d{2}):(\d{2})', entry_time)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}"
        # ISO format: truncate to minute
        return entry_time[:16]

    # ------------------------------------------------------------------
    # Dedup helpers
    # ------------------------------------------------------------------

    def _get_existing_trade_keys(self) -> Set[str]:
        """
        Build a set of (normalized_symbol, entry_time_prefix) keys for all
        trades already in the DB.  Used to skip duplicates during import.
        
        Symbols are normalized so Questrade and IBKR formats match.
        entry_time is truncated to the minute for fuzzy matching.
        """
        cursor = self.trade_db.conn.cursor()
        cursor.execute("SELECT symbol, entry_time FROM trades")
        keys = set()
        for row in cursor.fetchall():
            symbol = row[0] or ""
            entry_time = row[1] or ""
            key = self._make_key(symbol, entry_time)
            keys.add(key)
        return keys

    @staticmethod
    def _make_key(symbol: str, entry_time: str) -> str:
        """Create a dedup key from normalized symbol + normalized time."""
        norm_sym = TradeSync._normalize_symbol(symbol)
        norm_t = TradeSync._normalize_time(entry_time)
        return f"{norm_sym}|{norm_t}"

    def _refresh_keys(self):
        """Reload existing keys from DB."""
        self._existing_keys = self._get_existing_trade_keys()

    def _is_duplicate(self, symbol: str, entry_time: str) -> bool:
        """Check if a trade with this symbol+entry_time already exists."""
        if self._existing_keys is None:
            self._refresh_keys()
        return self._make_key(symbol, entry_time) in self._existing_keys

    @staticmethod
    def _parse_db_path_list(raw: str) -> List[str]:
        """Parse comma/semicolon-separated db path list."""
        if not raw:
            return []
        parts = re.split(r"[;,]", raw)
        return [p.strip() for p in parts if p and p.strip()]

    def _get_existing_flex_exec_ids(self) -> Set[str]:
        """Return exec IDs already imported from Flex ledger rows."""
        cursor = self.trade_db.conn.cursor()
        cursor.execute(
            "SELECT notes FROM trades WHERE notes LIKE 'imported:ibkr_flex_exec:%'"
        )
        ids: Set[str] = set()
        for row in cursor.fetchall():
            notes = (row[0] or "").strip()
            if notes.startswith("imported:ibkr_flex_exec:"):
                ids.add(notes.split("imported:ibkr_flex_exec:", 1)[1])
        return ids

    def _import_flex_realized_rows(self, flex_trades: List[Dict[str, Any]]) -> int:
        """
        Import Flex rows using IBKR's own realized PnL per row.

        This keeps dashboard/accounting totals aligned with Flex XML and avoids
        reconstruction drift from local FIFO pairing.
        """
        existing_exec_ids = self._get_existing_flex_exec_ids()
        imported = 0

        def _safe_float(v: Any, default: float = 0.0) -> float:
            try:
                return float(v) if v not in ("", None, "-") else default
            except (ValueError, TypeError):
                return default

        def _safe_int(v: Any, default: int = 0) -> int:
            try:
                if v in ("", None, "-"):
                    return default
                return int(float(v))
            except (ValueError, TypeError):
                return default

        rows = sorted(
            flex_trades,
            key=lambda t: t.get("dateTime", t.get("tradeDate", "")),
        )
        for ft in rows:
            pnl = _safe_float(ft.get("fifoPnlRealized", ft.get("realizedPnl", 0)))
            # Keep ledger focused on realized events; zero rows are not needed for totals.
            if abs(pnl) < 1e-9:
                continue

            exec_id = (ft.get("ibExecID", ft.get("execID", "")) or "").strip()
            if not exec_id:
                # Stable fallback key if exec ID is missing.
                exec_id = (
                    f"{ft.get('orderID', ft.get('ibOrderID', '0'))}|"
                    f"{ft.get('dateTime', ft.get('tradeDate', ''))}|"
                    f"{ft.get('symbol', '')}|"
                    f"{ft.get('quantity', '')}|"
                    f"{ft.get('tradePrice', ft.get('price', ''))}"
                )

            if exec_id in existing_exec_ids:
                continue

            asset = (ft.get("assetCategory", "") or "").upper()

            # Skip non-option trades (STK, FUT, etc.) — we only trade 0DTE options
            if asset and asset != "OPT":
                continue

            side_raw = (ft.get("buySell", ft.get("side", "")) or "").upper()
            action = "buy" if side_raw in ("BUY", "BOT") else "sell"
            qty = abs(_safe_int(ft.get("quantity", 0), 0))
            price = _safe_float(ft.get("tradePrice", ft.get("price", 0)), 0.0)
            dt = ft.get("dateTime", ft.get("tradeDate", "")) or ""
            sym = (ft.get("symbol", "") or "").strip()

            put_call = (ft.get("putCall", "") or "").upper()
            strike_val = _safe_float(ft.get("strike", 0), 0.0)
            expiry = ft.get("expiry", ft.get("lastTradeDateOrContractMonth", "")) or ""
            option_type = "call" if put_call.startswith("C") else ("put" if put_call.startswith("P") else "")
            trade_type = "option" if asset == "OPT" else "stock"

            if trade_type == "option" and sym and put_call and expiry:
                strike_str = str(int(strike_val)) if strike_val else ""
                symbol = f"{sym}{expiry}{put_call[:1]}{strike_str}"
            else:
                symbol = sym or "IBKR_ADJUSTMENT"

            commission = abs(_safe_float(ft.get("ibCommission", ft.get("commission", 0)), 0.0))
            qty_eff = qty if qty > 0 else 1
            price_eff = price if price > 0 else 1.0
            mult = 100 if trade_type == "option" else 1
            denom = price_eff * qty_eff * mult
            pnl_pct = (pnl / denom * 100.0) if denom > 0 else 0.0

            trade = self._make_trade(
                symbol=symbol,
                underlying=sym,
                trade_type=trade_type,
                option_type=option_type or None,
                strike=(strike_val if strike_val > 0 else None),
                expiration=expiry or None,
                action=action,
                quantity=qty_eff,
                entry_price=price_eff,
                entry_time=dt,
                exit_price=price_eff,
                exit_time=dt,
                pnl=pnl,
                pnl_percent=pnl_pct,
                commission=commission,
                status="closed",
                entry_order_id=_safe_int(ft.get("orderID", ft.get("ibOrderID", 0)), 0) or None,
                exit_order_id=_safe_int(ft.get("orderID", ft.get("ibOrderID", 0)), 0) or None,
                account_id=ft.get("accountId", "") or "",
                notes=f"imported:ibkr_flex_exec:{exec_id}",
            )

            try:
                tid = self.trade_db.insert_trade(trade)
                imported += 1
                existing_exec_ids.add(exec_id)
                logger.info(
                    "TradeSync: Imported Flex ledger row #%s %s PnL=$%.2f",
                    tid,
                    symbol,
                    pnl,
                )
            except Exception as e:
                logger.warning(f"TradeSync: Failed to import Flex row {symbol}: {e}")

        if imported:
            logger.info("TradeSync: Imported %d realized rows from IBKR Flex ledger", imported)
        return imported

    # ------------------------------------------------------------------
    # IBKR execution sync
    # ------------------------------------------------------------------

    def sync_from_ibkr(
        self,
        ibkr_client=None,
        lookback_days: int = 7,
        full_refresh: bool = False,
    ) -> int:
        """
        Fetch IBKR execution history and import closed round-trip trades.

        Priority:
          1. Flex Web Service (real historical trades — survives TWS restarts)
          2. TWS session executions (fallback — only current gateway session)

        Session fills may be per-fill (reqExecutions) or one row per order
        (reqCompletedOrders).  We pair BUY/SELL
        fills on the same symbol (localSymbol) into round-trip trades.

        Supports both raw IBKRClient and IBKRAdapter wrappers.

        Args:
            ibkr_client: Connected IBKRClient or IBKRAdapter instance
            lookback_days: How many days back to fetch (max ~7 for session execs)

        Returns:
            Number of new trades imported
        """
        # --- Try Flex first (historical, survives restarts) ---
        flex_imported = self._sync_from_flex(full_refresh=full_refresh)
        if flex_imported > 0:
            return flex_imported

        if ibkr_client is None:
            logger.debug("TradeSync: No IBKR client, skipping execution sync")
            return 0

        try:
            # Try calling get_executions — adapter takes no kwargs,
            # raw client accepts time_filter/timeout
            try:
                executions = ibkr_client.get_executions()
            except TypeError:
                # Fallback: raw client needs kwargs
                since = datetime.now() - timedelta(days=lookback_days)
                time_filter = since.strftime("%Y%m%d-00:00:00")
                if hasattr(ibkr_client, "get_merged_session_fills"):
                    executions = ibkr_client.get_merged_session_fills(
                        time_filter=time_filter,
                        timeout=20.0,
                    )
                else:
                    executions = ibkr_client.get_executions(
                        time_filter=time_filter, timeout=20.0
                    )

            if not executions:
                logger.info("TradeSync: No IBKR executions found")
                return 0

            logger.info(f"TradeSync: Fetched {len(executions)} IBKR executions")
            return self._pair_and_import_executions(executions)

        except Exception as e:
            logger.warning(f"TradeSync: IBKR execution sync failed: {e}")
            return 0

    def _sync_from_flex(self, full_refresh: bool = False) -> int:
        """
        Fetch historical trades from IBKR Flex Web Service and import them.

        Requires IBKR_FLEX_TOKEN and IBKR_FLEX_QUERY_ID env vars.
        Returns 0 if Flex is not configured or fails.
        """
        try:
            token = os.environ.get("IBKR_FLEX_TOKEN", "")
            query_id = os.environ.get("IBKR_FLEX_QUERY_ID", "")
            if not token or not query_id:
                return 0

            from clients.ibkr_flex import IBKRFlexClient
            flex = IBKRFlexClient(token=token)
            flex_trades = flex.fetch_trades(int(query_id))
            if not flex_trades:
                logger.info("TradeSync: No Flex trades returned")
                return 0

            logger.info(f"TradeSync: Fetched {len(flex_trades)} trades from IBKR Flex")

            # Primary path: import IBKR's authoritative realized ledger rows.
            # If realized fields exist, do NOT fall back to FIFO pairing on later runs;
            # otherwise duplicates with different semantics can be inserted.
            has_realized_fields = any(
                ("fifoPnlRealized" in ft) or ("realizedPnl" in ft)
                for ft in flex_trades
            )
            if has_realized_fields and full_refresh:
                cursor = self.trade_db.conn.cursor()
                cursor.execute(
                    "DELETE FROM trades WHERE notes='imported:ibkr_executions' "
                    "OR notes LIKE 'imported:ibkr_flex_exec:%'"
                )
                self.trade_db.conn.commit()
                self._existing_keys = None
                logger.info(
                    "TradeSync: Full refresh enabled — cleared previous IBKR imported rows"
                )

            if has_realized_fields:
                return self._import_flex_realized_rows(flex_trades)

            # Fallback path: normalize and pair only when realized fields are unavailable.
            normalised = []
            for ft in flex_trades:
                buy_sell = ft.get("buySell", ft.get("side", ""))
                if buy_sell.upper() in ("BUY", "BOT"):
                    side = "BOT"
                elif buy_sell.upper() in ("SELL", "SLD"):
                    side = "SLD"
                else:
                    side = buy_sell

                asset = ft.get("assetCategory", "")
                sym = ft.get("symbol", "")
                put_call = ft.get("putCall", "")
                strike = ft.get("strike", "")
                expiry = ft.get("expiry", ft.get("lastTradeDateOrContractMonth", ""))
                if asset == "OPT" and put_call and expiry:
                    strike_str = str(int(float(strike))) if strike else ""
                    right = put_call[0].upper() if put_call else ""
                    trade_sym = f"{sym}{expiry}{right}{strike_str}"
                else:
                    trade_sym = sym

                def _safe_float(v, default=0.0):
                    try:
                        return float(v) if v else default
                    except (ValueError, TypeError):
                        return default

                normalised.append({
                    "symbol": sym,
                    "trade_symbol": trade_sym,
                    "side": side,
                    "shares": abs(int(float(ft.get("quantity", 0)))),
                    "price": float(ft.get("tradePrice", ft.get("price", 0))),
                    "time": ft.get("dateTime", ft.get("tradeDate", "")),
                    "order_id": int(ft.get("orderID", ft.get("ibOrderID", 0)) or 0),
                    "exec_id": ft.get("ibExecID", ft.get("execID", "")),
                    "acct_number": ft.get("accountId", ""),
                    "right": put_call[0].upper() if put_call else "",
                    "strike": float(strike) if strike else 0,
                    "expiry": expiry,
                    "secType": asset or ft.get("secType", ""),
                    "realized_pnl": _safe_float(ft.get("fifoPnlRealized", ft.get("realizedPnl", 0))),
                    "commission": _safe_float(ft.get("ibCommission", ft.get("commission", 0))),
                })

            return self._pair_and_import_executions(normalised)

        except Exception as e:
            logger.warning(f"TradeSync: Flex sync failed: {e}")
            return 0

    def _pair_and_import_executions(self, executions: List[Dict]) -> int:
        """
        Pair IBKR executions into round-trip trades and import them.

        Groups by trade_symbol/localSymbol, pairs BUY with SELL chronologically.
        Handles both IBKRAdapter output (trade_symbol, side=BOT/SLD) and
        raw IBKRClient output (localSymbol, right, strike, expiry).
        """
        self._refresh_keys()

        # Group by unique option contract identifier
        # Adapter uses "trade_symbol", raw uses "localSymbol"
        by_symbol: Dict[str, List[Dict]] = {}
        for ex in executions:
            sym = ex.get("trade_symbol") or ex.get("localSymbol") or ex.get("symbol", "")
            by_symbol.setdefault(sym, []).append(ex)

        imported = 0
        for sym, fills in by_symbol.items():
            # Sort by time
            fills.sort(key=lambda f: f.get("time", ""))

            buys = [f for f in fills if f.get("side", "").upper() == "BOT"]
            sells = [f for f in fills if f.get("side", "").upper() == "SLD"]

            # Pair buys with sells (FIFO)
            pairs = min(len(buys), len(sells))
            for i in range(pairs):
                buy = buys[i]
                sell = sells[i]

                entry_time = buy.get("time", "")
                exit_time = sell.get("time", "")
                entry_price = float(buy.get("price", 0))
                exit_price = float(sell.get("price", 0))
                quantity = int(buy.get("shares", 0))

                if self._is_duplicate(sym, entry_time):
                    continue

                # Determine option type from contract info or trade_symbol
                right = buy.get("right", "")
                option_type = ""
                if right == "C":
                    option_type = "call"
                elif right == "P":
                    option_type = "put"
                elif "C" in sym[3:]:  # e.g. SPY20260318C669
                    option_type = "call"
                elif "P" in sym[3:]:
                    option_type = "put"

                strike = float(buy.get("strike", 0))
                expiry = buy.get("expiry", "")
                underlying = buy.get("symbol", "SPY")

                # Parse strike/expiry from trade_symbol if not in raw fields
                # Format: SPY20260318C669
                if (not strike or not expiry) and len(sym) > 3:
                    m = re.match(r'([A-Z]+)(\d{8})([CP])(\d+)', sym)
                    if m:
                        underlying = m.group(1)
                        expiry = expiry or m.group(2)
                        strike = strike or float(m.group(4))

                # Use IBKR's authoritative realized PnL when available
                ibkr_pnl_buy = float(buy.get("realized_pnl", 0) or 0)
                ibkr_pnl_sell = float(sell.get("realized_pnl", 0) or 0)
                ibkr_pnl = ibkr_pnl_buy + ibkr_pnl_sell

                if ibkr_pnl != 0:
                    pnl = ibkr_pnl
                else:
                    # Fallback: calculate from prices with correct multiplier
                    is_option = bool(option_type) or buy.get("secType") == "OPT"
                    multiplier = 100 if is_option else 1
                    pnl = (exit_price - entry_price) * quantity * multiplier

                pnl_pct = (pnl / (entry_price * quantity * 100)) * 100 if entry_price > 0 and quantity > 0 else 0

                # Use IBKR commission when available
                ibkr_comm_buy = abs(float(buy.get("commission", 0) or 0))
                ibkr_comm_sell = abs(float(sell.get("commission", 0) or 0))
                commission = ibkr_comm_buy + ibkr_comm_sell

                trade = self._make_trade(
                    symbol=sym,
                    underlying=underlying,
                    option_type=option_type,
                    strike=strike,
                    expiration=expiry,
                    action="buy",
                    quantity=quantity,
                    entry_price=entry_price,
                    entry_time=entry_time,
                    exit_price=exit_price,
                    exit_time=exit_time,
                    pnl=pnl,
                    pnl_percent=pnl_pct,
                    commission=commission,
                    status="closed",
                    entry_order_id=buy.get("order_id"),
                    exit_order_id=sell.get("order_id"),
                    account_id=buy.get("acct_number", ""),
                    notes="imported:ibkr_executions",
                )
                try:
                    tid = self.trade_db.insert_trade(trade)
                    key = self._make_key(sym, entry_time)
                    self._existing_keys.add(key)
                    imported += 1
                    logger.info(f"TradeSync: Imported IBKR trade #{tid} {sym} PnL=${pnl:.2f}")
                except Exception as e:
                    logger.warning(f"TradeSync: Failed to import IBKR trade {sym}: {e}")

        if imported:
            logger.info(f"TradeSync: Imported {imported} trades from IBKR executions")
        return imported

    # ------------------------------------------------------------------
    # External DB sync
    # ------------------------------------------------------------------

    def sync_from_db(self, db_path: str) -> int:
        """
        Import closed trades from another SQLite database.

        Reads from the standard `trades` table schema and inserts
        missing trades into the current DB.

        Args:
            db_path: Path to the external SQLite database

        Returns:
            Number of new trades imported
        """
        if not db_path or not os.path.exists(db_path):
            logger.debug(f"TradeSync: External DB not found: {db_path}")
            return 0

        try:
            ext_conn = sqlite3.connect(db_path, timeout=10)
            ext_conn.row_factory = sqlite3.Row
            cursor = ext_conn.cursor()

            # Verify trades table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
            )
            if not cursor.fetchone():
                logger.debug(f"TradeSync: No trades table in {db_path}")
                ext_conn.close()
                return 0

            cursor.execute("""
                SELECT * FROM trades 
                WHERE status = 'closed' AND pnl IS NOT NULL
                ORDER BY exit_time
            """)
            rows = cursor.fetchall()
            ext_conn.close()

            if not rows:
                logger.info(f"TradeSync: No closed trades in {db_path}")
                return 0

            self._refresh_keys()
            imported = 0

            for row in rows:
                row_dict = dict(row)
                symbol = row_dict.get("symbol", "")
                entry_time = row_dict.get("entry_time", "")

                if self._is_duplicate(symbol, entry_time):
                    continue

                trade = self._make_trade(
                    symbol=symbol,
                    underlying=row_dict.get("underlying", ""),
                    option_type=row_dict.get("option_type", ""),
                    strike=row_dict.get("strike"),
                    expiration=row_dict.get("expiration", ""),
                    action=row_dict.get("action", "buy"),
                    quantity=row_dict.get("quantity", 0),
                    entry_price=row_dict.get("entry_price", 0),
                    entry_time=entry_time,
                    exit_price=row_dict.get("exit_price"),
                    exit_time=row_dict.get("exit_time", ""),
                    pnl=row_dict.get("pnl", 0),
                    pnl_percent=row_dict.get("pnl_percent", 0),
                    status="closed",
                    commission=row_dict.get("commission", 0),
                    delta=row_dict.get("delta"),
                    gamma=row_dict.get("gamma"),
                    theta=row_dict.get("theta"),
                    vega=row_dict.get("vega"),
                    iv=row_dict.get("iv"),
                    underlying_price_entry=row_dict.get("underlying_price_entry"),
                    underlying_price_exit=row_dict.get("underlying_price_exit"),
                    entry_order_id=row_dict.get("entry_order_id"),
                    exit_order_id=row_dict.get("exit_order_id"),
                    strategy_name=row_dict.get("strategy_name", ""),
                    strategy_params=row_dict.get("strategy_params", ""),
                    account_id=row_dict.get("account_id", ""),
                    notes=f"imported:db:{Path(db_path).name}",
                )
                try:
                    tid = self.trade_db.insert_trade(trade)
                    key = self._make_key(symbol, entry_time)
                    self._existing_keys.add(key)
                    imported += 1
                    logger.info(
                        f"TradeSync: Imported DB trade #{tid} {symbol} "
                        f"PnL=${row_dict.get('pnl', 0):.2f} from {Path(db_path).name}"
                    )
                except Exception as e:
                    logger.warning(f"TradeSync: Failed to import DB trade {symbol}: {e}")

            if imported:
                logger.info(f"TradeSync: Imported {imported} trades from {Path(db_path).name}")
            return imported

        except Exception as e:
            logger.warning(f"TradeSync: DB sync failed for {db_path}: {e}")
            return 0

    # ------------------------------------------------------------------
    # CSV sync
    # ------------------------------------------------------------------

    def sync_from_csv(self, csv_path: str) -> int:
        """
        Import trades from a CSV file.

        Expected columns (flexible — maps common names):
            symbol, entry_time, exit_time, entry_price, exit_price,
            quantity, pnl, option_type, action, status

        Args:
            csv_path: Path to the CSV file

        Returns:
            Number of new trades imported
        """
        if not csv_path or not os.path.exists(csv_path):
            return 0

        try:
            with open(csv_path, "r", newline="") as f:
                # Check if file is empty
                first_line = f.readline().strip()
                if not first_line:
                    return 0
                f.seek(0)
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                return 0

            self._refresh_keys()
            imported = 0

            for row in rows:
                symbol = row.get("symbol", "")
                entry_time = row.get("entry_time", "")
                pnl_str = row.get("pnl", "0")

                if not symbol or not entry_time:
                    continue

                try:
                    pnl = float(pnl_str) if pnl_str else 0
                except ValueError:
                    pnl = 0

                if self._is_duplicate(symbol, entry_time):
                    continue

                # Parse optional fields
                def safe_float(val):
                    try:
                        return float(val) if val else None
                    except (ValueError, TypeError):
                        return None

                def safe_int(val):
                    try:
                        return int(val) if val else 0
                    except (ValueError, TypeError):
                        return 0

                trade = self._make_trade(
                    symbol=symbol,
                    underlying=row.get("underlying", "SPY"),
                    option_type=row.get("option_type", ""),
                    strike=safe_float(row.get("strike")),
                    expiration=row.get("expiration", ""),
                    action=row.get("action", "buy"),
                    quantity=safe_int(row.get("quantity", "1")),
                    entry_price=safe_float(row.get("entry_price")) or 0,
                    entry_time=entry_time,
                    exit_price=safe_float(row.get("exit_price")),
                    exit_time=row.get("exit_time", ""),
                    pnl=pnl,
                    pnl_percent=safe_float(row.get("pnl_percent")),
                    status=row.get("status", "closed"),
                    account_id=row.get("account_id", ""),
                    notes=f"imported:csv:{Path(csv_path).name}",
                )
                try:
                    tid = self.trade_db.insert_trade(trade)
                    key = self._make_key(symbol, entry_time)
                    self._existing_keys.add(key)
                    imported += 1
                except Exception as e:
                    logger.warning(f"TradeSync: Failed to import CSV trade {symbol}: {e}")

            if imported:
                logger.info(f"TradeSync: Imported {imported} trades from {Path(csv_path).name}")
            return imported

        except Exception as e:
            logger.warning(f"TradeSync: CSV sync failed for {csv_path}: {e}")
            return 0

    # ------------------------------------------------------------------
    # Full sync (startup)
    # ------------------------------------------------------------------

    def sync_all(
        self,
        ibkr_client=None,
        external_db_paths: Optional[List[str]] = None,
        csv_paths: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Run all sync sources.  Call at engine/dashboard startup.

        Auto-discovers:
            - data/*.db files (external DBs)
            - data/*.csv files (trade exports)

        Args:
            ibkr_client: Optional connected IBKR client
            external_db_paths: Additional DB paths to sync from
            csv_paths: Additional CSV paths to sync from

        Returns:
            Dict with import counts per source
        """
        results = {}
        total = 0

        logger.info("TradeSync: Starting full sync...")

        # 1. IBKR trades — Flex first (historical), then session executions
        n = self.sync_from_ibkr(ibkr_client, full_refresh=True)
        results["ibkr"] = n
        total += n

        # 2. Auto-discover external DBs in data/
        db_paths = list(external_db_paths or [])
        env_db_paths = self._parse_db_path_list(
            os.environ.get("LOCAL_TRADE_DB_PATHS", "")
        )
        for p in env_db_paths:
            if p not in db_paths:
                db_paths.append(p)

        # Backward-compatible single-path env var
        single_env_db = (os.environ.get("LOCAL_TRADE_DB_PATH", "") or "").strip()
        if single_env_db and single_env_db not in db_paths:
            db_paths.append(single_env_db)

        main_db_abs = os.path.abspath(str(self.trade_db.db_path))
        for db_file in DATA_DIR.glob("*.db"):
            db_abs = os.path.abspath(str(db_file))
            # Skip the main DB (don't import from self)
            if db_abs == main_db_abs:
                continue
            # Skip known non-trade DBs
            if db_file.name in ("market_data.db", "ibkr_data.db"):
                continue
            if str(db_file) not in db_paths:
                db_paths.append(str(db_file))

        for db_path in db_paths:
            if os.path.abspath(str(db_path)) == main_db_abs:
                continue
            n = self.sync_from_db(db_path)
            results[f"db:{Path(db_path).name}"] = n
            total += n

        # 3. Auto-discover CSV files (skip gcloud_trades.csv — it's the
        #    system's own export and importing from it creates a circular loop)
        csv_list = list(csv_paths or [])
        for csv_file in DATA_DIR.glob("*trades*.csv"):
            if csv_file.name == "gcloud_trades.csv":
                continue
            if str(csv_file) not in csv_list:
                csv_list.append(str(csv_file))

        for csv_path in csv_list:
            n = self.sync_from_csv(csv_path)
            results[f"csv:{Path(csv_path).name}"] = n
            total += n

        if total > 0:
            logger.info(f"TradeSync: Sync complete — {total} new trades imported: {results}")
        else:
            logger.info("TradeSync: Sync complete — no new trades to import")

        return results

    # ------------------------------------------------------------------
    # Trade factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_trade(**kwargs):
        """Create a Trade dataclass from keyword args."""
        from live.trade_database import Trade
        return Trade(**{k: v for k, v in kwargs.items() if v is not None})
