"""
ibkr_flex.py — IBKR Flex Web Service client for historical trade data.

The TWS API only returns executions from the current gateway session.
For historical trades (days/weeks/months back), use IBKR's Flex Web Service.

Setup (one-time, in IBKR Account Management):
  1. Go to Reports > Flex Queries > Activity Flex Query
  2. Create a new query with: Trades section enabled, XML format
  3. Note the Query ID (e.g., 123456)
  4. Go to Settings > Flex Web Service > Generate token
  5. Note the token (valid 1 year)

Usage:
  python clients/ibkr_flex.py --token YOUR_TOKEN --query-id YOUR_QUERY_ID
  python clients/ibkr_flex.py --token YOUR_TOKEN --query-id YOUR_QUERY_ID --days 30
"""

import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

# IBKR Flex Web Service endpoints
FLEX_BASE = "https://gdcdyn.interactivebrokers.com/Universal/servlet"
SEND_REQUEST_URL = f"{FLEX_BASE}/FlexStatementService.SendRequest"
GET_STATEMENT_URL = f"{FLEX_BASE}/FlexStatementService.GetStatement"


class FlexQueryError(Exception):
    """Raised when Flex Web Service returns an error."""
    pass


class IBKRFlexClient:
    """
    Client for IBKR Flex Web Service — fetches historical trade data.
    
    Two-step process:
      1. SendRequest: submit query → get reference code
      2. GetStatement: poll with reference code → get XML report
    """

    def __init__(self, token: str, max_retries: int = 5, retry_delay: int = 10):
        """
        Args:
            token: Flex Web Service token (from IBKR Account Management)
            max_retries: Max polling attempts for GetStatement
            retry_delay: Seconds between polling attempts
        """
        if not token:
            raise ValueError("Flex Web Service token is required")
        self.token = token
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _http_get(self, url: str, timeout: int = 30) -> str:
        """Make an HTTP GET request and return response text."""
        req = Request(url)
        req.add_header("User-Agent", "IBKRFlexClient/1.0")
        try:
            with urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except (URLError, HTTPError) as e:
            raise FlexQueryError(f"HTTP request failed: {e}")

    def send_request(self, query_id: int) -> str:
        """
        Step 1: Submit a Flex Query and get a reference code.
        
        Args:
            query_id: The Flex Query ID from IBKR Account Management
            
        Returns:
            Reference code string for GetStatement
        """
        url = f"{SEND_REQUEST_URL}?t={self.token}&q={query_id}&v=3"
        logger.info(f"Flex SendRequest: query_id={query_id}")
        
        response_text = self._http_get(url)
        
        # Parse XML response
        try:
            root = ET.fromstring(response_text)
        except ET.ParseError:
            raise FlexQueryError(f"Invalid XML response: {response_text[:200]}")
        
        status = root.findtext("Status")
        if status == "Success":
            ref_code = root.findtext("ReferenceCode")
            if not ref_code:
                raise FlexQueryError(f"No ReferenceCode in response: {response_text[:200]}")
            logger.info(f"Flex SendRequest success: ref={ref_code}")
            return ref_code
        else:
            error_code = root.findtext("ErrorCode", "?")
            error_msg = root.findtext("ErrorMessage", "Unknown error")
            raise FlexQueryError(f"Flex SendRequest failed: [{error_code}] {error_msg}")

    def get_statement(self, reference_code: str) -> str:
        """
        Step 2: Download the Flex statement XML using the reference code.
        Polls until ready.
        
        Args:
            reference_code: From send_request()
            
        Returns:
            Raw XML statement string
        """
        url = f"{GET_STATEMENT_URL}?q={reference_code}&t={self.token}&v=3"
        
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Flex GetStatement: attempt {attempt}/{self.max_retries}")
            response_text = self._http_get(url)
            
            # Check if it's still generating (XML with Status=Warn)
            if response_text.strip().startswith("<"):
                try:
                    root = ET.fromstring(response_text)
                    status = root.findtext("Status")
                    if status == "Warn":
                        error_code = root.findtext("ErrorCode", "")
                        # 1019 = statement being generated, try again
                        if error_code == "1019":
                            logger.info(f"Statement still generating, waiting {self.retry_delay}s...")
                            time.sleep(self.retry_delay)
                            continue
                    elif status == "Fail":
                        error_msg = root.findtext("ErrorMessage", "Unknown error")
                        raise FlexQueryError(f"GetStatement failed: {error_msg}")
                except ET.ParseError:
                    pass  # Not a status XML — might be the actual statement
            
            # If we get here, we have the actual statement
            logger.info(f"Flex statement received ({len(response_text)} bytes)")
            return response_text
        
        raise FlexQueryError(f"Statement not ready after {self.max_retries} attempts")

    def fetch_trades(self, query_id: int) -> List[Dict[str, Any]]:
        """
        Fetch and parse trade records from a Flex Query.
        
        Args:
            query_id: Flex Query ID configured in IBKR Account Management
            
        Returns:
            List of trade dicts with keys like:
              symbol, dateTime, quantity, tradePrice, proceeds, 
              commissions, netCash, buySell, assetCategory, 
              putCall, strike, expiry, underlyingSymbol, etc.
        """
        ref_code = self.send_request(query_id)
        xml_text = self.get_statement(ref_code)
        return self.parse_trades(xml_text)
    
    @staticmethod
    def parse_trades(xml_text: str) -> List[Dict[str, Any]]:
        """
        Parse trade records from Flex statement XML.
        
        Handles both the full FlexQueryResponse format and
        the simpler FlexStatement format.
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise FlexQueryError(f"Failed to parse statement XML: {e}")
        
        trades = []
        
        # Find Trade elements — could be under various paths
        trade_elements = (
            root.findall(".//Trade") or
            root.findall(".//Trades/Trade") or 
            root.findall(".//TradeConfirm") or
            root.findall(".//Order")
        )
        
        for elem in trade_elements:
            trade = dict(elem.attrib)
            trades.append(trade)
        
        logger.info(f"Parsed {len(trades)} trades from Flex statement")
        return trades


def print_trades(trades: List[Dict], days_filter: int = None):
    """Pretty-print trade records."""
    if days_filter:
        cutoff = (datetime.now() - timedelta(days=days_filter)).strftime("%Y%m%d")
        trades = [t for t in trades 
                  if t.get("tradeDate", t.get("dateTime", ""))[:8] >= cutoff]
    
    if not trades:
        print("  (no trades found)")
        return
    
    # Group by date
    by_date: Dict[str, List] = {}
    for t in trades:
        date_str = t.get("tradeDate", t.get("dateTime", ""))[:8]
        by_date.setdefault(date_str, []).append(t)
    
    total_pnl = 0.0
    total_comm = 0.0
    
    for date_str in sorted(by_date.keys()):
        day_trades = by_date[date_str]
        formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}" if len(date_str) == 8 else date_str
        print(f"\n  --- {formatted} ({len(day_trades)} fills) ---")
        
        for t in sorted(day_trades, key=lambda x: x.get("dateTime", x.get("tradeDate", ""))):
            sym = t.get("symbol", "?")
            asset = t.get("assetCategory", "")
            side = t.get("buySell", t.get("side", "?"))
            qty = t.get("quantity", "0")
            price = t.get("tradePrice", t.get("price", "0"))
            comm = float(t.get("ibCommission", t.get("commission", "0")))
            pnl = float(t.get("fifoPnlRealized", t.get("realizedPnl", "0")))
            put_call = t.get("putCall", "")
            strike = t.get("strike", "")
            expiry = t.get("expiry", t.get("lastTradeDateOrContractMonth", ""))
            time_str = t.get("dateTime", t.get("tradeDate", ""))
            
            total_pnl += pnl
            total_comm += comm
            
            # Build label
            if asset == "OPT" and put_call:
                label = f"{sym}{expiry}{put_call[0]}{strike}"
            else:
                label = sym
            
            pnl_str = f"pnl=${pnl:+.2f}" if pnl != 0 else ""
            print(f"    {side:>4s} {qty:>6s}x {label:<30s} @ ${float(price):>8.2f}  "
                  f"comm=${comm:.2f}  {pnl_str}")
    
    print(f"\n  TOTAL: {len(trades)} fills | PnL=${total_pnl:+,.2f} | Commission=${total_comm:,.2f} | Net=${total_pnl + total_comm:+,.2f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch historical IBKR trades via Flex Web Service")
    parser.add_argument("--token", default=os.environ.get("IBKR_FLEX_TOKEN", ""),
                        help="Flex Web Service token (or set IBKR_FLEX_TOKEN env var)")
    parser.add_argument("--query-id", type=int, default=int(os.environ.get("IBKR_FLEX_QUERY_ID", "0")),
                        help="Flex Query ID (or set IBKR_FLEX_QUERY_ID env var)")
    parser.add_argument("--days", type=int, default=None,
                        help="Only show trades from last N days")
    parser.add_argument("--raw", action="store_true",
                        help="Print raw XML instead of parsed output")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    
    if not args.token:
        print("ERROR: --token required (or set IBKR_FLEX_TOKEN env var)")
        print("\nSetup instructions:")
        print("  1. Log in to IBKR Account Management (portal.interactivebrokers.com)")
        print("  2. Go to: Reports > Flex Queries")
        print("  3. Create an Activity Flex Query with 'Trades' section enabled, XML format")
        print("  4. Note the Query ID")
        print("  5. Go to: Settings > Flex Web Service Configuration > Generate token")
        print("  6. Run: python clients/ibkr_flex.py --token YOUR_TOKEN --query-id YOUR_QUERY_ID")
        sys.exit(1)
    
    if not args.query_id:
        print("ERROR: --query-id required (or set IBKR_FLEX_QUERY_ID env var)")
        sys.exit(1)
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    
    client = IBKRFlexClient(token=args.token)
    
    if args.raw:
        ref = client.send_request(args.query_id)
        xml = client.get_statement(ref)
        print(xml)
    else:
        print(f"Fetching trades from IBKR Flex Query (ID={args.query_id})...")
        trades = client.fetch_trades(args.query_id)
        
        print(f"\n=== IBKR HISTORICAL TRADES ===")
        print_trades(trades, days_filter=args.days)


if __name__ == "__main__":
    import sys
    main()
