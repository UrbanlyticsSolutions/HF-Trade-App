#!/usr/bin/env python3
"""
Fetch actual trade records from IBKR — both current session and historical.

Session trades (TWS API):
  python scripts/query_ibkr_trades.py --host 127.0.0.1 --port 4004

Historical trades (Flex Web Service):
  python scripts/query_ibkr_trades.py --flex --token TOKEN --query-id 123456
  python scripts/query_ibkr_trades.py --flex --days 7

Both:
  python scripts/query_ibkr_trades.py --host 127.0.0.1 --port 4004 --flex --token TOKEN --query-id 123456

Env vars (alternative to CLI args):
  IBKR_FLEX_TOKEN, IBKR_FLEX_QUERY_ID
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clients.ibkr_client import IBKRClient


def query_session_trades(args):
    """Query current TWS/Gateway session executions via TWS API."""
    print(f"Connecting to IBKR at {args.host}:{args.port} ...")
    client = IBKRClient(host=args.host, port=args.port, client_id=args.client_id)
    try:
        client.connect()
    except ConnectionError as e:
        print(f"ERROR: {e}")
        return
    time.sleep(1)

    # ── Account summary ──
    print("\n=== ACCOUNT SUMMARY ===")
    summary = client.get_account_summary(timeout=10)
    for key in ("NetLiquidation", "TotalCashValue", "UnrealizedPnL",
                "RealizedPnL", "BuyingPower", "DayTradesRemaining"):
        val = summary.get(key, "N/A")
        print(f"  {key}: {val}")

    # ── Positions ──
    print("\n=== POSITIONS ===")
    positions = client.get_positions(timeout=10)
    if not positions:
        print("  (none)")
    for key, pos in positions.items():
        print(f"  {key}: qty={pos.get('qty', 0)}  avg_cost={pos.get('avg_cost', 0):.4f}")

    # ── Session fills: reqExecutions + reqCompletedOrders merge ──
    print("\n=== EXECUTIONS (merged session fills) ===")
    execs = client.get_merged_session_fills(
        acct_code=args.account,
        timeout=15,
    )
    if not execs:
        print("  (none — use Flex for full history)")
    else:
        for ex in sorted(execs, key=lambda e: e.get("time", "")):
            sym = ex.get("localSymbol") or ex.get("symbol", "?")
            side = ex.get("side", "?")
            shares = ex.get("shares", 0)
            price = ex.get("price", 0)
            t = ex.get("time", "")
            oid = ex.get("order_id", "")
            eid = ex.get("exec_id", "")
            sec = ex.get("secType", "")
            strike = ex.get("strike", 0)
            right = ex.get("right", "")
            expiry = ex.get("expiry", "")

            label = sym
            if sec == "OPT" and expiry:
                label = f"{ex.get('symbol','')}{expiry}{right}{int(strike) if strike == int(strike) else strike}"

            print(f"  {t}  {side:>5s} {shares:>4.0f}x {label:<30s} @ ${price:.2f}  "
                  f"order={oid}  exec={eid}")
        print(f"\n  Total executions: {len(execs)}")

    # ── Open orders ──
    print("\n=== OPEN ORDERS ===")
    open_orders = client.wrapper.open_orders if hasattr(client.wrapper, "open_orders") else {}
    if not open_orders:
        print("  (none)")
    else:
        for oid, info in open_orders.items():
            print(f"  order={oid}  {info}")

    client.disconnect()


def query_historical_trades(args):
    """Query historical trades via IBKR Flex Web Service."""
    from clients.ibkr_flex import IBKRFlexClient, print_trades, FlexQueryError
    
    token = args.token or os.environ.get("IBKR_FLEX_TOKEN", "")
    query_id = args.query_id or int(os.environ.get("IBKR_FLEX_QUERY_ID", "0"))
    
    if not token:
        print("\nERROR: Flex token required. Use --token or set IBKR_FLEX_TOKEN env var.")
        print("\nSetup (one-time in IBKR Account Management):")
        print("  1. portal.interactivebrokers.com > Reports > Flex Queries")
        print("  2. Create Activity Flex Query: enable 'Trades' section, XML format")
        print("  3. Note the Query ID")
        print("  4. Settings > Flex Web Service Configuration > Generate token")
        print("  5. Re-run with: --flex --token YOUR_TOKEN --query-id QUERY_ID")
        return
    
    if not query_id:
        print("\nERROR: Flex Query ID required. Use --query-id or set IBKR_FLEX_QUERY_ID env var.")
        return
    
    try:
        client = IBKRFlexClient(token=token)
        print(f"Fetching historical trades (Flex Query ID={query_id})...")
        trades = client.fetch_trades(query_id)
        print(f"\n=== IBKR HISTORICAL TRADES (Flex Web Service) ===")
        print_trades(trades, days_filter=args.days)
    except FlexQueryError as e:
        print(f"\nFlex Query ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Query IBKR for actual trade records")
    # TWS API args
    parser.add_argument("--host", default=os.environ.get("IBKR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("IBKR_PAPER_PORT", "7497")))
    parser.add_argument("--client-id", type=int, default=98)
    parser.add_argument("--account", default="")
    # Flex Web Service args
    parser.add_argument("--flex", action="store_true", help="Also query Flex Web Service for historical trades")
    parser.add_argument("--flex-only", action="store_true", help="ONLY query Flex (skip TWS API)")
    parser.add_argument("--token", default="", help="Flex Web Service token")
    parser.add_argument("--query-id", type=int, default=0, help="Flex Query ID")
    parser.add_argument("--days", type=int, default=None, help="Filter: last N days only")
    args = parser.parse_args()

    if not args.flex_only:
        query_session_trades(args)
    
    if args.flex or args.flex_only:
        query_historical_trades(args)
    
    print("\nDone.")


if __name__ == "__main__":
    main()
