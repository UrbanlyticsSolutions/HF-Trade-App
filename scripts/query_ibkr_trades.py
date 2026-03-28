#!/usr/bin/env python3
"""
Fetch actual trade executions and positions directly from IBKR TWS/Gateway.

Usage:
  python scripts/query_ibkr_trades.py                    # default: ib-gateway:4004
  python scripts/query_ibkr_trades.py --host 127.0.0.1 --port 7497
  python scripts/query_ibkr_trades.py --account DUP964745
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clients.ibkr_client import IBKRClient


def main():
    parser = argparse.ArgumentParser(description="Query IBKR for actual trade records")
    parser.add_argument("--host", default=os.environ.get("IBKR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("IBKR_PAPER_PORT", "7497")))
    parser.add_argument("--client-id", type=int, default=98)
    parser.add_argument("--account", default="")
    args = parser.parse_args()

    print(f"Connecting to IBKR at {args.host}:{args.port} ...")
    client = IBKRClient(host=args.host, port=args.port, client_id=args.client_id)
    try:
        client.connect()
    except ConnectionError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
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

    # ── Executions (current TWS session) ──
    print("\n=== EXECUTIONS (current session) ===")
    execs = client.get_executions(
        acct_code=args.account,
        timeout=15,
    )
    if not execs:
        print("  (none — only shows fills from the current TWS/Gateway session)")
    else:
        total_pnl = 0.0
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
    print("\nDone.")


if __name__ == "__main__":
    main()
