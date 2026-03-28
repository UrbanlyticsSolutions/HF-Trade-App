#!/usr/bin/env python3
"""Print open IBKR positions and session executions for today's date (US/Eastern)."""
import os
import sys
import time

if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clients.ibkr_client import IBKRClient


def main() -> None:
    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    port = int(os.environ.get("IBKR_PAPER_PORT", os.environ.get("IBKR_PORT", "4004")))
    client_id = int(os.environ.get("IBKR_SNAPSHOT_CLIENT_ID", "97"))

    try:
        from zoneinfo import ZoneInfo

        today = __import__("datetime").datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
    except Exception:
        today = __import__("datetime").datetime.utcnow().strftime("%Y%m%d")

    print(f"Connecting {host}:{port} clientId={client_id} ...")
    print(f"Today (ET yyyymmdd): {today}\n")

    client = IBKRClient(host=host, port=port, client_id=client_id)
    client.connect()
    time.sleep(1.5)

    print("=== OPEN POSITIONS (IBKR) ===")
    positions = client.get_positions(timeout=15.0)
    if not positions:
        print("  (none)")
    else:
        for key in sorted(positions.keys()):
            p = positions[key]
            qty = float(p.get("position", p.get("qty", 0)) or 0)
            ac = float(p.get("avg_cost", p.get("average_cost", 0)) or 0)
            c = p.get("contract")
            sym = getattr(c, "localSymbol", None) or getattr(c, "symbol", None) or key
            print(f"  {sym}  position={qty}  avg_cost={ac:.4f}  key={key}")

    print("\n=== SESSION EXECUTIONS — TODAY ONLY ===")
    execs = client.get_merged_session_fills(timeout=25.0)
    today_e = [e for e in execs if (e.get("time") or "").startswith(today)]
    if not today_e:
        print(
            "  (none for today in this Gateway session — "
            "restart clears history; use Flex for full day)"
        )
    for e in sorted(today_e, key=lambda x: x.get("time", "")):
        sym = e.get("localSymbol") or e.get("symbol", "?")
        print(
            f"  {e.get('time', '')}  {str(e.get('side', '?')):>3}  "
            f"{float(e.get('shares', 0) or 0):.0f}x  {sym:<28}  "
            f"@ ${float(e.get('price', 0) or 0):.4f}  "
            f"order={e.get('order_id', '')}  exec={e.get('exec_id', '')}"
        )
    print(f"\n  Today count: {len(today_e)}  |  All session fills: {len(execs)}")

    client.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
