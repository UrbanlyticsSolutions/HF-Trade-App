#!/usr/bin/env python3
"""Print Flex query rows for today's trade date (US/Eastern), from env token/query id."""
import os
import re
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime

try:
    from zoneinfo import ZoneInfo

    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
except Exception:
    today = datetime.utcnow().strftime("%Y%m%d")

token = os.environ.get("IBKR_FLEX_TOKEN", "")
qid = int(os.environ.get("IBKR_FLEX_QUERY_ID", "0") or "0")
if not token or not qid:
    print("ERROR: IBKR_FLEX_TOKEN and IBKR_FLEX_QUERY_ID must be set")
    sys.exit(1)

from clients.ibkr_flex import IBKRFlexClient  # noqa: E402


def dkey(t) -> str:
    raw = (t.get("dateTime") or t.get("tradeDate") or "").strip()
    raw = raw.replace("-", "")
    m = re.match(r"^(\d{8})", raw)
    if m:
        return m.group(1)
    m2 = re.search(r"(\d{8})", raw)
    return m2.group(1) if m2 else ""


def main():
    client = IBKRFlexClient(token=token)
    rows = client.fetch_trades(qid)
    day = [t for t in rows if dkey(t) == today]
    dates = sorted({dkey(t) for t in rows if dkey(t)})
    print(f"=== FLEX ACTIVITY FOR {today} (ET) ===")
    print(f"rows_today: {len(day)}  |  all_rows: {len(rows)}  |  distinct_dates: {dates[-5:] if dates else []}")
    for t in sorted(day, key=lambda x: x.get("dateTime", x.get("tradeDate", ""))):
        sym = t.get("symbol", "")
        bs = t.get("buySell", t.get("side", ""))
        q = t.get("quantity", "")
        px = t.get("tradePrice", t.get("price", ""))
        dt = t.get("dateTime", t.get("tradeDate", ""))
        pnl = t.get("fifoPnlRealized", t.get("realizedPnl", ""))
        print(f"  {dt}  {bs:>4} {str(q):>4} @ {px}  sym={sym}  fifoPnl={pnl}")


if __name__ == "__main__":
    main()
