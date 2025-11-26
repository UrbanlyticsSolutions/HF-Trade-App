"""
Continuous real-time quote ingestion from FMP Stable API.
Fetches both regular-session quotes and after-market quotes
and stores them in the SQLite database for downstream consumers.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Allow running as script without installation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients.fmp_stable_client import FMPStableClient
from clients.database import MarketDatabase

EST = ZoneInfo("America/New_York")


def is_market_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(tz=EST)
    # Weekends
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= now <= close_time


def to_iso_timestamp(raw_ts: int | float | None) -> str:
    if raw_ts is None:
        return datetime.now(tz=timezone.utc).isoformat()
    # FMP mixes seconds + milliseconds
    if raw_ts > 1e12:
        raw_ts /= 1000.0
    dt = datetime.fromtimestamp(raw_ts, tz=timezone.utc)
    return dt.isoformat()


def fetch_regular_quote(client: FMPStableClient, symbol: str):
    data = client.quote(symbol)
    if not data:
        return None
    payload = data[0]
    return {
        "price": payload.get("price"),
        "bid_price": payload.get("price"),
        "ask_price": payload.get("price"),
        "volume": payload.get("volume"),
        "timestamp": to_iso_timestamp(payload.get("timestamp"))
    }


def fetch_aftermarket_quote(client: FMPStableClient, symbol: str):
    data = client.aftermarket_quote(symbol)
    if not data:
        return None
    payload = data[0]
    bid = payload.get("bidPrice")
    ask = payload.get("askPrice")
    price = None
    if bid is not None and ask is not None:
        price = (bid + ask) / 2
    elif bid is not None:
        price = bid
    elif ask is not None:
        price = ask
    return {
        "price": price,
        "bid_price": bid,
        "ask_price": ask,
        "volume": payload.get("volume"),
        "timestamp": to_iso_timestamp(payload.get("timestamp"))
    }


def main():
    parser = argparse.ArgumentParser(description="Stream real-time quotes into the DB")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol to stream")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds")
    parser.add_argument("--iterations", type=int, default=0, help="Number of iterations (0 = infinite)")
    parser.add_argument("--aftermarket-only", action="store_true", help="Skip regular session calls")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise SystemExit("FMP_API_KEY missing; set it in .env")

    client = FMPStableClient(api_key)
    db = MarketDatabase()

    print(f"Starting real-time stream for {args.symbol} (interval={args.interval}s)")

    count = 0
    try:
        while args.iterations == 0 or count < args.iterations:
            count += 1
            source = "AFTERMARKET"
            quote = None

            if not args.aftermarket_only and is_market_open():
                source = "REGULAR"
                quote = fetch_regular_quote(client, args.symbol)
            else:
                quote = fetch_aftermarket_quote(client, args.symbol)

            if quote:
                db.insert_realtime_quote(
                    ticker=args.symbol,
                    price=quote.get("price"),
                    bid_price=quote.get("bid_price"),
                    ask_price=quote.get("ask_price"),
                    volume=quote.get("volume"),
                    source=source,
                    quote_timestamp=quote.get("timestamp")
                )
                print(f"[{source}] {args.symbol} price={quote.get('price')} ts={quote.get('timestamp')}")
            else:
                print(f"No {source.lower()} data returned for {args.symbol}")

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopping stream...")
    finally:
        db.close()


if __name__ == "__main__":
    main()
