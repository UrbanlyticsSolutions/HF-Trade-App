"""
Sync missing option + underlying intraday data for backtesting.

Usage:
    python scripts/sync_data.py                    # Sync missing dates from default start through today
    python scripts/sync_data.py --start 2024-07-01 --end 2026-03-23
"""
import sys
sys.path.insert(0, '.')

import os
import time
import sqlite3
import argparse
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()


def get_trading_days(start: date, end: date) -> list:
    """Return weekdays between start and end (inclusive), excluding known holidays."""
    # 2026 US market holidays (NYSE)
    holidays_2026 = {
        date(2026, 1, 1),   # New Year's Day
        date(2026, 1, 19),  # MLK Day
        date(2026, 2, 16),  # Presidents' Day
        date(2026, 4, 3),   # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 7, 3),   # Independence Day observed
        date(2026, 9, 7),   # Labor Day
        date(2026, 11, 26), # Thanksgiving
        date(2026, 12, 25), # Christmas
    }
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5 and cur not in holidays_2026:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def get_cached_option_dates(conn, underlying: str) -> set:
    """Get dates already in options_intraday."""
    rows = conn.execute(
        "SELECT DISTINCT date FROM options_intraday WHERE underlying=?",
        (underlying,)
    ).fetchall()
    return {r[0] for r in rows}


def get_cached_underlying_dates(conn, ticker: str) -> set:
    """Get dates already in intraday_5min_data."""
    rows = conn.execute(
        "SELECT DISTINCT date FROM intraday_5min_data WHERE ticker=?",
        (ticker,)
    ).fetchall()
    return {r[0] for r in rows}


def sync_underlying_5min(conn, ticker: str, missing_dates: list):
    """Fetch and store 5-min underlying data from FMP."""
    from clients.fmp_stable_client import FMPStableClient

    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        print("ERROR: FMP_API_KEY not set in .env")
        return 0

    fmp = FMPStableClient(api_key)
    total = 0

    # FMP returns max ~5 days of 5min data per call, so batch by week
    batch_size = 5
    for i in range(0, len(missing_dates), batch_size):
        batch = missing_dates[i:i+batch_size]
        start = str(batch[0])
        end = str(batch[-1])
        print(f"  FMP 5min {ticker}: {start} to {end} ...")

        try:
            data = fmp.historical_chart_5min(ticker, start, end)
            if data:
                from clients.database import MarketDatabase
                db = MarketDatabase('data/market_data.db')
                db.insert_intraday_5min(ticker, data)
                db.conn.close()
                total += len(data)
                print(f"    -> {len(data)} bars")
            else:
                print(f"    -> no data returned")
        except Exception as e:
            print(f"    -> ERROR: {e}")

        time.sleep(0.5)  # Rate limit

    return total


def sync_options_intraday(conn, underlying: str, missing_dates: list):
    """Fetch and store 5-min option intraday bars from Polygon (Massive)."""
    from massive import RESTClient

    api_key = os.getenv('MASSIVE_API_KEY')
    if not api_key:
        print("ERROR: MASSIVE_API_KEY not set in .env")
        return 0

    client = RESTClient(api_key)
    total = 0

    for trade_date in missing_dates:
        date_str = str(trade_date)
        print(f"\n  Polygon options for {date_str}...")

        try:
            # Get 0DTE contracts expiring on this date (as_of needed for past dates)
            contracts = list(client.list_options_contracts(
                underlying_ticker=underlying,
                expiration_date=date_str,
                as_of=date_str,
                limit=1000
            ))
            print(f"    {len(contracts)} 0DTE contracts found")

            date_records = 0
            for contract in contracts:
                ticker = contract.ticker
                try:
                    bars = list(client.list_aggs(
                        ticker=ticker,
                        multiplier=5,
                        timespan="minute",
                        from_=date_str,
                        to=date_str,
                        limit=500
                    ))

                    if not bars:
                        continue

                    rows = []
                    for bar in bars:
                        ts = getattr(bar, 'timestamp', None)
                        if not ts:
                            continue
                        rows.append({
                            'option_ticker': ticker,
                            'underlying': underlying,
                            'timestamp': ts,
                            'open': getattr(bar, 'open', None),
                            'high': getattr(bar, 'high', None),
                            'low': getattr(bar, 'low', None),
                            'close': getattr(bar, 'close', None),
                            'volume': getattr(bar, 'volume', 0),
                            'vwap': getattr(bar, 'vwap', None),
                            'transactions': getattr(bar, 'transactions', None),
                            'timespan': 'minute',
                            'expiration': date_str,
                            'strike': contract.strike_price,
                            'option_type': contract.contract_type,
                        })

                    if rows:
                        cursor = conn.cursor()
                        for row in rows:
                            dt = datetime.fromtimestamp(row['timestamp'] / 1000.0)
                            d_str = dt.strftime('%Y-%m-%d')
                            t_str = dt.strftime('%H:%M:%S')
                            cursor.execute('''
                                INSERT OR REPLACE INTO options_intraday
                                (option_ticker, underlying, timestamp, date, time,
                                 open, high, low, close, volume, vwap, transactions,
                                 timespan, expiration, strike, option_type)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                row['option_ticker'], row['underlying'], row['timestamp'],
                                d_str, t_str,
                                row['open'], row['high'], row['low'], row['close'],
                                row['volume'], row['vwap'], row['transactions'],
                                row['timespan'], row['expiration'], row['strike'],
                                row['option_type']
                            ))
                        conn.commit()
                        date_records += len(rows)

                except Exception as e:
                    continue  # Skip contracts with no data

                # Polygon rate limit: 5 requests/min on free, higher on paid
                time.sleep(0.15)

            total += date_records
            print(f"    -> {date_records} bars stored")

        except Exception as e:
            print(f"    -> ERROR: {e}")

    return total


def main():
    parser = argparse.ArgumentParser(description='Sync missing backtest data')
    parser.add_argument('--start', default='2026-02-11', help='Start date YYYY-MM-DD')
    parser.add_argument(
        '--end',
        default=None,
        help='End date YYYY-MM-DD (default: today, US Eastern trading calendar)',
    )
    parser.add_argument('--underlying-only', action='store_true', help='Only sync underlying 5min')
    parser.add_argument('--options-only', action='store_true', help='Only sync options intraday')
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    underlying = 'SPY'

    trading_days = get_trading_days(start, end)
    print(f"Date range: {start} to {end}")
    print(f"Trading days: {len(trading_days)}")

    conn = sqlite3.connect('data/market_data.db')

    # Find missing dates
    cached_opt = get_cached_option_dates(conn, underlying)
    cached_und = get_cached_underlying_dates(conn, underlying)

    missing_opt = [d for d in trading_days if str(d) not in cached_opt]
    missing_und = [d for d in trading_days if str(d) not in cached_und]

    print(f"\nMissing option dates: {len(missing_opt)}")
    print(f"Missing underlying dates: {len(missing_und)}")

    if not missing_opt and not missing_und:
        print("All data already cached!")
        conn.close()
        return

    # Sync underlying 5min (FMP) - fast
    if not args.options_only and missing_und:
        print(f"\n{'='*50}")
        print(f"SYNCING UNDERLYING 5MIN DATA ({len(missing_und)} days)")
        print(f"{'='*50}")
        n = sync_underlying_5min(conn, underlying, missing_und)
        print(f"Total underlying bars added: {n}")

    # Sync options intraday (Polygon) - slower due to per-contract fetches
    if not args.underlying_only and missing_opt:
        print(f"\n{'='*50}")
        print(f"SYNCING OPTIONS INTRADAY DATA ({len(missing_opt)} days)")
        print(f"{'='*50}")
        n = sync_options_intraday(conn, underlying, missing_opt)
        print(f"Total option bars added: {n}")

    conn.close()
    print("\nSync complete!")


if __name__ == '__main__':
    main()
