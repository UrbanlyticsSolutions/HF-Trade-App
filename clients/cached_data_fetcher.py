"""
Cached Data Fetcher - Uses DB as cache, only downloads missing data.
"""
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import os
from dotenv import load_dotenv

load_dotenv()


class CachedDataFetcher:
    """
    Smart data fetcher that uses SQLite as a cache.
    - Checks what dates are already in DB
    - Only fetches missing dates from API
    - Stores new data in DB for future use
    """
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()
        
        # Lazy load API client
        self._api_client = None
    
    def _init_tables(self):
        """Initialize cache tracking tables"""
        cursor = self.conn.cursor()
        
        # Table for tracking what data we have cached
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_metadata (
                cache_key TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                data_type TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                record_count INTEGER DEFAULT 0
            )
        ''')
        
        # Table for option chain daily snapshots (for backtesting)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS option_chain_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                underlying TEXT NOT NULL,
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                strike_price REAL,
                expiration_date TEXT,
                contract_type TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                open_interest INTEGER,
                implied_volatility REAL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(underlying, date, ticker)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ocd_underlying_date ON option_chain_daily(underlying, date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ocd_ticker ON option_chain_daily(ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ocd_volume ON option_chain_daily(volume DESC)')
        
        # Table for underlying price history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS underlying_price_history (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                vwap REAL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, date)
            )
        ''')
        
        self.conn.commit()
    
    @property
    def api_client(self):
        """Lazy load API client"""
        if self._api_client is None:
            from clients.massive_options_client import MassiveOptionsClient
            self._api_client = MassiveOptionsClient()
        return self._api_client
    
    def get_cached_dates(self, underlying: str, data_type: str = "option_chain") -> List[str]:
        """Get list of dates we already have cached for this underlying"""
        cursor = self.conn.cursor()
        
        if data_type == "option_chain":
            cursor.execute('''
                SELECT DISTINCT date FROM option_chain_daily 
                WHERE underlying = ? 
                ORDER BY date
            ''', (underlying.upper(),))
        elif data_type == "underlying_price":
            cursor.execute('''
                SELECT DISTINCT date FROM underlying_price_history 
                WHERE ticker = ? 
                ORDER BY date
            ''', (underlying.upper(),))
        else:
            return []
        
        return [row[0] for row in cursor.fetchall()]
    
    def get_missing_dates(self, underlying: str, start_date: str, end_date: str, 
                          data_type: str = "option_chain") -> List[str]:
        """
        Get list of dates that are NOT in the cache.
        Only returns trading days (excludes weekends).
        """
        cached_dates = set(self.get_cached_dates(underlying, data_type))
        
        # Generate all dates in range
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        missing = []
        current = start
        while current <= end:
            # Skip weekends
            if current.weekday() < 5:  # Mon=0, Fri=4
                date_str = current.strftime("%Y-%m-%d")
                if date_str not in cached_dates:
                    missing.append(date_str)
            current += timedelta(days=1)
        
        return missing
    
    def get_cache_stats(self, underlying: str = None) -> Dict:
        """Get statistics about what's in the cache"""
        cursor = self.conn.cursor()
        
        if underlying:
            cursor.execute('''
                SELECT 
                    underlying,
                    MIN(date) as first_date,
                    MAX(date) as last_date,
                    COUNT(DISTINCT date) as num_days,
                    COUNT(*) as total_records,
                    MAX(cached_at) as last_updated
                FROM option_chain_daily
                WHERE underlying = ?
                GROUP BY underlying
            ''', (underlying.upper(),))
        else:
            cursor.execute('''
                SELECT 
                    underlying,
                    MIN(date) as first_date,
                    MAX(date) as last_date,
                    COUNT(DISTINCT date) as num_days,
                    COUNT(*) as total_records,
                    MAX(cached_at) as last_updated
                FROM option_chain_daily
                GROUP BY underlying
            ''')
        
        rows = cursor.fetchall()
        stats = {}
        for row in rows:
            stats[row[0]] = {
                'first_date': row[1],
                'last_date': row[2],
                'num_days': row[3],
                'total_records': row[4],
                'last_updated': row[5]
            }
        return stats
    
    def fetch_and_cache_option_chain(self, underlying: str, date: str) -> int:
        """
        Fetch option chain for a single date and cache it.
        Returns number of records cached.
        """
        print(f"  Fetching {underlying} chain for {date}...")
        
        try:
            # Get historical aggregates for each contract
            records = []
            
            # First get list of contracts that existed on that date
            # Note: Polygon API max limit is 1000
            contracts = list(self.api_client.list_options_contracts(
                underlying_ticker=underlying,
                as_of=date,
                limit=1000
            ))
            
            print(f"    Found {len(contracts)} contracts")
            
            # Fetch daily bars for each contract
            for contract in contracts[:500]:  # Limit to avoid rate limits
                ticker = contract.ticker
                try:
                    bars = list(self.api_client.get_historical_aggregates(
                        ticker=ticker,
                        multiplier=1,
                        timespan="day",
                        from_date=date,
                        to_date=date,
                        limit=1
                    ))
                    
                    if bars:
                        bar = bars[0]
                        records.append({
                            'underlying': underlying.upper(),
                            'date': date,
                            'ticker': ticker,
                            'strike_price': contract.strike_price,
                            'expiration_date': contract.expiration_date,
                            'contract_type': contract.contract_type,
                            'open': getattr(bar, 'open', None),
                            'high': getattr(bar, 'high', None),
                            'low': getattr(bar, 'low', None),
                            'close': getattr(bar, 'close', None),
                            'volume': getattr(bar, 'volume', 0),
                            'open_interest': None,  # Need snapshot for this
                        })
                except Exception as e:
                    continue  # Skip contracts with no data
            
            # Bulk insert
            if records:
                self._insert_option_chain_records(records)
                print(f"    Cached {len(records)} records")
            
            return len(records)
            
        except Exception as e:
            print(f"    Error: {e}")
            return 0
    
    def _insert_option_chain_records(self, records: List[Dict]):
        """Bulk insert option chain records"""
        cursor = self.conn.cursor()
        
        for r in records:
            cursor.execute('''
                INSERT OR REPLACE INTO option_chain_daily 
                (underlying, date, ticker, strike_price, expiration_date, contract_type,
                 open, high, low, close, volume, open_interest, implied_volatility,
                 delta, gamma, theta, vega)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                r['underlying'], r['date'], r['ticker'], r.get('strike_price'),
                r.get('expiration_date'), r.get('contract_type'),
                r.get('open'), r.get('high'), r.get('low'), r.get('close'),
                r.get('volume', 0), r.get('open_interest'),
                r.get('implied_volatility'), r.get('delta'), r.get('gamma'),
                r.get('theta'), r.get('vega')
            ))
        
        self.conn.commit()
    
    def fetch_and_cache_underlying(self, ticker: str, start_date: str, end_date: str) -> int:
        """
        Fetch underlying price history and cache it.
        Only fetches missing dates.
        """
        missing_dates = self.get_missing_dates(ticker, start_date, end_date, "underlying_price")
        
        if not missing_dates:
            print(f"All dates already cached for {ticker}")
            return 0
        
        print(f"Fetching {len(missing_dates)} missing dates for {ticker}...")
        
        try:
            bars = list(self.api_client.get_historical_aggregates(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_date=missing_dates[0],
                to_date=missing_dates[-1],
                limit=5000
            ))
            
            cursor = self.conn.cursor()
            count = 0
            
            for bar in bars:
                # Convert timestamp to date
                ts = getattr(bar, 'timestamp', None) or getattr(bar, 't', None)
                if ts:
                    date_str = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")
                else:
                    continue
                
                cursor.execute('''
                    INSERT OR REPLACE INTO underlying_price_history 
                    (ticker, date, open, high, low, close, volume, vwap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ticker.upper(),
                    date_str,
                    getattr(bar, 'open', None),
                    getattr(bar, 'high', None),
                    getattr(bar, 'low', None),
                    getattr(bar, 'close', None),
                    getattr(bar, 'volume', None),
                    getattr(bar, 'vwap', None)
                ))
                count += 1
            
            self.conn.commit()
            print(f"Cached {count} price records for {ticker}")
            return count
            
        except Exception as e:
            print(f"Error fetching underlying data: {e}")
            return 0
    
    def smart_sync(self, underlying: str, start_date: str, end_date: str, 
                   include_chain: bool = True, include_underlying: bool = True) -> Dict:
        """
        Smart sync - only downloads what's missing.
        
        Args:
            underlying: Ticker symbol (e.g., "QQQ")
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
            include_chain: Whether to sync option chain data
            include_underlying: Whether to sync underlying price data
            
        Returns:
            Summary of what was synced
        """
        result = {
            'underlying': underlying,
            'start_date': start_date,
            'end_date': end_date,
            'chain_records_added': 0,
            'underlying_records_added': 0,
            'dates_processed': 0
        }
        
        print(f"\n{'='*60}")
        print(f"SMART SYNC: {underlying}")
        print(f"Range: {start_date} to {end_date}")
        print(f"{'='*60}")
        
        # Check what we already have
        stats = self.get_cache_stats(underlying)
        if underlying.upper() in stats:
            s = stats[underlying.upper()]
            print(f"\nExisting cache:")
            print(f"  Dates: {s['first_date']} to {s['last_date']}")
            print(f"  Days cached: {s['num_days']}")
            print(f"  Total records: {s['total_records']}")
        else:
            print(f"\nNo existing cache for {underlying}")
        
        # Sync underlying price data
        if include_underlying:
            print(f"\n[1/2] Syncing underlying price data...")
            result['underlying_records_added'] = self.fetch_and_cache_underlying(
                underlying, start_date, end_date
            )
        
        # Sync option chain data
        if include_chain:
            print(f"\n[2/2] Syncing option chain data...")
            missing_dates = self.get_missing_dates(underlying, start_date, end_date, "option_chain")
            
            if not missing_dates:
                print("All chain dates already cached!")
            else:
                print(f"Need to fetch {len(missing_dates)} dates")
                
                for i, date in enumerate(missing_dates):
                    print(f"\n[{i+1}/{len(missing_dates)}] {date}")
                    records = self.fetch_and_cache_option_chain(underlying, date)
                    result['chain_records_added'] += records
                    result['dates_processed'] += 1
        
        print(f"\n{'='*60}")
        print(f"SYNC COMPLETE")
        print(f"  Chain records added: {result['chain_records_added']}")
        print(f"  Underlying records added: {result['underlying_records_added']}")
        print(f"{'='*60}\n")
        
        return result
    
    def get_chain_for_date(self, underlying: str, date: str) -> List[Dict]:
        """Get cached option chain for a specific date"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM option_chain_daily
            WHERE underlying = ? AND date = ?
            ORDER BY volume DESC
        ''', (underlying.upper(), date))
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_underlying_price(self, ticker: str, date: str) -> Optional[Dict]:
        """Get cached underlying price for a specific date"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM underlying_price_history
            WHERE ticker = ? AND date = ?
        ''', (ticker.upper(), date))
        
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def get_high_volume_options(self, underlying: str, date: str, 
                                 min_volume: int = 1000, limit: int = 50) -> List[Dict]:
        """Get high volume options (potential whale activity) for a date"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM option_chain_daily
            WHERE underlying = ? AND date = ? AND volume >= ?
            ORDER BY volume DESC
            LIMIT ?
        ''', (underlying.upper(), date, min_volume, limit))
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def fetch_intraday_data(self, ticker: str, start_date: str, end_date: str, 
                            timespan: str = "minute") -> int:
        """
        Fetch intraday price data and store in intraday_ticker_data table.
        Uses Polygon aggregates endpoint.
        Stores timestamps as UTC ISO format for proper timezone handling.
        Includes: Pre-market (4 AM - 9:30 AM EST), Regular (9:30 AM - 4 PM EST), After-hours (4 PM - 8 PM EST)
        """
        print(f"Fetching intraday data for {ticker} from {start_date} to {end_date}...")
        
        try:
            bars = list(self.api_client.get_historical_aggregates(
                ticker=ticker,
                multiplier=1,
                timespan=timespan,
                from_date=start_date,
                to_date=end_date,
                limit=50000
            ))
            
            if not bars:
                print(f"  No intraday data returned")
                return 0
            
            cursor = self.conn.cursor()
            count = 0
            
            # Session counters
            sessions = {'pre-market': 0, 'regular': 0, 'after-hours': 0}
            est_offset = timedelta(hours=-5)
            
            for bar in bars:
                ts = getattr(bar, 'timestamp', None) or getattr(bar, 't', None)
                if not ts:
                    continue
                
                # Store as UTC timestamp string
                dt_utc = datetime.utcfromtimestamp(ts / 1000)
                timestamp_str = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
                
                # Calculate EST time for date grouping
                dt_est = dt_utc + est_offset
                date_str = dt_est.strftime("%Y-%m-%d")
                hour_est = dt_est.hour
                
                # Count sessions
                if hour_est < 9 or (hour_est == 9 and dt_est.minute < 30):
                    sessions['pre-market'] += 1
                elif hour_est >= 16:
                    sessions['after-hours'] += 1
                else:
                    sessions['regular'] += 1
                
                cursor.execute('''
                    INSERT OR REPLACE INTO intraday_ticker_data 
                    (timestamp, ticker, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp_str,
                    ticker.upper(),
                    date_str,
                    getattr(bar, 'open', None),
                    getattr(bar, 'high', None),
                    getattr(bar, 'low', None),
                    getattr(bar, 'close', None),
                    getattr(bar, 'volume', 0)
                ))
                count += 1
            
            self.conn.commit()
            print(f"  Cached {count} intraday bars for {ticker}")
            print(f"    Sessions: Pre-market: {sessions['pre-market']}, Regular: {sessions['regular']}, After-hours: {sessions['after-hours']}")
            return count
            
        except Exception as e:
            print(f"  Error fetching intraday data: {e}")
            return 0
    
    def get_intraday_date_range(self, ticker: str) -> tuple:
        """Get the date range of cached intraday data"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MIN(date), MAX(date), COUNT(*) 
            FROM intraday_ticker_data 
            WHERE ticker = ?
        ''', (ticker.upper(),))
        row = cursor.fetchone()
        return row[0], row[1], row[2] if row else (None, None, 0)
    
    def get_intraday_cached_dates(self, ticker: str) -> set:
        """Get set of dates that have intraday data cached"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT DISTINCT date FROM intraday_ticker_data 
            WHERE ticker = ?
        ''', (ticker.upper(),))
        return set(row[0] for row in cursor.fetchall())
    
    def get_missing_intraday_dates(self, ticker: str, start_date: str, end_date: str) -> List[str]:
        """Get list of trading days missing intraday data"""
        cached = self.get_intraday_cached_dates(ticker)
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        missing = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # Mon-Fri only
                date_str = current.strftime("%Y-%m-%d")
                if date_str not in cached:
                    missing.append(date_str)
            current += timedelta(days=1)
        
        return sorted(missing)
    
    def sync_intraday_complete(self, ticker: str, days_back: int = 14) -> int:
        """
        Complete intraday sync - fills ALL gaps and syncs to today.
        This is the main method to call for complete data.
        
        Args:
            ticker: Stock ticker (e.g., 'QQQ')
            days_back: How many days back to ensure complete data
            
        Returns:
            Total bars synced
        """
        print(f"\n{'='*60}")
        print(f"COMPLETE INTRADAY SYNC: {ticker}")
        print(f"{'='*60}")
        
        # Calculate date range
        today = datetime.now()
        # Skip to last trading day if weekend
        while today.weekday() >= 5:
            today -= timedelta(days=1)
        
        end_date = today.strftime("%Y-%m-%d")
        start_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        print(f"Target range: {start_date} to {end_date}")
        
        # Check current state
        min_d, max_d, count = self.get_intraday_date_range(ticker)
        print(f"Current cache: {min_d} to {max_d} ({count} bars)")
        
        # Find missing dates
        missing = self.get_missing_intraday_dates(ticker, start_date, end_date)
        
        if not missing:
            print(f"✅ No missing dates! Data is complete.")
            return 0
        
        print(f"⚠️ Missing {len(missing)} trading days: {missing}")
        
        # Fetch in batches (group consecutive dates)
        total_synced = 0
        
        # Simple approach: fetch entire range at once (API handles it efficiently)
        print(f"\nFetching {start_date} to {end_date}...")
        synced = self.fetch_intraday_data(ticker, start_date, end_date)
        total_synced += synced
        
        # Verify
        still_missing = self.get_missing_intraday_dates(ticker, start_date, end_date)
        if still_missing:
            print(f"⚠️ Still missing (likely market holidays): {still_missing}")
        else:
            print(f"✅ All gaps filled!")
        
        print(f"\n{'='*60}")
        print(f"SYNC COMPLETE: {total_synced} bars added")
        print(f"{'='*60}\n")
        
        return total_synced
    
    def sync_intraday_to_today(self, ticker: str) -> int:
        """
        Sync intraday data up to today (legacy method).
        Use sync_intraday_complete() for full gap-filling.
        """
        return self.sync_intraday_complete(ticker, days_back=14)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    # Run complete sync for QQQ
    fetcher = CachedDataFetcher()
    
    print("=" * 60)
    print("CACHED DATA FETCHER - AUTO SYNC")
    print("=" * 60)
    
    # Sync intraday data (fills gaps + syncs to today)
    fetcher.sync_intraday_complete("QQQ", days_back=14)
    
    # Show final stats
    min_d, max_d, count = fetcher.get_intraday_date_range("QQQ")
    print(f"\nFinal state: {min_d} to {max_d} ({count} bars)")
    
    fetcher.close()
