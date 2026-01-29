"""
Database for market data with SQLite persistence for grouped daily data.
In-memory cache for other data during the session.
"""
import sqlite3
import json

class MarketDatabase:
    """Database with SQLite persistence for grouped daily data and in-memory cache for other data"""

    def __init__(self, db_path: str = "market_data.db"):
        self.db_path = db_path
        # Initialize SQLite connection
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

        # In-memory caches
        self._stock_aggs = {}  # {(ticker, timespan): [bars]}
        self._ticker_info = {}  # {ticker: info}
        self._ticker_sectors = {}  # {ticker: {sic_code, sic_description, sector, last_updated}}

    def _init_tables(self):
        """Initialize database tables"""
        cursor = self.conn.cursor()

        # Normalized table for daily ticker data (one row per ticker per date)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_ticker_data (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                vwap REAL,
                transactions INTEGER,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, ticker)
            )
        ''')

        # Indexes for efficient querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON daily_ticker_data(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker ON daily_ticker_data(ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date_volume ON daily_ticker_data(date, volume)')

        # Table for ticker sector mappings (persisted)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ticker_sectors (
                ticker TEXT PRIMARY KEY,
                sic_code TEXT,
                sic_description TEXT,
                sector TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Table for Trump social posts (persisted with timeline)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trump_social_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_text TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                likes TEXT,
                retweets TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(post_text, timestamp)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trump_timestamp ON trump_social_posts(timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trump_scraped_at ON trump_social_posts(scraped_at DESC)')

        self.conn.commit()

        # Table for intraday ticker data (1-minute intervals)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intraday_ticker_data (
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, timestamp)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_intraday_date ON intraday_ticker_data(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_intraday_ticker_date ON intraday_ticker_data(ticker, date)')

        # Table for 5-minute intraday data (FMP)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intraday_5min_data (
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                timeframe TEXT DEFAULT '5min',
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, timestamp)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_5min_ticker_date ON intraday_5min_data(ticker, date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_5min_date ON intraday_5min_data(date)')

        self.conn.commit()

        # Table for real-time quotes collected via REST streaming
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realtime_quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                price REAL,
                bid_price REAL,
                ask_price REAL,
                volume REAL,
                source TEXT NOT NULL,
                quote_timestamp TEXT NOT NULL,
                received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_realtime_ticker ON realtime_quotes(ticker, quote_timestamp)')

        self.conn.commit()

        # Table for intraday ticker data (1-minute intervals)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intraday_ticker_data (
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, timestamp)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_intraday_date ON intraday_ticker_data(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_intraday_ticker_date ON intraday_ticker_data(ticker, date)')

        # Table for historical option prices (OHLCV)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS option_price_history (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                vwap REAL,
                transactions INTEGER,
                timespan TEXT DEFAULT 'day',
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, timestamp, timespan)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_opt_hist_ticker ON option_price_history(ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_opt_hist_date ON option_price_history(date)')

        # Table for raw option trades
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS option_trades (
                ticker TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                price REAL,
                size INTEGER,
                side TEXT,
                action TEXT,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_opt_trades_ticker_ts ON option_trades(ticker, timestamp)')

        # Table for option greeks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS option_greeks (
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                rho REAL,
                vanna REAL,
                charm REAL,
                vomma REAL,
                veta REAL,
                vera REAL,
                speed REAL,
                zomma REAL,
                color REAL,
                ultima REAL,
                d1 REAL,
                d2 REAL,
                dual_delta REAL,
                dual_gamma REAL,
                implied_vol REAL,
                underlying_price REAL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, timestamp)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_opt_greeks_ticker_date ON option_greeks(ticker, date)')

        # Table for option open interest
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS option_open_interest (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open_interest INTEGER,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, date)
            )
        ''')

        # Table for Massive options chain snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS massive_option_chain (
                ticker TEXT NOT NULL,
                underlying_asset TEXT NOT NULL,
                expiration_date TEXT,
                strike_price REAL,
                contract_type TEXT,
                details TEXT, -- JSON string of full details
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, fetched_at)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_massive_underlying ON massive_option_chain(underlying_asset)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_massive_fetched ON massive_option_chain(fetched_at)')

        self.conn.commit()

    def get_stock_aggregates(self, ticker: str, timespan: str, limit: int = 100):
        """Get cached stock aggregates"""
        key = (ticker.upper(), timespan)
        bars = self._stock_aggs.get(key, [])
        return bars[:limit] if bars else []

    def insert_stock_aggregates(self, ticker: str, aggregates: list, timespan: str):
        """Cache stock aggregates in memory"""
        key = (ticker.upper(), timespan)
        # Convert to dict format with both 't' and 'timestamp' keys
        formatted_bars = []
        for bar in aggregates:
            formatted_bar = dict(bar) if isinstance(bar, dict) else bar
            if 't' in formatted_bar and 'timestamp' not in formatted_bar:
                formatted_bar['timestamp'] = formatted_bar['t']
            formatted_bars.append(formatted_bar)
        self._stock_aggs[key] = formatted_bars

    def insert_polygon_news(self, news: list):
        """Stub - no-op"""
        pass

    def insert_dividends(self, dividends: list):
        """Stub - no-op"""
        pass

    def insert_stock_splits(self, splits: list):
        """Stub - no-op"""
        pass

    def get_ticker_info(self, ticker: str):
        """Get cached ticker info"""
        return self._ticker_info.get(ticker.upper())

    def insert_ticker_info(self, info: dict):
        """Cache ticker info in memory"""
        if info and 'ticker' in info:
            self._ticker_info[info['ticker'].upper()] = info

    def insert_ticker_sector(self, ticker: str, sic_code: str, sic_description: str, sector: str):
        """Cache ticker sector information in both SQLite and memory"""
        from datetime import datetime

        # In-memory cache
        self._ticker_sectors[ticker.upper()] = {
            'sic_code': sic_code,
            'sic_description': sic_description,
            'sector': sector,
            'last_updated': datetime.now().isoformat()
        }

        # SQLite persistence
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO ticker_sectors (ticker, sic_code, sic_description, sector)
            VALUES (?, ?, ?, ?)
        ''', (ticker.upper(), sic_code, sic_description, sector))
        self.conn.commit()

    def get_ticker_sector(self, ticker: str):
        """Get cached ticker sector from memory or SQLite"""
        # Check in-memory cache first
        result = self._ticker_sectors.get(ticker.upper())
        if result:
            return result

        # Fallback to SQLite
        cursor = self.conn.cursor()
        cursor.execute('SELECT sic_code, sic_description, sector, last_updated FROM ticker_sectors WHERE ticker = ?', (ticker.upper(),))
        row = cursor.fetchone()

        if row:
            result = {
                'sic_code': row[0],
                'sic_description': row[1],
                'sector': row[2],
                'last_updated': row[3]
            }
            # Cache in memory for faster subsequent access
            self._ticker_sectors[ticker.upper()] = result
            return result

        return None

    def get_all_ticker_sectors(self):
        """Get all cached ticker sectors"""
        return dict(self._ticker_sectors)

    def get_tickers_by_sector(self, sector: str):
        """Get all tickers in a specific sector"""
        return [
            ticker for ticker, data in self._ticker_sectors.items()
            if data.get('sector') == sector
        ]

    def get_sector_cache_stats(self):
        """Get statistics about sector cache"""
        sectors = {}
        for ticker, data in self._ticker_sectors.items():
            sector = data.get('sector', 'Unknown')
            sectors[sector] = sectors.get(sector, 0) + 1

        return {
            'total_tickers': len(self._ticker_sectors),
            'sectors': sectors,
            'sector_count': len(sectors)
        }

    def get_grouped_daily(self, date: str):
        """Get cached grouped daily data for a specific date from normalized table"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT ticker, open, high, low, close, volume, vwap, transactions
            FROM daily_ticker_data
            WHERE date = ?
        ''', (date,))
        rows = cursor.fetchall()

        if not rows:
            return None

        # Convert to Polygon API format
        results = []
        for row in rows:
            results.append({
                'T': row[0],      # ticker
                'o': row[1],      # open
                'h': row[2],      # high
                'l': row[3],      # low
                'c': row[4],      # close
                'v': row[5],      # volume
                'vw': row[6],     # vwap
                'n': row[7]       # transactions
            })
        return results

    def insert_grouped_daily(self, date: str, results: list):
        """Cache grouped daily data for a specific date to normalized table"""
        cursor = self.conn.cursor()

        # Bulk insert with INSERT OR REPLACE
        for ticker_data in results:
            cursor.execute('''
                INSERT OR REPLACE INTO daily_ticker_data
                (date, ticker, open, high, low, close, volume, vwap, transactions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date,
                ticker_data.get('T'),
                ticker_data.get('o'),
                ticker_data.get('h'),
                ticker_data.get('l'),
                ticker_data.get('c'),
                ticker_data.get('v'),
                ticker_data.get('vw'),
                ticker_data.get('n')
            ))

        self.conn.commit()

    def get_grouped_daily_cache_stats(self):
        """Get statistics about daily ticker data cache"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT DISTINCT date FROM daily_ticker_data ORDER BY date DESC')
        rows = cursor.fetchall()

        cursor.execute('SELECT COUNT(*) FROM daily_ticker_data')
        total_rows = cursor.fetchone()[0]

        return {
            'total_dates_cached': len(rows),
            'total_ticker_records': total_rows,
            'dates': [row[0] for row in rows]
        }

    def get_trump_social_posts(self, limit: int = 50, topic: str = None):
        """
        Get cached Trump social posts from database.

        Args:
            limit: Maximum number of posts to return
            topic: Optional keyword filter

        Returns:
            List of post dictionaries
        """
        cursor = self.conn.cursor()

        if topic:
            query = '''
                SELECT post_text, timestamp, likes, retweets, scraped_at
                FROM trump_social_posts
                WHERE post_text LIKE ?
                ORDER BY scraped_at DESC
                LIMIT ?
            '''
            cursor.execute(query, (f'%{topic}%', limit))
        else:
            query = '''
                SELECT post_text, timestamp, likes, retweets, scraped_at
                FROM trump_social_posts
                ORDER BY scraped_at DESC
                LIMIT ?
            '''
            cursor.execute(query, (limit,))

        rows = cursor.fetchall()

        posts = []
        for row in rows:
            posts.append({
                'text': row[0],
                'timestamp': row[1],
                'likes': row[2],
                'retweets': row[3],
                'scraped_at': row[4]
            })

        return posts

    def get_latest_trump_post_timestamp(self):
        """
        Get the timestamp of the most recent Trump social post in the database.

        Returns:
            Timestamp string or None if no posts exist
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT timestamp
            FROM trump_social_posts
            ORDER BY scraped_at DESC
            LIMIT 1
        ''')
        row = cursor.fetchone()
        return row[0] if row else None

    def insert_trump_social_posts(self, posts: list):
        """
        Insert Trump social posts into database.

        Args:
            posts: List of post dictionaries with keys: text, timestamp, likes, retweets
        """
        cursor = self.conn.cursor()

        for post in posts:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO trump_social_posts (post_text, timestamp, likes, retweets)
                    VALUES (?, ?, ?, ?)
                ''', (
                    post.get('text', ''),
                    post.get('timestamp', 'Recent'),
                    post.get('likes', 'N/A'),
                    post.get('retweets', 'N/A')
                ))
            except Exception as e:
                # Skip duplicates
                continue

        self.conn.commit()

    def count_trump_social_posts(self):
        """Get total count of Trump social posts in database."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM trump_social_posts')
        row = cursor.fetchone()
        return row[0] if row else 0

    def insert_option_trades_bulk(self, trades: list):
        """
        Bulk insert option trades.
        trades: list of tuples (ticker, timestamp, price, size, side, action)
        """
        cursor = self.conn.cursor()
        cursor.executemany('''
            INSERT INTO option_trades (ticker, timestamp, price, size, side, action)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', trades)
        self.conn.commit()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def insert_intraday_data(self, ticker: str, data: list):
        """
        Insert intraday (1min) data into database.
        
        Args:
            ticker: Stock symbol
            data: List of dictionaries with keys: date (timestamp), open, high, low, close, volume
        """
        cursor = self.conn.cursor()
        
        for row in data:
            # FMP 1min data usually has 'date' as the timestamp string "YYYY-MM-DD HH:MM:SS"
            timestamp = row.get('date')
            if not timestamp:
                continue
                
            # Extract just the date part "YYYY-MM-DD" for indexing
            date_part = timestamp.split(' ')[0]
            
            cursor.execute('''
                INSERT OR REPLACE INTO intraday_ticker_data 
                (ticker, timestamp, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker.upper(),
                timestamp,
                date_part,
                row.get('open'),
                row.get('high'),
                row.get('low'),
                row.get('close'),
                row.get('volume')
            ))
            
        self.conn.commit()

    def get_intraday_data(self, ticker: str, date: str):
        """
        Get intraday data for a specific date.
        
        Args:
            ticker: Stock symbol
            date: Date string "YYYY-MM-DD"
            
        Returns:
            List of dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT timestamp, open, high, low, close, volume
            FROM intraday_ticker_data
            WHERE ticker = ? AND date = ?
            ORDER BY timestamp ASC
        ''', (ticker.upper(), date))
        
        rows = cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                'date': row[0], # Keep 'date' key to match FMP format for compatibility
                'open': row[1],
                'high': row[2],
                'low': row[3],
                'close': row[4],
                'volume': row[5]
            })
            
        return results

    def insert_option_bars(self, ticker: str, bars: list, timespan: str = 'day'):
        """
        Insert historical option bars into database.
        
        Args:
            ticker: Option ticker (e.g., O:SPY...)
            bars: List of bar dictionaries (Polygon format)
            timespan: Time interval (default: 'day')
        """
        cursor = self.conn.cursor()
        
        for bar in bars:
            # Polygon bars usually have 't' (timestamp ms), 'o', 'h', 'l', 'c', 'v', 'vw', 'n'
            ts_ms = bar.get('t')
            if not ts_ms:
                continue
                
            # Convert timestamp to date string YYYY-MM-DD
            from datetime import datetime
            date_str = datetime.fromtimestamp(ts_ms / 1000.0).strftime('%Y-%m-%d')
            
            cursor.execute('''
                INSERT OR REPLACE INTO option_price_history 
                (ticker, date, timestamp, open, high, low, close, volume, vwap, transactions, timespan)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker,
                date_str,
                ts_ms,
                bar.get('o'),
                bar.get('h'),
                bar.get('l'),
                bar.get('c'),
                bar.get('v'),
                bar.get('vw'),
                bar.get('n'),
                timespan
            ))
            
        self.conn.commit()

    def get_option_bars(self, ticker: str, start_date: str = None, end_date: str = None, timespan: str = 'day'):
        """
        Get historical option bars from database.
        
        Args:
            ticker: Option ticker
            start_date: Optional start date YYYY-MM-DD
            end_date: Optional end date YYYY-MM-DD
            timespan: Time interval
            
        Returns:
            List of bar dictionaries
        """
        cursor = self.conn.cursor()
        
        query = '''
            SELECT timestamp, open, high, low, close, volume, vwap, transactions
            FROM option_price_history
            WHERE ticker = ? AND timespan = ?
        '''
        params = [ticker, timespan]
        
        if start_date:
            query += ' AND date >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND date <= ?'
            params.append(end_date)
            
        query += ' ORDER BY timestamp ASC'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                't': row[0],
                'o': row[1],
                'h': row[2],
                'l': row[3],
                'c': row[4],
                'v': row[5],
                'vw': row[6],
                'n': row[7]
            })
            
        return results

    def insert_option_history_bulk(self, history_data: list):
        """
        Bulk insert option price history (OHLC).
        history_data: list of tuples (ticker, date, timestamp, open, high, low, close, volume, vwap, transactions, timespan)
        """
        cursor = self.conn.cursor()
        cursor.executemany('''
            INSERT OR REPLACE INTO option_price_history 
            (ticker, date, timestamp, open, high, low, close, volume, vwap, transactions, timespan)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', history_data)
        self.conn.commit()

    def insert_realtime_quote(self, ticker: str, price: float, bid_price: float,
                               ask_price: float, volume: float, source: str,
                               quote_timestamp: str):
        """Insert a single real-time quote snapshot"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO realtime_quotes
            (ticker, price, bid_price, ask_price, volume, source, quote_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker.upper(),
            price,
            bid_price,
            ask_price,
            volume,
            source.upper(),
            quote_timestamp
        ))
        self.conn.commit()

    def insert_option_greeks(self, greeks_data: list):
        """
        Insert option greeks data.
        greeks_data: list of dicts
        """
        cursor = self.conn.cursor()
        for item in greeks_data:
            # Check if columns exist (simple migration for dev)
            # In production, use proper migrations. 
            # Here we just try to insert with all columns.
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO option_greeks 
                    (ticker, timestamp, date, delta, gamma, theta, vega, rho, 
                     vanna, charm, vomma, veta, vera, speed, zomma, color, ultima, 
                     d1, d2, dual_delta, dual_gamma, implied_vol, underlying_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    item['ticker'],
                    item['timestamp'],
                    item['date'],
                    item.get('delta'),
                    item.get('gamma'),
                    item.get('theta'),
                    item.get('vega'),
                    item.get('rho'),
                    item.get('vanna'),
                    item.get('charm'),
                    item.get('vomma'),
                    item.get('veta'),
                    item.get('vera'),
                    item.get('speed'),
                    item.get('zomma'),
                    item.get('color'),
                    item.get('ultima'),
                    item.get('d1'),
                    item.get('d2'),
                    item.get('dual_delta'),
                    item.get('dual_gamma'),
                    item.get('implied_vol'),
                    item.get('underlying_price')
                ))
            except sqlite3.OperationalError as e:
                # If column missing, we might need to alter table or just ignore for now
                # For this session, let's assume the table is recreated or we just log
                print(f"Error inserting greeks: {e}")
                break
        self.conn.commit()

    def insert_option_open_interest(self, oi_data: list):
        """
        Insert option open interest data.
        oi_data: list of dicts with keys: ticker, date, open_interest
        """
        cursor = self.conn.cursor()
        for item in oi_data:
            cursor.execute('''
                INSERT OR REPLACE INTO option_open_interest 
                (ticker, date, open_interest)
                VALUES (?, ?, ?)
            ''', (
                item['ticker'],
                item['date'],
                item['open_interest']
            ))
        self.conn.commit()

    def insert_massive_option_chain(self, chain_data: list):
        """
        Insert data from Massive options chain.
        chain_data: list of dicts containing option details.
        Expected keys in dict: ticker, underlying_asset, expiration_date, strike_price, contract_type, details (dict/json)
        """
        cursor = self.conn.cursor()
        import json
        
        for item in chain_data:
            details_json = json.dumps(item.get('details', {})) if isinstance(item.get('details'), dict) else item.get('details', '{}')
            
            cursor.execute('''
                INSERT OR REPLACE INTO massive_option_chain 
                (ticker, underlying_asset, expiration_date, strike_price, contract_type, details, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                item.get('ticker'),
                item.get('underlying_asset'),
                item.get('expiration_date'),
                item.get('strike_price'),
                item.get('contract_type'),
                details_json
            ))
        self.conn.commit()

    def get_massive_option_chain(self, ticker: str = None, underlying_asset: str = None):
        """
        Get Massive options chain data.
        """
        cursor = self.conn.cursor()
        query = "SELECT * FROM massive_option_chain WHERE 1=1"
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        
        if underlying_asset:
            query += " AND underlying_asset = ?"
            params.append(underlying_asset)
            
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        import json
        for row in rows:
            # row keys: ticker, underlying_asset, expiration_date, strike_price, contract_type, details, fetched_at
            results.append({
                'ticker': row[0],
                'underlying_asset': row[1],
                'expiration_date': row[2],
                'strike_price': row[3],
                'contract_type': row[4],
                'details': json.loads(row[5]) if row[5] else {},
                'fetched_at': row[6]
            })
        return results

    # =============================================
    # 5-MINUTE INTRADAY DATA (FMP)
    # =============================================
    
    def insert_intraday_5min(self, ticker: str, data: list):
        """
        Insert 5-minute intraday data from FMP.
        
        Args:
            ticker: Stock symbol (e.g., SPY)
            data: List of dicts with keys: date, open, high, low, close, volume
        """
        cursor = self.conn.cursor()
        
        for row in data:
            timestamp = row.get('date')
            if not timestamp:
                continue
            
            # Extract date part for indexing
            date_part = timestamp.split(' ')[0] if ' ' in timestamp else timestamp[:10]
            
            cursor.execute('''
                INSERT OR REPLACE INTO intraday_5min_data
                (ticker, timestamp, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker.upper(),
                timestamp,
                date_part,
                row.get('open'),
                row.get('high'),
                row.get('low'),
                row.get('close'),
                row.get('volume')
            ))
        
        self.conn.commit()
    
    def get_intraday_5min(self, ticker: str, start_date: str = None, end_date: str = None) -> list:
        """
        Get 5-minute intraday data from cache.
        
        Args:
            ticker: Stock symbol
            start_date: Optional start date YYYY-MM-DD
            end_date: Optional end date YYYY-MM-DD
            
        Returns:
            List of dicts matching FMP format
        """
        cursor = self.conn.cursor()
        
        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM intraday_5min_data
            WHERE ticker = ?
        '''
        params = [ticker.upper()]
        
        if start_date:
            query += ' AND date >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND date <= ?'
            params.append(end_date)
        
        query += ' ORDER BY timestamp ASC'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                'date': row[0],
                'open': row[1],
                'high': row[2],
                'low': row[3],
                'close': row[4],
                'volume': row[5]
            })
        
        return results
    
    def get_underlying_price_on_date(self, ticker: str, date: str) -> float:
        """Get the close price of underlying on a specific date (for filtering options)"""
        cursor = self.conn.cursor()
        # Try to get the average close price for that day from 5min data
        cursor.execute('''
            SELECT AVG(close) as avg_close
            FROM intraday_5min_data
            WHERE ticker = ? AND date = ?
        ''', (ticker.upper(), date))
        row = cursor.fetchone()
        if row and row[0]:
            return float(row[0])
        
        # Fall back to daily data
        cursor.execute('''
            SELECT close FROM daily_ticker_data
            WHERE ticker = ? AND date = ?
        ''', (ticker.upper(), date))
        row = cursor.fetchone()
        if row and row[0]:
            return float(row[0])
        
        return None
    
    def get_intraday_5min_cached_dates(self, ticker: str) -> set:
        """Get set of dates that have 5min data cached"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT DISTINCT date FROM intraday_5min_data
            WHERE ticker = ?
        ''', (ticker.upper(),))
        return set(row[0] for row in cursor.fetchall())
    
    def get_intraday_5min_stats(self, ticker: str = None) -> dict:
        """Get statistics about cached 5min data"""
        cursor = self.conn.cursor()
        
        if ticker:
            cursor.execute('''
                SELECT 
                    ticker,
                    MIN(date) as first_date,
                    MAX(date) as last_date,
                    COUNT(DISTINCT date) as num_days,
                    COUNT(*) as total_bars,
                    MAX(cached_at) as last_updated
                FROM intraday_5min_data
                WHERE ticker = ?
                GROUP BY ticker
            ''', (ticker.upper(),))
        else:
            cursor.execute('''
                SELECT 
                    ticker,
                    MIN(date) as first_date,
                    MAX(date) as last_date,
                    COUNT(DISTINCT date) as num_days,
                    COUNT(*) as total_bars,
                    MAX(cached_at) as last_updated
                FROM intraday_5min_data
                GROUP BY ticker
            ''')
        
        rows = cursor.fetchall()
        stats = {}
        for row in rows:
            stats[row[0]] = {
                'first_date': row[1],
                'last_date': row[2],
                'num_days': row[3],
                'total_bars': row[4],
                'last_updated': row[5]
            }
        return stats

    # =============================================
    # OPTIONS INTRADAY DATA (Massive/Polygon)
    # =============================================
    
    def _init_options_intraday_table(self):
        """Initialize options intraday tables"""
        cursor = self.conn.cursor()
        
        # Options minute-level OHLCV data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS options_intraday (
                option_ticker TEXT NOT NULL,
                underlying TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                vwap REAL,
                transactions INTEGER,
                timespan TEXT DEFAULT 'minute',
                expiration TEXT,
                strike REAL,
                option_type TEXT,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (option_ticker, timestamp, timespan)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_opt_intra_underlying ON options_intraday(underlying, date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_opt_intra_date ON options_intraday(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_opt_intra_expiration ON options_intraday(expiration)')
        
        # Track which option contracts we've cached
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS options_contracts_cached (
                option_ticker TEXT NOT NULL,
                underlying TEXT NOT NULL,
                expiration TEXT NOT NULL,
                strike REAL NOT NULL,
                option_type TEXT NOT NULL,
                first_date TEXT,
                last_date TEXT,
                bar_count INTEGER DEFAULT 0,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (option_ticker)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_opt_cached_underlying ON options_contracts_cached(underlying, expiration)')
        
        self.conn.commit()
    
    def insert_options_intraday(self, data: list):
        """
        Insert options intraday OHLCV data.
        
        Args:
            data: List of dicts with keys:
                - option_ticker: e.g., O:SPY250117C00500000
                - underlying: e.g., SPY
                - timestamp: Unix timestamp in ms
                - open, high, low, close, volume, vwap, transactions
                - expiration: YYYY-MM-DD
                - strike: float
                - option_type: call/put
        """
        self._init_options_intraday_table()
        cursor = self.conn.cursor()
        
        from datetime import datetime
        
        for row in data:
            ts_ms = row.get('timestamp') or row.get('t')
            if not ts_ms:
                continue
            
            # Convert timestamp
            dt = datetime.fromtimestamp(ts_ms / 1000.0)
            date_str = dt.strftime('%Y-%m-%d')
            time_str = dt.strftime('%H:%M:%S')
            
            cursor.execute('''
                INSERT OR REPLACE INTO options_intraday
                (option_ticker, underlying, timestamp, date, time, open, high, low, close, 
                 volume, vwap, transactions, timespan, expiration, strike, option_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row.get('option_ticker'),
                row.get('underlying'),
                ts_ms,
                date_str,
                time_str,
                row.get('open') or row.get('o'),
                row.get('high') or row.get('h'),
                row.get('low') or row.get('l'),
                row.get('close') or row.get('c'),
                row.get('volume') or row.get('v'),
                row.get('vwap') or row.get('vw'),
                row.get('transactions') or row.get('n'),
                row.get('timespan', 'minute'),
                row.get('expiration'),
                row.get('strike'),
                row.get('option_type')
            ))
        
        self.conn.commit()
    
    def get_options_intraday(self, underlying: str = None, option_ticker: str = None,
                             date: str = None, start_date: str = None, end_date: str = None,
                             expiration: str = None, option_type: str = None,
                             strike_min: float = None, strike_max: float = None) -> list:
        """
        Get options intraday data with flexible filtering.
        
        Args:
            underlying: Filter by underlying (e.g., SPY)
            option_ticker: Filter by specific option ticker
            date: Filter by specific date
            start_date/end_date: Date range filter
            expiration: Filter by expiration date
            option_type: Filter by call/put
            strike_min/strike_max: Strike price range
            
        Returns:
            List of bar dicts
        """
        self._init_options_intraday_table()
        cursor = self.conn.cursor()
        
        query = 'SELECT * FROM options_intraday WHERE 1=1'
        params = []
        
        if option_ticker:
            query += ' AND option_ticker = ?'
            params.append(option_ticker)
        
        if underlying:
            query += ' AND underlying = ?'
            params.append(underlying.upper())
        
        if date:
            query += ' AND date = ?'
            params.append(date)
        
        if start_date:
            query += ' AND date >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND date <= ?'
            params.append(end_date)
        
        if expiration:
            query += ' AND expiration = ?'
            params.append(expiration)
        
        if option_type:
            query += ' AND option_type = ?'
            params.append(option_type.lower())
        
        if strike_min is not None:
            query += ' AND strike >= ?'
            params.append(strike_min)
        
        if strike_max is not None:
            query += ' AND strike <= ?'
            params.append(strike_max)
        
        query += ' ORDER BY timestamp ASC'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        
        results = []
        for row in rows:
            results.append(dict(zip(columns, row)))
        
        return results
    
    def get_0dte_options_for_date(self, underlying: str, date: str) -> list:
        """
        Get all 0DTE options data for a specific date.
        0DTE = options expiring on the same day.
        
        Args:
            underlying: e.g., SPY
            date: YYYY-MM-DD
            
        Returns:
            List of option bar dicts where expiration == date
        """
        return self.get_options_intraday(
            underlying=underlying,
            date=date,
            expiration=date
        )
    
    def update_options_contract_cache_info(self, option_ticker: str, underlying: str,
                                           expiration: str, strike: float, option_type: str,
                                           first_date: str, last_date: str, bar_count: int):
        """Update the cache tracking table for an option contract"""
        self._init_options_intraday_table()
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO options_contracts_cached
            (option_ticker, underlying, expiration, strike, option_type, first_date, last_date, bar_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (option_ticker, underlying.upper(), expiration, strike, option_type.lower(),
              first_date, last_date, bar_count))
        
        self.conn.commit()
    
    def get_cached_option_contracts(self, underlying: str = None, expiration: str = None) -> list:
        """Get list of cached option contracts"""
        self._init_options_intraday_table()
        cursor = self.conn.cursor()
        
        query = 'SELECT * FROM options_contracts_cached WHERE 1=1'
        params = []
        
        if underlying:
            query += ' AND underlying = ?'
            params.append(underlying.upper())
        
        if expiration:
            query += ' AND expiration = ?'
            params.append(expiration)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def get_options_intraday_stats(self, underlying: str = None) -> dict:
        """Get statistics about cached options intraday data"""
        self._init_options_intraday_table()
        cursor = self.conn.cursor()
        
        if underlying:
            cursor.execute('''
                SELECT 
                    underlying,
                    COUNT(DISTINCT option_ticker) as contracts,
                    COUNT(DISTINCT date) as trading_days,
                    COUNT(*) as total_bars,
                    MIN(date) as first_date,
                    MAX(date) as last_date,
                    COUNT(DISTINCT expiration) as expirations
                FROM options_intraday
                WHERE underlying = ?
                GROUP BY underlying
            ''', (underlying.upper(),))
        else:
            cursor.execute('''
                SELECT 
                    underlying,
                    COUNT(DISTINCT option_ticker) as contracts,
                    COUNT(DISTINCT date) as trading_days,
                    COUNT(*) as total_bars,
                    MIN(date) as first_date,
                    MAX(date) as last_date,
                    COUNT(DISTINCT expiration) as expirations
                FROM options_intraday
                GROUP BY underlying
            ''')
        
        rows = cursor.fetchall()
        stats = {}
        for row in rows:
            stats[row[0]] = {
                'contracts': row[1],
                'trading_days': row[2],
                'total_bars': row[3],
                'first_date': row[4],
                'last_date': row[5],
                'expirations': row[6]
            }
        return stats


