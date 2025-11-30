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

        self.conn.commit()

        # Table for historical predictions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_predictions (
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                prediction REAL,
                direction TEXT,
                confidence REAL,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (timestamp, ticker)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pred_ticker_ts ON historical_predictions(ticker, timestamp)')

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

    def insert_prediction(self, ticker: str, timestamp: str, prediction: float, direction: str, confidence: float):
        """Insert a historical prediction"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO historical_predictions 
            (timestamp, ticker, prediction, direction, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, ticker.upper(), prediction, direction, confidence))
        self.conn.commit()

    def get_predictions(self, ticker: str, limit: int = 100):
        """Get historical predictions"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT timestamp, prediction, direction, confidence, generated_at
            FROM historical_predictions
            WHERE ticker = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (ticker.upper(), limit))
        
        rows = cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                'timestamp': row[0],
                'prediction': row[1],
                'direction': row[2],
                'confidence': row[3],
                'generated_at': row[4]
            })
        return results