from massive import RESTClient
try:
    from massive.websocket import WebSocketClient
    from massive.websocket.models import Feed, Market
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    WebSocketClient = None
    Feed = None
    Market = None

# Database import is optional
try:
    from clients.database import MarketDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    MarketDatabase = None

import datetime
import json
import os
import threading
import time
from typing import Callable, Optional, List, Dict, Any
from queue import Queue
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



class OptionsWebSocketClient:
    """
    WebSocket client for streaming real-time options data from Massive API.
    Data has 15-minute delay for non-professional subscribers.
    """
    
    def __init__(self, api_key: str, subscriptions: List[str] = None):
        if not WEBSOCKET_AVAILABLE:
            raise ImportError("WebSocket support not available. Install: pip install massive[websocket]")
        
        self.api_key = api_key
        self.subscriptions = subscriptions or []
        # Keep the feed/market params so we can recreate the client on reconnect
        self._feed = Feed.Delayed  # 15-min delayed data (most common plan)
        self._market = Market.Options
        self.ws_client = None
        self.is_connected = False
        self.last_message_time = None
        self.message_count = 0
        self.error_message = None
        self._running = False
        self._thread = None
        self._message_queue = Queue()
        self._callbacks = []
        
    def add_callback(self, callback: Callable):
        """Add a callback function to receive messages"""
        self._callbacks.append(callback)
        
    def remove_callback(self, callback: Callable):
        """Remove a callback function"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _handle_message(self, msgs):
        """Internal message handler"""
        self.last_message_time = datetime.datetime.now()
        self.message_count += len(msgs) if isinstance(msgs, list) else 1
        
        # Put messages in queue for async processing
        for msg in (msgs if isinstance(msgs, list) else [msgs]):
            self._message_queue.put(msg)
            
        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(msgs)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def connect(self, subscriptions: List[str] = None):
        """
        Connect to websocket and start streaming.
        
        Args:
            subscriptions: List of subscription strings, e.g.:
                - "T.O:QQQ*" for all QQQ options trades
                - "Q.O:QQQ*" for all QQQ options quotes
                - "T.O:QQQ230616C00400000" for specific contract trades
        """
        if subscriptions:
            self.subscriptions = subscriptions
            
        if not self.subscriptions:
            raise ValueError("No subscriptions provided")
        
        try:
            # Use Delayed feed for 15-min delayed data (most common plan)
            self.ws_client = self._build_ws_client()
            
            self._running = True
            self.is_connected = True
            self.error_message = None
            
            # Run in background thread
            self._thread = threading.Thread(target=self._run_websocket, daemon=True)
            self._thread.start()
            
            print(f"[OK] WebSocket connected with {len(self.subscriptions)} subscriptions")
            return True
            
        except Exception as e:
            self.is_connected = False
            self.error_message = str(e)
            print(f"[ERR] WebSocket connection failed: {e}")
            return False
    
    def _build_ws_client(self):
        """Create a WebSocketClient with stored params (for reconnects)."""
        return WebSocketClient(
            api_key=self.api_key,
            feed=self._feed,
            market=self._market,
            subscriptions=self.subscriptions,
            max_reconnects=5,
            verbose=False
        )

    def _run_websocket(self):
        """Run websocket in background thread"""
        while self._running:
            try:
                if not self.ws_client:
                    self.ws_client = self._build_ws_client()
                # mark as connected before entering run loop
                self.is_connected = True
                self.error_message = None
                self.ws_client.run(self._handle_message)
            except Exception as e:
                self.is_connected = False
                self.error_message = str(e)
                print(f"WebSocket error: {e}")
                if not self._running:
                    break
                # brief backoff then try to recreate client
                time.sleep(2)
                self.ws_client = None
                continue
            else:
                # run() exited cleanly; stop loop
                break
        self._running = False
        self.is_connected = False
    
    def disconnect(self):
        """Disconnect from websocket"""
        self._running = False
        self.is_connected = False
        if self.ws_client:
            try:
                self.ws_client = None
            except:
                pass
        print("WebSocket disconnected")
    
    def subscribe(self, *subscriptions: str):
        """Subscribe to additional data streams"""
        if self.ws_client and self.is_connected:
            self.ws_client.subscribe(*subscriptions)
            self.subscriptions.extend(subscriptions)
    
    def unsubscribe(self, *subscriptions: str):
        """Unsubscribe from data streams"""
        if self.ws_client and self.is_connected:
            self.ws_client.unsubscribe(*subscriptions)
            for sub in subscriptions:
                if sub in self.subscriptions:
                    self.subscriptions.remove(sub)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            'connected': self.is_connected,
            'last_message': self.last_message_time.isoformat() if self.last_message_time else None,
            'message_count': self.message_count,
            'subscriptions': len(self.subscriptions),
            'error': self.error_message,
            'data_delay': '15-min delayed'
        }
    
    def get_messages(self, max_messages: int = 100) -> List:
        """Get pending messages from queue"""
        messages = []
        while not self._message_queue.empty() and len(messages) < max_messages:
            messages.append(self._message_queue.get_nowait())
        return messages


class MassiveOptionsClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")  # Support both
        if not self.api_key:
            raise ValueError("API Key not found. Please set MASSIVE_API_KEY in .env or pass it to the constructor.")
        self.client = RESTClient(self.api_key)
        self.db = MarketDatabase() if DATABASE_AVAILABLE else None
        self.ws_client: Optional[OptionsWebSocketClient] = None

    def test_connectivity(self):
        """
        Test connectivity by fetching simple ticker details.
        """
        print("Testing API connectivity with ticker details (AAPL)...")
        try:
            details = self.client.get_ticker_details("AAPL")
            print("Connectivity Success! Ticker details fetched.")
            return True
        except Exception as e:
            print(f"Connectivity Check Failed: {e}")
            return False

    def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status using real-time quote data.
        Returns market status info based on quote timestamps and trading activity.
        """
        try:
            # Get a real-time quote for SPY (most liquid ETF)
            snapshot = self.get_universal_snapshot("SPY")
            
            now = datetime.datetime.now()
            hour = now.hour
            minute = now.minute
            day_of_week = now.weekday()  # 0=Monday, 6=Sunday
            
            # Check for weekend
            if day_of_week >= 5:
                return {
                    'status': 'CLOSED',
                    'session': 'Weekend',
                    'is_trading': False,
                    'message': 'Markets closed for weekend',
                    'timestamp': now.isoformat()
                }
            
            # Determine session based on time (EST)
            if hour < 4:
                session = 'Overnight'
                is_trading = False
                status = 'CLOSED'
            elif hour < 9 or (hour == 9 and minute < 30):
                session = 'Pre-Market'
                is_trading = True
                status = 'PRE-MARKET'
            elif hour < 16:
                session = 'Regular'
                is_trading = True
                status = 'OPEN'
            elif hour < 20:
                session = 'After-Hours'
                is_trading = True
                status = 'AFTER-HOURS'
            else:
                session = 'Overnight'
                is_trading = False
                status = 'CLOSED'
            
            # Verify with actual quote data if available
            quote_time = None
            last_price = None
            if snapshot:
                if hasattr(snapshot, 'last_trade'):
                    lt = snapshot.last_trade
                    if hasattr(lt, 'timestamp'):
                        quote_time = lt.timestamp
                    if hasattr(lt, 'price'):
                        last_price = lt.price
                elif hasattr(snapshot, 'session'):
                    s = snapshot.session
                    if hasattr(s, 'close'):
                        last_price = s.close
            
            return {
                'status': status,
                'session': session,
                'is_trading': is_trading,
                'message': f'{session} Session ({status})',
                'last_price': last_price,
                'quote_time': str(quote_time) if quote_time else None,
                'timestamp': now.isoformat()
            }
            
        except Exception as e:
            # Fallback to time-based calculation
            now = datetime.datetime.now()
            hour = now.hour
            day_of_week = now.weekday()
            
            if day_of_week >= 5:
                status = 'CLOSED'
                session = 'Weekend'
            elif hour < 4 or hour >= 20:
                status = 'CLOSED'
                session = 'Overnight'
            elif hour < 9 or (hour == 9 and now.minute < 30):
                status = 'PRE-MARKET'
                session = 'Pre-Market'
            elif hour < 16:
                status = 'OPEN'
                session = 'Regular'
            else:
                status = 'AFTER-HOURS'
                session = 'After-Hours'
            
            return {
                'status': status,
                'session': session,
                'is_trading': status != 'CLOSED',
                'message': f'{session} Session ({status})',
                'error': str(e),
                'timestamp': now.isoformat()
            }

    def start_options_stream(self, underlying: str, callback: Callable = None) -> bool:
        """
        Start WebSocket stream for options on an underlying asset.
        
        Args:
            underlying: Ticker symbol (e.g., 'QQQ', 'SPY')
            callback: Optional callback function for messages
            
        Returns:
            True if connection successful
        """
        if not WEBSOCKET_AVAILABLE:
            print("[WARN] WebSocket not available. Install: pip install massive[websocket]")
            return False
        
        # Create subscriptions for trades and quotes on all options
        subscriptions = [
            f"T.O:{underlying}*",  # All options trades
            f"Q.O:{underlying}*",  # All options quotes
        ]
        
        self.ws_client = OptionsWebSocketClient(self.api_key, subscriptions)
        
        if callback:
            self.ws_client.add_callback(callback)
        
        return self.ws_client.connect()
    
    def stop_options_stream(self):
        """Stop the WebSocket stream"""
        if self.ws_client:
            self.ws_client.disconnect()
            self.ws_client = None
    
    def get_stream_status(self) -> Dict[str, Any]:
        """Get WebSocket stream status"""
        if self.ws_client:
            return self.ws_client.get_status()
        return {
            'connected': False,
            'last_message': None,
            'message_count': 0,
            'subscriptions': 0,
            'error': 'Not initialized',
            'data_delay': '15-min delayed'
        }

    def fetch_and_store_all_contracts(self, underlying_ticker, limit=1000):
        """
        Fetches the list of options contracts for a given underlying ticker
        and stores them in the database.
        Uses list_options_contracts (Reference Data) instead of Snapshots.
        """
        print(f"Fetching options contracts for {underlying_ticker}...")
        try:
            contracts = []
            count = 0
            # Iterate over contracts
            for c in self.client.list_options_contracts(underlying_ticker=underlying_ticker, limit=1000):
                contracts.append(c)
                count += 1
                if count >= limit:
                    print(f"Reached limit of {limit} contracts.")
                    break
            
            print(f"Fetched {len(contracts)} contracts.")
            
            # Process and store
            db_data = []
            for c in contracts:
                # c is likely a Contract object
                c_dict = c.__dict__ if hasattr(c, '__dict__') else c
                
                ticker = c_dict.get('ticker')
                underlying = c_dict.get('underlying_ticker')
                expiration = c_dict.get('expiration_date')
                strike = c_dict.get('strike_price')
                ctype = c_dict.get('contract_type')
                
                db_data.append({
                    'ticker': ticker,
                    'underlying_asset': underlying,
                    'expiration_date': expiration,
                    'strike_price': strike,
                    'contract_type': ctype,
                    'details': c_dict
                })
            
            if db_data:
                self.db.insert_massive_option_chain(db_data)
                print(f"Stored {len(db_data)} records in database.")
            else:
                print("No data to store.")
                
        except Exception as e:
            print(f"Error fetching/storing options contracts: {e}")

    def list_options_contracts(self, **kwargs):
        """
        List options contracts with filtering.
        Wraps client.list_options_contracts.
        """
        return self.client.list_options_contracts(**kwargs)

    def get_historical_aggregates(self, ticker, multiplier, timespan, from_date, to_date, **kwargs):
        """
        Get historical aggregates (bars).
        Wraps client.list_aggs.
        """
        return self.client.list_aggs(ticker, multiplier, timespan, from_date, to_date, **kwargs)

    def get_universal_snapshot(self, ticker):
        """
        Get a snapshot for a specific ticker using the Universal Snapshot endpoint.
        This endpoint often includes Open Interest and works where others fail.
        """
        try:
            it = self.client.list_universal_snapshots(
                ticker_any_of=[ticker],
                limit=1
            )
            snapshots = [s for s in it]
            return snapshots[0] if snapshots else None
        except Exception as e:
            print(f"Error fetching universal snapshot for {ticker}: {e}")
            return None

    def fetch_chain_snapshots(self, underlying_ticker, limit=1000):
        """
        Fetches snapshots for the options chain of an underlying using Universal Snapshots.
        This is the best way to get Open Interest for many contracts.
        Strategy: 
        1. List contracts to get valid tickers.
        2. Batch fetch snapshots for those tickers.
        """
        print(f"Fetching chain snapshots for {underlying_ticker}...")
        try:
            # Step 1: Get valid tickers
            tickers = []
            # We limit the contract listing to avoid fetching the entire universe if not needed
            # But for a full chain analysis, we might need more. 
            # Let's stick to the limit passed in.
            for c in self.client.list_options_contracts(underlying_ticker=underlying_ticker, limit=limit):
                tickers.append(c.ticker)
            
            if not tickers:
                print("No contracts found.")
                return []
                
            print(f"Found {len(tickers)} contracts. Fetching snapshots...")
            
            # Step 2: Fetch snapshots in batches
            snapshots = []
            chunk_size = 50 # Safe batch size
            
            for i in range(0, len(tickers), chunk_size):
                chunk = tickers[i:i+chunk_size]
                try:
                    it = self.client.list_universal_snapshots(
                        ticker_any_of=chunk,
                        limit=len(chunk)
                    )
                    for s in it:
                        snapshots.append(s)
                except Exception as e:
                    print(f"Error fetching batch {i}: {e}")
            
            print(f"Fetched {len(snapshots)} snapshots.")
            return snapshots
            
        except Exception as e:
            print(f"Error fetching chain snapshots: {e}")
            return []
            
        except Exception as e:
            print(f"Error fetching chain snapshots: {e}")
            return []

    def get_sma(self, ticker, **kwargs):
        """
        Get Simple Moving Average (SMA).
        """
        return self.client.get_sma(ticker, **kwargs)

    def get_ema(self, ticker, **kwargs):
        """
        Get Exponential Moving Average (EMA).
        """
        return self.client.get_ema(ticker, **kwargs)

    def get_macd(self, ticker, **kwargs):
        """
        Get Moving Average Convergence Divergence (MACD).
        """
        return self.client.get_macd(ticker, **kwargs)

    def get_rsi(self, ticker, **kwargs):
        """
        Get Relative Strength Index (RSI).
        """
        return self.client.get_rsi(ticker, **kwargs)

    def get_last_option_quote(self, option_ticker: str) -> Dict[str, Any]:
        """
        Get the latest quote for an options contract.
        Returns the most recent bid/ask with timestamp.
        
        Args:
            option_ticker: Options contract ticker (e.g., 'O:AAPL250117C00200000')
            
        Returns:
            Dict with bid, ask, mid, timestamp, and data_age info
        """
        try:
            quote = self.client.get_last_quote(option_ticker)
            
            if quote:
                q_dict = quote.__dict__ if hasattr(quote, '__dict__') else quote
                bid = q_dict.get('bid_price') or q_dict.get('bid') or 0
                ask = q_dict.get('ask_price') or q_dict.get('ask') or 0
                bid_size = q_dict.get('bid_size', 0)
                ask_size = q_dict.get('ask_size', 0)
                timestamp = q_dict.get('participant_timestamp') or q_dict.get('sip_timestamp') or q_dict.get('timestamp')
                
                # Calculate data age
                data_age_seconds = None
                if timestamp:
                    # Handle nanosecond timestamps
                    if isinstance(timestamp, (int, float)):
                        if timestamp > 1e15:  # Nanoseconds
                            ts_datetime = datetime.datetime.fromtimestamp(timestamp / 1e9)
                        elif timestamp > 1e12:  # Milliseconds
                            ts_datetime = datetime.datetime.fromtimestamp(timestamp / 1e3)
                        else:
                            ts_datetime = datetime.datetime.fromtimestamp(timestamp)
                        data_age_seconds = (datetime.datetime.now() - ts_datetime).total_seconds()
                    else:
                        data_age_seconds = None
                
                return {
                    'ticker': option_ticker,
                    'bid': bid,
                    'ask': ask,
                    'mid': (bid + ask) / 2 if bid and ask else 0,
                    'bid_size': bid_size,
                    'ask_size': ask_size,
                    'timestamp': timestamp,
                    'data_age_seconds': data_age_seconds,
                    'is_realtime': data_age_seconds is not None and data_age_seconds < 60,
                    'data_delay_note': '15-min delayed for non-professional' if data_age_seconds and data_age_seconds > 900 else None
                }
            return {'ticker': option_ticker, 'error': 'No quote data available'}
        except Exception as e:
            return {'ticker': option_ticker, 'error': str(e)}

    def get_last_option_trade(self, option_ticker: str) -> Dict[str, Any]:
        """
        Get the latest trade for an options contract.
        Returns the most recent trade price with timestamp.
        
        Args:
            option_ticker: Options contract ticker (e.g., 'O:AAPL250117C00200000')
            
        Returns:
            Dict with price, size, timestamp, and data_age info
        """
        try:
            trade = self.client.get_last_trade(option_ticker)
            
            if trade:
                t_dict = trade.__dict__ if hasattr(trade, '__dict__') else trade
                price = t_dict.get('price') or t_dict.get('p') or 0
                size = t_dict.get('size') or t_dict.get('s') or 0
                timestamp = t_dict.get('participant_timestamp') or t_dict.get('sip_timestamp') or t_dict.get('timestamp')
                
                # Calculate data age
                data_age_seconds = None
                if timestamp:
                    if isinstance(timestamp, (int, float)):
                        if timestamp > 1e15:  # Nanoseconds
                            ts_datetime = datetime.datetime.fromtimestamp(timestamp / 1e9)
                        elif timestamp > 1e12:  # Milliseconds
                            ts_datetime = datetime.datetime.fromtimestamp(timestamp / 1e3)
                        else:
                            ts_datetime = datetime.datetime.fromtimestamp(timestamp)
                        data_age_seconds = (datetime.datetime.now() - ts_datetime).total_seconds()
                    else:
                        data_age_seconds = None
                
                return {
                    'ticker': option_ticker,
                    'price': price,
                    'size': size,
                    'timestamp': timestamp,
                    'data_age_seconds': data_age_seconds,
                    'is_realtime': data_age_seconds is not None and data_age_seconds < 60,
                    'data_delay_note': '15-min delayed for non-professional' if data_age_seconds and data_age_seconds > 900 else None
                }
            return {'ticker': option_ticker, 'error': 'No trade data available'}
        except Exception as e:
            return {'ticker': option_ticker, 'error': str(e)}

    def get_option_snapshot(self, option_ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive snapshot for an options contract.
        Includes quote, trade, greeks, and IV with freshness info.
        
        Args:
            option_ticker: Options contract ticker (e.g., 'O:AAPL250117C00200000')
            
        Returns:
            Dict with comprehensive option data and freshness timestamps
        """
        try:
            # Use universal snapshot (works with basic Massive tier)
            snapshot = self.get_universal_snapshot(option_ticker)
            
            if snapshot:
                snap_dict = snapshot.__dict__ if hasattr(snapshot, '__dict__') else {}
                
                result = {
                    'ticker': option_ticker,
                    'fetched_at': datetime.datetime.now().isoformat(),
                }
                
                # Extract quote data
                if hasattr(snapshot, 'last_quote') and snapshot.last_quote:
                    lq = snapshot.last_quote
                    result['bid'] = getattr(lq, 'bid', 0) or getattr(lq, 'bid_price', 0) or 0
                    result['ask'] = getattr(lq, 'ask', 0) or getattr(lq, 'ask_price', 0) or 0
                    result['mid'] = (result['bid'] + result['ask']) / 2 if result['bid'] and result['ask'] else 0
                    result['quote_timestamp'] = getattr(lq, 'timestamp', None) or getattr(lq, 'sip_timestamp', None)
                
                # Extract trade data
                if hasattr(snapshot, 'last_trade') and snapshot.last_trade:
                    lt = snapshot.last_trade
                    result['last_price'] = getattr(lt, 'price', 0) or getattr(lt, 'p', 0) or 0
                    result['last_size'] = getattr(lt, 'size', 0) or getattr(lt, 's', 0) or 0
                    result['trade_timestamp'] = getattr(lt, 'timestamp', None) or getattr(lt, 'sip_timestamp', None)
                
                # Session data (for after-hours reference)
                if hasattr(snapshot, 'session') and snapshot.session:
                    sess = snapshot.session
                    result['session_close'] = getattr(sess, 'close', 0) or 0
                    result['session_high'] = getattr(sess, 'high', 0) or 0
                    result['session_low'] = getattr(sess, 'low', 0) or 0
                    result['session_volume'] = getattr(sess, 'volume', 0) or 0
                
                # Day data
                if hasattr(snapshot, 'day') and snapshot.day:
                    d = snapshot.day
                    result['day_volume'] = getattr(d, 'volume', 0) or getattr(d, 'v', 0) or 0
                    result['day_open'] = getattr(d, 'open', 0) or getattr(d, 'o', 0) or 0
                    result['day_high'] = getattr(d, 'high', 0) or getattr(d, 'h', 0) or 0
                    result['day_low'] = getattr(d, 'low', 0) or getattr(d, 'l', 0) or 0
                
                # Open Interest
                result['open_interest'] = getattr(snapshot, 'open_interest', 0) or 0
                
                # Greeks
                if hasattr(snapshot, 'greeks') and snapshot.greeks:
                    g = snapshot.greeks
                    result['delta'] = getattr(g, 'delta', 0) or 0
                    result['gamma'] = getattr(g, 'gamma', 0) or 0
                    result['theta'] = getattr(g, 'theta', 0) or 0
                    result['vega'] = getattr(g, 'vega', 0) or 0
                
                # IV
                result['implied_volatility'] = getattr(snapshot, 'implied_volatility', 0) or 0
                
                # Calculate best available price (prefer mid, then last, then session close)
                result['best_price'] = (
                    result.get('mid') or 
                    result.get('last_price') or 
                    result.get('session_close') or 
                    0
                )
                
                # Data freshness indicator
                trade_ts = result.get('trade_timestamp')
                if trade_ts:
                    if isinstance(trade_ts, (int, float)):
                        if trade_ts > 1e15:
                            ts_dt = datetime.datetime.fromtimestamp(trade_ts / 1e9)
                        elif trade_ts > 1e12:
                            ts_dt = datetime.datetime.fromtimestamp(trade_ts / 1e3)
                        else:
                            ts_dt = datetime.datetime.fromtimestamp(trade_ts)
                        result['data_age_seconds'] = (datetime.datetime.now() - ts_dt).total_seconds()
                        result['data_freshness'] = 'real-time' if result['data_age_seconds'] < 60 else (
                            'delayed' if result['data_age_seconds'] < 900 else '15-min+ delayed'
                        )
                
                return result
            
            return {'ticker': option_ticker, 'error': 'No snapshot data available'}
        except Exception as e:
            return {'ticker': option_ticker, 'error': str(e)}

    def get_options_chain_snapshot(self, underlying: str, expiration: str = None, 
                                   option_type: str = None, limit: int = 2000) -> List[Dict[str, Any]]:
        """
        Get snapshot for entire options chain.
        Uses list_options_contracts + list_universal_snapshots for compatibility
        with basic Massive subscription tiers.
        
        Args:
            underlying: Underlying ticker (e.g., 'AAPL')
            expiration: Filter by expiration date (YYYY-MM-DD)
            option_type: Filter by 'call' or 'put'
            limit: Max contracts to fetch (default 2000)
            
        Returns:
            List of option snapshots with pricing, greeks, and timestamps
        """
        print(f"Fetching options chain snapshot for {underlying}...")
        
        # Step 1: Get contract list
        contracts = []
        kwargs = {'underlying_ticker': underlying, 'limit': limit}
        if expiration:
            kwargs['expiration_date'] = expiration
        if option_type:
            kwargs['contract_type'] = option_type
        
        try:
            for c in self.client.list_options_contracts(**kwargs):
                contracts.append(c)
        except Exception as e:
            print(f"  Error listing contracts: {e}")
            return []
        
        if not contracts:
            print(f"  No contracts found for {underlying}")
            return []
        
        print(f"  Found {len(contracts)} contracts, fetching snapshots...")
        
        # Step 2: Get snapshots in batches using universal snapshots
        tickers = [c.ticker for c in contracts]
        snapshots = []
        chunk_size = 50
        
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            try:
                for snap in self.client.list_universal_snapshots(ticker_any_of=chunk, limit=len(chunk)):
                    result = self._parse_option_snapshot(snap, underlying)
                    if result:
                        snapshots.append(result)
            except Exception as e:
                print(f"  Error fetching batch {i//chunk_size + 1}: {e}")
                # Continue with next batch instead of failing completely
                continue
        
        print(f"  Retrieved {len(snapshots)} option snapshots")
        return snapshots
    
    def _parse_option_snapshot(self, snap, underlying: str) -> Dict[str, Any]:
        """Parse a universal snapshot into standardized option data dict."""
        try:
            result = {
                'ticker': getattr(snap, 'ticker', None),
                'underlying': underlying,
                'fetched_at': datetime.datetime.now().isoformat(),
            }
            
            # Contract details
            if hasattr(snap, 'details') and snap.details:
                d = snap.details
                result['strike'] = getattr(d, 'strike_price', 0) or 0
                result['expiration'] = getattr(d, 'expiration_date', '') or ''
                result['option_type'] = getattr(d, 'contract_type', '') or ''
            
            # Quote data
            if hasattr(snap, 'last_quote') and snap.last_quote:
                lq = snap.last_quote
                result['bid'] = getattr(lq, 'bid', 0) or 0
                result['ask'] = getattr(lq, 'ask', 0) or 0
                result['mid'] = (result['bid'] + result['ask']) / 2 if result['bid'] and result['ask'] else 0
                result['quote_timestamp'] = getattr(lq, 'timestamp', None)
            
            # Trade data
            if hasattr(snap, 'last_trade') and snap.last_trade:
                lt = snap.last_trade
                result['last_price'] = getattr(lt, 'price', 0) or 0
                result['last_size'] = getattr(lt, 'size', 0) or 0
                result['trade_timestamp'] = getattr(lt, 'timestamp', None)
            
            # Day data
            if hasattr(snap, 'day') and snap.day:
                d = snap.day
                result['volume'] = getattr(d, 'volume', 0) or 0
            
            # Open Interest
            result['open_interest'] = getattr(snap, 'open_interest', 0) or 0
            
            # Greeks
            if hasattr(snap, 'greeks') and snap.greeks:
                g = snap.greeks
                result['delta'] = getattr(g, 'delta', 0) or 0
                result['gamma'] = getattr(g, 'gamma', 0) or 0
                result['theta'] = getattr(g, 'theta', 0) or 0
                result['vega'] = getattr(g, 'vega', 0) or 0
            
            # IV
            result['implied_volatility'] = getattr(snap, 'implied_volatility', 0) or 0
            
            # Session data (fallback for price)
            if hasattr(snap, 'session') and snap.session:
                sess = snap.session
                result['session_close'] = getattr(sess, 'close', 0) or 0
                result['volume'] = getattr(sess, 'volume', 0) or result.get('volume', 0)

            # Best price
            result['best_price'] = result.get('mid') or result.get('last_price') or result.get('session_close') or 0
            
            return result
        except Exception as e:
            return None




if __name__ == "__main__":
    # Example usage
    try:
        client = MassiveOptionsClient()
        if client.test_connectivity():
            # Fetch a small sample of contracts
            client.fetch_and_store_all_contracts("QQQ", limit=10)
        else:
            print("Skipping options fetch due to connectivity failure.")
    except ValueError as e:
        print(e)
