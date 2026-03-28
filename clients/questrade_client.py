"""
Questrade API Client - Complete Implementation
Features:
- Auto token refresh (tokens expire every 5 minutes)
- Persistent token storage for 7-day refresh tokens
- All account, order, market data, and position endpoints
- Thread-safe token management
"""
import os
import json
import time
import logging
import threading
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TokenInfo:
    """Token information from Questrade OAuth"""
    access_token: str
    refresh_token: str
    api_server: str
    token_type: str
    expires_at: float  # Unix timestamp when access token expires


class QuestradeClient:
    """
    Complete Questrade API Client with auto-refresh functionality.
    
    Features:
    - Automatic token refresh before expiration
    - Persistent token storage across restarts
    - Full account, order, position, and market data support
    - Thread-safe operations
    
    Usage:
        client = QuestradeClient(refresh_token="your_refresh_token")
        accounts = client.get_accounts()
        positions = client.get_positions(account_id)
    """
    
    # OAuth endpoints
    OAUTH_URL = "https://login.questrade.com/oauth2/token"
    PRACTICE_OAUTH_URL = "https://practicelogin.questrade.com/oauth2/token"
    
    # Token file location
    TOKEN_FILE = Path(__file__).parent / ".questrade_token.json"
    
    # Token refresh buffer (refresh 2 minutes before expiry)
    REFRESH_BUFFER_SECONDS = 120
    
    # Periodic refresh interval (refresh every 10 minutes to keep token alive)
    PERIODIC_REFRESH_SECONDS = 600
    
    def __init__(
        self, 
        refresh_token: Optional[str] = None,
        token_file: Optional[Path] = None,
        practice_mode: bool = False,
        auto_refresh: bool = True
    ):
        """
        Initialize Questrade client.
        
        Args:
            refresh_token: Initial refresh token from Questrade API Centre
            token_file: Custom path for token storage file
            practice_mode: Use practice/paper trading environment
            auto_refresh: Automatically refresh tokens before expiry
        """
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()  # Separate lock for token refresh
        self._refresh_in_progress = False
        self._last_refresh_attempt = 0
        self._token_info: Optional[TokenInfo] = None
        self._session = requests.Session()
        self._practice_mode = practice_mode
        self._auto_refresh = auto_refresh
        self._refresh_timer: Optional[threading.Timer] = None
        self._periodic_timer: Optional[threading.Timer] = None
        
        if token_file:
            self.TOKEN_FILE = token_file
        
        # Try to load existing token first
        loaded = False
        try:
            loaded = self._load_token_from_file()
        except ConnectionError:
            # File token's refresh token is dead — surface a clear error.
            # Don't fall back to the .env token because it was consumed on
            # first use and is guaranteed to be invalid.
            raise ConnectionError(
                "Stored Questrade refresh token has expired (tokens last 7 days). "
                "Generate a new token at Questrade App Hub -> API Access, "
                "then update QUESTRADE_API_KEY in .env and delete "
                f"{self.TOKEN_FILE} to force re-authentication."
            )
        
        # Only authenticate with new refresh token if no valid stored token
        if not loaded:
            if refresh_token:
                self._authenticate(refresh_token)
            else:
                raise ValueError(
                    "No refresh token provided and no stored token found. "
                    "Please provide a refresh_token from Questrade API Centre."
                )
        
        # Schedule auto-refresh if enabled
        if self._auto_refresh:
            self._schedule_token_refresh()
    
    def _get_oauth_url(self) -> str:
        """Get appropriate OAuth URL based on mode"""
        return self.PRACTICE_OAUTH_URL if self._practice_mode else self.OAUTH_URL

    @staticmethod
    def _sanitize_error_message(message: str) -> str:
        """Redact token-like values that may be present in HTTP error text."""
        if not message:
            return message
        redacted = re.sub(r"(?i)(refresh_token=)[^&\s]+", r"\1<redacted>", message)
        redacted = re.sub(r"(?i)(access_token=)[^&\s]+", r"\1<redacted>", redacted)
        return redacted
    
    def _authenticate(self, refresh_token: str) -> None:
        """
        Authenticate with Questrade and get access token.
        
        Args:
            refresh_token: The refresh token to exchange for access token
        """
        url = f"{self._get_oauth_url()}?grant_type=refresh_token&refresh_token={refresh_token}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Calculate expiration time
            expires_in = data.get("expires_in", 300)  # Default 5 minutes
            expires_at = time.time() + expires_in
            
            # Store token info
            self._token_info = TokenInfo(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                api_server=data["api_server"].rstrip("/"),
                token_type=data.get("token_type", "Bearer"),
                expires_at=expires_at
            )
            
            # Save to file for persistence
            self._save_token_to_file()
            
            logger.info(f"Successfully authenticated. Token expires in {expires_in} seconds.")
            
        except requests.exceptions.RequestException as e:
            safe_error = self._sanitize_error_message(str(e))
            logger.error(f"Authentication failed: {safe_error}")
            if "400" in safe_error:
                logger.error("="*60)
                logger.error("REFRESH TOKEN EXPIRED!")
                logger.error("Go to Questrade App Hub -> API Access -> Generate New Token")
                logger.error("Then update the .questrade_token.json file or pass new token")
                logger.error("="*60)
            raise ConnectionError(f"Failed to authenticate with Questrade: {safe_error}")
    
    def _robust_token_refresh(self, max_retries: int = 3) -> bool:
        """
        Robustly refresh the token with retry logic and race condition handling.
        
        This method:
        1. Uses a lock to prevent concurrent refresh attempts
        2. Checks if another thread already refreshed by reloading from file
        3. Retries with exponential backoff on transient failures
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if refresh succeeded, False otherwise
        """
        # Try to acquire refresh lock (non-blocking first check)
        if self._refresh_in_progress:
            # Another thread is refreshing, wait and reload from file
            logger.debug("Token refresh already in progress, waiting...")
            time.sleep(2)
            return self._reload_token_from_file_if_newer()
        
        with self._refresh_lock:
            # Double-check after acquiring lock
            if self._refresh_in_progress:
                time.sleep(1)
                return self._reload_token_from_file_if_newer()
            
            self._refresh_in_progress = True
            
            try:
                # --- Cross-process coordination ---
                # Always re-read the file BEFORE attempting refresh.
                # Another process may have already refreshed (consuming the old
                # single-use token).  If the file's refresh_token differs from
                # our in-memory copy, the file has a newer token — use it.
                if self._reload_token_from_file_if_newer():
                    return True

                # --- File-lock protected refresh ---
                # Use a lockfile so only one process refreshes at a time.
                lock_path = str(self.TOKEN_FILE) + ".lock"
                old_access = self._token_info.access_token if self._token_info else None
                last_error = None

                for attempt in range(max_retries):
                    # Before every attempt, re-check file (another process may
                    # have won the lock and refreshed while we were waiting).
                    if attempt > 0 and self._reload_token_from_file_if_newer():
                        return True

                    try:
                        # Try to acquire file lock with timeout
                        try:
                            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                            os.close(fd)
                        except FileExistsError:
                            # Another process holds the lock — wait, then reload
                            logger.debug("Another process is refreshing, waiting for lock...")
                            for _ in range(10):  # wait up to 5s
                                time.sleep(0.5)
                                if not os.path.exists(lock_path):
                                    break
                            else:
                                # Lock file stale (>5s), remove it
                                try:
                                    os.remove(lock_path)
                                except OSError:
                                    pass
                            # After waiting, check if file token is now new
                            if self._reload_token_from_file_if_newer():
                                return True
                            # Re-try to acquire lock
                            try:
                                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                                os.close(fd)
                            except FileExistsError:
                                # Still held — reload from file and give up this attempt
                                time.sleep(1)
                                if self._reload_token_from_file_if_newer():
                                    return True
                                continue

                        try:
                            if self._token_info and self._token_info.refresh_token:
                                self._authenticate(self._token_info.refresh_token)
                                return True
                            else:
                                logger.error("No refresh token available")
                                return False
                        finally:
                            # Always release file lock
                            try:
                                os.remove(lock_path)
                            except OSError:
                                pass

                    except ConnectionError as e:
                        last_error = e
                        # Release lock on failure
                        try:
                            os.remove(lock_path)
                        except OSError:
                            pass

                        if "400" in str(e):
                            # 400 = consumed token.  Another process likely refreshed.
                            # Re-read file one more time before giving up.
                            time.sleep(0.5)
                            if self._reload_token_from_file_if_newer():
                                return True
                            logger.error("Refresh token expired, cannot recover automatically")
                            return False
                        
                        # Transient error, retry with backoff
                        wait_time = (2 ** attempt) + (time.time() % 1)
                        logger.warning(f"Token refresh attempt {attempt + 1} failed, retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                
                if last_error:
                    logger.error(f"Token refresh failed after {max_retries} attempts: {last_error}")
                return False
                
            finally:
                self._refresh_in_progress = False

    def _reload_token_from_file_if_newer(self) -> bool:
        """Check if the token file was updated by another process and reload."""
        if not self.TOKEN_FILE.exists():
            return False
        try:
            with open(self.TOKEN_FILE, "r") as f:
                data = json.load(f)
            file_refresh = data.get("refresh_token", "")
            current_refresh = self._token_info.refresh_token if self._token_info else ""
            file_expires = data.get("expires_at", 0)
            
            # If the file has a different refresh token OR a later expiry, it's newer
            if file_refresh and file_refresh != current_refresh and file_expires > time.time():
                logger.info("Loading newer token from file (refreshed by another process)")
                self._token_info = TokenInfo(
                    access_token=data["access_token"],
                    refresh_token=data["refresh_token"],
                    api_server=data["api_server"],
                    token_type=data.get("token_type", "Bearer"),
                    expires_at=file_expires
                )
                return True
        except Exception as e:
            logger.debug(f"Could not check file token: {e}")
        return False
    
    def _save_token_to_file(self) -> None:
        """Save current token to file atomically for persistence"""
        if not self._token_info:
            return
        
        try:
            token_data = {
                "access_token": self._token_info.access_token,
                "refresh_token": self._token_info.refresh_token,
                "api_server": self._token_info.api_server,
                "token_type": self._token_info.token_type,
                "expires_at": self._token_info.expires_at,
                "saved_at": datetime.now().isoformat()
            }
            
            # Atomic write: write to temp file then rename
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.TOKEN_FILE.parent), suffix=".tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(token_data, f, indent=2)
                # Atomic rename (same filesystem)
                os.replace(tmp_path, str(self.TOKEN_FILE))
            except Exception:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise
            
            logger.debug(f"Token saved to {self.TOKEN_FILE}")
            
        except Exception as e:
            logger.warning(f"Failed to save token to file: {e}")
    
    def _load_token_from_file(self) -> bool:
        """
        Load token from file if it exists and is still valid.
        
        Returns:
            True if token was loaded successfully, False otherwise
        """
        if not self.TOKEN_FILE.exists():
            return False
        
        try:
            with open(self.TOKEN_FILE, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Failed to read token file: {e}")
            return False
        
        try:
            self._token_info = TokenInfo(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                api_server=data["api_server"],
                token_type=data.get("token_type", "Bearer"),
                expires_at=data["expires_at"]
            )
        except KeyError as e:
            logger.warning(f"Token file missing required field: {e}")
            return False
        
        # Check if access token is still valid
        if time.time() < self._token_info.expires_at:
            logger.info("Loaded valid token from file")
            return True
        
        # Access token expired — try refreshing with the file's refresh token
        logger.info("Stored access token expired, refreshing with file refresh token...")
        try:
            self._authenticate(self._token_info.refresh_token)
            return True
        except ConnectionError as e:
            # File refresh token is dead — let caller decide what to do.
            # Do NOT fall through to __init__'s .env token since that original
            # token was consumed on first use and is guaranteed invalid.
            logger.error(
                f"Token file refresh failed: {e}. "
                f"The stored refresh token (saved {data.get('saved_at', 'unknown')}) "
                f"has likely expired. Generate a new token at Questrade App Hub."
            )
            raise
    
    def _schedule_token_refresh(self) -> None:
        """Schedule automatic token refresh before expiry"""
        if not self._token_info:
            return
        
        # Cancel any existing timers
        if self._refresh_timer:
            self._refresh_timer.cancel()
        if self._periodic_timer:
            self._periodic_timer.cancel()
        
        # Calculate time until refresh needed
        time_until_expiry = self._token_info.expires_at - time.time()
        refresh_delay = max(0, time_until_expiry - self.REFRESH_BUFFER_SECONDS)
        
        if refresh_delay > 0:
            self._refresh_timer = threading.Timer(refresh_delay, self._auto_refresh_token)
            self._refresh_timer.daemon = True
            self._refresh_timer.start()
            logger.debug(f"Token refresh scheduled in {refresh_delay:.0f} seconds")
        
        # Also schedule periodic refresh to keep token alive
        self._periodic_timer = threading.Timer(self.PERIODIC_REFRESH_SECONDS, self._periodic_refresh_token)
        self._periodic_timer.daemon = True
        self._periodic_timer.start()
        logger.debug(f"Periodic token refresh scheduled in {self.PERIODIC_REFRESH_SECONDS} seconds")
    
    def _auto_refresh_token(self) -> None:
        """Automatically refresh the token before expiry"""
        if self._token_info:
            logger.info("Auto-refreshing token (pre-expiry)...")
            if self._robust_token_refresh():
                self._schedule_token_refresh()
            else:
                logger.error("Auto-refresh failed")
    
    def _periodic_refresh_token(self) -> None:
        """Periodically check for token updates from other processes.
        
        Instead of proactively authenticating (which consumes the single-use
        refresh token and invalidates other processes' access tokens), just
        reload from the shared token file.  If the file has a newer token
        (refreshed by the engine or another process), adopt it.  Only
        authenticate if the current token is near expiry and no one else
        has refreshed yet.
        """
        if self._token_info:
            logger.info("Periodic token refresh (keep-alive)...")
            # First, try to pick up a newer token from file
            reloaded = self._reload_token_from_file_if_newer()
            if reloaded:
                logger.info("Periodic refresh: loaded newer token from file")
            elif time.time() >= self._token_info.expires_at - self.REFRESH_BUFFER_SECONDS:
                # Token is actually near expiry — perform a real refresh
                self._robust_token_refresh()
            # else: token still valid, nothing to do

            # Reschedule periodic refresh
            self._periodic_timer = threading.Timer(self.PERIODIC_REFRESH_SECONDS, self._periodic_refresh_token)
            self._periodic_timer.daemon = True
            self._periodic_timer.start()
    
    def _ensure_valid_token(self) -> None:
        """Ensure we have a valid access token"""
        if not self._token_info:
            raise ConnectionError("Not authenticated. Please provide a refresh token.")
        
        # Check if token is about to expire
        if time.time() >= self._token_info.expires_at - self.REFRESH_BUFFER_SECONDS:
            logger.info("Token expired or expiring soon, refreshing...")
            if not self._robust_token_refresh():
                raise ConnectionError("Failed to refresh token")
            if self._auto_refresh:
                self._schedule_token_refresh()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers"""
        self._ensure_valid_token()
        return {
            "Authorization": f"{self._token_info.token_type} {self._token_info.access_token}",
            "Content-Type": "application/json"
        }
    
    def _request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        _retry_401: int = 0
    ) -> Any:
        """
        Make an API request to Questrade.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body for POST/PUT
            _retry_401: Internal counter to prevent infinite 401 retry loops
            
        Returns:
            JSON response data
        """
        self._ensure_valid_token()
        
        url = f"{self._token_info.api_server}/v1/{endpoint}"
        headers = self._get_headers()
        
        try:
            response = self._session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data,
                timeout=30
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 1))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self._request(method, endpoint, params, data, _retry_401=_retry_401)
            
            # Handle 401 Unauthorized - token may have been revoked server-side
            if response.status_code == 401:
                if _retry_401 >= 1:
                    raise ConnectionError(
                        "Persistent 401 after token refresh — refresh token is likely "
                        "expired. Generate a new token at Questrade App Hub -> API Access."
                    )
                logger.warning("401 Unauthorized - attempting token refresh...")
                if self._robust_token_refresh():
                    # Retry the request with new token (at most once)
                    return self._request(method, endpoint, params, data, _retry_401=_retry_401 + 1)
                else:
                    raise ConnectionError("Failed to refresh token after 401")
            
            response.raise_for_status()
            
            if response.text:
                return response.json()
            return {}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {method} {endpoint} - {e}")
            raise
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Make a GET request"""
        return self._request("GET", endpoint, params=params)
    
    def _post(self, endpoint: str, data: Optional[Dict] = None) -> Any:
        """Make a POST request"""
        return self._request("POST", endpoint, data=data)
    
    def _delete(self, endpoint: str) -> Any:
        """Make a DELETE request"""
        return self._request("DELETE", endpoint)
    
    # ==================== ACCOUNT OPERATIONS ====================
    
    def get_accounts(self) -> List[Dict]:
        """
        Get list of all accounts for the user.
        
        Returns:
            List of account objects with type, number, status, etc.
        """
        response = self._get("accounts")
        return response.get("accounts", [])
    
    def get_account_positions(self, account_id: str) -> List[Dict]:
        """
        Get all positions for an account.
        
        Args:
            account_id: The account number
            
        Returns:
            List of position objects
        """
        response = self._get(f"accounts/{account_id}/positions")
        return response.get("positions", [])
    
    def get_account_balances(self, account_id: str) -> Dict:
        """
        Get account balances including cash, buying power, etc.
        
        Args:
            account_id: The account number
            
        Returns:
            Balance information object
        """
        return self._get(f"accounts/{account_id}/balances")
    
    def get_account_executions(
        self, 
        account_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get trade executions for an account.
        
        Args:
            account_id: The account number
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of execution objects
        """
        params = {}
        if start_time:
            params["startTime"] = start_time.isoformat()
        if end_time:
            params["endTime"] = end_time.isoformat()
        
        response = self._get(f"accounts/{account_id}/executions", params)
        return response.get("executions", [])
    
    def get_account_orders(
        self, 
        account_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        state_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Get orders for an account.
        
        Args:
            account_id: The account number
            start_time: Start of time range
            end_time: End of time range
            state_filter: Filter by order state (All, Open, Closed)
            
        Returns:
            List of order objects
        """
        params = {}
        if start_time:
            params["startTime"] = start_time.isoformat()
        if end_time:
            params["endTime"] = end_time.isoformat()
        if state_filter:
            params["stateFilter"] = state_filter
        
        response = self._get(f"accounts/{account_id}/orders", params)
        return response.get("orders", [])
    
    def get_order(self, account_id: str, order_id: int) -> Dict:
        """
        Get a specific order by ID.
        
        Args:
            account_id: The account number
            order_id: The order ID
            
        Returns:
            Order object
        """
        response = self._get(f"accounts/{account_id}/orders/{order_id}")
        orders = response.get("orders", [])
        return orders[0] if orders else {}
    
    def get_account_activities(
        self,
        account_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """
        Get account activities (cash transactions, dividends, etc.).
        
        Args:
            account_id: The account number
            start_time: Start of time range
            end_time: End of time range (max 31 days from start)
            
        Returns:
            List of activity objects
        """
        params = {
            "startTime": start_time.isoformat(),
            "endTime": end_time.isoformat()
        }
        response = self._get(f"accounts/{account_id}/activities", params)
        return response.get("activities", [])
    
    # ==================== ORDER OPERATIONS ====================
    
    def place_order(
        self,
        account_id: str,
        symbol_id: int,
        quantity: int,
        action: str = None,
        order_type: str = "Market",
        time_in_force: str = "Day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        is_all_or_none: bool = False,
        is_anonymous: bool = False,
        iceberg_quantity: Optional[int] = None,
        primary_route: str = "AUTO",
        secondary_route: str = "AUTO",
        is_buy: Optional[bool] = None,
        **kwargs,
    ) -> Dict:
        """
        Place a new order.
        
        Args:
            account_id: The account number
            symbol_id: Internal symbol ID (get from search_symbols)
            quantity: Number of shares
            action: Buy, Sell
            order_type: Market, Limit, Stop, StopLimit, TrailStopInPercentage, TrailStopInDollar
            time_in_force: Day, GoodTillCanceled, GoodTillExtendedDay, GoodTillDate, ImmediateOrCancel, FillOrKill
            limit_price: Limit price (required for Limit orders)
            stop_price: Stop price (required for Stop orders)
            is_all_or_none: All-or-none instruction
            is_anonymous: Anonymous order
            iceberg_quantity: Iceberg order quantity
            primary_route: Primary routing (AUTO, etc.)
            secondary_route: Secondary routing
            
        Returns:
            Order response with orderId
        """
        # Support is_buy (used by order_manager) in addition to action string
        if action is None and is_buy is not None:
            action = "Buy" if is_buy else "Sell"
        elif action is None:
            raise ValueError("Either 'action' or 'is_buy' must be provided")

        order_data = {
            "symbolId": symbol_id,
            "quantity": quantity,
            "action": action,
            "orderType": order_type,
            "timeInForce": time_in_force,
            "primaryRoute": primary_route,
            "secondaryRoute": secondary_route,
            "isAllOrNone": is_all_or_none,
            "isAnonymous": is_anonymous
        }
        
        if limit_price is not None:
            order_data["limitPrice"] = limit_price
        if stop_price is not None:
            order_data["stopPrice"] = stop_price
        if iceberg_quantity is not None:
            order_data["icebergQuantity"] = iceberg_quantity
        
        return self._post(f"accounts/{account_id}/orders", order_data)
    
    def place_market_order(
        self,
        account_id: str,
        symbol_id: int,
        quantity: int,
        action: str,
        time_in_force: str = "Day"
    ) -> Dict:
        """
        Place a market order (convenience method).
        
        Args:
            account_id: The account number
            symbol_id: Internal symbol ID
            quantity: Number of shares
            action: Buy or Sell
            time_in_force: Day, GoodTillCanceled, etc.
            
        Returns:
            Order response
        """
        return self.place_order(
            account_id=account_id,
            symbol_id=symbol_id,
            quantity=quantity,
            action=action,
            order_type="Market",
            time_in_force=time_in_force
        )
    
    def place_limit_order(
        self,
        account_id: str,
        symbol_id: int,
        quantity: int,
        action: str,
        limit_price: float,
        time_in_force: str = "Day"
    ) -> Dict:
        """
        Place a limit order (convenience method).
        
        Args:
            account_id: The account number
            symbol_id: Internal symbol ID
            quantity: Number of shares
            action: Buy or Sell
            limit_price: Limit price
            time_in_force: Day, GoodTillCanceled, etc.
            
        Returns:
            Order response
        """
        return self.place_order(
            account_id=account_id,
            symbol_id=symbol_id,
            quantity=quantity,
            action=action,
            order_type="Limit",
            time_in_force=time_in_force,
            limit_price=limit_price
        )
    
    def place_stop_order(
        self,
        account_id: str,
        symbol_id: int,
        quantity: int,
        action: str,
        stop_price: float,
        time_in_force: str = "Day"
    ) -> Dict:
        """
        Place a stop order (convenience method).
        
        Args:
            account_id: The account number
            symbol_id: Internal symbol ID
            quantity: Number of shares
            action: Buy or Sell
            stop_price: Stop price
            time_in_force: Day, GoodTillCanceled, etc.
            
        Returns:
            Order response
        """
        return self.place_order(
            account_id=account_id,
            symbol_id=symbol_id,
            quantity=quantity,
            action=action,
            order_type="Stop",
            time_in_force=time_in_force,
            stop_price=stop_price
        )
    
    def place_stop_limit_order(
        self,
        account_id: str,
        symbol_id: int,
        quantity: int,
        action: str,
        stop_price: float,
        limit_price: float,
        time_in_force: str = "Day"
    ) -> Dict:
        """
        Place a stop-limit order (convenience method).
        
        Args:
            account_id: The account number
            symbol_id: Internal symbol ID
            quantity: Number of shares
            action: Buy or Sell
            stop_price: Stop trigger price
            limit_price: Limit price after stop triggers
            time_in_force: Day, GoodTillCanceled, etc.
            
        Returns:
            Order response
        """
        return self.place_order(
            account_id=account_id,
            symbol_id=symbol_id,
            quantity=quantity,
            action=action,
            order_type="StopLimit",
            time_in_force=time_in_force,
            stop_price=stop_price,
            limit_price=limit_price
        )
    
    def replace_order(
        self,
        account_id: str,
        order_id: int,
        quantity: int,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Dict:
        """
        Modify an existing order.
        
        Args:
            account_id: The account number
            order_id: ID of order to modify
            quantity: New quantity
            limit_price: New limit price
            stop_price: New stop price
            
        Returns:
            Updated order response
        """
        order_data = {"quantity": quantity}
        if limit_price is not None:
            order_data["limitPrice"] = limit_price
        if stop_price is not None:
            order_data["stopPrice"] = stop_price
        
        return self._post(f"accounts/{account_id}/orders/{order_id}", order_data)
    
    def cancel_order(self, account_id: str, order_id: int) -> Dict:
        """
        Cancel an existing order.
        
        Args:
            account_id: The account number
            order_id: ID of order to cancel
            
        Returns:
            Cancellation response
        """
        return self._delete(f"accounts/{account_id}/orders/{order_id}")
    
    def get_order_impact(
        self,
        account_id: str,
        symbol_id: int,
        quantity: int,
        action: str,
        order_type: str,
        limit_price: Optional[float] = None
    ) -> Dict:
        """
        Get the estimated impact of an order before placing it.
        
        Args:
            account_id: The account number
            symbol_id: Internal symbol ID
            quantity: Number of shares
            action: Buy or Sell
            order_type: Market, Limit, etc.
            limit_price: Limit price
            
        Returns:
            Order impact estimate
        """
        order_data = {
            "symbolId": symbol_id,
            "quantity": quantity,
            "action": action,
            "orderType": order_type
        }
        if limit_price is not None:
            order_data["limitPrice"] = limit_price
        
        return self._post(f"accounts/{account_id}/orders/impact", order_data)
    
    # ==================== MARKET DATA OPERATIONS ====================
    
    def search_symbols(self, prefix: str, offset: int = 0) -> List[Dict]:
        """
        Search for symbols by prefix.
        
        Args:
            prefix: Symbol or name prefix to search
            offset: Pagination offset
            
        Returns:
            List of matching symbol objects
        """
        response = self._get("symbols/search", {"prefix": prefix, "offset": offset})
        return response.get("symbols", [])
    
    def get_symbols(self, ids: Optional[List[int]] = None, names: Optional[List[str]] = None) -> List[Dict]:
        """
        Get detailed symbol information.
        
        Args:
            ids: List of symbol IDs
            names: List of symbol names
            
        Returns:
            List of symbol detail objects
        """
        params = {}
        if ids:
            params["ids"] = ",".join(str(i) for i in ids)
        if names:
            params["names"] = ",".join(names)
        
        response = self._get("symbols", params)
        return response.get("symbols", [])
    
    def get_symbol_by_name(self, symbol: str) -> Optional[Dict]:
        """
        Get symbol details by name (convenience method).
        
        Args:
            symbol: Symbol name (e.g., "AAPL", "MSFT")
            
        Returns:
            Symbol details or None
        """
        symbols = self.get_symbols(names=[symbol])
        return symbols[0] if symbols else None
    
    def get_symbol_id(self, symbol: str) -> Optional[int]:
        """
        Get the internal symbol ID for a ticker.
        
        Args:
            symbol: Symbol name
            
        Returns:
            Symbol ID or None
        """
        sym = self.get_symbol_by_name(symbol)
        return sym.get("symbolId") if sym else None
    
    def get_option_chain(self, symbol_id: int) -> Dict:
        """
        Get option chain for a symbol (expiry dates and strike prices).
        
        Args:
            symbol_id: Internal symbol ID of the underlying
            
        Returns:
            Option chain data with expiry dates and available strikes
        """
        return self._get(f"symbols/{symbol_id}/options")
    
    def get_option_chain_by_symbol(self, symbol: str) -> Dict:
        """
        Get option chain by underlying symbol name.
        
        Args:
            symbol: Underlying symbol name (e.g., "AAPL", "SPY")
            
        Returns:
            Option chain data
        """
        symbol_id = self.get_symbol_id(symbol)
        if not symbol_id:
            raise ValueError(f"Symbol not found: {symbol}")
        return self.get_option_chain(symbol_id)
    
    def get_quotes(self, ids: List[int]) -> List[Dict]:
        """
        Get real-time quotes for symbols.
        
        Args:
            ids: List of symbol IDs
            
        Returns:
            List of quote objects
        """
        params = {"ids": ",".join(str(i) for i in ids)}
        response = self._get("markets/quotes", params)
        return response.get("quotes", [])
    
    def get_quote(self, symbol_id: int) -> Optional[Dict]:
        """
        Get quote for a single symbol.
        
        Args:
            symbol_id: Internal symbol ID
            
        Returns:
            Quote object or None
        """
        quotes = self.get_quotes([symbol_id])
        return quotes[0] if quotes else None
    
    def get_quote_by_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Get quote by symbol name (convenience method).
        
        Args:
            symbol: Symbol name
            
        Returns:
            Quote object or None
        """
        symbol_id = self.get_symbol_id(symbol)
        if symbol_id:
            return self.get_quote(symbol_id)
        return None
    
    # ==================== REAL-TIME OPTIONS DATA ====================
    
    def get_option_quotes(
        self, 
        option_ids: Optional[List[int]] = None, 
        filters: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Get real-time quotes for option contracts with full Greeks.
        
        This is the main method for retrieving real-time option data including:
        - Bid/Ask prices and sizes
        - Last trade price/size/time
        - Volume and open interest
        - Greeks (delta, gamma, theta, vega, rho)
        - Implied volatility
        
        Args:
            option_ids: List of specific option symbol IDs to quote
            filters: List of filter dictionaries to find options. Each filter requires:
                - underlyingId (int): Required - underlying symbol ID
                - expiryDate (str): Required - expiry in ISO format
                - optionType (str): Optional - "Call" or "Put"
                - minstrikePrice (float): Optional - minimum strike
                - maxstrikePrice (float): Optional - maximum strike
                
        Returns:
            List of option quote objects with Greeks
            
        Example:
            # Get quotes by option IDs
            quotes = client.get_option_quotes(option_ids=[9907637, 9907638])
            
            # Get quotes using filters
            quotes = client.get_option_quotes(filters=[{
                "underlyingId": 27426,
                "expiryDate": "2026-02-21T00:00:00.000000-05:00",
                "optionType": "Call",
                "minstrikePrice": 150,
                "maxstrikePrice": 160
            }])
        """
        payload = {}
        
        if option_ids:
            payload["optionIds"] = option_ids
        if filters:
            payload["filters"] = filters
        
        if not payload:
            raise ValueError("Must provide either option_ids or filters")
        
        response = self._post("markets/quotes/options", payload)
        return response.get("optionQuotes", [])
    
    def get_option_quotes_by_symbol(
        self,
        symbol: str,
        expiry_date: Union[str, datetime],
        option_type: Optional[str] = None,
        min_strike: Optional[float] = None,
        max_strike: Optional[float] = None
    ) -> List[Dict]:
        """
        Get real-time option quotes by underlying symbol name (convenience method).
        
        Args:
            symbol: Underlying symbol name (e.g., "AAPL", "SPY", "QQQ")
            expiry_date: Option expiry date (datetime or ISO string)
            option_type: "Call" or "Put" (None for both)
            min_strike: Minimum strike price filter
            max_strike: Maximum strike price filter
            
        Returns:
            List of option quotes with real-time data and Greeks
            
        Example:
            # Get all AAPL calls expiring Feb 21, 2026
            calls = client.get_option_quotes_by_symbol(
                symbol="AAPL",
                expiry_date="2026-02-21",
                option_type="Call"
            )
            
            # Get SPY options near the money
            options = client.get_option_quotes_by_symbol(
                symbol="SPY",
                expiry_date=datetime(2026, 2, 21),
                min_strike=480,
                max_strike=520
            )
        """
        # Get underlying symbol ID
        symbol_id = self.get_symbol_id(symbol)
        if not symbol_id:
            raise ValueError(f"Symbol not found: {symbol}")
        
        # Format expiry date
        if isinstance(expiry_date, datetime):
            expiry_str = expiry_date.strftime("%Y-%m-%dT00:00:00.000000-05:00")
        elif isinstance(expiry_date, str):
            # Handle various date formats
            if "T" not in expiry_date:
                expiry_str = f"{expiry_date}T00:00:00.000000-05:00"
            else:
                expiry_str = expiry_date
        else:
            raise ValueError("expiry_date must be datetime or string")
        
        # Build filter
        filter_dict = {
            "underlyingId": symbol_id,
            "expiryDate": expiry_str
        }
        
        if option_type:
            filter_dict["optionType"] = option_type
        if min_strike is not None:
            filter_dict["minstrikePrice"] = min_strike
        if max_strike is not None:
            filter_dict["maxstrikePrice"] = max_strike
        
        return self.get_option_quotes(filters=[filter_dict])
    
    def get_calls(
        self,
        symbol: str,
        expiry_date: Union[str, datetime],
        min_strike: Optional[float] = None,
        max_strike: Optional[float] = None
    ) -> List[Dict]:
        """
        Get real-time call option quotes (convenience method).
        
        Args:
            symbol: Underlying symbol name
            expiry_date: Option expiry date
            min_strike: Minimum strike price
            max_strike: Maximum strike price
            
        Returns:
            List of call option quotes with Greeks
        """
        return self.get_option_quotes_by_symbol(
            symbol=symbol,
            expiry_date=expiry_date,
            option_type="Call",
            min_strike=min_strike,
            max_strike=max_strike
        )
    
    def get_puts(
        self,
        symbol: str,
        expiry_date: Union[str, datetime],
        min_strike: Optional[float] = None,
        max_strike: Optional[float] = None
    ) -> List[Dict]:
        """
        Get real-time put option quotes (convenience method).
        
        Args:
            symbol: Underlying symbol name
            expiry_date: Option expiry date
            min_strike: Minimum strike price
            max_strike: Maximum strike price
            
        Returns:
            List of put option quotes with Greeks
        """
        return self.get_option_quotes_by_symbol(
            symbol=symbol,
            expiry_date=expiry_date,
            option_type="Put",
            min_strike=min_strike,
            max_strike=max_strike
        )
    
    def get_option_expiries(self, symbol: str) -> List[Dict]:
        """
        Get available option expiry dates for a symbol.
        
        Args:
            symbol: Underlying symbol name
            
        Returns:
            List of expiry date objects with available strikes
        """
        chain = self.get_option_chain_by_symbol(symbol)
        return chain.get("optionChain", [])
    
    def get_atm_options(
        self,
        symbol: str,
        expiry_date: Union[str, datetime],
        num_strikes: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Get at-the-money options (strikes near current price).
        
        Args:
            symbol: Underlying symbol name
            expiry_date: Option expiry date
            num_strikes: Number of strikes above and below ATM
            
        Returns:
            Dict with 'calls' and 'puts' lists
        """
        # Get current price
        quote = self.get_quote_by_symbol(symbol)
        if not quote:
            raise ValueError(f"Could not get quote for {symbol}")
        
        current_price = quote.get("lastTradePrice") or quote.get("bidPrice", 0)
        
        # Estimate strike width (usually $1, $2.50, or $5 depending on price)
        if current_price < 50:
            strike_width = 1
        elif current_price < 200:
            strike_width = 2.5
        else:
            strike_width = 5
        
        # Calculate strike range
        min_strike = current_price - (num_strikes * strike_width)
        max_strike = current_price + (num_strikes * strike_width)
        
        calls = self.get_calls(symbol, expiry_date, min_strike, max_strike)
        puts = self.get_puts(symbol, expiry_date, min_strike, max_strike)
        
        return {"calls": calls, "puts": puts}
    
    def get_option_greeks(
        self,
        symbol: str,
        expiry_date: Union[str, datetime],
        strike: float,
        option_type: str = "Call"
    ) -> Optional[Dict]:
        """
        Get Greeks for a specific option contract.
        
        Args:
            symbol: Underlying symbol name
            expiry_date: Option expiry date
            strike: Strike price
            option_type: "Call" or "Put"
            
        Returns:
            Option data including delta, gamma, theta, vega, rho, IV
        """
        options = self.get_option_quotes_by_symbol(
            symbol=symbol,
            expiry_date=expiry_date,
            option_type=option_type,
            min_strike=strike,
            max_strike=strike
        )
        
        # Find the exact strike
        for opt in options:
            if abs(float(opt.get("strikePrice", 0)) - strike) < 0.01:
                return {
                    "symbol": opt.get("symbol"),
                    "symbolId": opt.get("symbolId"),
                    "underlying": opt.get("underlying"),
                    "strikePrice": opt.get("strikePrice"),
                    "bidPrice": opt.get("bidPrice"),
                    "askPrice": opt.get("askPrice"),
                    "lastTradePrice": opt.get("lastTradePrice"),
                    "volume": opt.get("volume"),
                    "openInterest": opt.get("openInterest"),
                    "impliedVolatility": opt.get("volatility"),
                    "delta": opt.get("delta"),
                    "gamma": opt.get("gamma"),
                    "theta": opt.get("theta"),
                    "vega": opt.get("vega"),
                    "rho": opt.get("rho")
                }
        
        return None

    def get_candles(
        self,
        symbol_id: int,
        start_time: datetime,
        end_time: datetime,
        interval: str = "OneDay"
    ) -> List[Dict]:
        """
        Get historical OHLC candlesticks.
        
        Args:
            symbol_id: Internal symbol ID
            start_time: Start of range
            end_time: End of range
            interval: Candle interval:
                - OneMinute, TwoMinutes, ThreeMinutes, FourMinutes, FiveMinutes
                - TenMinutes, FifteenMinutes, TwentyMinutes, HalfHour
                - OneHour, TwoHours, FourHours
                - OneDay, OneWeek, OneMonth, OneYear
                
        Returns:
            List of candle objects
        """
        # Questrade requires ISO 8601 with Eastern timezone offset
        # Strip microseconds — API rejects them
        fmt = "%Y-%m-%dT%H:%M:%S-05:00"
        params = {
            "startTime": start_time.replace(microsecond=0).strftime(fmt),
            "endTime": end_time.replace(microsecond=0).strftime(fmt),
            "interval": interval
        }
        response = self._get(f"markets/candles/{symbol_id}", params)
        return response.get("candles", [])
    
    def get_markets(self) -> List[Dict]:
        """
        Get list of supported markets/exchanges.
        
        Returns:
            List of market objects
        """
        response = self._get("markets")
        return response.get("markets", [])
    
    def get_market_hours(self, market: str, date: Optional[datetime] = None) -> Dict:
        """
        Get market hours for a date.
        
        Args:
            market: Market name (e.g., "NYSE", "NASDAQ")
            date: Date to check (default: today)
            
        Returns:
            Market hours information
        """
        params = {}
        if date:
            params["date"] = date.strftime("%Y-%m-%d")
        
        return self._get(f"markets/{market}/hours", params)
    
    def get_strategy_quotes(
        self,
        account_id: str,
        variants: List[Dict]
    ) -> List[Dict]:
        """
        Get quotes for multi-leg option strategies.
        
        Args:
            account_id: The account number
            variants: List of strategy leg definitions
            
        Returns:
            List of strategy quote objects
        """
        response = self._post(
            f"accounts/{account_id}/markets/quotes/strategies",
            {"variants": variants}
        )
        return response.get("strategyQuotes", [])
    
    # ==================== UTILITY METHODS ====================
    
    def get_server_time(self) -> datetime:
        """
        Get current server time.
        
        Returns:
            Server datetime
        """
        response = self._get("time")
        return datetime.fromisoformat(response["time"].replace("Z", "+00:00"))
    
    def refresh_token_now(self) -> None:
        """Force an immediate token refresh"""
        with self._lock:
            if self._token_info:
                self._authenticate(self._token_info.refresh_token)
                if self._auto_refresh:
                    self._schedule_token_refresh()
    
    def set_new_refresh_token(self, new_token: str) -> None:
        """
        Set a new refresh token when the old one has completely expired.
        
        Get a new token from: Questrade App Hub -> API Access -> Generate New Token
        
        Args:
            new_token: The new refresh token from Questrade
        """
        with self._lock:
            logger.info("Setting new refresh token...")
            self._authenticate(new_token)
            if self._auto_refresh:
                self._schedule_token_refresh()
            logger.info("New token authenticated successfully!")
    
    def get_token_expiry(self) -> Optional[datetime]:
        """Get when the current access token expires"""
        if self._token_info:
            return datetime.fromtimestamp(self._token_info.expires_at)
        return None
    
    def is_authenticated(self) -> bool:
        """Check if client is authenticated with valid token"""
        return self._token_info is not None and time.time() < self._token_info.expires_at
    
    def close(self) -> None:
        """Clean up resources"""
        if self._refresh_timer:
            self._refresh_timer.cancel()
        if self._periodic_timer:
            self._periodic_timer.cancel()
        self._session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            accounts = self.get_accounts()
            return accounts is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Factory function with your token
def create_questrade_client(
    refresh_token: Optional[str] = None,
    practice_mode: bool = False,
    auto_refresh: bool = True
) -> QuestradeClient:
    """
    Factory function to create Questrade client.
    
    Args:
        refresh_token: Your Questrade API refresh token (falls back to
                       QUESTRADE_API_KEY env var, then hardcoded default)
        practice_mode: Use practice/paper trading environment
        auto_refresh: Automatically refresh tokens before expiry
        
    Returns:
        Configured QuestradeClient instance
    """
    if refresh_token is None:
        refresh_token = os.environ.get("QUESTRADE_API_KEY", DEFAULT_REFRESH_TOKEN)
    token_file_env = os.environ.get("QUESTRADE_TOKEN_FILE")
    return QuestradeClient(
        refresh_token=refresh_token,
        token_file=Path(token_file_env) if token_file_env else None,
        practice_mode=practice_mode,
        auto_refresh=auto_refresh
    )


# Default instance with your token
DEFAULT_REFRESH_TOKEN = "pCGi0Pz5EU_50iMJRxGaQJ-nO_eqAKFa0"


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Questrade Client...")
    print("-" * 50)
    
    try:
        client = create_questrade_client()
        
        print(f"Authenticated: {client.is_authenticated()}")
        print(f"Token expires: {client.get_token_expiry()}")
        
        # Get accounts
        accounts = client.get_accounts()
        print(f"\nAccounts: {len(accounts)}")
        for acc in accounts:
            print(f"  - {acc.get('type')}: {acc.get('number')} ({acc.get('status')})")
        
        # Get server time
        server_time = client.get_server_time()
        print(f"\nServer time: {server_time}")
        
        print("\n" + "-" * 50)
        print("Connection test successful!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure your refresh token is valid.")
        print("Tokens expire after 7 days and must be regenerated in Questrade API Centre.")
