"""
Wealthsimple Trade API Client (v2) — OAuth v2 + GraphQL

A complete Python client for the Wealthsimple platform using
OAuth v2 authentication and the GraphQL API.

Features:
- OAuth v2 authentication with auto token refresh
- GraphQL-based data API for all operations
- Account management (non-registered, TFSA, RRSP, crypto)
- Market, limit, and stop-limit orders
- Security/symbol search with real-time quotes
- Position and holdings tracking with P&L
- Activity and transaction history
- Account financials and performance
- Identity/user profile

API Endpoints:
- Auth:    https://api.production.wealthsimple.com/v1/oauth/v2/token
- GraphQL: https://my.wealthsimple.com/graphql

Note: This is an unofficial client based on reverse-engineered APIs.
Cloudflare bypass (cloudscraper) is no longer needed.
"""

import os
import time
import json
import uuid
import base64
import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

try:
    import requests
except ImportError:
    requests = None


logger = logging.getLogger(__name__)


# ==================== Constants ====================

AUTH_URL = "https://api.production.wealthsimple.com/v1/oauth/v2/token"
GRAPHQL_URL = "https://my.wealthsimple.com/graphql"
DEFAULT_CLIENT_ID = "4da53ac2b03225bed1550eba8e4611e086c7b905a3855e6ed12ea08c246758fa"

# Token expiry buffer (refresh 5 minutes before expiry)
TOKEN_REFRESH_BUFFER = 300

# Default token file path
DEFAULT_TOKEN_FILE = ".ws_tokens.json"


# ==================== Enums ====================

class WSOrderType(str, Enum):
    """Wealthsimple order types for GraphQL API"""
    BUY = "BUY_QUANTITY"
    SELL = "SELL_QUANTITY"
    DIY_BUY = "DIY_BUY"
    DIY_SELL = "DIY_SELL"
    CRYPTO_BUY = "CRYPTO_BUY"
    CRYPTO_SELL = "CRYPTO_SELL"


class WSExecutionType(str, Enum):
    """Order execution types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class WSTimeInForce(str, Enum):
    """Time-in-force options"""
    DAY = "DAY"
    GTC = "GTC"


class WSOrderStatus(str, Enum):
    """Order status values"""
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    POSTED = "posted"


class WSAccountType(str, Enum):
    """Wealthsimple account types"""
    PERSONAL = "CA_NON_REGISTERED"
    TFSA = "CA_TFSA"
    RRSP = "CA_RRSP"
    CRYPTO = "CA_CRYPTO"
    RESP = "CA_RESP"
    LIRA = "CA_LIRA"
    RRIF = "CA_RRIF"
    JOINT = "CA_JOINT"
    FHSA = "CA_FHSA"


class WSSecurityType(str, Enum):
    """Security types"""
    EQUITY = "EQUITY"
    OPTION = "OPTION"
    CRYPTO = "CRYPTO"
    ETF = "ETF"
    MUTUAL_FUND = "MUTUAL_FUND"


class WSActivityType(str, Enum):
    """Activity/transaction types"""
    DIY_BUY = "DIY_BUY"
    DIY_SELL = "DIY_SELL"
    CRYPTO_BUY = "CRYPTO_BUY"
    CRYPTO_SELL = "CRYPTO_SELL"
    DIVIDEND = "DIVIDEND"
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    OPTIONS_BUY = "OPTIONS_BUY"
    OPTIONS_SELL = "OPTIONS_SELL"


# ==================== Configuration ====================

@dataclass
class WealthsimpleConfig:
    """Configuration for Wealthsimple API client (v2)"""
    email: str = ""
    password: str = ""
    otp: Optional[str] = None
    otp_callback: Optional[Callable[[], str]] = None
    client_id: str = DEFAULT_CLIENT_ID
    auth_url: str = AUTH_URL
    graphql_url: str = GRAPHQL_URL
    token_file: Optional[str] = DEFAULT_TOKEN_FILE  # Set to None to disable persistence

    @classmethod
    def from_env(cls) -> 'WealthsimpleConfig':
        """Load configuration from environment variables"""
        return cls(
            email=os.getenv("WEALTHSIMPLE_EMAIL", ""),
            password=os.getenv("WEALTHSIMPLE_PASSWORD", ""),
            otp=os.getenv("WEALTHSIMPLE_OTP"),
            client_id=os.getenv("WEALTHSIMPLE_CLIENT_ID", DEFAULT_CLIENT_ID),
            auth_url=os.getenv("WEALTHSIMPLE_AUTH_URL", AUTH_URL),
            graphql_url=os.getenv("WEALTHSIMPLE_GRAPHQL_URL", GRAPHQL_URL),
        )

    def validate(self) -> Dict[str, bool]:
        """Validate configuration"""
        return {
            "has_email": bool(self.email),
            "has_password": bool(self.password),
        }

    def is_ready(self) -> bool:
        """Check if required credentials are present"""
        v = self.validate()
        return all(v.values())


# ==================== Token Management ====================

@dataclass
class AuthTokens:
    """Authentication token storage (OAuth v2)"""
    access_token: str = ""
    refresh_token: str = ""
    expires_at: float = 0.0
    identity_id: str = ""

    @property
    def is_expired(self) -> bool:
        """Check if the access token is expired or about to expire"""
        return time.time() >= (self.expires_at - TOKEN_REFRESH_BUFFER)

    @property
    def is_valid(self) -> bool:
        """Check if we have a valid, non-expired token"""
        return bool(self.access_token) and not self.is_expired


# ==================== GraphQL Queries ====================

GQL_FETCH_ALL_ACCOUNTS = """
query FetchAllAccounts($identityId: ID!, $filter: AccountsFilter = {}, $pageSize: Int = 25) {
  identity(id: $identityId) {
    id
    accounts(filter: $filter, first: $pageSize) {
      edges {
        node {
          id
          branch
          currency
          nickname
          status
          unifiedAccountType
          type
          createdAt
          custodianAccounts {
            id
            branch
            custodian
            status
          }
          accountFeatures {
            name
            enabled
            functional
          }
        }
      }
    }
  }
}
"""

GQL_SEARCH_SECURITIES = """
query FetchSecuritySearchResult($query: String!, $securityGroupIds: [String!]) {
  securitySearch(input: {query: $query, securityGroupIds: $securityGroupIds}) {
    results {
      id
      buyable
      sellable
      optionsEligible
      securityType
      allowedOrderSubtypes
      status
      stock {
        symbol
        name
        primaryExchange
      }
      features
      logoUrl
      quoteV2(currency: null) {
        securityId
        currency
        price
        ... on EquityQuote {
          marketStatus
          close
          high
          low
          open
          volume: vol
        }
      }
    }
  }
}
"""

GQL_FETCH_SECURITY = """
query FetchSecurity($securityId: ID!, $currency: Currency) {
  security(id: $securityId) {
    id
    active
    activeDate
    allowedOrderSubtypes
    buyable
    currency
    features
    logoUrl
    securityType
    sellable
    status
    wsTradeEligible
    optionsEligible
    stock {
      description
      dividendFrequency
      name
      primaryExchange
      primaryMic
      symbol
    }
    quoteV2(currency: $currency) {
      securityId
      ask
      bid
      currency
      price
      sessionPrice
      quotedAsOf
      previousBaseline
      ... on EquityQuote {
        marketStatus
        askSize
        bidSize
        close
        high
        last
        lastSize
        low
        open
        mid
        volume: vol
        referenceClose
      }
    }
  }
}
"""

GQL_FETCH_QUOTE = """
query FetchSecurityQuoteV2($id: ID!, $currency: Currency = null) {
  security(id: $id) {
    id
    quoteV2(currency: $currency) {
      securityId
      ask
      bid
      currency
      price
      sessionPrice
      quotedAsOf
      previousBaseline
      ... on EquityQuote {
        marketStatus
        askSize
        bidSize
        close
        high
        last
        lastSize
        low
        open
        mid
        volume: vol
        referenceClose
      }
    }
  }
}
"""

GQL_FETCH_OPTION_QUOTE = """
query FetchOptionQuote($id: ID!) {
  security(id: $id) {
    id
    securityType
    quoteV2(currency: null) {
      securityId
      price
      ... on OptionQuote {
        strikePrice
        expiryDate
        contractType
        underlyingSecurityId
      }
    }
  }
}
"""

GQL_FETCH_POSITIONS = """
query FetchIdentityPositions($identityId: ID!, $currency: Currency!, $first: Int, $cursor: String,
                             $accountIds: [ID!], $aggregated: Boolean, $currencyOverride: CurrencyOverride,
                             $filter: PositionFilter, $includeSecurity: Boolean = true) {
  identity(id: $identityId) {
    id
    financials(filter: {accounts: $accountIds}) {
      current(currency: $currency) {
        id
        positions(first: $first, after: $cursor, aggregated: $aggregated, filter: $filter) {
          edges {
            node {
              id
              quantity
              percentageOfAccount
              positionDirection
              bookValue {
                amount
                currency
                __typename
              }
              averagePrice {
                amount
                currency
                __typename
              }
              marketAveragePrice: averagePrice(currencyOverride: $currencyOverride) {
                amount
                currency
                __typename
              }
              marketBookValue: bookValue(currencyOverride: $currencyOverride) {
                amount
                currency
                __typename
              }
              totalValue(currencyOverride: $currencyOverride) {
                amount
                currency
                __typename
              }
              unrealizedReturns {
                amount
                currency
                __typename
              }
              marketUnrealizedReturns: unrealizedReturns(currencyOverride: $currencyOverride) {
                amount
                currency
                __typename
              }
              security {
                id
                securityType
                currency
                status
                logoUrl
                features
                stock @include(if: $includeSecurity) {
                  name
                  symbol
                  primaryExchange
                  primaryMic
                  __typename
                }
                quoteV2(currency: null) @include(if: $includeSecurity) {
                  securityId
                  currency
                  price
                  sessionPrice
                  ask
                  bid
                  quotedAsOf
                  previousBaseline
                  ... on OptionQuote {
                    strikePrice
                    expiryDate
                    contractType
                    underlyingSecurityId
                  }
                  __typename
                }
                __typename
              }
              __typename
            }
            __typename
          }
          pageInfo {
            hasNextPage
            endCursor
            __typename
          }
          totalCount
          status
          __typename
        }
        __typename
      }
      __typename
    }
    __typename
  }
}
"""

GQL_FETCH_ACTIVITIES = """
query FetchActivityFeedItems($first: Int, $cursor: Cursor, $condition: ActivityCondition,
                             $orderBy: [ActivitiesOrderBy!] = OCCURRED_AT_DESC) {
  activityFeedItems(first: $first, after: $cursor, condition: $condition, orderBy: $orderBy) {
    edges {
      node {
        ...Activity
        __typename
      }
      __typename
    }
    pageInfo {
      hasNextPage
      endCursor
      __typename
    }
    __typename
  }
}

fragment Activity on ActivityFeedItem {
  accountId
  aftOriginatorName
  aftTransactionCategory
  aftTransactionType
  amount
  amountSign
  assetQuantity
  assetSymbol
  canonicalId
  currency
  eTransferEmail
  eTransferName
  externalCanonicalId
  identityId
  institutionName
  occurredAt
  p2pHandle
  p2pMessage
  spendMerchant
  securityId
  status
  subType
  type
  strikePrice
  contractType
  expiryDate
  fxRate
  fees
  reference
  __typename
}
"""

GQL_FETCH_ACCOUNT_FINANCIALS = """
query FetchAccountFinancials($ids: [String!]!, $startDate: Date, $currency: Currency) {
  accounts(ids: $ids) {
    id
    ...AccountFinancials
    __typename
  }
}

fragment AccountFinancials on Account {
  id
  custodianAccounts {
    id
    branch
    financials {
      current {
        ...CustodianAccountCurrentFinancialValues
        __typename
      }
      __typename
    }
    __typename
  }
  financials {
    currentCombined(currency: $currency) {
      id
      ...AccountCurrentFinancials
      __typename
    }
    __typename
  }
  __typename
}

fragment CustodianAccountCurrentFinancialValues on CustodianAccountCurrentFinancialValues {
  deposits {
    ...Money
    __typename
  }
  earnings {
    ...Money
    __typename
  }
  netDeposits {
    ...Money
    __typename
  }
  netLiquidationValue {
    ...Money
    __typename
  }
  withdrawals {
    ...Money
    __typename
  }
  __typename
}

fragment Money on Money {
  amount
  cents
  currency
  __typename
}

fragment AccountCurrentFinancials on AccountCurrentFinancials {
  id
  netLiquidationValueV2 {
    ...Money
    __typename
  }
  netDeposits: netDepositsV2 {
    ...Money
    __typename
  }
  simpleReturns(referenceDate: $startDate) {
    ...SimpleReturns
    __typename
  }
  totalDeposits: totalDepositsV2 {
    ...Money
    __typename
  }
  totalWithdrawals: totalWithdrawalsV2 {
    ...Money
    __typename
  }
  __typename
}

fragment SimpleReturns on SimpleReturns {
  amount {
    ...Money
    __typename
  }
  asOf
  rate
  referenceDate
  __typename
}
"""

GQL_FETCH_FUNDING_BALANCES = """
query FetchAccountFundingBalances($accountIds: [ID!]!) {
  account_funding_balances(account_ids: $accountIds) {
    ...AccountFundingBalance
    __typename
  }
}

fragment AccountFundingBalance on AccountFundingBalance {
  id
  trading_balances {
    amount
    currency
    __typename
  }
  __typename
}
"""

GQL_CREATE_ORDER = """
mutation SoOrdersOrderCreate($input: SoOrders_CreateOrderInput!) {
  soOrdersCreateOrder(input: $input) {
    errors {
      code
      message
      __typename
    }
    order {
      orderId
      externalCanonicalId
      status
      createdAt
      __typename
    }
    __typename
  }
}
"""

GQL_CANCEL_ORDER = """
mutation SoOrdersOrderCancel($cancelOrderRequest: CancelOrderRequest!) {
  orderServiceCancelOrder(cancelOrderRequest: $cancelOrderRequest) {
    externalId
    errors {
      code
      message
      __typename
    }
    __typename
  }
}
"""

GQL_EXTENDED_ORDER = """
query SoOrdersExtendedOrder($branchId: String!, $externalId: String!) {
  soOrdersExtendedOrder(branchId: $branchId, externalId: $externalId) {
    ...SoOrdersExtendedOrder
    __typename
  }
}

fragment SoOrdersExtendedOrder on SoOrders_ExtendedOrderResponse {
  averageFilledPrice
  filledExchangeRate
  filledQuantity
  filledCommissionFee
  filledTotalFee
  firstFilledAtUtc
  lastFilledAtUtc
  limitPrice
  openClose
  orderType
  rejectionCause
  rejectionCode
  securityCurrency
  status
  stopPrice
  submittedAtUtc
  submittedExchangeRate
  submittedNetValue
  submittedQuantity
  submittedTotalFee
  timeInForce
  accountId
  canonicalAccountId
  cancellationCutoff
  tradingSession
  expiredAtUtc
  __typename
}
"""

GQL_FETCH_IDENTITY = """
query FetchIdentity($id: ID!) {
  identity(id: $id) {
    id
    createdAt
    email
    givenName
    familyName
    phoneNumber
  }
}
"""


# ==================== Main Client ====================

class WealthsimpleClient:
    """
    Wealthsimple API v2 Client (OAuth v2 + GraphQL)

    Uses OAuth v2 authentication at api.production.wealthsimple.com
    and GraphQL API at my.wealthsimple.com/graphql for all data operations.

    Example:
        config = WealthsimpleConfig(
            email="user@example.com",
            password="password123",
        )
        client = WealthsimpleClient(config)
        client.login()
        accounts = client.list_accounts()
    """

    def __init__(self, config: Optional[WealthsimpleConfig] = None):
        """
        Initialize the Wealthsimple client.

        Args:
            config: WealthsimpleConfig object (loads from env if not provided)
        """
        if requests is None:
            raise ImportError(
                "The 'requests' library is required. "
                "Install with: pip install requests"
            )

        self.config = config or WealthsimpleConfig.from_env()
        self.tokens = AuthTokens()
        self._session = requests.Session()

        # Try to load saved tokens
        if self.config.token_file:
            self._load_tokens()

        logger.info("Wealthsimple client initialized (OAuth v2 + GraphQL)")

    # ==================== HTTP / GraphQL Helpers ====================

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for GraphQL requests."""
        self._ensure_auth()
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.tokens.access_token}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
            "x-ws-api-version": "12",
            "x-platform-os": "web",
            "x-ws-locale": "en-CA",
            "x-ws-profile": "trade",
        }

    def _ensure_auth(self) -> None:
        """Ensure we have valid authentication tokens, refreshing if needed"""
        if not self.tokens.access_token:
            raise RuntimeError(
                "Not authenticated. Call login() first."
            )
        if self.tokens.is_expired:
            logger.info("Access token expired, refreshing...")
            self.refresh_token()

    # ==================== Token Persistence ====================

    def _get_token_path(self) -> Optional[str]:
        """Get the absolute path to the token file."""
        if not self.config.token_file:
            return None
        # If relative, resolve from the directory of this module
        p = self.config.token_file
        if not os.path.isabs(p):
            p = os.path.join(os.path.dirname(os.path.abspath(__file__)), p)
        return p

    def _save_tokens(self) -> None:
        """Save current tokens to disk for session persistence."""
        path = self._get_token_path()
        if not path:
            return
        try:
            data = {
                "access_token": self.tokens.access_token,
                "refresh_token": self.tokens.refresh_token,
                "expires_at": self.tokens.expires_at,
                "identity_id": self.tokens.identity_id,
            }
            with open(path, "w") as f:
                json.dump(data, f)
            logger.debug(f"Tokens saved to {path}")
        except Exception as e:
            logger.warning(f"Failed to save tokens: {e}")

    def _load_tokens(self) -> bool:
        """Load tokens from disk. Returns True if valid tokens were loaded."""
        path = self._get_token_path()
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.tokens.access_token = data.get("access_token", "")
            self.tokens.refresh_token = data.get("refresh_token", "")
            self.tokens.expires_at = data.get("expires_at", 0.0)
            self.tokens.identity_id = data.get("identity_id", "")

            if self.tokens.access_token:
                if self.tokens.is_expired and self.tokens.refresh_token:
                    logger.info("Saved access token expired, refreshing...")
                    try:
                        self.refresh_token()
                        return True
                    except Exception as e:
                        logger.warning(f"Token refresh failed: {e}")
                        self.tokens = AuthTokens()
                        return False
                elif not self.tokens.is_expired:
                    logger.info(f"Loaded saved session (identity: {self.tokens.identity_id})")
                    return True
            return False
        except Exception as e:
            logger.warning(f"Failed to load tokens: {e}")
            return False

    def clear_saved_tokens(self) -> None:
        """Delete the saved token file."""
        path = self._get_token_path()
        if path and os.path.exists(path):
            os.remove(path)
            logger.info(f"Removed saved tokens: {path}")

    def graphql_query(
        self,
        operation_name: str,
        query: str,
        variables: Optional[Dict] = None,
    ) -> Dict:
        """
        Execute a GraphQL query or mutation.

        Args:
            operation_name: The operation name
            query: The GraphQL query string
            variables: Optional variables for the query

        Returns:
            Response data dictionary

        Raises:
            Exception if the request fails
        """
        payload = {
            "operationName": operation_name,
            "query": query,
            "variables": variables or {},
        }

        response = self._session.post(
            self.config.graphql_url,
            json=payload,
            headers=self._get_headers(),
        )

        if response.status_code == 200:
            data = response.json()
            if "errors" in data:
                raise Exception(f"GraphQL errors: {data['errors']}")
            return data
        else:
            raise Exception(
                f"GraphQL request failed: {response.status_code} - {response.text}"
            )

    # ==================== Authentication ====================

    def _extract_identity_from_token(self) -> None:
        """Extract the identity ID from the JWT access token."""
        try:
            if self.tokens.access_token:
                parts = self.tokens.access_token.split(".")
                if len(parts) >= 2:
                    payload = parts[1]
                    # Add padding for base64
                    payload += "=" * (4 - len(payload) % 4)
                    decoded = base64.urlsafe_b64decode(payload)
                    token_data = json.loads(decoded)

                    for key in [
                        "identity_canonical_id",
                        "identity_id",
                        "sub",
                        "user_id",
                    ]:
                        if key in token_data:
                            value = token_data[key]
                            if isinstance(value, str) and value.startswith(
                                "identity-"
                            ):
                                self.tokens.identity_id = value
                                return
        except Exception:
            pass

    def login(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        otp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Authenticate with Wealthsimple using OAuth v2.

        Args:
            email: Email address (uses config if not provided)
            password: Password (uses config if not provided)
            otp: OTP code for 2FA (uses callback if not provided)

        Returns:
            Authentication response data

        Raises:
            ValueError: If email or password is missing
            RuntimeError: If login fails
        """
        email = email or self.config.email
        password = password or self.config.password
        otp = otp or self.config.otp

        if not email or not password:
            raise ValueError(
                "Email and password are required. Set WEALTHSIMPLE_EMAIL "
                "and WEALTHSIMPLE_PASSWORD environment variables."
            )

        payload = {
            "grant_type": "password",
            "username": email,
            "password": password,
            "skip_provision": True,
            "scope": "invest.read invest.write trade.read trade.write tax.read tax.write",
            "client_id": self.config.client_id,
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
        }

        if otp:
            headers["x-wealthsimple-otp"] = f"{otp};remember=true"

        response = self._session.post(
            self.config.auth_url,
            json=payload,
            headers=headers,
        )

        # Check if OTP is required (401 response with OTP header)
        if response.status_code == 401:
            # Server may use either x-wealthsimple-otp-required: true
            # or x-wealthsimple-otp: required; method=app
            otp_required_header = response.headers.get("x-wealthsimple-otp-required", "").lower()
            otp_combined_header = response.headers.get("x-wealthsimple-otp", "").lower()
            otp_required = (
                otp_required_header == "true"
                or "required" in otp_combined_header
            )

            if otp_required and not otp:
                # Determine OTP method from response headers
                otp_method = response.headers.get(
                    "x-wealthsimple-otp-method",
                    response.headers.get("X-Wealthsimple-OTP-Method", "unknown"),
                )
                logger.info(f"OTP required (method: {otp_method})")

                if self.config.otp_callback:
                    otp = self.config.otp_callback()
                    headers["x-wealthsimple-otp"] = f"{otp};remember=true"
                    response = self._session.post(
                        self.config.auth_url,
                        json=payload,
                        headers=headers,
                    )
                else:
                    raise RuntimeError(
                        f"2FA is required (method: {otp_method}) but no OTP provided "
                        "and no otp_callback configured. "
                        "Pass otp= or set otp_callback in WealthsimpleConfig."
                    )
            elif otp_required and otp:
                raise RuntimeError(
                    f"OTP code was rejected (expired or invalid). "
                    f"OTP codes are valid for ~30 seconds. "
                    f"Server response: {response.text}"
                )
            else:
                # Not OTP-related — likely wrong credentials
                raise RuntimeError(
                    f"Authentication failed (wrong email/password): "
                    f"{response.status_code} - {response.text}"
                )

        if response.status_code == 200:
            data = response.json()
            self.tokens.access_token = data.get("access_token", "")
            self.tokens.refresh_token = data.get("refresh_token", "")
            expires_in = data.get("expires_in", 1800)
            self.tokens.expires_at = time.time() + expires_in

            # Extract identity ID from response or token
            self.tokens.identity_id = data.get("identity_canonical_id", "")
            if not self.tokens.identity_id:
                self._extract_identity_from_token()

            # Save tokens to disk for persistence
            self._save_tokens()

            logger.info(f"Logged in as {email} (identity: {self.tokens.identity_id})")
            return data
        else:
            raise RuntimeError(
                f"Authentication failed: {response.status_code} - {response.text}"
            )

    def refresh_token(self) -> None:
        """Refresh the access token using the refresh token."""
        if not self.tokens.refresh_token:
            raise RuntimeError("No refresh token available. Call login() first.")

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self.tokens.refresh_token,
            "client_id": self.config.client_id,
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        }

        response = self._session.post(
            self.config.auth_url,
            json=payload,
            headers=headers,
        )

        if response.status_code == 200:
            data = response.json()
            self.tokens.access_token = data.get("access_token", "")
            self.tokens.refresh_token = data.get("refresh_token", self.tokens.refresh_token)
            expires_in = data.get("expires_in", 1800)
            self.tokens.expires_at = time.time() + expires_in
            self._save_tokens()
            logger.info("Token refreshed successfully")
        else:
            raise RuntimeError(
                f"Token refresh failed: {response.status_code} - {response.text}"
            )

    def logout(self) -> None:
        """Clear authentication tokens and remove saved session."""
        self.tokens = AuthTokens()
        self.clear_saved_tokens()
        logger.info("Logged out")

    @property
    def is_authenticated(self) -> bool:
        """Check if currently authenticated with valid tokens"""
        return self.tokens.is_valid

    @property
    def identity_id(self) -> str:
        """Get the identity ID of the authenticated user"""
        return self.tokens.identity_id

    # ==================== User / Identity ====================

    def get_identity(self) -> Dict[str, Any]:
        """
        Get identity/profile information for the authenticated user.

        Returns:
            Identity information dictionary
        """
        if not self.tokens.identity_id:
            raise RuntimeError("No identity ID available. Please authenticate first.")

        result = self.graphql_query(
            "FetchIdentity",
            GQL_FETCH_IDENTITY,
            {"id": self.tokens.identity_id},
        )
        return result.get("data", {}).get("identity", {})

    # ==================== Account Management ====================

    def list_accounts(self) -> List[Dict[str, Any]]:
        """
        List all Wealthsimple accounts.

        Returns:
            List of account dictionaries with id, type, nickname, etc.
        """
        if not self.tokens.identity_id:
            raise RuntimeError("No identity ID available. Please authenticate first.")

        result = self.graphql_query(
            "FetchAllAccounts",
            GQL_FETCH_ALL_ACCOUNTS,
            {
                "identityId": self.tokens.identity_id,
                "filter": {},
                "pageSize": 100,
            },
        )
        edges = (
            result.get("data", {})
            .get("identity", {})
            .get("accounts", {})
            .get("edges", [])
        )
        return [edge.get("node", {}) for edge in edges]

    def get_account_financials(
        self,
        account_ids: List[str],
        currency: str = "CAD",
        start_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get financial information for specific accounts.

        Args:
            account_ids: List of account IDs
            currency: Currency for the financials (default 'CAD')
            start_date: Optional start date for returns calculation (YYYY-MM-DD)

        Returns:
            List of account financial data
        """
        result = self.graphql_query(
            "FetchAccountFinancials",
            GQL_FETCH_ACCOUNT_FINANCIALS,
            {
                "ids": account_ids,
                "currency": currency,
                "startDate": start_date,
            },
        )
        return result.get("data", {}).get("accounts", [])

    def get_account_funding_balances(
        self,
        account_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Get account funding balances (available trading cash).

        Args:
            account_ids: List of account IDs

        Returns:
            List of account funding balances
        """
        result = self.graphql_query(
            "FetchAccountFundingBalances",
            GQL_FETCH_FUNDING_BALANCES,
            {"accountIds": account_ids},
        )
        return result.get("data", {}).get("account_funding_balances", [])

    # ==================== Positions & Holdings ====================

    def get_positions(
        self,
        account_ids: Optional[List[str]] = None,
        currency: Optional[str] = None,
        security_type: Optional[str] = None,
        include_security: bool = True,
        first: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Get current positions/holdings.

        Args:
            account_ids: Optional list of account IDs to filter
            currency: Currency for position values (default: None -> MARKET)
            security_type: Optional security type filter ('EQUITY', 'OPTION', 'CRYPTO')
            include_security: Include full security details in response
            first: Maximum number of positions to return

        Returns:
            List of position dictionaries
        """
        if not self.tokens.identity_id:
            raise RuntimeError("No identity ID available. Please authenticate first.")

        # If currency not set, use MARKET override
        if currency is None:
            currency_override = "MARKET"
            currency = "CAD"  # Default for GraphQL required field
        else:
            currency_override = None

        position_filter = {}
        if security_type:
            position_filter["positionSecurityType"] = security_type

        result = self.graphql_query(
            "FetchIdentityPositions",
            GQL_FETCH_POSITIONS,
            {
                "identityId": self.tokens.identity_id,
                "currency": currency,
                "currencyOverride": currency_override,
                "accountIds": account_ids,
                "filter": position_filter if position_filter else None,
                "first": first,
                "aggregated": False,
                "includeSecurity": include_security,
                "cursor": None,
            },
        )
        positions_data = (
            result.get("data", {})
            .get("identity", {})
            .get("financials", {})
            .get("current", {})
            .get("positions", {})
        )
        edges = positions_data.get("edges", [])
        return [edge.get("node", {}) for edge in edges]

    # ==================== Securities / Market Data ====================

    def search_securities(
        self,
        query: str,
        security_group_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for securities by ticker symbol or name.

        Args:
            query: Search query (e.g., 'AAPL', 'Apple')
            security_group_ids: Optional security group IDs to filter

        Returns:
            List of matching security dictionaries
        """
        result = self.graphql_query(
            "FetchSecuritySearchResult",
            GQL_SEARCH_SECURITIES,
            {
                "query": query,
                "securityGroupIds": security_group_ids,
            },
        )
        return (
            result.get("data", {})
            .get("securitySearch", {})
            .get("results", [])
        )

    def get_security(
        self,
        security_id: str,
        currency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get detailed information about a security.

        Args:
            security_id: The security ID (e.g., 'sec-s-xxxxx')
            currency: Optional currency for fundamentals

        Returns:
            Security details dictionary
        """
        result = self.graphql_query(
            "FetchSecurity",
            GQL_FETCH_SECURITY,
            {
                "securityId": security_id,
                "currency": currency,
            },
        )
        return result.get("data", {}).get("security", {})

    def get_quote(
        self,
        security_id: str,
        currency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get real-time quote for a security.

        Args:
            security_id: The security ID
            currency: Optional currency

        Returns:
            Quote data dictionary
        """
        result = self.graphql_query(
            "FetchSecurityQuoteV2",
            GQL_FETCH_QUOTE,
            {
                "id": security_id,
                "currency": currency,
            },
        )
        return (
            result.get("data", {})
            .get("security", {})
            .get("quoteV2", {})
        )

    def get_option_quote(
        self,
        security_id: str,
    ) -> Dict[str, Any]:
        """
        Get option-specific quote data (strike, expiry, contractType) for an option security.

        Uses the OptionQuote inline fragment on quoteV2.

        Args:
            security_id: The option security ID (e.g., 'sec-o-xxxxx')

        Returns:
            Quote dict with strikePrice, expiryDate, contractType if available
        """
        result = self.graphql_query(
            "FetchOptionQuote",
            GQL_FETCH_OPTION_QUOTE,
            {"id": security_id},
        )
        return (
            result.get("data", {})
            .get("security", {})
            .get("quoteV2", {})
        )

    def find_security_id(
        self,
        ticker: str,
        exchange: Optional[str] = None,
    ) -> Optional[str]:
        """
        Find the security ID for a given ticker symbol.

        Args:
            ticker: Ticker symbol (e.g., 'AAPL')
            exchange: Optional exchange filter (e.g., 'NASDAQ', 'TSX')

        Returns:
            Security ID string, or None if not found
        """
        results = self.search_securities(ticker)
        for result in results:
            stock = result.get("stock", {})
            if stock.get("symbol") == ticker:
                if exchange is None or stock.get("primaryExchange") == exchange:
                    return result.get("id")
        # Return first result if no exact match
        if results:
            return results[0].get("id")
        return None

    def get_quotes_for_symbols(
        self,
        symbols: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of ticker symbols

        Returns:
            List of quote data dictionaries
        """
        results = []
        for symbol in symbols:
            sec_id = self.find_security_id(symbol)
            if sec_id:
                try:
                    quote = self.get_quote(sec_id)
                    quote["_symbol"] = symbol
                    quote["_security_id"] = sec_id
                    results.append(quote)
                except Exception as e:
                    logger.warning(f"Failed to get quote for {symbol}: {e}")
        return results

    # ==================== Activities / Transactions ====================

    def get_activities(
        self,
        account_ids: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None,
        sub_types: Optional[List[str]] = None,
        security_ids: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get activity feed items (orders, trades, deposits, etc.).

        Args:
            account_ids: Optional list of account IDs to filter
            types: Optional list of activity types ('DIY_BUY', 'DIY_SELL', etc.)
            statuses: Optional list of statuses ('PENDING', 'COMPLETED', 'CANCELLED')
            sub_types: Optional sub-types ('LIMIT_ORDER', 'MARKET_ORDER', etc.)
            security_ids: Optional list of security IDs
            start_date: Optional start date in ISO format
            end_date: Optional end date in ISO format
            limit: Maximum number of items to return
            cursor: Optional cursor for pagination

        Returns:
            Dictionary with 'items' and 'pageInfo'
        """
        condition = {}
        if account_ids:
            condition["accountIds"] = account_ids
        if types:
            condition["types"] = types
        if statuses:
            condition["unifiedStatuses"] = statuses
        if sub_types:
            condition["subTypes"] = sub_types
        if security_ids:
            condition["securityIds"] = security_ids
        if start_date:
            condition["startDate"] = start_date
        if end_date:
            condition["endDate"] = end_date

        result = self.graphql_query(
            "FetchActivityFeedItems",
            GQL_FETCH_ACTIVITIES,
            {
                "first": limit,
                "cursor": cursor,
                "condition": condition if condition else None,
                "orderBy": "OCCURRED_AT_DESC",
            },
        )
        activity_data = result.get("data", {}).get("activityFeedItems", {})
        edges = activity_data.get("edges", [])

        return {
            "items": [edge.get("node", {}) for edge in edges],
            "pageInfo": activity_data.get("pageInfo", {}),
        }

    def get_all_activities(
        self,
        account_ids: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        max_pages: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get all activities across multiple pages.

        Args:
            account_ids: Optional account ID filter
            types: Optional activity type filter
            max_pages: Maximum pages to fetch

        Returns:
            Complete list of activities
        """
        all_activities = []
        cursor = None

        for _ in range(max_pages):
            result = self.get_activities(
                account_ids=account_ids,
                types=types,
                limit=100,
                cursor=cursor,
            )
            all_activities.extend(result.get("items", []))

            page_info = result.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")

        return all_activities

    def get_pending_orders(
        self,
        account_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all pending orders.

        Args:
            account_ids: Optional list of account IDs to filter

        Returns:
            List of pending order items
        """
        pending_statuses = {"SUBMITTED", "PENDING", "PARTIALLY_FILLED"}
        result = self.get_activities(
            account_ids=account_ids,
            limit=100,
        )
        # Filter client-side — server-side type/status filters are unreliable
        return [
            item for item in result.get("items", [])
            if item.get("status") in pending_statuses
        ]

    # ==================== Orders / Trading ====================

    def create_order(
        self,
        account_id: str,
        security_id: str,
        quantity: int,
        order_type: str = "BUY_QUANTITY",
        execution_type: str = "LIMIT",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "DAY",
    ) -> Dict[str, Any]:
        """
        Create a new order (unified interface).

        Args:
            account_id: Account ID to place the order in
            security_id: Security ID to trade
            quantity: Number of shares
            order_type: 'BUY_QUANTITY' or 'SELL_QUANTITY'
            execution_type: 'MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'
            limit_price: Limit price (required for LIMIT and STOP_LIMIT)
            stop_price: Stop price (required for STOP and STOP_LIMIT)
            time_in_force: 'DAY' or 'GTC'

        Returns:
            Order creation response
        """
        order_input = {
            "canonicalAccountId": account_id,
            "externalId": f"order-{uuid.uuid4()}",
            "executionType": execution_type,
            "orderType": order_type,
            "quantity": quantity,
            "securityId": security_id,
            "timeInForce": time_in_force,
        }

        if limit_price is not None:
            order_input["limitPrice"] = limit_price
        if stop_price is not None:
            order_input["stopPrice"] = stop_price

        result = self.graphql_query(
            "SoOrdersOrderCreate",
            GQL_CREATE_ORDER,
            {"input": order_input},
        )
        return result.get("data", {}).get("soOrdersCreateOrder", {})

    def market_buy(
        self,
        account_id: str,
        security_id: str,
        quantity: int,
    ) -> Dict[str, Any]:
        """Place a market buy order."""
        return self.create_order(
            account_id, security_id, quantity,
            order_type="BUY_QUANTITY",
            execution_type="MARKET",
        )

    def market_sell(
        self,
        account_id: str,
        security_id: str,
        quantity: int,
    ) -> Dict[str, Any]:
        """Place a market sell order."""
        return self.create_order(
            account_id, security_id, quantity,
            order_type="SELL_QUANTITY",
            execution_type="MARKET",
        )

    def limit_buy(
        self,
        account_id: str,
        security_id: str,
        quantity: int,
        limit_price: float,
        time_in_force: str = "DAY",
    ) -> Dict[str, Any]:
        """Place a limit buy order."""
        return self.create_order(
            account_id, security_id, quantity,
            order_type="BUY_QUANTITY",
            execution_type="LIMIT",
            limit_price=limit_price,
            time_in_force=time_in_force,
        )

    def limit_sell(
        self,
        account_id: str,
        security_id: str,
        quantity: int,
        limit_price: float,
        time_in_force: str = "DAY",
    ) -> Dict[str, Any]:
        """Place a limit sell order."""
        return self.create_order(
            account_id, security_id, quantity,
            order_type="SELL_QUANTITY",
            execution_type="LIMIT",
            limit_price=limit_price,
            time_in_force=time_in_force,
        )

    def stop_limit_buy(
        self,
        account_id: str,
        security_id: str,
        quantity: int,
        limit_price: float,
        stop_price: float,
        time_in_force: str = "DAY",
    ) -> Dict[str, Any]:
        """Place a stop-limit buy order."""
        return self.create_order(
            account_id, security_id, quantity,
            order_type="BUY_QUANTITY",
            execution_type="STOP_LIMIT",
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
        )

    def stop_limit_sell(
        self,
        account_id: str,
        security_id: str,
        quantity: int,
        limit_price: float,
        stop_price: float,
        time_in_force: str = "DAY",
    ) -> Dict[str, Any]:
        """Place a stop-limit sell order."""
        return self.create_order(
            account_id, security_id, quantity,
            order_type="SELL_QUANTITY",
            execution_type="STOP_LIMIT",
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
        )

    def cancel_order(self, external_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order.

        Args:
            external_id: The external order ID
                (e.g., 'order-da8e68b0-6a66-4783-...')

        Returns:
            Cancel order response
        """
        result = self.graphql_query(
            "SoOrdersOrderCancel",
            GQL_CANCEL_ORDER,
            {"cancelOrderRequest": {"externalId": external_id}},
        )
        cancel_data = result.get("data", {}).get("orderServiceCancelOrder", {})
        errors = cancel_data.get("errors", [])
        if errors:
            raise Exception(f"Cancel order failed: {errors}")
        logger.info(f"Order cancelled: {external_id}")
        return cancel_data

    def get_extended_order(
        self,
        external_id: str,
        branch_id: str = "TR",
    ) -> Dict[str, Any]:
        """
        Get extended order details including fill information.

        Args:
            external_id: The external order ID
            branch_id: Branch ID (default: 'TR')

        Returns:
            Extended order details
        """
        result = self.graphql_query(
            "SoOrdersExtendedOrder",
            GQL_EXTENDED_ORDER,
            {
                "branchId": branch_id,
                "externalId": external_id,
            },
        )
        return result.get("data", {}).get("soOrdersExtendedOrder", {})

    # ==================== Convenience Methods ====================

    def get_portfolio_summary(
        self,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get a complete portfolio summary.

        Args:
            account_id: Account ID (uses first account if not provided)

        Returns:
            Portfolio summary with account, positions, and totals
        """
        accounts = self.list_accounts()
        if not accounts:
            return {"error": "No accounts found"}

        if account_id:
            account = next(
                (a for a in accounts if a.get("id") == account_id), None
            )
            if not account:
                return {"error": f"Account {account_id} not found"}
        else:
            account = accounts[0]
            account_id = account.get("id")

        positions = self.get_positions(account_ids=[account_id])

        total_market_value = 0.0
        position_details = []

        for pos in positions:
            qty = float(pos.get("quantity", 0))
            security = pos.get("security", {})
            stock = security.get("stock", {})
            quote = security.get("quoteV2", {})
            price = float(quote.get("price") or 0)
            total_val = pos.get("totalValue", {})
            market_val = float(total_val.get("amount") or 0) if total_val else qty * price
            total_market_value += market_val

            book_val = pos.get("bookValue", {})
            book_amount = float(book_val.get("amount") or 0) if book_val else 0

            unrealized = pos.get("unrealizedReturns", {})
            pnl = float(unrealized.get("amount") or 0) if unrealized else None
            pnl_pct = ((pnl / book_amount) * 100) if pnl and book_amount else None

            position_details.append({
                "symbol": stock.get("symbol", "N/A"),
                "name": stock.get("name", "N/A"),
                "exchange": stock.get("primaryExchange", "N/A"),
                "quantity": qty,
                "price": price,
                "currency": quote.get("currency", "N/A"),
                "market_value": market_val,
                "book_value": book_amount,
                "pnl": pnl,
                "pnl_percent": pnl_pct,
                "security_id": security.get("id"),
            })

        # Get financials
        financials = {}
        try:
            fin_data = self.get_account_financials([account_id])
            if fin_data:
                financials = fin_data[0]
        except Exception:
            pass

        return {
            "account_id": account_id,
            "account_type": account.get("unifiedAccountType", "N/A"),
            "nickname": account.get("nickname", "N/A"),
            "currency": account.get("currency", "CAD"),
            "status": account.get("status", "N/A"),
            "positions_count": len(position_details),
            "total_market_value": total_market_value,
            "positions": position_details,
            "financials": financials,
        }


# ==================== Convenience Functions ====================

def create_wealthsimple_client(
    email: Optional[str] = None,
    password: Optional[str] = None,
    otp: Optional[str] = None,
    otp_callback: Optional[Callable[[], str]] = None,
    auto_login: bool = False,
) -> WealthsimpleClient:
    """
    Create a Wealthsimple client from params or environment variables.

    Args:
        email: Email address (or WEALTHSIMPLE_EMAIL env var)
        password: Password (or WEALTHSIMPLE_PASSWORD env var)
        otp: OTP code for 2FA
        otp_callback: Callback function to get OTP code
        auto_login: Whether to login immediately

    Returns:
        Configured WealthsimpleClient instance
    """
    config = WealthsimpleConfig(
        email=email or os.getenv("WEALTHSIMPLE_EMAIL", ""),
        password=password or os.getenv("WEALTHSIMPLE_PASSWORD", ""),
        otp=otp,
        otp_callback=otp_callback,
    )
    client = WealthsimpleClient(config)

    if auto_login and config.is_ready():
        client.login()

    return client


def quick_login(
    email: Optional[str] = None,
    password: Optional[str] = None,
    otp: Optional[str] = None,
) -> WealthsimpleClient:
    """
    Quick login to Wealthsimple.

    Args:
        email: Email (or WEALTHSIMPLE_EMAIL env var)
        password: Password (or WEALTHSIMPLE_PASSWORD env var)
        otp: OTP code if 2FA is enabled

    Returns:
        Authenticated WealthsimpleClient instance
    """
    config = WealthsimpleConfig(
        email=email or os.getenv("WEALTHSIMPLE_EMAIL", ""),
        password=password or os.getenv("WEALTHSIMPLE_PASSWORD", ""),
        otp=otp,
    )
    client = WealthsimpleClient(config)
    client.login()
    return client
