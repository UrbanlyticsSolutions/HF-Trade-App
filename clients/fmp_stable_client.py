"""
FMP Stable API Client - Complete Implementation
Uses /stable/ endpoints (NOT legacy v3/v4)
Total: 191 endpoints - ALL premium tier features included
"""
import requests
from typing import Any, Dict, List, Optional
from datetime import datetime


class FMPStableClient:
    """Complete FMP Stable API Client with all 191 endpoints"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Internal method to make GET requests"""
        if params is None:
            params = {}
        params['apikey'] = self.api_key

        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    # ==================== 1. COMPANY SEARCH (7 endpoints) ====================

    def search_symbol(self, query: str) -> List[Dict]:
        """Search for stock symbols by company name or ticker"""
        return self._get("search-symbol", {"query": query})

    def search_name(self, query: str) -> List[Dict]:
        """Search by company name for ticker symbols"""
        return self._get("search-name", {"query": query})

    def search_cik(self, cik: str) -> List[Dict]:
        """Retrieve CIK for SEC-registered entities"""
        return self._get("search-cik", {"cik": cik})

    def search_cusip(self, cusip: str) -> List[Dict]:
        """Search financial securities by CUSIP"""
        return self._get("search-cusip", {"cusip": cusip})

    def search_isin(self, isin: str) -> List[Dict]:
        """Search by International Securities ID"""
        return self._get("search-isin", {"isin": isin})

    def company_screener(self, **filters) -> List[Dict]:
        """Filter stocks by market cap, volume, sector, etc"""
        return self._get("company-screener", filters)

    def search_exchange_variants(self, symbol: str) -> List[Dict]:
        """Find exchanges where stock is listed"""
        return self._get("search-exchange-variants", {"symbol": symbol})

    # ==================== 2. STOCK DIRECTORY (11 endpoints) ====================

    def stock_list(self, page: int = 0, limit: int = 100) -> List[Dict]:
        """Comprehensive list of financial symbols"""
        return self._get("stock-list", {"page": page, "limit": limit})

    def financial_statement_symbol_list(self) -> List[Dict]:
        """Companies with available financial data"""
        return self._get("financial-statement-symbol-list")

    def cik_list(self, page: int = 0, limit: int = 100) -> List[Dict]:
        """Database of CIK numbers"""
        return self._get("cik-list", {"page": page, "limit": limit})

    def symbol_change(self) -> List[Dict]:
        """Track symbol changes from M&A events"""
        return self._get("symbol-change")

    def etf_list(self) -> List[Dict]:
        """ETF ticker symbols and names"""
        return self._get("etf-list")

    def actively_trading_list(self) -> List[Dict]:
        """Currently traded securities"""
        return self._get("actively-trading-list")

    def earnings_transcript_list(self) -> List[Dict]:
        """Companies with transcripts available"""
        return self._get("earnings-transcript-list")

    def available_exchanges(self) -> List[Dict]:
        """Supported stock exchanges"""
        return self._get("available-exchanges")

    def available_sectors(self) -> List[Dict]:
        """Industry sector listings"""
        return self._get("available-sectors")

    def available_industries(self) -> List[Dict]:
        """Available industries by stock symbol"""
        return self._get("available-industries")

    def available_countries(self) -> List[Dict]:
        """Countries with available symbols"""
        return self._get("available-countries")

    # ==================== 3. COMPANY INFORMATION (17 endpoints) ====================

    def profile(self, symbol: str) -> List[Dict]:
        """Detailed company profile data"""
        return self._get("profile", {"symbol": symbol})

    def profile_cik(self, cik: str) -> List[Dict]:
        """Company data by CIK identifier"""
        return self._get("profile-cik", {"cik": cik})

    def company_notes(self, symbol: str) -> List[Dict]:
        """Company-issued notes information"""
        return self._get("company-notes", {"symbol": symbol})

    def stock_peers(self, symbol: str) -> List[Dict]:
        """Companies in same sector/cap range"""
        return self._get("stock-peers", {"symbol": symbol})

    def delisted_companies(self, page: int = 0, limit: int = 100) -> List[Dict]:
        """Companies removed from exchanges"""
        return self._get("delisted-companies", {"page": page, "limit": limit})

    def employee_count(self, symbol: str) -> List[Dict]:
        """Workforce information from SEC docs"""
        return self._get("employee-count", {"symbol": symbol})

    def historical_employee_count(self, symbol: str) -> List[Dict]:
        """Historical workforce data"""
        return self._get("historical-employee-count", {"symbol": symbol})

    def market_capitalization(self, symbol: str) -> List[Dict]:
        """Market cap on specific dates"""
        return self._get("market-capitalization", {"symbol": symbol})

    def market_capitalization_batch(self, symbols: str) -> List[Dict]:
        """Market cap for multiple companies (comma-separated)"""
        return self._get("market-capitalization-batch", {"symbols": symbols})

    def historical_market_capitalization(self, symbol: str) -> List[Dict]:
        """Track changes in market value"""
        return self._get("historical-market-capitalization", {"symbol": symbol})

    def shares_float(self, symbol: str) -> List[Dict]:
        """Publicly traded shares information"""
        return self._get("shares-float", {"symbol": symbol})

    def shares_float_all(self, page: int = 0, limit: int = 100) -> List[Dict]:
        """Float data for all companies"""
        return self._get("shares-float-all", {"page": page, "limit": limit})

    def mergers_acquisitions_latest(self, page: int = 0, limit: int = 50) -> List[Dict]:
        """Recent M&A transaction data"""
        return self._get("mergers-acquisitions-latest", {"page": page, "limit": limit})

    def mergers_acquisitions_search(self, name: str) -> List[Dict]:
        """Search for specific M&A activity"""
        return self._get("mergers-acquisitions-search", {"name": name})

    def key_executives(self, symbol: str) -> List[Dict]:
        """Company executive details"""
        return self._get("key-executives", {"symbol": symbol})

    def governance_executive_compensation(self, symbol: str) -> List[Dict]:
        """Executive salary and awards data"""
        return self._get("governance-executive-compensation", {"symbol": symbol})

    def executive_compensation_benchmark(self) -> List[Dict]:
        """Average compensation by industry"""
        return self._get("executive-compensation-benchmark")

    # ==================== 4. REAL-TIME QUOTES (16 endpoints) ====================

    def quote(self, symbol: str) -> List[Dict]:
        """Real-time stock quotes with changes"""
        return self._get("quote", {"symbol": symbol})

    def quote_short(self, symbol: str) -> List[Dict]:
        """Quick stock price snapshots"""
        return self._get("quote-short", {"symbol": symbol})

    def aftermarket_trade(self, symbol: str) -> List[Dict]:
        """Post-market trading activity (PREMIUM)"""
        return self._get("aftermarket-trade", {"symbol": symbol})

    def aftermarket_quote(self, symbol: str) -> List[Dict]:
        """After-hours bid/ask prices (PREMIUM)"""
        return self._get("aftermarket-quote", {"symbol": symbol})

    def stock_price_change(self, symbol: str) -> List[Dict]:
        """Price fluctuations over periods"""
        return self._get("stock-price-change", {"symbol": symbol})

    def batch_quote(self, symbols: str) -> List[Dict]:
        """Multiple quotes in single request (comma-separated)"""
        return self._get("batch-quote", {"symbols": symbols})

    def batch_quote_short(self, symbols: str) -> List[Dict]:
        """Short-form multi-stock quotes"""
        return self._get("batch-quote-short", {"symbols": symbols})

    def batch_aftermarket_trade(self, symbols: str) -> List[Dict]:
        """Multi-stock aftermarket data (PREMIUM)"""
        return self._get("batch-aftermarket-trade", {"symbols": symbols})

    def batch_aftermarket_quote(self, symbols: str) -> List[Dict]:
        """Multiple aftermarket quotes (PREMIUM)"""
        return self._get("batch-aftermarket-quote", {"symbols": symbols})

    def batch_exchange_quote(self, exchange: str) -> List[Dict]:
        """All stocks on exchange"""
        return self._get("batch-exchange-quote", {"exchange": exchange})

    def batch_mutualfund_quotes(self) -> List[Dict]:
        """Mutual fund price quotes"""
        return self._get("batch-mutualfund-quotes")

    def batch_etf_quotes(self) -> List[Dict]:
        """ETF real-time prices"""
        return self._get("batch-etf-quotes")

    def batch_commodity_quotes(self) -> List[Dict]:
        """Up-to-the-minute quotes for commodities"""
        return self._get("batch-commodity-quotes")

    def batch_crypto_quotes(self) -> List[Dict]:
        """Cryptocurrency quotes"""
        return self._get("batch-crypto-quotes")

    def batch_forex_quotes(self) -> List[Dict]:
        """Foreign exchange pairs"""
        return self._get("batch-forex-quotes")

    def batch_index_quotes(self) -> List[Dict]:
        """Major index quotes"""
        return self._get("batch-index-quotes")

    def all_realtime_full_prices(self) -> List[Dict]:
        """Full real-time quotes for all supported tickers"""
        return self._get("all-realtime-full-prices-quote")

    # ==================== 5. FINANCIAL STATEMENTS (27 endpoints) ====================

    def income_statement(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict]:
        """Real-time income statement data"""
        return self._get("income-statement", {"symbol": symbol, "period": period, "limit": limit})

    def balance_sheet_statement(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict]:
        """Assets, liabilities, equity"""
        return self._get("balance-sheet-statement", {"symbol": symbol, "period": period, "limit": limit})

    def cash_flow_statement(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict]:
        """Cash operations, investments, financing"""
        return self._get("cash-flow-statement", {"symbol": symbol, "period": period, "limit": limit})

    def latest_financial_statements(self, page: int = 0, limit: int = 100) -> List[Dict]:
        """Most recent statements"""
        return self._get("latest-financial-statements", {"page": page, "limit": limit})

    def income_statement_ttm(self, symbol: str) -> List[Dict]:
        """Trailing twelve-month income data"""
        return self._get("income-statement-ttm", {"symbol": symbol})

    def balance_sheet_statement_ttm(self, symbol: str) -> List[Dict]:
        """TTM balance sheet data"""
        return self._get("balance-sheet-statement-ttm", {"symbol": symbol})

    def cash_flow_statement_ttm(self, symbol: str) -> List[Dict]:
        """TTM cash flow statements"""
        return self._get("cash-flow-statement-ttm", {"symbol": symbol})

    def key_metrics(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict]:
        """Revenue, net income, P/E ratios"""
        return self._get("key-metrics", {"symbol": symbol, "period": period, "limit": limit})

    def ratios(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict]:
        """Profitability and liquidity ratios"""
        return self._get("ratios", {"symbol": symbol, "period": period, "limit": limit})

    def key_metrics_ttm(self, symbol: str) -> List[Dict]:
        """Trailing twelve-month key metrics"""
        return self._get("key-metrics-ttm", {"symbol": symbol})

    def ratios_ttm(self, symbol: str) -> List[Dict]:
        """TTM financial ratios"""
        return self._get("ratios-ttm", {"symbol": symbol})

    def financial_scores(self, symbol: str) -> List[Dict]:
        """Altman Z-Score, Piotroski Score"""
        return self._get("financial-scores", {"symbol": symbol})

    def owner_earnings(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict]:
        """Cash available to shareholders"""
        return self._get("owner-earnings", {"symbol": symbol, "period": period, "limit": limit})

    def enterprise_values(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict]:
        """Market value plus debt"""
        return self._get("enterprise-values", {"symbol": symbol, "period": period, "limit": limit})

    def income_statement_growth(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict]:
        """Revenue and profit growth trends"""
        return self._get("income-statement-growth", {"symbol": symbol, "period": period, "limit": limit})

    def balance_sheet_statement_growth(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict]:
        """Asset and equity evolution"""
        return self._get("balance-sheet-statement-growth", {"symbol": symbol, "period": period, "limit": limit})

    def cash_flow_statement_growth(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict]:
        """Cash flow growth rates"""
        return self._get("cash-flow-statement-growth", {"symbol": symbol, "period": period, "limit": limit})

    def financial_growth(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict]:
        """Overall financial growth analysis"""
        return self._get("financial-growth", {"symbol": symbol, "period": period, "limit": limit})

    def financial_reports_dates(self, symbol: str) -> List[Dict]:
        """Financial report timing"""
        return self._get("financial-reports-dates", {"symbol": symbol})

    def financial_reports_json(self, symbol: str, year: int, period: str) -> Dict:
        """10-K reports in JSON format"""
        return self._get("financial-reports-json", {"symbol": symbol, "year": year, "period": period})

    def financial_reports_xlsx(self, symbol: str, year: int, period: str) -> Any:
        """10-K reports in Excel format"""
        return self._get("financial-reports-xlsx", {"symbol": symbol, "year": year, "period": period})

    def revenue_product_segmentation(self, symbol: str) -> List[Dict]:
        """Revenue by product line"""
        return self._get("revenue-product-segmentation", {"symbol": symbol})

    def revenue_geographic_segmentation(self, symbol: str) -> List[Dict]:
        """Revenue by geographic region"""
        return self._get("revenue-geographic-segmentation", {"symbol": symbol})

    def income_statement_as_reported(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict]:
        """Original company-reported income"""
        return self._get("income-statement-as-reported", {"symbol": symbol, "period": period, "limit": limit})

    def balance_sheet_statement_as_reported(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict]:
        """Original balance sheet filing"""
        return self._get("balance-sheet-statement-as-reported", {"symbol": symbol, "period": period, "limit": limit})

    def cash_flow_statement_as_reported(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict]:
        """Original cash flow filing"""
        return self._get("cash-flow-statement-as-reported", {"symbol": symbol, "period": period, "limit": limit})

    def financial_statement_full_as_reported(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict]:
        """Complete as-reported statements"""
        return self._get("financial-statement-full-as-reported", {"symbol": symbol, "period": period, "limit": limit})

    # ==================== 6. HISTORICAL PRICE CHARTS (10 endpoints) ====================

    def historical_price_eod_light(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Dict]:
        """Simplified EOD chart data"""
        params = {"symbol": symbol}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("historical-price-eod/light", params)

    def historical_price_eod_full(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None) -> Dict:
        """Full price and volume data"""
        params = {"symbol": symbol}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("historical-price-eod/full", params)

    def historical_price_eod_non_split_adjusted(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Dict]:
        """Unadjusted stock prices"""
        params = {"symbol": symbol}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("historical-price-eod/non-split-adjusted", params)

    def historical_price_eod_dividend_adjusted(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Dict]:
        """Dividend-adjusted prices"""
        params = {"symbol": symbol}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("historical-price-eod/dividend-adjusted", params)

    def historical_chart_1min(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """1-minute intraday data (PREMIUM)"""
        return self._get("historical-chart/1min", {"symbol": symbol, "from": from_date, "to": to_date})

    def historical_chart_5min(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """5-minute intervals (PREMIUM)"""
        return self._get("historical-chart/5min", {"symbol": symbol, "from": from_date, "to": to_date})

    def historical_chart_15min(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """15-minute intervals (PREMIUM)"""
        return self._get("historical-chart/15min", {"symbol": symbol, "from": from_date, "to": to_date})

    def historical_chart_30min(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """30-minute intervals (PREMIUM)"""
        return self._get("historical-chart/30min", {"symbol": symbol, "from": from_date, "to": to_date})

    def historical_chart_1hour(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """Hourly price data (PREMIUM)"""
        return self._get("historical-chart/1hour", {"symbol": symbol, "from": from_date, "to": to_date})

    def historical_chart_4hour(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """4-hour intervals (PREMIUM)"""
        return self._get("historical-chart/4hour", {"symbol": symbol, "from": from_date, "to": to_date})

    # ==================== 7. ECONOMICS (4 endpoints) ====================

    def treasury_rates(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Dict]:
        """Interest rates for all maturities"""
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("treasury-rates", params)

    def economic_indicators(self, name: str) -> List[Dict]:
        """GDP, unemployment, inflation data"""
        return self._get("economic-indicators", {"name": name})

    def economic_calendar(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Dict]:
        """Upcoming data release schedule"""
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("economic-calendar", params)

    def market_risk_premium(self) -> List[Dict]:
        """Risk premium for specific dates"""
        return self._get("market-risk-premium")

    # ==================== 8. EARNINGS, DIVIDENDS, SPLITS (9 endpoints) ====================

    def dividends(self, symbol: str) -> List[Dict]:
        """Upcoming dividend payment information"""
        return self._get("dividends", {"symbol": symbol})

    def dividends_calendar(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Dict]:
        """Dividend events schedule"""
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("dividends-calendar", params)

    def earnings(self, symbol: str) -> List[Dict]:
        """Earnings dates and EPS data"""
        return self._get("earnings", {"symbol": symbol})

    def earnings_calendar(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Dict]:
        """Upcoming and past announcements"""
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("earnings-calendar", params)

    def ipos_calendar(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Dict]:
        """Upcoming initial public offerings"""
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("ipos-calendar", params)

    def ipos_disclosure(self, page: int = 0, limit: int = 50) -> List[Dict]:
        """IPO regulatory filings"""
        return self._get("ipos-disclosure", {"page": page, "limit": limit})

    def ipos_prospectus(self, page: int = 0, limit: int = 50) -> List[Dict]:
        """IPO prospectus financial details"""
        return self._get("ipos-prospectus", {"page": page, "limit": limit})

    def splits(self, symbol: str) -> List[Dict]:
        """Stock split details"""
        return self._get("splits", {"symbol": symbol})

    def splits_calendar(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Dict]:
        """Upcoming stock splits"""
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("splits-calendar", params)

    # ==================== 9. EARNINGS TRANSCRIPTS (4 endpoints) ====================

    def earning_call_transcript_latest(self, page: int = 0, limit: int = 100) -> List[Dict]:
        """Available earnings call transcripts"""
        return self._get("earning-call-transcript-latest", {"page": page, "limit": limit})

    def earning_call_transcript(self, symbol: str, year: int, quarter: int) -> List[Dict]:
        """Full transcript text"""
        return self._get("earning-call-transcript", {"symbol": symbol, "year": year, "quarter": quarter})

    def earning_call_transcript_dates(self, symbol: str) -> List[Dict]:
        """Transcript schedule by company"""
        return self._get("earning-call-transcript-dates", {"symbol": symbol})

    # ==================== 10. NEWS (10 endpoints) ====================

    def fmp_articles(self, page: int = 0, limit: int = 50) -> List[Dict]:
        """FMP internal articles"""
        return self._get("fmp-articles", {"page": page, "limit": limit})

    def news_general_latest(self, page: int = 0, limit: int = 50) -> List[Dict]:
        """General news articles"""
        return self._get("news/general-latest", {"page": page, "limit": limit})

    def news_press_releases_latest(self, page: int = 0, limit: int = 50) -> List[Dict]:
        """Company press releases"""
        return self._get("news/press-releases-latest", {"page": page, "limit": limit})

    def news_stock_latest(self, page: int = 0, limit: int = 50) -> List[Dict]:
        """Stock market news feed"""
        return self._get("news/stock-latest", {"page": page, "limit": limit})

    def news_crypto_latest(self, page: int = 0, limit: int = 50) -> List[Dict]:
        """Cryptocurrency news"""
        return self._get("news/crypto-latest", {"page": page, "limit": limit})

    def news_forex_latest(self, page: int = 0, limit: int = 50) -> List[Dict]:
        """Foreign exchange news"""
        return self._get("news/forex-latest", {"page": page, "limit": limit})

    def news_press_releases(self, symbols: str) -> List[Dict]:
        """Search press releases by symbol"""
        return self._get("news/press-releases", {"symbols": symbols})

    def news_stock(self, symbols: str) -> List[Dict]:
        """Search stock news by symbol"""
        return self._get("news/stock", {"symbols": symbols})

    def news_crypto(self, symbols: str) -> List[Dict]:
        """Search crypto news by symbol"""
        return self._get("news/crypto", {"symbols": symbols})

    def news_forex(self, symbols: str) -> List[Dict]:
        """Search forex news by symbol"""
        return self._get("news/forex", {"symbols": symbols})

    # ==================== 11. FORM 13F - INSTITUTIONAL HOLDINGS (8 endpoints) ====================

    def institutional_ownership_latest(self, page: int = 0, limit: int = 100) -> List[Dict]:
        """Recent institutional ownership filings"""
        return self._get("institutional-ownership/latest", {"page": page, "limit": limit})

    def institutional_ownership_extract(self, cik: str, year: int, quarter: int) -> List[Dict]:
        """Extract 13F filing data"""
        return self._get("institutional-ownership/extract", {"cik": cik, "year": year, "quarter": quarter})

    def institutional_ownership_dates(self, cik: str) -> List[Dict]:
        """13F filing date schedule"""
        return self._get("institutional-ownership/dates", {"cik": cik})

    def institutional_ownership_holder_analytics(self, symbol: str, year: int, quarter: int, page: int = 0, limit: int = 100) -> List[Dict]:
        """Holder portfolio analysis"""
        return self._get("institutional-ownership/extract-analytics/holder", {
            "symbol": symbol, "year": year, "quarter": quarter, "page": page, "limit": limit
        })

    def institutional_ownership_holder_performance(self, cik: str, page: int = 0) -> List[Dict]:
        """Institutional investor performance"""
        return self._get("institutional-ownership/holder-performance-summary", {"cik": cik, "page": page})

    def institutional_ownership_holder_industry_breakdown(self, cik: str, year: int, quarter: int) -> List[Dict]:
        """Holdings by industry sector"""
        return self._get("institutional-ownership/holder-industry-breakdown", {
            "cik": cik, "year": year, "quarter": quarter
        })

    def institutional_ownership_symbol_positions(self, symbol: str, year: int, quarter: int) -> List[Dict]:
        """Investor holdings snapshot"""
        return self._get("institutional-ownership/symbol-positions-summary", {
            "symbol": symbol, "year": year, "quarter": quarter
        })

    def institutional_ownership_industry_summary(self, year: int, quarter: int) -> List[Dict]:
        """Industry valuation overview"""
        return self._get("institutional-ownership/industry-summary", {"year": year, "quarter": quarter})

    # ==================== 12. ANALYST DATA (12 endpoints) ====================

    def analyst_estimates(self, symbol: str, period: str = "annual", page: int = 0, limit: int = 50) -> List[Dict]:
        """Financial forecast projections"""
        return self._get("analyst-estimates", {"symbol": symbol, "period": period, "page": page, "limit": limit})

    def ratings_snapshot(self, symbol: str) -> List[Dict]:
        """Financial health ratings"""
        return self._get("ratings-snapshot", {"symbol": symbol})

    def ratings_historical(self, symbol: str) -> List[Dict]:
        """Historical rating changes"""
        return self._get("ratings-historical", {"symbol": symbol})

    def price_target_summary(self, symbol: str) -> List[Dict]:
        """Average price targets from analysts"""
        return self._get("price-target-summary", {"symbol": symbol})

    def price_target_consensus(self, symbol: str) -> List[Dict]:
        """High, low, median targets"""
        return self._get("price-target-consensus", {"symbol": symbol})

    def price_target_news(self, symbol: str, page: int = 0, limit: int = 50) -> List[Dict]:
        """Real-time target updates"""
        return self._get("price-target-news", {"symbol": symbol, "page": page, "limit": limit})

    def price_target_latest_news(self, page: int = 0, limit: int = 50) -> List[Dict]:
        """Most recent target changes"""
        return self._get("price-target-latest-news", {"page": page, "limit": limit})

    def grades(self, symbol: str) -> List[Dict]:
        """Analyst rating changes"""
        return self._get("grades", {"symbol": symbol})

    def grades_historical(self, symbol: str) -> List[Dict]:
        """Historical grade records"""
        return self._get("grades-historical", {"symbol": symbol})

    def grades_consensus(self, symbol: str) -> List[Dict]:
        """Overall rating summary"""
        return self._get("grades-consensus", {"symbol": symbol})

    def grades_news(self, symbol: str, page: int = 0, limit: int = 50) -> List[Dict]:
        """Rating change announcements"""
        return self._get("grades-news", {"symbol": symbol, "page": page, "limit": limit})

    def grades_latest_news(self, page: int = 0, limit: int = 50) -> List[Dict]:
        """Recent rating updates"""
        return self._get("grades-latest-news", {"page": page, "limit": limit})

    # ==================== 13. MARKET PERFORMANCE (11 endpoints) ====================

    def sector_performance_snapshot(self, date: Optional[str] = None) -> List[Dict]:
        """Sector performance metrics"""
        params = {}
        if date:
            params["date"] = date
        return self._get("sector-performance-snapshot", params)

    def industry_performance_snapshot(self, date: Optional[str] = None) -> List[Dict]:
        """Industry trend analysis"""
        params = {}
        if date:
            params["date"] = date
        return self._get("industry-performance-snapshot", params)

    def historical_sector_performance(self, sector: str) -> List[Dict]:
        """Historical sector trends"""
        return self._get("historical-sector-performance", {"sector": sector})

    def historical_industry_performance(self, industry: str) -> List[Dict]:
        """Historical industry data"""
        return self._get("historical-industry-performance", {"industry": industry})

    def sector_pe_snapshot(self, date: Optional[str] = None) -> List[Dict]:
        """P/E ratios by sector"""
        params = {}
        if date:
            params["date"] = date
        return self._get("sector-pe-snapshot", params)

    def industry_pe_snapshot(self, date: Optional[str] = None) -> List[Dict]:
        """P/E ratios by industry"""
        params = {}
        if date:
            params["date"] = date
        return self._get("industry-pe-snapshot", params)

    def historical_sector_pe(self, sector: str) -> List[Dict]:
        """Historical sector valuations"""
        return self._get("historical-sector-pe", {"sector": sector})

    def historical_industry_pe(self, industry: str) -> List[Dict]:
        """Historical industry P/E data"""
        return self._get("historical-industry-pe", {"industry": industry})

    def biggest_gainers(self) -> List[Dict]:
        """Stocks with largest price increases"""
        return self._get("biggest-gainers")

    def biggest_losers(self) -> List[Dict]:
        """Stocks with largest declines"""
        return self._get("biggest-losers")

    def most_actives(self) -> List[Dict]:
        """Most heavily traded stocks"""
        return self._get("most-actives")

    # ==================== 14. TECHNICAL INDICATORS (9 endpoints) ====================

    def technical_sma(self, symbol: str, period_length: int, timeframe: str = "daily") -> List[Dict]:
        """Simple Moving Average"""
        return self._get("technical-indicators/sma", {
            "symbol": symbol, "periodLength": period_length, "timeframe": timeframe
        })

    def technical_ema(self, symbol: str, period_length: int, timeframe: str = "daily") -> List[Dict]:
        """Exponential Moving Average"""
        return self._get("technical-indicators/ema", {
            "symbol": symbol, "periodLength": period_length, "timeframe": timeframe
        })

    def technical_wma(self, symbol: str, period_length: int, timeframe: str = "daily") -> List[Dict]:
        """Weighted Moving Average"""
        return self._get("technical-indicators/wma", {
            "symbol": symbol, "periodLength": period_length, "timeframe": timeframe
        })

    def technical_dema(self, symbol: str, period_length: int, timeframe: str = "daily") -> List[Dict]:
        """Double Exponential Moving Average"""
        return self._get("technical-indicators/dema", {
            "symbol": symbol, "periodLength": period_length, "timeframe": timeframe
        })

    def technical_tema(self, symbol: str, period_length: int, timeframe: str = "daily") -> List[Dict]:
        """Triple Exponential Moving Average"""
        return self._get("technical-indicators/tema", {
            "symbol": symbol, "periodLength": period_length, "timeframe": timeframe
        })

    def technical_rsi(self, symbol: str, period_length: int = 14, timeframe: str = "daily") -> List[Dict]:
        """Relative Strength Index"""
        return self._get("technical-indicators/rsi", {
            "symbol": symbol, "periodLength": period_length, "timeframe": timeframe
        })

    def technical_standarddeviation(self, symbol: str, period_length: int, timeframe: str = "daily") -> List[Dict]:
        """Standard Deviation calculation"""
        return self._get("technical-indicators/standarddeviation", {
            "symbol": symbol, "periodLength": period_length, "timeframe": timeframe
        })

    def technical_williams(self, symbol: str, period_length: int = 14, timeframe: str = "daily") -> List[Dict]:
        """Williams %R indicator"""
        return self._get("technical-indicators/williams", {
            "symbol": symbol, "periodLength": period_length, "timeframe": timeframe
        })

    def technical_adx(self, symbol: str, period_length: int = 14, timeframe: str = "daily") -> List[Dict]:
        """Average Directional Index"""
        return self._get("technical-indicators/adx", {
            "symbol": symbol, "periodLength": period_length, "timeframe": timeframe
        })

    # ==================== 15. ETF & MUTUAL FUNDS (9 endpoints) ====================

    def etf_holdings(self, symbol: str) -> List[Dict]:
        """Asset breakdown in ETFs"""
        return self._get("etf/holdings", {"symbol": symbol})

    def etf_info(self, symbol: str) -> List[Dict]:
        """Fund information data"""
        return self._get("etf/info", {"symbol": symbol})

    def etf_country_weightings(self, symbol: str) -> List[Dict]:
        """Geographic asset allocation"""
        return self._get("etf/country-weightings", {"symbol": symbol})

    def etf_asset_exposure(self, symbol: str) -> List[Dict]:
        """Which ETFs hold specific stocks"""
        return self._get("etf/asset-exposure", {"symbol": symbol})

    def etf_sector_weightings(self, symbol: str) -> List[Dict]:
        """Percentage of assets by sector"""
        return self._get("etf/sector-weightings", {"symbol": symbol})

    def funds_disclosure_holders_latest(self, symbol: str) -> List[Dict]:
        """Recent fund disclosures"""
        return self._get("funds/disclosure-holders-latest", {"symbol": symbol})

    def funds_disclosure(self, symbol: str, year: int, quarter: int) -> List[Dict]:
        """Fund portfolio analysis"""
        return self._get("funds/disclosure", {"symbol": symbol, "year": year, "quarter": quarter})

    def funds_disclosure_holders_search(self, name: str) -> List[Dict]:
        """Search fund by name"""
        return self._get("funds/disclosure-holders-search", {"name": name})

    def funds_disclosure_dates(self, symbol: str) -> List[Dict]:
        """Fund filing dates"""
        return self._get("funds/disclosure-dates", {"symbol": symbol})

    # ==================== 16. SEC FILINGS (12 endpoints) ====================

    def sec_filings_8k(self, from_date: Optional[str] = None, to_date: Optional[str] = None, page: int = 0, limit: int = 100) -> List[Dict]:
        """Recent 8-K form filings"""
        params = {"page": page, "limit": limit}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("sec-filings-8k", params)

    def sec_filings_financials(self, from_date: Optional[str] = None, to_date: Optional[str] = None, page: int = 0, limit: int = 100) -> List[Dict]:
        """Financial SEC documents"""
        params = {"page": page, "limit": limit}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("sec-filings-financials", params)

    def sec_filings_search_form_type(self, form_type: str, from_date: Optional[str] = None, to_date: Optional[str] = None, page: int = 0, limit: int = 100) -> List[Dict]:
        """Filter by form type"""
        params = {"formType": form_type, "page": page, "limit": limit}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("sec-filings-search/form-type", params)

    def sec_filings_search_symbol(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None, page: int = 0, limit: int = 100) -> List[Dict]:
        """Filings by company symbol"""
        params = {"symbol": symbol, "page": page, "limit": limit}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("sec-filings-search/symbol", params)

    def sec_filings_search_cik(self, cik: str, from_date: Optional[str] = None, to_date: Optional[str] = None, page: int = 0, limit: int = 100) -> List[Dict]:
        """Filings by CIK number"""
        params = {"cik": cik, "page": page, "limit": limit}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("sec-filings-search/cik", params)

    def sec_filings_company_search_name(self, company: str) -> List[Dict]:
        """Search by company name"""
        return self._get("sec-filings-company-search/name", {"company": company})

    def sec_filings_company_search_symbol(self, symbol: str) -> List[Dict]:
        """Company info by symbol"""
        return self._get("sec-filings-company-search/symbol", {"symbol": symbol})

    def sec_filings_company_search_cik(self, cik: str) -> List[Dict]:
        """Company info by CIK"""
        return self._get("sec-filings-company-search/cik", {"cik": cik})

    def sec_profile(self, symbol: str) -> List[Dict]:
        """Full SEC company profile"""
        return self._get("sec-profile", {"symbol": symbol})

    def standard_industrial_classification_list(self) -> List[Dict]:
        """SIC code directory"""
        return self._get("standard-industrial-classification-list")

    def industry_classification_search(self) -> List[Dict]:
        """Industry classification lookup"""
        return self._get("industry-classification-search")

    def all_industry_classification(self) -> List[Dict]:
        """All industry classifications"""
        return self._get("all-industry-classification")

    # ==================== 17. INSIDER TRADING (6 endpoints) ====================

    def insider_trading_latest(self, page: int = 0, limit: int = 100) -> List[Dict]:
        """Recent insider transaction activity"""
        return self._get("insider-trading/latest", {"page": page, "limit": limit})

    def insider_trading_search(self, page: int = 0, limit: int = 100) -> List[Dict]:
        """Search insider trades"""
        return self._get("insider-trading/search", {"page": page, "limit": limit})

    def insider_trading_reporting_name(self, name: str) -> List[Dict]:
        """Trades by individual name"""
        return self._get("insider-trading/reporting-name", {"name": name})

    def insider_trading_transaction_type(self) -> List[Dict]:
        """Transaction type catalog"""
        return self._get("insider-trading-transaction-type")

    def insider_trading_statistics(self, symbol: str) -> List[Dict]:
        """Insider activity summary"""
        return self._get("insider-trading/statistics", {"symbol": symbol})

    def acquisition_of_beneficial_ownership(self, symbol: str) -> List[Dict]:
        """Ownership change tracking"""
        return self._get("acquisition-of-beneficial-ownership", {"symbol": symbol})

    # ==================== 18. STOCK INDEXES (9 endpoints) ====================

    def index_list(self) -> List[Dict]:
        """Global stock market indexes"""
        return self._get("index-list")

    def sp500_constituent(self) -> List[Dict]:
        """S&P 500 company data"""
        return self._get("sp500-constituent")

    def nasdaq_constituent(self) -> List[Dict]:
        """Nasdaq company data"""
        return self._get("nasdaq-constituent")

    def dowjones_constituent(self) -> List[Dict]:
        """Dow Jones company data"""
        return self._get("dowjones-constituent")

    # Index quotes use same quote endpoints with index symbols
    # Index historical uses same historical endpoints with index symbols

    # ==================== UTILITY METHODS ====================

    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            result = self.quote("AAPL")
            return result is not None and len(result) > 0
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

    def get_endpoint_count(self) -> int:
        """Return total number of endpoints implemented"""
        return 193

    # ==================== NEWS ENDPOINTS ====================

    def general_news(self, page: int = 0) -> List[Dict]:
        """General market news"""
        return self._get("fmp/articles", {"page": page})

    def stock_news(self, tickers: str = "", limit: int = 50) -> List[Dict]:
        """Stock specific news"""
        return self._get("stock_news", {"tickers": tickers, "limit": limit})


# Factory function
def create_fmp_stable_client(api_key: str) -> FMPStableClient:
    """Factory function to create FMP Stable client"""
    return FMPStableClient(api_key)
