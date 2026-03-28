"""
Regime Detector — Pre-market preliminary check.

Uses ONLY data available BEFORE the 10:00 AM trading window:
  1. VIX prior close        (from FMP, cached in DB)
  2. SPY daily technicals   (ADX, daily RSI, SMA slope — from prior closes)
  3. Treasury yield curve   (2Y-10Y spread from FMP)
  4. Economic calendar      (high-impact events today)

All data is fetched once from FMP, cached in SQLite, and looked up by date.
No look-ahead bias: every feature uses data available at market open.
"""
import sqlite3
import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class RegimeConfig:
    """Configuration for the regime detector."""
    # VIX thresholds
    vix_low: float = 14.0           # Below = low-vol regime (choppy for momentum)
    vix_high: float = 22.0          # Above = high-vol (good for momentum, but size down)

    # ADX threshold — below this = no trend (bad for momentum strategy)
    adx_no_trend: float = 20.0

    # Daily RSI range — extreme readings = mean-reversion likely
    daily_rsi_overbought: float = 72.0
    daily_rsi_oversold: float = 28.0

    # Yield curve inversion flag — risk-off
    yield_curve_inversion_bps: float = 0.0  # Spread below this = inverted

    # High-impact econ event caution
    skip_high_impact_events: bool = False

    # Adjustments when regime is unfavorable
    size_reduction: float = 0.40        # Cut position size by 40%
    skip_first_bar: bool = True         # Skip 10:00 bar
    rsi_buffer: int = 5                 # Require RSI 5 pts beyond threshold
    tighter_stop_pct: Optional[float] = None  # Override stop loss (None=no change)


class RegimeDetector:
    """
    Pre-market regime detector.

    Fetches and caches macro data from FMP.  For backtesting, call
    `load_regime_data(start, end)` once to pre-fetch everything.
    Then call `get_regime(date)` for each trading day.
    """

    def __init__(self, db_path: str = "data/market_data.db",
                 config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.db_path = db_path
        self._conn = None
        self._fmp = None

        # Cached lookups (date str -> value)
        self._vix_cache: Dict[str, float] = {}
        self._adx_cache: Dict[str, float] = {}
        self._daily_rsi_cache: Dict[str, float] = {}
        self._sma_slope_cache: Dict[str, float] = {}
        self._yield_spread_cache: Dict[str, float] = {}
        self._econ_events_cache: Dict[str, List[dict]] = {}

    # ------------------------------------------------------------------
    # DB connection
    # ------------------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._init_tables()
        return self._conn

    def _init_tables(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS regime_vix_daily (
                date TEXT PRIMARY KEY,
                vix_close REAL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS regime_treasury (
                date TEXT PRIMARY KEY,
                y2 REAL, y5 REAL, y10 REAL, y30 REAL,
                spread_2_10 REAL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS regime_econ_calendar (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT, event TEXT, country TEXT,
                impact TEXT, actual TEXT, estimate TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # FMP client (lazy)
    # ------------------------------------------------------------------
    @property
    def fmp(self):
        if self._fmp is None:
            from dotenv import load_dotenv
            load_dotenv()
            from clients.fmp_stable_client import FMPStableClient
            api_key = os.getenv("FMP_API_KEY", "")
            if not api_key:
                raise ValueError("FMP_API_KEY not set in environment")
            self._fmp = FMPStableClient(api_key)
        return self._fmp

    # ------------------------------------------------------------------
    # 1. VIX daily close
    # ------------------------------------------------------------------
    def _fetch_vix(self, start: str, end: str):
        """Fetch VIX daily close from FMP and cache."""
        existing = set()
        for row in self.conn.execute("SELECT date FROM regime_vix_daily"):
            existing.add(row[0])

        try:
            data = self.fmp.historical_price_eod_light("^VIX", from_date=start, to_date=end)
        except Exception as e:
            print(f"  [RegimeDetector] VIX fetch failed: {e}")
            return

        if not data:
            print("  [RegimeDetector] No VIX data returned")
            return

        inserted = 0
        for item in data:
            d = item.get("date", "")
            close = item.get("close") or item.get("price")
            if d and close is not None and d not in existing:
                self.conn.execute(
                    "INSERT OR IGNORE INTO regime_vix_daily (date, vix_close) VALUES (?, ?)",
                    (d, float(close))
                )
                inserted += 1
        self.conn.commit()
        print(f"  [RegimeDetector] VIX: {inserted} new days cached ({len(data)} total from API)")

    def _load_vix_cache(self, start: str, end: str):
        rows = self.conn.execute(
            "SELECT date, vix_close FROM regime_vix_daily WHERE date >= ? AND date <= ? ORDER BY date",
            (start, end)
        ).fetchall()
        self._vix_cache = {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------
    # 2. SPY daily technicals (ADX, RSI, SMA slope from underlying DB)
    # ------------------------------------------------------------------
    def _compute_daily_technicals(self, start: str, end: str):
        """
        Compute daily ADX, RSI, SMA slope from 5-min data already in DB.
        Uses PRIOR days only — each day's features come from closes up to
        the previous trading day.
        """
        # Get daily closes from intraday_5min_data
        rows = self.conn.execute("""
            SELECT date, MAX(close) as high, MIN(close) as low,
                   -- last close of the day (15:55 bar)
                   (SELECT close FROM intraday_5min_data t2
                    WHERE t2.ticker = 'SPY' AND t2.date = t1.date
                    ORDER BY timestamp DESC LIMIT 1) as close,
                   -- first open of the day
                   (SELECT open FROM intraday_5min_data t2
                    WHERE t2.ticker = 'SPY' AND t2.date = t1.date
                    ORDER BY timestamp ASC LIMIT 1) as open
            FROM intraday_5min_data t1
            WHERE ticker = 'SPY' AND date >= ? AND date <= ?
            GROUP BY date
            ORDER BY date
        """, (start, end)).fetchall()

        if not rows:
            print("  [RegimeDetector] No SPY daily data found in DB")
            return

        dates = [r[0] for r in rows]
        highs = np.array([r[1] for r in rows], dtype=float)
        lows = np.array([r[2] for r in rows], dtype=float)
        closes = np.array([r[3] for r in rows], dtype=float)

        n = len(dates)

        # --- ADX (14-period) ---
        adx_period = 14
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)

            up = highs[i] - highs[i-1]
            down = lows[i-1] - lows[i]
            plus_dm[i] = up if (up > down and up > 0) else 0
            minus_dm[i] = down if (down > up and down > 0) else 0

        # Wilder smoothing
        atr14 = np.zeros(n)
        plus_di14 = np.zeros(n)
        minus_di14 = np.zeros(n)
        dx = np.zeros(n)
        adx = np.zeros(n)

        if n > adx_period:
            atr14[adx_period] = np.mean(tr[1:adx_period+1])
            plus_di14[adx_period] = np.mean(plus_dm[1:adx_period+1])
            minus_di14[adx_period] = np.mean(minus_dm[1:adx_period+1])

            for i in range(adx_period + 1, n):
                atr14[i] = (atr14[i-1] * (adx_period - 1) + tr[i]) / adx_period
                plus_di14[i] = (plus_di14[i-1] * (adx_period - 1) + plus_dm[i]) / adx_period
                minus_di14[i] = (minus_di14[i-1] * (adx_period - 1) + minus_dm[i]) / adx_period

            for i in range(adx_period, n):
                if atr14[i] > 0:
                    pdi = 100 * plus_di14[i] / atr14[i]
                    mdi = 100 * minus_di14[i] / atr14[i]
                    dsum = pdi + mdi
                    dx[i] = 100 * abs(pdi - mdi) / dsum if dsum > 0 else 0
                else:
                    dx[i] = 0

            # ADX = smoothed DX
            start_idx = 2 * adx_period
            if n > start_idx:
                adx[start_idx] = np.mean(dx[adx_period:start_idx])
                for i in range(start_idx + 1, n):
                    adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period

        # --- RSI (14-period) ---
        rsi_period = 14
        rsi = np.full(n, 50.0)
        if n > rsi_period + 1:
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[:rsi_period])
            avg_loss = np.mean(losses[:rsi_period])

            for i in range(rsi_period, len(deltas)):
                avg_gain = (avg_gain * (rsi_period - 1) + gains[i]) / rsi_period
                avg_loss = (avg_loss * (rsi_period - 1) + losses[i]) / rsi_period
                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                rsi[i + 1] = 100 - 100 / (1 + rs)

        # --- SMA 20 slope (normalized) ---
        sma_period = 20
        sma_slope = np.zeros(n)
        if n > sma_period + 1:
            sma = pd.Series(closes).rolling(sma_period).mean().values
            for i in range(sma_period + 1, n):
                if sma[i-1] > 0:
                    sma_slope[i] = (sma[i] - sma[i-1]) / sma[i-1] * 100

        # Store with 1-day lag (each day uses PREVIOUS day's value)
        for i in range(1, n):
            prev_date = dates[i - 1]
            cur_date = dates[i]
            self._adx_cache[cur_date] = adx[i - 1]
            self._daily_rsi_cache[cur_date] = rsi[i - 1]
            self._sma_slope_cache[cur_date] = sma_slope[i - 1]

        # For first trading day, use its own data (no prior available)
        if n > 0:
            self._adx_cache[dates[0]] = adx[0]
            self._daily_rsi_cache[dates[0]] = rsi[0]
            self._sma_slope_cache[dates[0]] = sma_slope[0]

        print(f"  [RegimeDetector] Daily technicals computed: {n} days (ADX/RSI/SMA slope)")

    # ------------------------------------------------------------------
    # 3. Treasury yield curve (2Y-10Y spread)
    # ------------------------------------------------------------------
    def _fetch_treasury(self, start: str, end: str):
        existing = set()
        for row in self.conn.execute("SELECT date FROM regime_treasury"):
            existing.add(row[0])

        try:
            data = self.fmp.treasury_rates(from_date=start, to_date=end)
        except Exception as e:
            print(f"  [RegimeDetector] Treasury fetch failed: {e}")
            return

        if not data:
            print("  [RegimeDetector] No treasury data returned")
            return

        inserted = 0
        for item in data:
            d = item.get("date", "")
            if d and d not in existing:
                y2 = item.get("year2", item.get("month6"))
                y5 = item.get("year5")
                y10 = item.get("year10")
                y30 = item.get("year30")
                spread = (y10 - y2) if (y10 is not None and y2 is not None) else None
                self.conn.execute(
                    "INSERT OR IGNORE INTO regime_treasury (date, y2, y5, y10, y30, spread_2_10) VALUES (?,?,?,?,?,?)",
                    (d, y2, y5, y10, y30, spread)
                )
                inserted += 1
        self.conn.commit()
        print(f"  [RegimeDetector] Treasury: {inserted} new days cached")

    def _load_treasury_cache(self, start: str, end: str):
        rows = self.conn.execute(
            "SELECT date, spread_2_10 FROM regime_treasury WHERE date >= ? AND date <= ? ORDER BY date",
            (start, end)
        ).fetchall()
        self._yield_spread_cache = {}
        last_spread = None
        for r in rows:
            spread = r[1]
            if spread is not None:
                last_spread = spread
            # Forward-fill missing spreads
            if last_spread is not None:
                self._yield_spread_cache[r[0]] = last_spread

    # ------------------------------------------------------------------
    # 4. Economic calendar (high-impact events)
    # ------------------------------------------------------------------
    def _fetch_econ_calendar(self, start: str, end: str):
        existing_dates = set()
        for row in self.conn.execute("SELECT DISTINCT date FROM regime_econ_calendar"):
            existing_dates.add(row[0])

        try:
            data = self.fmp.economic_calendar(from_date=start, to_date=end)
        except Exception as e:
            print(f"  [RegimeDetector] Econ calendar fetch failed: {e}")
            return

        if not data:
            return

        inserted = 0
        for item in data:
            d = item.get("date", "")[:10]  # Date part only
            if d:
                self.conn.execute(
                    "INSERT INTO regime_econ_calendar (date, event, country, impact, actual, estimate) VALUES (?,?,?,?,?,?)",
                    (d, item.get("event", ""), item.get("country", "US"),
                     item.get("impact", ""), str(item.get("actual", "")),
                     str(item.get("estimate", "")))
                )
                inserted += 1
        self.conn.commit()
        print(f"  [RegimeDetector] Econ calendar: {inserted} events cached")

    def _load_econ_cache(self, start: str, end: str):
        rows = self.conn.execute(
            "SELECT date, event, impact FROM regime_econ_calendar WHERE date >= ? AND date <= ? AND country = 'US'",
            (start, end)
        ).fetchall()
        self._econ_events_cache = {}
        for r in rows:
            d = r[0]
            if d not in self._econ_events_cache:
                self._econ_events_cache[d] = []
            self._econ_events_cache[d].append({'event': r[1], 'impact': r[2]})

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def load_regime_data(self, start: str, end: str, fetch_from_api: bool = True):
        """
        Pre-fetch and cache all regime data for the date range.
        Call once before backtesting.

        Args:
            start: Start date YYYY-MM-DD
            end: End date YYYY-MM-DD
            fetch_from_api: If True, fetch from FMP API. If False, use only cached data.
        """
        # Extend start back 30 days for lookback calculations
        start_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=45)
        ext_start = start_dt.strftime("%Y-%m-%d")

        print(f"  [RegimeDetector] Loading regime data {ext_start} to {end}")

        if fetch_from_api:
            self._fetch_vix(ext_start, end)
            self._fetch_treasury(ext_start, end)
            self._fetch_econ_calendar(start, end)

        self._load_vix_cache(ext_start, end)
        self._load_treasury_cache(ext_start, end)
        self._load_econ_cache(start, end)
        self._compute_daily_technicals(ext_start, end)

        print(f"  [RegimeDetector] Loaded: VIX={len(self._vix_cache)} days, "
              f"ADX={len(self._adx_cache)}, RSI={len(self._daily_rsi_cache)}, "
              f"Treasury={len(self._yield_spread_cache)}, "
              f"EconDays={len(self._econ_events_cache)}")

    def get_regime(self, date: str) -> dict:
        """
        Get the regime assessment for a specific date.
        Returns dict with regime signals and overall is_unfavorable flag.
        All data used is from PRIOR day (no look-ahead).
        """
        cfg = self.config
        vix = self._vix_cache.get(date)
        adx = self._adx_cache.get(date)
        rsi = self._daily_rsi_cache.get(date)
        sma_slope = self._sma_slope_cache.get(date)
        spread = self._yield_spread_cache.get(date)
        events = self._econ_events_cache.get(date, [])
        high_impact = [e for e in events if e.get('impact', '').lower() in ('high', 'medium')]

        # Score unfavorable conditions (each adds 1 point)
        reasons = []

        # 1. VIX too low = low-vol choppy environment
        vix_low = False
        vix_high = False
        if vix is not None:
            if vix < cfg.vix_low:
                vix_low = True
                reasons.append(f"VIX={vix:.1f}<{cfg.vix_low} (low-vol choppy)")
            elif vix > cfg.vix_high:
                vix_high = True
                reasons.append(f"VIX={vix:.1f}>{cfg.vix_high} (elevated risk)")

        # 2. ADX below threshold = no trend
        adx_weak = False
        if adx is not None and adx < cfg.adx_no_trend:
            adx_weak = True
            reasons.append(f"ADX={adx:.1f}<{cfg.adx_no_trend} (no trend)")

        # 3. Daily RSI extreme = mean-reversion risk
        rsi_extreme = False
        if rsi is not None:
            if rsi > cfg.daily_rsi_overbought or rsi < cfg.daily_rsi_oversold:
                rsi_extreme = True
                reasons.append(f"DailyRSI={rsi:.1f} (extreme)")

        # 4. Yield curve inverted
        curve_inverted = False
        if spread is not None and spread < cfg.yield_curve_inversion_bps:
            curve_inverted = True
            reasons.append(f"2Y-10Y={spread:.2f}% (inverted)")

        # 5. High-impact event
        has_event = len(high_impact) > 0

        # ---- Decision: unfavorable if multiple signals converge ----
        # Low VIX + weak ADX = classic choppy market (the August scenario)
        # Any 2+ of the 4 core signals = unfavorable
        score = sum([vix_low, adx_weak, rsi_extreme, curve_inverted])

        is_unfavorable = score >= 2

        return {
            'date': date,
            'vix': vix,
            'adx': adx,
            'daily_rsi': rsi,
            'sma_slope': sma_slope,
            'yield_spread': spread,
            'high_impact_events': len(high_impact),
            'vix_low': vix_low,
            'vix_high': vix_high,
            'adx_weak': adx_weak,
            'rsi_extreme': rsi_extreme,
            'curve_inverted': curve_inverted,
            'has_event': has_event,
            'score': score,
            'is_unfavorable': is_unfavorable,
            'reasons': reasons,
        }

    def get_all_regimes(self, dates: List[str]) -> Dict[str, dict]:
        """Get regime for all dates (batch)."""
        return {d: self.get_regime(d) for d in dates}

    def summary(self, dates: List[str]) -> str:
        """Print summary of regime classifications."""
        regimes = self.get_all_regimes(dates)
        unfav = sum(1 for r in regimes.values() if r['is_unfavorable'])
        lines = [
            f"Regime Summary: {unfav}/{len(dates)} days unfavorable",
            f"  VIX low:    {sum(1 for r in regimes.values() if r['vix_low'])} days",
            f"  ADX weak:   {sum(1 for r in regimes.values() if r['adx_weak'])} days",
            f"  RSI extreme:{sum(1 for r in regimes.values() if r['rsi_extreme'])} days",
            f"  Curve inv:  {sum(1 for r in regimes.values() if r['curve_inverted'])} days",
        ]
        return "\n".join(lines)
