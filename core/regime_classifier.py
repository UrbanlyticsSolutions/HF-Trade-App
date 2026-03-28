"""
Shared regime classifier for 0DTE trading.

Used by both backtest (engine.py) and live (strategy_0dte.py) to ensure
identical regime detection logic. Single source of truth.

Detects 6 regimes (priority order):
  VOLATILE   — High intraday vol (top quartile of expanding history)
  STEADY_UP  — Persistent uptrend + directional return + not high vol
  STEADY_DN  — Persistent downtrend + negative return + not high vol
  TRENDING   — Strong trend strength + significant return
  CHOPPY     — Low vol + no directional trend
  NORMAL     — None of the above
"""
import numpy as np
from typing import Dict, List, Optional, Tuple


# Default thresholds (overridden by strategy.json)
DEFAULT_CONFIG = {
    'lookback': 5,
    'vol_percentile': 0.30,
    'trend_percentile': 0.25,
    'up_day_pct': 0.70,
    'dn_day_pct': 0.70,
    'momentum_threshold': 0.012,
    'high_vol_percentile': 0.75,
    'adx_trend_threshold': 25.0,
}


def _compute_day_stats(prices: List[float]) -> Optional[dict]:
    """
    Compute per-day statistics from intraday close prices.

    Args:
        prices: List of intraday close prices for one trading day.

    Returns:
        Dict with intra_vol, trend_strength, daily_ret, is_up, adx_proxy, close.
        None if insufficient data (<5 bars).
    """
    if len(prices) < 5:
        return None

    open_px = prices[0]
    close_px = prices[-1]
    hi = max(prices)
    lo = min(prices)
    rng = hi - lo

    # Intraday volatility: std of bar-to-bar returns
    arr = np.array(prices, dtype=np.float64)
    rets = np.diff(arr) / arr[:-1]
    intra_vol = float(np.std(rets) * 100) if len(rets) > 1 else 0.0

    # Trend strength: |close - open| / range  (1.0 = pure trend, 0 = doji)
    trend_str = abs(close_px - open_px) / rng if rng > 0 else 0.0

    # Daily return
    daily_ret = (close_px - open_px) / open_px if open_px > 0 else 0.0

    # Is up day?
    is_up = 1.0 if close_px > open_px else 0.0

    # ADX-like proxy: trend strength weighted by vol
    adx_proxy = trend_str * intra_vol * 100 if intra_vol > 0 else 0.0

    return {
        'intra_vol': intra_vol,
        'trend_strength': trend_str,
        'daily_ret': daily_ret,
        'is_up': is_up,
        'adx_proxy': adx_proxy,
        'close': close_px,
    }


def _classify_single(
    vol_pctl: float,
    trend_pctl: float,
    up_pct: float,
    win_ret: float,
    adx_r: float,
    config: dict,
) -> Tuple[str, int]:
    """
    Classify a single day into one of 6 regimes.
    Priority order: VOLATILE > STEADY_UP > STEADY_DN > TRENDING > CHOPPY > NORMAL.

    Returns:
        (regime_type, direction) where direction is +1, -1, or 0.
    """
    # VOLATILE: high intraday vol (top quartile)
    if vol_pctl >= config['high_vol_percentile']:
        direction = 1 if win_ret > 0 else (-1 if win_ret < 0 else 0)
        return 'VOLATILE', direction

    # STEADY_UP: mostly up days + directional return + NOT high vol
    if (up_pct >= config['up_day_pct'] and
            win_ret > config['momentum_threshold'] and
            vol_pctl < config['high_vol_percentile']):
        return 'STEADY_UP', 1

    # STEADY_DN: mostly down days + negative return + NOT high vol
    if ((1 - up_pct) >= config['dn_day_pct'] and
            win_ret < -config['momentum_threshold'] and
            vol_pctl < config['high_vol_percentile']):
        return 'STEADY_DN', -1

    # TRENDING: strong trend strength + significant return (either direction)
    if (adx_r >= config['adx_trend_threshold'] and
            abs(win_ret) > config['momentum_threshold']):
        direction = 1 if win_ret > 0 else -1
        return 'TRENDING', direction

    # CHOPPY: low vol + no directional trend
    if (vol_pctl <= config['vol_percentile'] and
            trend_pctl <= config['trend_percentile']):
        return 'CHOPPY', 0

    return 'NORMAL', 0


def _make_result(regime_type: str, direction: int,
                 vol_r: float, trend_r: float, vol_pctl: float,
                 trend_pctl: float, up_pct: float, win_ret: float,
                 adx_r: float) -> dict:
    """Build the standard regime result dict."""
    return {
        'regime_type': regime_type,
        'is_choppy': regime_type in ('CHOPPY', 'STEADY_UP', 'STEADY_DN'),
        'direction': direction,
        'intra_vol': vol_r,
        'trend_strength': trend_r,
        'vol_pctl': vol_pctl,
        'trend_pctl': trend_pctl,
        'up_day_pct': up_pct,
        'window_return': win_ret,
        'adx_proxy': adx_r,
    }


def classify_regimes(
    daily_bars: List[dict],
    config: Optional[dict] = None,
) -> Dict[str, dict]:
    """
    Batch regime classifier — used by backtest engine.

    Classifies every day in daily_bars using only past data (no look-ahead).

    Args:
        daily_bars: List of dicts, each with:
            'date': str (date key, e.g. '2025-01-15')
            'prices': List[float] (intraday close prices for that day)
            Sorted chronologically.
        config: Regime classification thresholds (see DEFAULT_CONFIG).

    Returns:
        Dict[date_str, regime_result_dict] with keys:
          regime_type, is_choppy, direction, intra_vol, trend_strength,
          vol_pctl, trend_pctl, up_day_pct, window_return, adx_proxy
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    lookback = cfg['lookback']

    # Step 1: Compute per-day stats
    stats = []
    for bar in daily_bars:
        day_stats = _compute_day_stats(bar['prices'])
        if day_stats is None:
            continue
        day_stats['date'] = bar['date']
        stats.append(day_stats)

    if not stats:
        return {}

    # Step 2: Rolling features (past data only, using simple loops — no pandas)
    n = len(stats)
    vol_roll = [None] * n
    trend_roll = [None] * n
    up_day_roll = [None] * n
    win_return = [None] * n
    adx_roll = [None] * n

    for i in range(n):
        window_start = max(0, i - lookback + 1)
        window = stats[window_start:i + 1]
        if len(window) < 2:
            continue

        vol_roll[i] = float(np.mean([s['intra_vol'] for s in window]))
        trend_roll[i] = float(np.mean([s['trend_strength'] for s in window]))
        up_day_roll[i] = float(np.mean([s['is_up'] for s in window]))
        adx_roll[i] = float(np.mean([s['adx_proxy'] for s in window]))

        # Window return: pct_change over lookback periods
        if i >= lookback and stats[i - lookback]['close'] > 0:
            win_return[i] = (stats[i]['close'] - stats[i - lookback]['close']) / stats[i - lookback]['close']
        else:
            win_return[i] = 0.0

    # Step 3: Classify each day
    regime = {}
    all_vols: List[float] = []
    all_trends: List[float] = []

    for i in range(n):
        d = stats[i]['date']
        vr = vol_roll[i]
        tr = trend_roll[i]
        up_pct = up_day_roll[i]
        wr = win_return[i]
        ar = adx_roll[i]

        # Handle early days without enough history
        if vr is None or tr is None or up_pct is None:
            regime[d] = _make_result('NORMAL', 0, 0, 0, 0.5, 0.5, 0.5, 0, 0)
            if vr is not None:
                all_vols.append(vr)
            if tr is not None:
                all_trends.append(tr)
            continue

        all_vols.append(vr)
        all_trends.append(tr)
        if wr is None:
            wr = 0.0
        if ar is None:
            ar = 0.0

        # Expanding percentile (no look-ahead)
        if len(all_vols) >= lookback:
            vol_pctl = sum(1 for v in all_vols if v <= vr) / len(all_vols)
            trend_pctl = sum(1 for t in all_trends if t <= tr) / len(all_trends)
        else:
            vol_pctl = 0.5
            trend_pctl = 0.5

        regime_type, direction = _classify_single(
            vol_pctl, trend_pctl, up_pct, wr, ar, cfg
        )
        regime[d] = _make_result(
            regime_type, direction, vr, tr, vol_pctl, trend_pctl, up_pct, wr, ar
        )

    return regime


def classify_regime_incremental(
    recent_day_stats: List[dict],
    all_vols: List[float],
    all_trends: List[float],
    config: Optional[dict] = None,
) -> dict:
    """
    Incremental regime classifier — used by live strategy.

    Classifies the current day based on recently accumulated day stats
    and the expanding history of vol/trend values.

    This avoids recomputing all history each day: the live system maintains
    all_vols and all_trends lists across days and passes them in.

    Args:
        recent_day_stats: List of day stats dicts for the lookback window.
            Each dict has: intra_vol, trend_strength, is_up, adx_proxy, close.
            Should be the last `lookback` completed days.
        all_vols: Expanding list of all historical vol_roll values (mutated in place).
        all_trends: Expanding list of all historical trend_roll values (mutated in place).
        config: Regime classification thresholds.

    Returns:
        Regime result dict (same structure as classify_regimes output).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    lookback = cfg['lookback']

    if len(recent_day_stats) < 2:
        return _make_result('NORMAL', 0, 0, 0, 0.5, 0.5, 0.5, 0, 0)

    # Rolling averages over the lookback window
    window = recent_day_stats[-lookback:]
    vol_avg = float(np.mean([s['intra_vol'] for s in window]))
    trend_avg = float(np.mean([s['trend_strength'] for s in window]))
    up_pct = float(np.mean([s['is_up'] for s in window]))
    adx_avg = float(np.mean([s['adx_proxy'] for s in window]))

    # Window return (most recent close vs lookback-ago close)
    if len(recent_day_stats) >= lookback + 1:
        old_close = recent_day_stats[-(lookback + 1)]['close']
        new_close = recent_day_stats[-1]['close']
        win_ret = (new_close - old_close) / old_close if old_close > 0 else 0.0
    else:
        win_ret = 0.0

    # Append to expanding history
    all_vols.append(vol_avg)
    all_trends.append(trend_avg)

    # Expanding percentile
    if len(all_vols) >= lookback:
        vol_pctl = sum(1 for v in all_vols if v <= vol_avg) / len(all_vols)
        trend_pctl = sum(1 for t in all_trends if t <= trend_avg) / len(all_trends)
    else:
        vol_pctl = 0.5
        trend_pctl = 0.5

    regime_type, direction = _classify_single(
        vol_pctl, trend_pctl, up_pct, win_ret, adx_avg, cfg
    )
    return _make_result(
        regime_type, direction, vol_avg, trend_avg,
        vol_pctl, trend_pctl, up_pct, win_ret, adx_avg
    )
