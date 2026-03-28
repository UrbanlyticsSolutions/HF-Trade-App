#!/usr/bin/env python3
"""
Trend Slope Signal — Standalone runner using Questrade data.

Fetches historical candles, computes a suite of trend/slope indicators,
and displays a consolidated directional signal with component breakdown.

Usage:
    python scripts/trend_signal.py                          # terminal mode
    python scripts/trend_signal.py --dash                   # Dash web UI
    python scripts/trend_signal.py --dash --watch 30        # auto-refresh
    python scripts/trend_signal.py --symbol QQQ --interval FiveMinutes
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Ensure project root is on path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from clients.questrade_client import QuestradeClient
from indicators.trend import SMA, EMA, ADX
from indicators.momentum import RSI, MACD, Stochastic
from indicators.volatility import ATR, BollingerBands

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Linear-regression slope (not in existing indicators)
# ---------------------------------------------------------------------------

def linreg_slope(values: np.ndarray, window: int = 20) -> float:
    """Normalised linear-regression slope over the last *window* bars.

    Returns slope as a per-bar % change relative to the mean price in the
    window, so it's comparable across different price levels.
    """
    segment = values[-window:]
    if len(segment) < window:
        return 0.0
    x = np.arange(window, dtype=float)
    y = segment.astype(float)
    mean_y = np.mean(y)
    if mean_y == 0:
        return 0.0
    # Least-squares slope
    slope = (np.sum((x - x.mean()) * (y - y.mean())) /
             np.sum((x - x.mean()) ** 2))
    # Normalise to % per bar
    return (slope / mean_y) * 100.0


# ---------------------------------------------------------------------------
# Fetch candles from Questrade → numpy arrays
# ---------------------------------------------------------------------------

def fetch_candles(qt: QuestradeClient, symbol: str,
                  interval: str, bars: int) -> dict:
    """Return dict with numpy arrays: open, high, low, close, volume, times."""

    sym_id = qt.get_symbol_id(symbol)
    if sym_id is None:
        raise ValueError(f"Symbol '{symbol}' not found on Questrade")

    # Map interval → calendar days we need to look back.
    INTERVAL_MINUTES = {
        "OneMinute": 1, "TwoMinutes": 2, "ThreeMinutes": 3,
        "FiveMinutes": 5, "TenMinutes": 10, "FifteenMinutes": 15,
        "TwentyMinutes": 20, "HalfHour": 30, "OneHour": 60,
        "TwoHours": 120, "FourHours": 240,
        "OneDay": 1440, "OneWeek": 10080, "OneMonth": 43200,
    }
    mins_per_bar = INTERVAL_MINUTES.get(interval, 1440)

    if mins_per_bar < 1440:
        trading_mins_needed = bars * mins_per_bar
        trading_days_needed = max(2, int(trading_mins_needed / 390) + 1)
        calendar_days = int(trading_days_needed * 1.6) + 2
    elif mins_per_bar == 1440:
        # Daily: ~1.5 trading days per calendar day (weekends)
        calendar_days = int(bars * 1.5) + 5
    elif mins_per_bar == 10080:
        calendar_days = bars * 7 + 7
    else:
        calendar_days = bars * 35

    end = datetime.now().replace(microsecond=0)
    start = (end - timedelta(days=calendar_days)).replace(
        hour=0, minute=0, second=0)

    # Questrade candle API has a max range per request (~2000 candles).
    # For large daily ranges, chunk into 30-day blocks.
    MAX_CHUNK_DAYS = 30 if mins_per_bar < 1440 else 365
    all_candles = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=MAX_CHUNK_DAYS), end)
        try:
            chunk = qt.get_candles(sym_id, chunk_start, chunk_end, interval)
            if chunk:
                all_candles.extend(chunk)
        except Exception as e:
            logger.warning(f"Chunk {chunk_start.date()}→{chunk_end.date()} failed: {e}")
        chunk_start = chunk_end

    if not all_candles:
        raise RuntimeError(f"No candle data returned for {symbol} ({interval})")

    # Take last N bars
    candles = all_candles[-bars:]

    opens = np.array([c["open"] for c in candles], dtype=float)
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows = np.array([c["low"] for c in candles], dtype=float)
    closes = np.array([c["close"] for c in candles], dtype=float)
    volumes = np.array([c.get("volume", 0) for c in candles], dtype=float)
    times = [c.get("start", c.get("end", "")) for c in candles]

    return {
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes, "times": times,
    }


# ---------------------------------------------------------------------------
# Compute all slope / trend components
# ---------------------------------------------------------------------------

def compute_trend_components(data: dict) -> list:
    """
    Compute each trend-slope component.

    Returns list of dicts:
        name, value (normalised -100..+100), raw, weight, description
    """
    closes = data["close"]
    highs = data["high"]
    lows = data["low"]
    n = len(closes)
    components = []

    # --- 1. SMA slopes (short / medium / long) ---
    for period, weight, label in [(10, 0.10, "SMA-10 slope"),
                                   (20, 0.10, "SMA-20 slope"),
                                   (50, 0.08, "SMA-50 slope")]:
        if n >= period + 5:
            sma = SMA(period).calculate(closes)
            raw = sma.slope  # price per bar
            norm = (raw / closes[-1]) * 1000  # scale to readable
            components.append(dict(
                name=label, raw=round(raw, 4),
                norm=np.clip(norm, -100, 100),
                weight=weight,
                desc=f"{'↑' if raw > 0 else '↓'} ${raw:+.4f}/bar",
            ))

    # --- 2. EMA slopes ---
    for period, weight, label in [(8, 0.08, "EMA-8 slope"),
                                   (21, 0.08, "EMA-21 slope")]:
        if n >= period + 5:
            ema = EMA(period).calculate(closes)
            raw = ema.slope
            norm = (raw / closes[-1]) * 1000
            components.append(dict(
                name=label, raw=round(raw, 4),
                norm=np.clip(norm, -100, 100),
                weight=weight,
                desc=f"{'↑' if raw > 0 else '↓'} ${raw:+.4f}/bar",
            ))

    # --- 3. EMA-8 vs EMA-21 spread (trend strength) ---
    if n >= 21:
        ema8 = EMA(8).calculate(closes).current
        ema21 = EMA(21).calculate(closes).current
        spread_pct = (ema8 - ema21) / closes[-1] * 100
        norm = np.clip(spread_pct * 20, -100, 100)  # ±5% → ±100
        components.append(dict(
            name="EMA 8/21 spread", raw=round(spread_pct, 3),
            norm=norm, weight=0.10,
            desc=f"{'↑' if spread_pct > 0 else '↓'} {spread_pct:+.3f}%",
        ))

    # --- 4. ADX + direction ---
    if n >= 28:
        try:
            adx = ADX(14).calculate(highs, lows, closes)
            direction_score = (adx.plus_di[-1] - adx.minus_di[-1])
            strength = adx.current_adx
            raw = direction_score * (strength / 50)  # amplify with ADX
            norm = np.clip(raw, -100, 100)
            components.append(dict(
                name="ADX direction", raw=round(raw, 2),
                norm=norm, weight=0.10,
                desc=f"ADX={strength:.1f} +DI={adx.plus_di[-1]:.1f} -DI={adx.minus_di[-1]:.1f} ({adx.trend_strength})",
            ))
        except Exception:
            pass

    # --- 5. RSI position (centered at 50) ---
    if n >= 20:
        rsi = RSI(14).calculate(closes)
        raw = rsi.current - 50  # -50..+50
        norm = np.clip(raw * 2, -100, 100)
        components.append(dict(
            name="RSI(14)", raw=round(rsi.current, 1),
            norm=norm, weight=0.08,
            desc=f"RSI={rsi.current:.1f}" +
                 (" OB" if rsi.is_overbought else "") +
                 (" OS" if rsi.is_oversold else "") +
                 (f" div:{rsi.divergence}" if rsi.divergence else ""),
        ))

    # --- 6. MACD histogram ---
    if n >= 35:
        macd = MACD().calculate(closes)
        raw = macd.current_hist
        norm = np.clip((raw / closes[-1]) * 5000, -100, 100)
        components.append(dict(
            name="MACD hist", raw=round(raw, 4),
            norm=norm, weight=0.08,
            desc=f"hist={raw:+.4f} xover={macd.crossover or 'none'}",
        ))

    # --- 7. Bollinger %B ---
    if n >= 22:
        try:
            bb = BollingerBands(20, 2.0).calculate(closes)
            raw = bb.percent_b  # 0–1, >1 above upper, <0 below lower
            norm = np.clip((raw - 0.5) * 200, -100, 100)
            components.append(dict(
                name="BB %B", raw=round(raw, 3),
                norm=norm, weight=0.05,
                desc=f"%B={raw:.3f} BW={bb.bandwidth:.3f}" +
                     (" SQUEEZE" if bb.squeeze else ""),
            ))
        except Exception:
            pass

    # --- 8. Linear-regression slope (20-bar) ---
    if n >= 20:
        lr_slope = linreg_slope(closes, 20)
        norm = np.clip(lr_slope * 50, -100, 100)
        components.append(dict(
            name="LinReg slope(20)", raw=round(lr_slope, 4),
            norm=norm, weight=0.10,
            desc=f"{lr_slope:+.4f}%/bar",
        ))

    # --- 9. Price vs SMA-50 (positional bias) ---
    if n >= 50:
        sma50 = SMA(50).calculate(closes).current
        dist = (closes[-1] - sma50) / sma50 * 100
        norm = np.clip(dist * 10, -100, 100)
        components.append(dict(
            name="Price vs SMA-50", raw=round(dist, 2),
            norm=norm, weight=0.05,
            desc=f"{'above' if dist > 0 else 'below'} by {abs(dist):.2f}%",
        ))

    return components


# ---------------------------------------------------------------------------
# Composite signal
# ---------------------------------------------------------------------------

def composite_signal(components: list) -> dict:
    """Weighted combination → single score and label."""
    total_weight = sum(c["weight"] for c in components)
    if total_weight == 0:
        return {"score": 0.0, "label": "NO DATA", "color": "white"}

    score = sum(c["norm"] * c["weight"] for c in components) / total_weight

    # Map score → label
    if score >= 40:
        label, color = "STRONG UP", "\033[92m"       # green
    elif score >= 15:
        label, color = "UP", "\033[32m"               # light green
    elif score >= 5:
        label, color = "LEAN UP", "\033[33m"          # yellow
    elif score > -5:
        label, color = "NEUTRAL", "\033[37m"          # white
    elif score > -15:
        label, color = "LEAN DOWN", "\033[33m"        # yellow
    elif score > -40:
        label, color = "DOWN", "\033[91m"             # light red
    else:
        label, color = "STRONG DOWN", "\033[31m"      # red

    return {"score": round(score, 2), "label": label, "color": color}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"


def bar_chart(value: float, width: int = 30) -> str:
    """Render a horizontal bar from -100 to +100."""
    mid = width // 2
    fill = int(abs(value) / 100 * mid)
    fill = min(fill, mid)

    if value >= 0:
        left = " " * mid
        right = "█" * fill + "░" * (mid - fill)
        color = "\033[92m" if value >= 30 else "\033[32m" if value >= 10 else "\033[37m"
    else:
        pad = mid - fill
        left = " " * pad + "█" * fill
        right = "░" * mid
        color = "\033[91m" if value <= -30 else "\033[31m" if value <= -10 else "\033[37m"

    return f"{color}{left}│{right}{RESET}"


def display(symbol: str, interval: str, data: dict,
            components: list, result: dict):
    """Print the formatted signal report."""
    price = data["close"][-1]
    prev = data["close"][-2] if len(data["close"]) >= 2 else price
    change = price - prev
    change_pct = (change / prev * 100) if prev else 0
    high = float(np.max(data["high"][-20:]))
    low = float(np.min(data["low"][-20:]))
    last_time = data["times"][-1] if data["times"] else "?"

    os.system("cls" if os.name == "nt" else "clear")
    print()
    print(f"  {BOLD}{CYAN}TREND SLOPE SIGNAL{RESET}  —  {symbol}  {interval}")
    print(f"  {DIM}{last_time}{RESET}")
    print(f"  {'─' * 62}")

    chg_color = "\033[92m" if change >= 0 else "\033[91m"
    print(f"  Price: {BOLD}${price:.2f}{RESET}  "
          f"{chg_color}{change:+.2f} ({change_pct:+.2f}%){RESET}  "
          f"{DIM}H ${high:.2f}  L ${low:.2f}{RESET}")
    print()

    # Component table
    name_w = max(len(c["name"]) for c in components) + 1
    print(f"  {'COMPONENT':<{name_w}} {'SCORE':>6}  {'BAR':<32} DETAIL")
    print(f"  {'─' * (name_w + 48)}")

    for c in components:
        n = c["norm"]
        sign_color = "\033[92m" if n > 10 else "\033[91m" if n < -10 else "\033[37m"
        print(f"  {c['name']:<{name_w}} {sign_color}{n:>+6.1f}{RESET}  "
              f"{bar_chart(n)}  {DIM}{c['desc']}{RESET}")

    print(f"  {'─' * (name_w + 48)}")

    # Composite
    r = result
    print(f"\n  {BOLD}COMPOSITE:  {r['color']}{r['label']}  "
          f"({r['score']:+.1f}){RESET}")
    print()

    # Quick interpretation
    score = r["score"]
    if abs(score) >= 40:
        strength = "High conviction"
    elif abs(score) >= 15:
        strength = "Moderate conviction"
    elif abs(score) >= 5:
        strength = "Weak lean"
    else:
        strength = "No clear edge"

    if score > 5:
        bias = f"Bullish bias — favor CALL entries"
    elif score < -5:
        bias = f"Bearish bias — favor PUT entries"
    else:
        bias = "Flat — no directional bias, consider sitting out"

    print(f"  {DIM}{strength}. {bias}{RESET}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(symbol: str, interval: str, bars: int,
        watch: int = 0, qt: QuestradeClient = None):
    """Single run or watch loop (terminal mode)."""
    if qt is None:
        qt = QuestradeClient()

    while True:
        try:
            data = fetch_candles(qt, symbol, interval, bars)
            components = compute_trend_components(data)
            result = composite_signal(components)
            display(symbol, interval, data, components, result)
        except KeyboardInterrupt:
            print("\n  Stopped.")
            break
        except Exception as e:
            logger.exception("Error computing signal")
            print(f"\n  ERROR: {e}\n")

        if watch <= 0:
            break
        try:
            print(f"  {DIM}Refreshing in {watch}s ...  (Ctrl+C to stop){RESET}")
            time.sleep(watch)
        except KeyboardInterrupt:
            print("\n  Stopped.")
            break


# ---------------------------------------------------------------------------
# Dash web UI
# ---------------------------------------------------------------------------

def build_dash_app(symbol: str, interval: str, bars: int, watch: int):
    """Build and return a Plotly Dash app for the trend signal."""
    from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update
    import plotly.graph_objects as go

    # Questrade client — initialised lazily so dashboard starts even
    # when the token is expired (user can paste a new one in the UI).
    qt_holder = {"client": None, "error": None}

    def _get_qt():
        if qt_holder["client"] is None:
            try:
                qt_holder["client"] = QuestradeClient()
                qt_holder["error"] = None
            except Exception as e:
                qt_holder["error"] = str(e)
        return qt_holder["client"]

    _get_qt()  # attempt on startup

    app = Dash(__name__)
    app.title = f"Trend Signal — {symbol}"

    refresh_ms = max(watch, 30) * 1000  # minimum 30s

    _input_style = {
        "backgroundColor": "#161b22", "border": "1px solid #30363d",
        "color": "#e0e0e0", "borderRadius": "6px",
        "padding": "6px 10px", "fontSize": "14px",
    }

    app.layout = html.Div(style={
        "backgroundColor": "#0e1117", "minHeight": "100vh",
        "fontFamily": "'Segoe UI', system-ui, sans-serif",
        "color": "#e0e0e0", "padding": "20px 30px",
    }, children=[
        # Header
        html.Div(style={"display": "flex", "alignItems": "center",
                         "justifyContent": "space-between",
                         "marginBottom": "20px"}, children=[
            html.H1("TREND SLOPE SIGNAL", style={
                "margin": 0, "fontSize": "22px", "letterSpacing": "2px",
                "color": "#58a6ff",
            }),
            html.Div(id="header-info", style={"fontSize": "13px", "color": "#8b949e"}),
        ]),

        # Questrade API token row
        html.Div(id="qt-token-section", style={
            "marginBottom": "16px", "padding": "12px 16px",
            "backgroundColor": "#161b22", "borderRadius": "8px",
            "border": "1px solid #30363d",
        }, children=[
            html.Div(style={"display": "flex", "gap": "10px",
                             "alignItems": "center", "flexWrap": "wrap"}, children=[
                html.Span("Questrade API Token", style={
                    "fontSize": "12px", "color": "#8b949e",
                    "fontWeight": 600, "letterSpacing": "1px",
                    "textTransform": "uppercase", "minWidth": "160px",
                }),
                dcc.Input(
                    id="qt-token-input", type="text",
                    placeholder="Paste refresh token from Questrade App Hub",
                    style={**_input_style, "flex": "1", "minWidth": "280px"},
                ),
                html.Button("Connect", id="qt-token-btn",
                            style={"backgroundColor": "#1f6feb", "color": "#fff",
                                   "border": "none", "borderRadius": "6px",
                                   "padding": "7px 18px", "cursor": "pointer",
                                   "fontSize": "13px", "fontWeight": 600}),
            ]),
            html.Div(id="qt-token-status", style={"marginTop": "6px", "fontSize": "12px"}),
        ]),

        # Controls row
        html.Div(style={"display": "flex", "gap": "12px", "marginBottom": "20px",
                         "alignItems": "center"}, children=[
            dcc.Input(id="sym-input", value=symbol, type="text",
                      placeholder="Symbol", debounce=True,
                      style={**_input_style, "width": "80px",
                             "textTransform": "uppercase"}),
            dcc.Dropdown(id="interval-dd", value=interval, clearable=False,
                         options=[
                             {"label": "1 min", "value": "OneMinute"},
                             {"label": "5 min", "value": "FiveMinutes"},
                             {"label": "15 min", "value": "FifteenMinutes"},
                             {"label": "1 hour", "value": "OneHour"},
                             {"label": "Daily", "value": "OneDay"},
                             {"label": "Weekly", "value": "OneWeek"},
                         ],
                         style={"width": "130px", "backgroundColor": "#161b22",
                                "color": "#0e1117", "borderRadius": "6px"}),
            dcc.Input(id="bars-input", value=bars, type="number",
                      placeholder="Bars", debounce=True,
                      style={**_input_style, "width": "70px"}),
            html.Button("Refresh", id="refresh-btn",
                        style={"backgroundColor": "#238636", "color": "#fff",
                               "border": "none", "borderRadius": "6px",
                               "padding": "7px 18px", "cursor": "pointer",
                               "fontSize": "13px", "fontWeight": 600}),
        ]),

        # Composite signal banner
        html.Div(id="signal-banner", style={
            "borderRadius": "10px", "padding": "18px 24px",
            "marginBottom": "20px", "textAlign": "center",
        }),

        # Price info row
        html.Div(id="price-row", style={
            "display": "flex", "gap": "20px", "marginBottom": "20px",
            "flexWrap": "wrap",
        }),

        # Main content: chart left, components right
        html.Div(style={"display": "flex", "gap": "20px",
                         "flexWrap": "wrap"}, children=[
            # Price chart
            html.Div(style={"flex": "1.2", "minWidth": "400px"}, children=[
                dcc.Graph(id="price-chart",
                          config={"displayModeBar": False},
                          style={"height": "380px"}),
            ]),
            # Component bars
            html.Div(id="component-bars", style={
                "flex": "1", "minWidth": "350px",
            }),
        ]),

        # Auto-refresh
        dcc.Interval(id="auto-refresh", interval=refresh_ms, n_intervals=0),
    ])

    # ---- Callbacks ----

    # Token submit callback
    @app.callback(
        Output("qt-token-status", "children"),
        Input("qt-token-btn", "n_clicks"),
        State("qt-token-input", "value"),
        prevent_initial_call=True,
    )
    def submit_token(_clicks, token_val):
        if not token_val or not token_val.strip():
            return html.Span("Paste a token first.", style={"color": "#d29922"})
        token_val = token_val.strip()
        try:
            # Delete old token file so fresh auth happens
            token_file = Path(__file__).resolve().parent.parent / "clients" / ".questrade_token.json"
            if token_file.exists():
                token_file.unlink()
            qt_holder["client"] = QuestradeClient(refresh_token=token_val)
            qt_holder["error"] = None
            return html.Span("Connected.",
                             style={"color": "#3fb950", "fontWeight": 600})
        except Exception as e:
            qt_holder["client"] = None
            qt_holder["error"] = str(e)
            return html.Span(f"Failed: {e}", style={"color": "#f85149"})

    @app.callback(
        Output("signal-banner", "children"),
        Output("signal-banner", "style"),
        Output("price-row", "children"),
        Output("price-chart", "figure"),
        Output("component-bars", "children"),
        Output("header-info", "children"),
        Input("refresh-btn", "n_clicks"),
        Input("auto-refresh", "n_intervals"),
        State("sym-input", "value"),
        State("interval-dd", "value"),
        State("bars-input", "value"),
    )
    def update_all(_clicks, _n, sym, intv, nbars):
        sym = (sym or symbol).upper().strip()
        intv = intv or interval
        nbars = int(nbars or bars)

        qt = _get_qt()
        if qt is None:
            err_msg = qt_holder.get("error") or "No Questrade connection — paste API token above."
            err = html.Div(err_msg, style={"color": "#f85149"})
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_dark",
                                    paper_bgcolor="#0e1117",
                                    plot_bgcolor="#0e1117")
            return err, {"padding": "18px"}, [], empty_fig, [], "Not connected"

        try:
            data = fetch_candles(qt, sym, intv, nbars)
            components = compute_trend_components(data)
            result = composite_signal(components)
        except Exception as e:
            err = html.Div(f"Error: {e}", style={"color": "#f85149"})
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_dark",
                                    paper_bgcolor="#0e1117",
                                    plot_bgcolor="#0e1117")
            return err, {"padding": "18px"}, [], empty_fig, [], str(e)

        score = result["score"]
        label = result["label"]
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        price = data["close"][-1]
        prev = data["close"][-2] if len(data["close"]) >= 2 else price
        chg = price - prev
        chg_pct = (chg / prev * 100) if prev else 0

        # --- Signal banner ---
        if score >= 15:
            bg = "linear-gradient(135deg, #0d4429, #143d22)"
            txt_color = "#3fb950"
        elif score > 5:
            bg = "linear-gradient(135deg, #2a2000, #332800)"
            txt_color = "#d29922"
        elif score > -5:
            bg = "linear-gradient(135deg, #1c1c1c, #222)"
            txt_color = "#8b949e"
        elif score > -15:
            bg = "linear-gradient(135deg, #2a2000, #332800)"
            txt_color = "#d29922"
        else:
            bg = "linear-gradient(135deg, #3d0a0a, #2d0808)"
            txt_color = "#f85149"

        banner_children = [
            html.Div(label, style={"fontSize": "28px", "fontWeight": 700,
                                    "color": txt_color, "letterSpacing": "3px"}),
            html.Div(f"Score: {score:+.1f}", style={
                "fontSize": "16px", "color": txt_color, "opacity": 0.8,
                "marginTop": "4px"}),
        ]
        banner_style = {
            "borderRadius": "10px", "padding": "18px 24px",
            "marginBottom": "20px", "textAlign": "center",
            "background": bg, "border": f"1px solid {txt_color}33",
        }

        # --- Price info cards ---
        chg_color = "#3fb950" if chg >= 0 else "#f85149"
        h20 = float(np.max(data["high"][-20:]))
        l20 = float(np.min(data["low"][-20:]))
        vol = int(data["volume"][-1]) if data["volume"][-1] > 0 else None

        def _card(title, value, color="#e0e0e0"):
            return html.Div(style={
                "backgroundColor": "#161b22", "borderRadius": "8px",
                "padding": "10px 16px", "minWidth": "100px",
                "border": "1px solid #30363d",
            }, children=[
                html.Div(title, style={"fontSize": "11px", "color": "#8b949e",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "1px"}),
                html.Div(value, style={"fontSize": "18px", "fontWeight": 600,
                                        "color": color, "marginTop": "2px"}),
            ])

        price_cards = [
            _card("Price", f"${price:.2f}"),
            _card("Change", f"{chg:+.2f} ({chg_pct:+.2f}%)", chg_color),
            _card("20-bar High", f"${h20:.2f}"),
            _card("20-bar Low", f"${l20:.2f}"),
        ]
        if vol:
            price_cards.append(_card("Volume", f"{vol:,}"))

            # Conviction
        if abs(score) >= 40:
            conviction = "High conviction"
        elif abs(score) >= 15:
            conviction = "Moderate conviction"
        elif abs(score) >= 5:
            conviction = "Weak lean"
        else:
            conviction = "No clear edge"

        if score > 5:
            bias_txt = "Bullish — favor CALL"
        elif score < -5:
            bias_txt = "Bearish — favor PUT"
        else:
            bias_txt = "Flat — sit out"
        price_cards.append(_card("Bias", f"{conviction} / {bias_txt}",
                                  txt_color))

        # --- Price chart (candlestick) ---
        fig = go.Figure()
        times = data["times"]
        fig.add_trace(go.Candlestick(
            x=list(range(len(times))),
            open=data["open"], high=data["high"],
            low=data["low"], close=data["close"],
            increasing_line_color="#3fb950",
            decreasing_line_color="#f85149",
            name=sym,
        ))

        # Overlay SMA-20 and EMA-8
        if len(data["close"]) >= 20:
            sma20 = SMA(20).calculate(data["close"])
            fig.add_trace(go.Scatter(
                x=list(range(len(times))),
                y=sma20.values, mode="lines",
                line=dict(color="#58a6ff", width=1, dash="dot"),
                name="SMA-20",
            ))
        if len(data["close"]) >= 8:
            ema8 = EMA(8).calculate(data["close"])
            fig.add_trace(go.Scatter(
                x=list(range(len(times))),
                y=ema8.values, mode="lines",
                line=dict(color="#d29922", width=1),
                name="EMA-8",
            ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=40, r=20, t=30, b=30),
            xaxis=dict(showgrid=False, rangeslider_visible=False),
            yaxis=dict(showgrid=True, gridcolor="#21262d"),
            showlegend=True,
            legend=dict(orientation="h", y=1.08, x=0,
                        font=dict(size=11)),
            title=dict(text=f"{sym} — {intv}", font=dict(size=14),
                       x=0.5),
        )

        # --- Component horizontal bars ---
        comp_children = [
            html.Div("COMPONENTS", style={
                "fontSize": "12px", "letterSpacing": "2px",
                "color": "#8b949e", "marginBottom": "10px",
                "fontWeight": 600,
            }),
        ]
        for c in components:
            n = c["norm"]
            if n > 10:
                bar_color = "#3fb950"
            elif n < -10:
                bar_color = "#f85149"
            else:
                bar_color = "#8b949e"

            # Bar width: map -100..+100 to 0..100%
            pct = abs(n)
            is_pos = n >= 0

            comp_children.append(html.Div(style={
                "marginBottom": "8px", "backgroundColor": "#161b22",
                "borderRadius": "6px", "padding": "8px 12px",
                "border": "1px solid #30363d",
            }, children=[
                html.Div(style={"display": "flex",
                                 "justifyContent": "space-between",
                                 "marginBottom": "4px"}, children=[
                    html.Span(c["name"], style={"fontSize": "12px",
                                                 "fontWeight": 600}),
                    html.Span(f"{n:+.1f}", style={
                        "fontSize": "12px", "fontWeight": 700,
                        "color": bar_color}),
                ]),
                # Bar track
                html.Div(style={
                    "height": "6px", "borderRadius": "3px",
                    "backgroundColor": "#21262d", "position": "relative",
                    "overflow": "hidden",
                }, children=[
                    html.Div(style={
                        "position": "absolute",
                        "top": 0, "height": "100%",
                        "borderRadius": "3px",
                        "backgroundColor": bar_color,
                        "left": "50%" if is_pos else f"{50 - pct / 2}%",
                        "width": f"{pct / 2}%",
                    }),
                    # Center line
                    html.Div(style={
                        "position": "absolute", "left": "50%",
                        "top": 0, "height": "100%", "width": "1px",
                        "backgroundColor": "#484f58",
                    }),
                ]),
                html.Div(c["desc"], style={
                    "fontSize": "10px", "color": "#8b949e",
                    "marginTop": "3px"}),
            ]))

        header_info = f"{sym} | {intv} | {nbars} bars | {now_str}"

        return banner_children, banner_style, price_cards, fig, comp_children, header_info

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Trend Slope Signal — consolidated directional bias from Questrade data")
    parser.add_argument("--symbol", default="SPY",
                        help="Ticker symbol (default: SPY)")
    parser.add_argument("--interval", default="OneDay",
                        help="Candle interval (default: OneDay)")
    parser.add_argument("--bars", type=int, default=60,
                        help="Number of bars to fetch (default: 60)")
    parser.add_argument("--watch", type=int, default=0,
                        help="Auto-refresh seconds (0=once). Dash mode min 30s.")
    parser.add_argument("--dash", action="store_true",
                        help="Launch Dash web UI instead of terminal output")
    parser.add_argument("--port", type=int, default=8060,
                        help="Dash port (default: 8060)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.dash:
        app = build_dash_app(args.symbol, args.interval, args.bars, args.watch)
        print(f"\n  Trend Signal Dashboard: http://localhost:{args.port}")
        print(f"  Symbol: {args.symbol}  Interval: {args.interval}  Bars: {args.bars}")
        print(f"  Auto-refresh: {max(args.watch, 30)}s\n")
        app.run(debug=False, port=args.port, host="127.0.0.1")
    else:
        run(symbol=args.symbol, interval=args.interval,
            bars=args.bars, watch=args.watch)


if __name__ == "__main__":
    main()
