"""
Trading Dashboard - Real-time P&L visualization with Plotly Dash

Usage:
    python dashboard.py
    python dashboard.py --port 8050
"""
import json
import sqlite3
import os
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from functools import wraps
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, callback, Output, Input, State, no_update, ctx
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import threading
import subprocess
import signal
from flask import request, Response

# Project root (dashboard.py lives in live/, project root is one level up)
PROJECT_DIR = Path(__file__).parent.parent
import sys
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Load defaults from config
from config.defaults import (
    initial_capital as _default_capital,
    dashboard_port as _default_dashboard_port,
)

# Timezone definitions
TZ_ET = ZoneInfo("America/New_York")
TZ_PST = ZoneInfo("America/Los_Angeles")
TZ_UTC = ZoneInfo("UTC")

# ============================================================
# QUESTRADE REAL-TIME SPY PRICE
# ============================================================
_qt_client = None
_qt_init_attempted = False
_spy_price_cache = {"price": None, "change": None, "change_pct": None, "time": None, "fetched_at": 0}
_SPY_CACHE_TTL = 5  # seconds between API calls


def _get_qt_client():
    """Lazy-init Questrade client for dashboard real-time quotes."""
    global _qt_client, _qt_init_attempted
    if _qt_client is not None:
        return _qt_client
    if _qt_init_attempted:
        return None
    _qt_init_attempted = True
    try:
        from clients.questrade_client import create_questrade_client
        # auto_refresh=False: dashboard piggybacks on engine's token refresh
        # via the shared token file.  Reactive 401 handling in _request()
        # reloads from file when the engine has already refreshed.
        _qt_client = create_questrade_client(auto_refresh=False)
        _qt_client.get_accounts()  # validate token
        return _qt_client
    except Exception:
        _qt_client = None
        return None


def get_spy_live_price() -> dict:
    """Fetch real-time SPY price from Questrade (cached)."""
    import time as _time
    now = _time.time()
    if now - _spy_price_cache["fetched_at"] < _SPY_CACHE_TTL and _spy_price_cache["price"] is not None:
        return _spy_price_cache

    client = _get_qt_client()
    if client is None:
        return _spy_price_cache

    try:
        quote = client.get_quote_by_symbol("SPY")
        if quote:
            last = quote.get("lastTradePrice") or quote.get("lastTradePriceTrHrs", 0)
            prev_close = quote.get("prevDayClosePrice", 0)
            change = (last - prev_close) if prev_close else 0
            change_pct = (change / prev_close * 100) if prev_close else 0
            _spy_price_cache["price"] = last
            _spy_price_cache["change"] = change
            _spy_price_cache["change_pct"] = change_pct
            _spy_price_cache["time"] = datetime.now(TZ_ET).strftime("%H:%M:%S")
            _spy_price_cache["fetched_at"] = now
    except Exception as e:
        # Let the client's built-in 401 retry handle token refresh.
        # Only reset on truly fatal errors (ConnectionError = dead token).
        if isinstance(e, ConnectionError):
            global _qt_client, _qt_init_attempted
            _qt_client = None
            _qt_init_attempted = False

    return _spy_price_cache

# ============================================================
# AUTHENTICATION
# ============================================================
DASHBOARD_USERNAME = os.environ.get("DASHBOARD_USERNAME") or "realericzhu@gmail.com"
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD") or "admin"


def check_auth(username, password):
    """Check if username/password combo is valid."""
    return username == DASHBOARD_USERNAME and password == DASHBOARD_PASSWORD


def authenticate():
    """Send 401 response to enable basic auth."""
    return Response(
        'Login required. Enter your credentials.',
        401,
        {'WWW-Authenticate': 'Basic realm="Trading Dashboard"'}
    )


def requires_auth(f):
    """Decorator for routes that require authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# Process tracking for restart functionality
ENGINE_PROCESS = None


def _deferred_exit(delay: float = 1.5) -> None:
    """Exit the process after a short delay so the HTTP response can be flushed first.
    Docker restart=unless-stopped will bring the container back up automatically."""
    def _do_exit():
        import time
        time.sleep(delay)
        os._exit(1)
    t = threading.Thread(target=_do_exit, daemon=True)
    t.start()

# ============================================================
# STRATEGY CONFIGURATION
# ============================================================

def load_strategy_config():
    """Load strategy configuration from JSON file."""
    config_path = PROJECT_DIR / "config" / "strategy.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            return {}
    return {}


def save_strategy_config(trade_config, risk_config):
    """Save strategy configuration to JSON file."""
    config_path = PROJECT_DIR / "config" / "strategy.json"
    try:
        with open(config_path) as f:
            data = json.load(f)
        
        # Update trade_config fields
        for key, value in trade_config.items():
            if key in data.get('trade_config', {}):
                data['trade_config'][key] = value
        
        # Update risk_config fields
        for key, value in risk_config.items():
            if key in data.get('risk_config', {}):
                data['risk_config'][key] = value
        
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        return False


# ============================================================
# SYSTEM STATUS
# ============================================================

def check_ibkr_gateway():
    """Check if IBKR Gateway is reachable via TCP."""
    import socket
    try:
        host = os.environ.get("IBKR_HOST", "127.0.0.1")
        port = int(os.environ.get("IBKR_PAPER_PORT", 0))
        if not port:
            port = load_strategy_config().get("ibkr_paper_port", 7497)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0, host, port
    except Exception:
        return False, "127.0.0.1", 7497


def check_questrade_status():
    """Check Questrade API connectivity and token status."""
    try:
        client = _get_qt_client()
        if client is None:
            return "Disconnected", "Token expired or not configured"
        # Check if token is still valid
        client.get_accounts()
        return "Connected", "Token valid"
    except ConnectionError as e:
        # Only reset on truly fatal connection errors (dead token)
        global _qt_client, _qt_init_attempted
        _qt_client = None
        _qt_init_attempted = False
        err_msg = str(e)
        if "401" in err_msg or "token" in err_msg.lower():
            return "Token Expired", "Refresh token at Questrade App Hub"
        return "Error", err_msg[:80]
    except Exception as e:
        return "Error", str(e)[:80]


def get_system_status():
    """Get system status information."""
    status = {
        "ibkr_status": "Unknown",
        "ibkr_host": "",
        "ibkr_port": 0,
        "questrade_status": "Unknown",
        "questrade_detail": "",
        "engine_status": "Unknown",
        "db_status": "Unknown",
        "last_quote_time": None,
        "errors": []
    }
    
    # Check IBKR Gateway connectivity
    reachable, host, port = check_ibkr_gateway()
    status["ibkr_host"] = host
    status["ibkr_port"] = port
    if reachable:
        status["ibkr_status"] = "Connected"
    else:
        status["ibkr_status"] = "Unreachable"
        status["errors"].append(f"IBKR Gateway not reachable on {host}:{port}")
    
    # Check Questrade
    qt_status, qt_detail = check_questrade_status()
    status["questrade_status"] = qt_status
    status["questrade_detail"] = qt_detail
    if qt_status != "Connected":
        status["errors"].append(f"Questrade: {qt_detail}")

    # Check database
    db_path = PROJECT_DIR / "data" / "live_0dte_trades.db"
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades")
            count = cursor.fetchone()[0]
            status["db_status"] = f"OK ({count} trades)"
            
            # Get last trade time
            cursor.execute("SELECT MAX(entry_time) FROM trades")
            last_trade = cursor.fetchone()[0]
            if last_trade:
                status["last_trade_time"] = last_trade[:19]
            
            conn.close()
        except Exception as e:
            status["db_status"] = "Error"
            status["errors"].append(f"Database error: {str(e)}")
    else:
        status["db_status"] = "No database"
    
    # Check log file for engine status
    log_file = PROJECT_DIR / "logs" / f"live_0dte_{date.today().strftime('%Y%m%d')}.log"
    if log_file.exists():
        try:
            # Read last few lines of log
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            if lines:
                last_lines = lines[-20:]  # Last 20 lines
                last_log_time = None
                
                for line in reversed(last_lines):
                    if line.strip():
                        # Parse timestamp from log line
                        try:
                            timestamp_str = line[:23]
                            last_log_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                            break
                        except:
                            continue
                
                if last_log_time:
                    age_seconds = (datetime.now() - last_log_time).total_seconds()
                    status["last_quote_time"] = last_log_time.strftime("%H:%M:%S")
                    
                    if age_seconds < 60:
                        status["engine_status"] = "Running"
                    elif age_seconds < 300:
                        status["engine_status"] = "Slow"
                        status["errors"].append(f"No activity for {int(age_seconds)}s")
                    else:
                        status["engine_status"] = "Stale"
                        status["errors"].append(f"No activity for {int(age_seconds/60)}min")
                
                # Check for recent errors in logs
                for line in last_lines:
                    if 'ERROR' in line or 'Exception' in line:
                        status["errors"].append(line.strip()[:100])
        except Exception as e:
            pass
    else:
        status["engine_status"] = "No logs"
    
    return status


def get_recent_logs(num_lines=100):
    """Get recent log entries from journalctl (systemd) or local files."""
    lines_out = []
    
    # Primary: Try journalctl for systemd service logs (VM deployment)
    try:
        result = subprocess.run(
            ["journalctl", "-u", "trading-engine", "-n", str(num_lines), "--no-pager", "--output=short"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                # Clean up the log line - remove the systemd prefix
                if 'python[' in line:
                    # Extract just the log message after the process info
                    parts = line.split(':', 3)
                    if len(parts) >= 4:
                        line = parts[3].strip()
                    else:
                        line = line.strip()
                if line and len(line) > 200:
                    line = line[:200] + '...'
                if line:
                    lines_out.append(line)
    except Exception as e:
        pass
    
    # Fallback 1: Read from terminal_output.log (live backend output)
    if not lines_out:
        terminal_log = PROJECT_DIR / "logs" / "terminal_output.log"
        if terminal_log.exists():
            try:
                with open(terminal_log, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                for line in lines[-num_lines:]:
                    line = line.strip()
                    if line and len(line) > 200:
                        line = line[:200] + '...'
                    if line:
                        lines_out.append(line)
            except:
                pass
    
    # Fallback 2: Trading log file
    if not lines_out:
        trading_log = PROJECT_DIR / "logs" / f"live_0dte_{date.today().strftime('%Y%m%d')}.log"
        if trading_log.exists():
            try:
                with open(trading_log, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                for line in lines[-num_lines:]:
                    line = line.strip()
                    if line and len(line) > 200:
                        line = line[:200] + '...'
                    if line:
                        lines_out.append(line)
            except:
                pass
    
    # Reverse for most recent first
    lines_out.reverse()
    return lines_out if lines_out else ["No log output available - engine may not be running"]


# ============================================================
# DATA LOADING
# ============================================================

def get_db_stats():
    """Get stats directly from database (source of truth)."""
    db_path = PROJECT_DIR / "data" / "live_0dte_trades.db"
    if not db_path.exists():
        return None
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get aggregate stats from closed trades
        cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins,
                COUNT(CASE WHEN pnl <= 0 THEN 1 END) as losses,
                COALESCE(SUM(pnl), 0) as total_pnl
            FROM trades 
            WHERE status = 'closed' AND pnl IS NOT NULL
        """)
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "total_trades": row[0] or 0,
                "total_wins": row[1] or 0,
                "total_losses": row[2] or 0,
                "total_pnl": row[3] or 0.0
            }
    except Exception as e:
        pass
    
    return None


def get_state():
    """Load trading state from JSON file, synced with DB stats and real broker balance."""
    state_path = PROJECT_DIR / "trading_state.json"
    _cap = _default_capital()
    state = {
        "initial_capital": _cap,
        "current_capital": _cap,
        "total_pnl": 0,
        "total_trades": 0,
        "total_wins": 0,
        "total_losses": 0,
        "max_drawdown": 0,
        "equity_curve": [],
        "engine_status": "unknown",
        "broker_nlv": None,
        "broker_cash": None,
        "broker_positions_value": None,
        "broker_balance_time": None,
    }
    
    if state_path.exists():
        with open(state_path) as f:
            state.update(json.load(f))
    
    # Sync with DB stats (DB is source of truth for trade counts)
    db_stats = get_db_stats()
    if db_stats:
        state["total_trades"] = db_stats["total_trades"]
        state["total_wins"] = db_stats["total_wins"]
        state["total_losses"] = db_stats["total_losses"]
        state["total_pnl"] = db_stats["total_pnl"]
        state["current_capital"] = state["initial_capital"] + db_stats["total_pnl"]
    
    # Get real broker balance from balance_history table
    broker_bal = get_latest_broker_balance()
    if broker_bal:
        state["broker_nlv"] = broker_bal.get("net_liquidation")
        state["broker_cash"] = broker_bal.get("cash")
        state["broker_positions_value"] = broker_bal.get("positions_value")
        state["broker_balance_time"] = broker_bal.get("timestamp")
    
    return state


def get_latest_broker_balance():
    """Get the most recent broker balance from DB."""
    db_path = PROJECT_DIR / "data" / "live_0dte_trades.db"
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        # Check if balance_history table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='balance_history'")
        if not cursor.fetchone():
            conn.close()
            return None
        cursor.execute(
            "SELECT * FROM balance_history ORDER BY timestamp DESC LIMIT 1"
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            cols = [d[0] for d in cursor.description]
            return dict(zip(cols, row))
    except Exception:
        pass
    return None


def get_ibkr_positions():
    """Read current IBKR account positions from the database."""
    db_path = PROJECT_DIR / "data" / "live_0dte_trades.db"
    if not db_path.exists():
        return [], None
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Check table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='current_positions'")
        if not cursor.fetchone():
            conn.close()
            return [], None
        cursor.execute('SELECT * FROM current_positions ORDER BY symbol')
        rows = [dict(r) for r in cursor.fetchall()]
        # Get update time from the first row
        updated = rows[0]['updated_at'] if rows else None
        conn.close()
        return rows, updated
    except Exception:
        return [], None


def get_trades_df():
    """Load trades from database into DataFrame."""
    db_path = PROJECT_DIR / "data" / "live_0dte_trades.db"
    if not db_path.exists():
        return pd.DataFrame()
    
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY entry_time DESC", conn)
    conn.close()
    
    if not df.empty:
        # Use format='ISO8601' and utc=True to handle timestamps with mixed timezones
        df['entry_time'] = pd.to_datetime(df['entry_time'], format='ISO8601', utc=True, errors='coerce')
        df['exit_time'] = pd.to_datetime(df['exit_time'], format='ISO8601', utc=True, errors='coerce')
        df['trade_date'] = df['entry_time'].dt.date
    
    return df


def get_today_trades(df):
    """Filter to today's trades."""
    if df.empty:
        return df
    today = date.today()
    return df[df['trade_date'] == today]


# ============================================================
# CHARTS
# ============================================================

def get_option_type_from_symbol(symbol):
    """Extract CALL or PUT from option symbol."""
    import re
    if not symbol:
        return ""
    match = re.search(r'[CP]\d', str(symbol))
    if match:
        return "PUT" if match.group()[0] == 'P' else "CALL"
    return ""


def create_equity_curve(state, initial_capital=None):
    """Create equity curve chart from state JSON or DB."""
    if initial_capital is None:
        initial_capital = _default_capital()
    equity_curve = state.get('equity_curve', [])
    
    # Detect stale equity curve (capital mismatch from previous run)
    if equity_curve:
        first_equity = equity_curve[0].get('equity', 0)
        if first_equity != initial_capital and first_equity != 0:
            equity_curve = []  # Force rebuild from DB
    
    # If no equity curve in state, build from DB
    if not equity_curve or len(equity_curve) < 2:
        db_path = PROJECT_DIR / "data" / "live_0dte_trades.db"
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, symbol, pnl, exit_time 
                    FROM trades 
                    WHERE status = 'closed' AND pnl IS NOT NULL 
                    ORDER BY exit_time
                """)
                trades = cursor.fetchall()
                conn.close()
                
                if trades:
                    equity_curve = [{"trade_id": 0, "type": "-", "equity": initial_capital, "pnl": 0}]
                    running_capital = initial_capital
                    for trade_id, symbol, pnl, exit_time in trades:
                        running_capital += pnl
                        opt_type = get_option_type_from_symbol(symbol)
                        equity_curve.append({
                            "trade_id": trade_id,
                            "type": opt_type,
                            "equity": running_capital,
                            "pnl": pnl,
                            "time": exit_time
                        })
            except Exception as e:
                pass
    
    if not equity_curve or len(equity_curve) < 2:
        fig = go.Figure()
        fig.add_annotation(text="No trade data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color='#888'))
    else:
        # Build equity curve from state
        equities = [point.get('equity', initial_capital) for point in equity_curve]
        pnls = [point.get('pnl', 0) for point in equity_curve]
        types = [point.get('type', '-') for point in equity_curve]
        
        # Use datetime for x-axis, fall back to trade numbers if no time data
        times = [point.get('time') for point in equity_curve]
        has_times = any(t is not None for t in times)
        
        if has_times:
            # Parse datetime strings for x-axis
            x_values = []
            for i, t in enumerate(times):
                if t:
                    try:
                        x_values.append(pd.to_datetime(t))
                    except:
                        x_values.append(pd.Timestamp.now())
                else:
                    # For initial point without time, use first trade time minus 1 hour
                    first_time = next((pd.to_datetime(tt) for tt in times if tt), pd.Timestamp.now())
                    x_values.append(first_time - pd.Timedelta(hours=1))
        else:
            x_values = list(range(len(equity_curve)))
        
        # Color markers by type (PUT=blue, CALL=green)
        colors = ['#00d9ff' if t == 'PUT' else '#00ff88' if t == 'CALL' else '#888' for t in types]
        
        # Hover text
        if has_times:
            hover_text = [f"Trade #{equity_curve[i].get('trade_id', i)}: {types[i]}<br>Time: {x_values[i].strftime('%Y-%m-%d %H:%M') if hasattr(x_values[i], 'strftime') else x_values[i]}<br>Equity: ${equities[i]:,.0f}<br>P&L: ${pnls[i]:+,.0f}" 
                          for i in range(len(equity_curve))]
        else:
            hover_text = [f"Trade #{i}: {types[i]}<br>Equity: ${equities[i]:,.0f}<br>P&L: ${pnls[i]:+,.0f}" 
                          for i in range(len(equity_curve))]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values,
            y=equities,
            mode='lines+markers',
            name='Equity',
            line=dict(color='#00d9ff', width=3),
            marker=dict(size=10, color=colors),
            hovertext=hover_text,
            hoverinfo='text'
        ))
        
        # Add initial capital line
        fig.add_hline(y=initial_capital, line_dash="dash", line_color="#888", 
                     annotation_text=f"Initial: ${initial_capital:,}")
        
        # Add high water mark line
        hwm = state.get('high_water_mark', initial_capital)
        if hwm > initial_capital:
            fig.add_hline(y=hwm, line_dash="dot", line_color="#00ff88", 
                         annotation_text=f"HWM: ${hwm:,.0f}")
        
        # Auto-focus on recent trades if there are many
        num_trades = len(equity_curve)
        x_range = None
        
        if num_trades > 30:
            # Focus on last 30 trades
            if has_times:
                x_range = [x_values[-31], x_values[-1]]
            else:
                x_range = [num_trades - 31, num_trades]
        
        # Focus y-axis on the change (min/max with padding)
        y_min = min(equities)
        y_max = max(equities)
        y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 100
        y_range = [y_min - y_padding, y_max + y_padding]
        
        fig.update_layout(
            title="Equity Curve",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=60, r=20, t=50, b=40),
            showlegend=False,
            xaxis=dict(
                title="Time" if has_times else "Trade #",
                title_font=dict(size=12, color='#888'),
                range=x_range,
                rangeslider=dict(visible=True, thickness=0.05, bgcolor='rgba(50,50,50,0.3)'),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(100,100,100,0.3)',
                showline=True,
                linewidth=1,
                linecolor='#555',
                tickfont=dict(size=10, color='#aaa'),
                tickformat='%m/%d %H:%M' if has_times else None,
                zeroline=False,
                showspikes=True,
                spikecolor='#00d9ff',
                spikethickness=1,
                spikedash='dot',
                spikemode='across'
            ),
            yaxis=dict(
                title="Capital ($)",
                title_font=dict(size=12, color='#888'),
                range=y_range,
                fixedrange=False,
                tickprefix="$",
                tickformat=",.0f",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(100,100,100,0.3)',
                showline=True,
                linewidth=1,
                linecolor='#555',
                tickfont=dict(size=10, color='#aaa'),
                zeroline=False,
                showspikes=True,
                spikecolor='#00d9ff',
                spikethickness=1,
                spikedash='dot',
                spikemode='across'
            ),
            hovermode='x unified',
            height=280
        )
        return fig
    
    # Empty chart case - basic layout
    fig.update_layout(
        title="Equity Curve",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20),
        height=280
    )
    return fig


def create_pnl_bars(state):
    """Create P&L bar chart by trade from state or DB."""
    equity_curve = state.get('equity_curve', [])
    
    # Skip first entry (initial capital with pnl=0)
    trades = [t for t in equity_curve if t.get('trade_id', 0) > 0]
    
    # If no trades from state, get ALL from DB
    if not trades:
        db_path = PROJECT_DIR / "data" / "live_0dte_trades.db"
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, symbol, pnl, exit_time
                    FROM trades 
                    WHERE status = 'closed' AND pnl IS NOT NULL 
                    ORDER BY exit_time
                """)
                db_trades = cursor.fetchall()
                conn.close()
                
                trades = []
                for trade_id, symbol, pnl, exit_time in db_trades:
                    opt_type = get_option_type_from_symbol(symbol)
                    trades.append({
                        "trade_id": trade_id,
                        "type": opt_type,
                        "pnl": pnl,
                        "time": exit_time
                    })
            except:
                pass
    
    x_range = None
    if not trades:
        fig = go.Figure()
        fig.add_annotation(text="No trade data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color='#888'))
    else:
        pnls = [t.get('pnl', 0) for t in trades]
        types = [t.get('type', '-') for t in trades]
        trade_ids = [t.get('trade_id', 0) for t in trades]
        times = [t.get('time') for t in trades]
        
        # Cumulative P&L for running total line
        cum_pnl = []
        running = 0
        for p in pnls:
            running += p
            cum_pnl.append(running)
        
        # Green for profit, red for loss
        colors = ['#00ff88' if p > 0 else '#ff4757' for p in pnls]
        
        # Use trade numbers for x-axis
        x_vals = list(range(1, len(trades) + 1))
        
        fig = go.Figure()
        
        # P&L bars
        fig.add_trace(go.Bar(
            x=x_vals,
            y=pnls,
            marker_color=colors,
            text=[f"${p:+.0f}" for p in pnls],
            textposition='outside',
            textfont=dict(size=9),
            hovertemplate='Trade #%{customdata[0]}<br>Type: %{customdata[1]}<br>P&L: $%{y:.2f}<extra></extra>',
            customdata=list(zip(trade_ids, types)),
            name='P&L'
        ))
        
        # Cumulative P&L line
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=cum_pnl,
            mode='lines',
            line=dict(color='#00d9ff', width=2),
            yaxis='y2',
            name='Cumulative',
            hovertemplate='Cumulative: $%{y:+,.0f}<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)
        
        # Default view: last 30 trades with rangeslider to see all
        num_trades = len(trades)
        x_range = [max(1, num_trades - 29), num_trades + 1] if num_trades > 30 else None
    
    fig.update_layout(
        title=f"Trade P&L ({len(trades) if trades else 0} trades)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=40, b=40),
        showlegend=False,
        height=350,
        xaxis=dict(
            title="Trade #",
            title_font=dict(size=11, color='#888'),
            range=x_range,
            rangeslider=dict(visible=True, thickness=0.08, bgcolor='rgba(50,50,50,0.3)'),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(100,100,100,0.2)',
            tickfont=dict(size=10, color='#aaa'),
            showspikes=True,
            spikecolor='#00d9ff',
            spikethickness=1,
            spikedash='dot',
            spikemode='across'
        ),
        yaxis=dict(
            title="P&L ($)",
            title_font=dict(size=11, color='#888'),
            tickprefix="$",
            tickformat="+,.0f",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(100,100,100,0.2)',
            tickfont=dict(size=10, color='#aaa'),
            zeroline=True,
            zerolinecolor='#555',
            fixedrange=False
        ),
        yaxis2=dict(
            title="Cumulative ($)",
            title_font=dict(size=11, color='#00d9ff'),
            tickprefix="$",
            tickformat="+,.0f",
            overlaying='y',
            side='right',
            showgrid=False,
            tickfont=dict(size=10, color='#00d9ff'),
            fixedrange=False
        ),
        hovermode='x unified',
        dragmode='zoom'
    )
    return fig


def create_win_rate_gauge(wins, total):
    """Create win rate gauge chart."""
    win_rate = (wins / total * 100) if total > 0 else 0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=win_rate,
        number={'suffix': '%', 'font': {'size': 40, 'color': '#00d9ff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#888'},
            'bar': {'color': '#00ff88' if win_rate >= 50 else '#ff4757'},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(255,71,87,0.3)'},
                {'range': [40, 60], 'color': 'rgba(255,165,0,0.3)'},
                {'range': [60, 100], 'color': 'rgba(0,255,136,0.3)'}
            ],
            'threshold': {
                'line': {'color': '#fff', 'width': 2},
                'thickness': 0.75,
                'value': win_rate
            }
        },
        title={'text': 'Win Rate', 'font': {'color': '#888', 'size': 14}}
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=20),
        height=200
    )
    return fig


def create_today_pnl_indicator(today_pnl):
    """Create today's P&L indicator."""
    color = '#00ff88' if today_pnl >= 0 else '#ff4757'
    
    fig = go.Figure(go.Indicator(
        mode="number+delta",
        value=today_pnl,
        number={'prefix': '$', 'font': {'size': 48, 'color': color}},
        delta={'reference': 0, 'relative': False, 'valueformat': '.0f'},
        title={'text': "Today's P&L", 'font': {'color': '#888', 'size': 14}}
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=20),
        height=150
    )
    return fig


# ============================================================
# DASH APP
# ============================================================

app = Dash(__name__)
app.title = "0DTE Trading Dashboard"

# Apply HTTP Basic Auth to all routes
@app.server.before_request
def protect_dashboards():
    """Require authentication for all dashboard routes."""
    auth = request.authorization
    if not auth or not check_auth(auth.username, auth.password):
        return authenticate()

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                box-sizing: border-box;
            }
            body {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                min-height: 100vh;
                -webkit-text-size-adjust: 100%;
            }
            .dash-table-container {
                background: rgba(255,255,255,0.05) !important;
            }
            /* Mobile responsive styles */
            @media (max-width: 768px) {
                .stats-row {
                    flex-direction: column !important;
                    gap: 10px !important;
                }
                .stats-card {
                    padding: 15px !important;
                }
                .stats-card-value {
                    font-size: 1.5em !important;
                }
                .charts-row {
                    flex-direction: column !important;
                }
                .chart-container {
                    min-height: 250px !important;
                }
                .header-title {
                    font-size: 1.5em !important;
                }
                .status-bar {
                    flex-direction: column !important;
                    align-items: flex-start !important;
                    gap: 5px !important;
                }
                .main-container {
                    padding: 10px !important;
                }
                .section-container {
                    padding: 15px !important;
                }
                .log-container {
                    max-height: 200px !important;
                }
            }
            @media (max-width: 480px) {
                .stats-card-value {
                    font-size: 1.3em !important;
                }
                .header-title {
                    font-size: 1.2em !important;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def serve_layout():
    """Generate layout with fresh data."""
    state = get_state()
    df = get_trades_df()
    today_df = get_today_trades(df)
    # Filter out phantom trades (closed with no real exit and $0 PnL)
    if not today_df.empty:
        phantom_mask = (today_df['status'] == 'closed') & (today_df['pnl'].fillna(0) == 0) & (today_df['exit_price'].fillna(0) == 0)
        today_df = today_df[~phantom_mask]
    
    # Calculate metrics
    today_pnl = today_df[today_df['status'] == 'closed']['pnl'].sum() if not today_df.empty else 0
    today_wins = len(today_df[(today_df['status'] == 'closed') & (today_df['pnl'] > 0)]) if not today_df.empty else 0
    today_losses = len(today_df[(today_df['status'] == 'closed') & (today_df['pnl'] <= 0)]) if not today_df.empty else 0
    
    initial_cap = state.get('initial_capital', _default_capital())
    current_cap = state.get('current_capital', _default_capital())
    total_pnl = state.get('total_pnl', 0)
    total_return = ((current_cap - initial_cap) / initial_cap) * 100 if initial_cap > 0 else 0
    
    # Prepare table data — all historical trades
    table_data = []
    if not df.empty:
        for _, t in df.iterrows():
            pnl = t.get('pnl') or 0
            pnl_pct = t.get('pnl_percent') or 0
            status = t.get('status', 'open')
            
            if status == 'open':
                pnl_display = 'OPEN'
            elif pnl > 0:
                pnl_display = f'+${pnl:.0f} (+{pnl_pct:.1f}%)'
            else:
                pnl_display = f'-${abs(pnl):.0f} ({pnl_pct:.1f}%)'
            
            entry_time = t.get('entry_time', '')
            trade_date = str(entry_time)[:10] if entry_time else ''
            trade_time = str(entry_time)[11:16] if entry_time else ''
            table_data.append({
                'Date': trade_date,
                'Symbol': t.get('symbol', '')[-15:],  # Truncate for mobile
                'Type': str(t.get('option_type', '')).upper()[:4],
                'Qty': t.get('quantity', 0),
                'Entry': f"${t.get('entry_price', 0):.2f}",
                'Exit': f"${t.get('exit_price', 0):.2f}" if t.get('exit_price') else '-',
                'P&L': pnl_display,
                'Time': trade_time,
                'Status': status.upper()[:4]
            })
    
    # Get system status
    sys_status = get_system_status()
    
    # Status colors
    def get_status_color(status_text):
        if status_text in ['Valid', 'Running', 'OK', 'Connected'] or status_text.startswith('OK'):
            return '#00ff88'
        elif status_text in ['Expired', 'Error', 'Missing', 'Stale', 'Unreachable']:
            return '#ff4757'
        elif status_text in ['Slow', 'Unknown']:
            return '#ffa500'
        else:
            return '#00d9ff'
    
    return html.Div([
        # Header
        html.Div([
            html.H1("0DTE Trading Dashboard", className='header-title', style={
                'background': 'linear-gradient(90deg, #00d9ff, #00ff88)',
                'WebkitBackgroundClip': 'text',
                'WebkitTextFillColor': 'transparent',
                'fontSize': '2.5em',
                'marginBottom': '10px'
            }),
            html.Span("PAPER TRADING", style={
                'background': '#ffa500',
                'color': '#000',
                'padding': '5px 15px',
                'borderRadius': '20px',
                'fontWeight': 'bold',
                'fontSize': '0.8em'
            })
        ], style={'textAlign': 'center', 'padding': '15px', 'marginBottom': '15px'}),
        
        # System Status Bar
        html.Div([
            html.Div(id='status-bar', className='status-bar'),
            html.Div([
                html.Button("🔄 Restart", id='restart-btn', n_clicks=0, style={
                    'background': 'linear-gradient(90deg, #ff4757, #ff6b81)',
                    'color': '#fff',
                    'border': 'none',
                    'padding': '8px 15px',
                    'borderRadius': '20px',
                    'cursor': 'pointer',
                    'fontWeight': 'bold',
                    'fontSize': '0.9em'
                }),
                html.Button("⚙️ Config", id='config-toggle-btn', n_clicks=0, style={
                    'background': 'linear-gradient(90deg, #5f27cd, #7c3aed)',
                    'color': '#fff',
                    'border': 'none',
                    'padding': '8px 15px',
                    'borderRadius': '20px',
                    'cursor': 'pointer',
                    'fontWeight': 'bold',
                    'fontSize': '0.9em',
                    'marginLeft': '10px'
                }),
                html.Span(id='restart-status', style={'marginLeft': '10px', 'fontSize': '0.85em'})
            ], style={'marginTop': '10px', 'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap'})
        ], className='section-container', style={
            'background': 'rgba(255,255,255,0.05)',
            'borderRadius': '10px',
            'padding': '12px 15px',
            'marginBottom': '15px',
            'border': '1px solid rgba(255,255,255,0.1)'
        }),
        
        # Configuration Panel (hidden by default)
        html.Div(id='config-panel', children=[
            html.H3("Strategy Configuration — Phase 8 Momentum", style={'marginBottom': '15px', 'color': '#00d9ff'}),

            # ── Broker Connection ──────────────────────────────────────
            html.Div([
                html.H4("🔌 IBKR Gateway", style={'color': '#00d9ff', 'marginBottom': '10px', 'marginTop': '0'}),
                html.Div(id='ibkr-gateway-info', style={'fontSize': '0.9em'}),
            ], style={
                'padding': '15px',
                'background': 'rgba(0,217,255,0.07)',
                'borderRadius': '10px',
                'border': '1px solid rgba(0,217,255,0.25)',
                'marginBottom': '20px',
            }),

            # ── Questrade Status ──────────────────────────────────────
            html.Div([
                html.H4("📊 Questrade", style={'color': '#00d9ff', 'marginBottom': '10px', 'marginTop': '0'}),
                html.Div(id='questrade-status-info', style={'fontSize': '0.9em'}),
                html.Hr(style={'borderColor': 'rgba(0,217,255,0.2)', 'margin': '12px 0'}),
                html.Div([
                    html.Label("Refresh Token:", style={'color': '#aaa', 'fontSize': '0.85em', 'marginBottom': '5px', 'display': 'block'}),
                    html.Div([
                        dcc.Input(
                            id='qt-token-input',
                            type='text',
                            placeholder='Paste new refresh token from Questrade App Hub',
                            style={
                                'width': '100%', 'padding': '8px 12px',
                                'background': '#1a1a2e', 'color': '#fff',
                                'border': '1px solid rgba(0,217,255,0.3)',
                                'borderRadius': '6px', 'fontSize': '0.85em',
                                'fontFamily': 'monospace',
                            },
                            debounce=True,
                        ),
                        html.Button(
                            "🔄 Refresh Token",
                            id='qt-token-submit-btn',
                            n_clicks=0,
                            style={
                                'marginTop': '8px', 'padding': '8px 20px',
                                'background': '#00d9ff', 'color': '#000',
                                'border': 'none', 'borderRadius': '6px',
                                'cursor': 'pointer', 'fontWeight': 'bold',
                                'fontSize': '0.85em',
                            },
                        ),
                    ]),
                    html.Div(id='qt-token-result', style={'marginTop': '8px', 'fontSize': '0.85em'}),
                ]),
            ], style={
                'padding': '15px',
                'background': 'rgba(0,217,255,0.07)',
                'borderRadius': '10px',
                'border': '1px solid rgba(0,217,255,0.25)',
                'marginBottom': '20px',
            }),

            html.Div([
                # Risk Config
                html.Div([
                    html.H4("Risk Management", style={'color': '#ffa500', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Stop After First Loss:", style={'display': 'block', 'marginBottom': '5px'}),
                        dcc.Dropdown(id='cfg-stop-after-first-loss', options=[
                            {'label': 'Yes (Conservative)', 'value': True},
                            {'label': 'No (Aggressive)', 'value': False}
                        ], style={'color': '#000', 'width': '200px'})
                    ], style={'marginBottom': '15px'}),
                    html.Div([
                        html.Label("Kelly Fraction:", style={'display': 'block', 'marginBottom': '5px'}),
                        dcc.Input(id='cfg-kelly-fraction', type='number', min=0.05, max=1.0, step=0.05, 
                                  style={'width': '100px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff'})
                    ], style={'marginBottom': '15px'}),
                    html.Div([
                        html.Label("Max Position Value ($):", style={'display': 'block', 'marginBottom': '5px'}),
                        dcc.Input(id='cfg-max-position-value', type='number', min=100, max=50000, step=100,
                                  style={'width': '100px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff'})
                    ], style={'marginBottom': '15px'}),
                ], style={'flex': '1', 'minWidth': '250px', 'padding': '15px', 'background': 'rgba(255,255,255,0.03)', 'borderRadius': '10px', 'marginRight': '10px'}),
                
                # Trade Config
                html.Div([
                    html.H4("Exit Rules", style={'color': '#00ff88', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Default Profit Target:", style={'display': 'block', 'marginBottom': '5px'}),
                        dcc.Input(id='cfg-profit-target', type='number', min=0.05, max=2.0, step=0.01,
                                  style={'width': '100px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff'})
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Default Stop Loss:", style={'display': 'block', 'marginBottom': '5px'}),
                        dcc.Input(id='cfg-stop-loss', type='number', min=0.05, max=1.0, step=0.01,
                                  style={'width': '100px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff'})
                    ], style={'marginBottom': '10px'}),
                    html.Hr(style={'borderColor': '#333', 'margin': '8px 0'}),
                    html.Div("Asymmetric Overrides (blank = use default)", style={'color': '#888', 'fontSize': '0.8em', 'marginBottom': '8px'}),
                    html.Div([
                        html.Div([
                            html.Label("CALL PT:", style={'display': 'block', 'marginBottom': '3px', 'fontSize': '0.85em'}),
                            dcc.Input(id='cfg-call-pt', type='number', min=0.05, max=2.0, step=0.01, placeholder='—',
                                      style={'width': '80px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff', 'fontSize': '0.9em'})
                        ], style={'display': 'inline-block', 'marginRight': '8px'}),
                        html.Div([
                            html.Label("CALL SL:", style={'display': 'block', 'marginBottom': '3px', 'fontSize': '0.85em'}),
                            dcc.Input(id='cfg-call-sl', type='number', min=0.05, max=1.0, step=0.01, placeholder='—',
                                      style={'width': '80px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff', 'fontSize': '0.9em'})
                        ], style={'display': 'inline-block'}),
                    ], style={'marginBottom': '8px'}),
                    html.Div([
                        html.Div([
                            html.Label("PUT PT:", style={'display': 'block', 'marginBottom': '3px', 'fontSize': '0.85em'}),
                            dcc.Input(id='cfg-put-pt', type='number', min=0.05, max=2.0, step=0.01, placeholder='—',
                                      style={'width': '80px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff', 'fontSize': '0.9em'})
                        ], style={'display': 'inline-block', 'marginRight': '8px'}),
                        html.Div([
                            html.Label("PUT SL:", style={'display': 'block', 'marginBottom': '3px', 'fontSize': '0.85em'}),
                            dcc.Input(id='cfg-put-sl', type='number', min=0.05, max=1.0, step=0.01, placeholder='—',
                                      style={'width': '80px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff', 'fontSize': '0.9em'})
                        ], style={'display': 'inline-block'}),
                    ], style={'marginBottom': '10px'}),
                    html.Hr(style={'borderColor': '#333', 'margin': '8px 0'}),
                    html.Div([
                        html.Label("Max Hold (minutes):", style={'display': 'block', 'marginBottom': '5px'}),
                        dcc.Input(id='cfg-max-hold', type='number', min=1, max=120, step=5,
                                  style={'width': '100px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff'})
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Regime Detection:", style={'display': 'block', 'marginBottom': '5px'}),
                        dcc.Dropdown(id='cfg-regime-detection', options=[
                            {'label': 'Enabled', 'value': True},
                            {'label': 'Disabled', 'value': False}
                        ], style={'color': '#000', 'width': '150px'})
                    ], style={'marginBottom': '10px'}),
                ], style={'flex': '1', 'minWidth': '250px', 'padding': '15px', 'background': 'rgba(255,255,255,0.03)', 'borderRadius': '10px', 'marginRight': '10px'}),
                
                # Option Filter Config
                html.Div([
                    html.H4("Option Filters", style={'color': '#ff6b81', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Min Option Price ($):", style={'display': 'block', 'marginBottom': '5px'}),
                        dcc.Input(id='cfg-min-option-price', type='number', min=0.10, max=5.0, step=0.05,
                                  style={'width': '100px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff'})
                    ], style={'marginBottom': '15px'}),
                    html.Div([
                        html.Label("Max Option Price ($):", style={'display': 'block', 'marginBottom': '5px'}),
                        dcc.Input(id='cfg-max-option-price', type='number', min=0.50, max=10.0, step=0.10,
                                  style={'width': '100px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff'})
                    ], style={'marginBottom': '15px'}),
                    html.Div([
                        html.Label("Max Consecutive Losses:", style={'display': 'block', 'marginBottom': '5px'}),
                        dcc.Input(id='cfg-max-consec-losses', type='number', min=1, max=10, step=1,
                                  style={'width': '100px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff'})
                    ], style={'marginBottom': '15px'}),
                    html.Div([
                        html.Label("Max Daily Loss (%):", style={'display': 'block', 'marginBottom': '5px'}),
                        dcc.Input(id='cfg-max-daily-loss', type='number', min=0.1, max=5.0, step=0.1,
                                  style={'width': '100px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff'})
                    ], style={'marginBottom': '15px'}),
                ], style={'flex': '1', 'minWidth': '250px', 'padding': '15px', 'background': 'rgba(255,255,255,0.03)', 'borderRadius': '10px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px', 'marginBottom': '20px'}),
            
            # Save Button
            html.Div([
                html.Button("💾 Save & Restart Engine", id='save-config-btn', n_clicks=0, style={
                    'background': 'linear-gradient(90deg, #00d9ff, #00ff88)',
                    'color': '#000',
                    'border': 'none',
                    'padding': '12px 25px',
                    'borderRadius': '25px',
                    'cursor': 'pointer',
                    'fontWeight': 'bold',
                    'fontSize': '1em'
                }),
                html.Span(id='config-save-status', style={'marginLeft': '15px', 'fontSize': '0.9em'})
            ], style={'textAlign': 'center'})
        ], style={
            'display': 'none',
            'background': 'rgba(255,255,255,0.05)',
            'borderRadius': '10px',
            'padding': '20px',
            'marginBottom': '15px',
            'border': '1px solid rgba(255,255,255,0.1)'
        }),
        
        # Stats Cards Row
        html.Div([
            # SPY Live Price (from Questrade)
            html.Div([
                html.Div("SPY LIVE", style={'color': '#888', 'fontSize': '0.75em', 'marginBottom': '3px'}),
                html.Div(id='spy-live-price', className='stats-card-value', style={'fontSize': '1.8em', 'fontWeight': 'bold', 'color': '#ffa500'}),
                html.Div(id='spy-live-change', style={'fontSize': '0.8em', 'marginTop': '2px'}),
                html.Div(id='spy-live-time', style={'color': '#666', 'fontSize': '0.65em', 'marginTop': '2px'})
            ], className='stats-card', style={'background': 'rgba(255,165,0,0.08)', 'borderRadius': '12px', 'padding': '15px', 'textAlign': 'center', 'flex': '1', 'minWidth': '100px', 'border': '1px solid rgba(255,165,0,0.2)'}),
            
            # Today's P&L
            html.Div([
                html.Div("TODAY", style={'color': '#888', 'fontSize': '0.75em', 'marginBottom': '3px'}),
                html.Div(id='today-pnl-value', className='stats-card-value', style={'fontSize': '1.8em', 'fontWeight': 'bold'}),
                html.Div(id='today-pnl-dots')
            ], className='stats-card', style={'background': 'rgba(255,255,255,0.08)', 'borderRadius': '12px', 'padding': '15px', 'textAlign': 'center', 'flex': '1', 'minWidth': '100px'}),
            
            # Total P&L
            html.Div([
                html.Div("TOTAL P&L", style={'color': '#888', 'fontSize': '0.75em', 'marginBottom': '3px'}),
                html.Div(id='total-pnl-value', className='stats-card-value', style={'fontSize': '1.8em', 'fontWeight': 'bold'})
            ], className='stats-card', style={'background': 'rgba(255,255,255,0.08)', 'borderRadius': '12px', 'padding': '15px', 'textAlign': 'center', 'flex': '1', 'minWidth': '100px'}),
            
            # Current Capital
            html.Div([
                html.Div("IBKR BALANCE", style={'color': '#888', 'fontSize': '0.75em', 'marginBottom': '3px'}),
                html.Div(id='current-capital-value', className='stats-card-value', style={'fontSize': '1.8em', 'fontWeight': 'bold', 'color': '#00d9ff'}),
                html.Div(id='broker-balance-time', style={'color': '#666', 'fontSize': '0.65em', 'marginTop': '2px'})
            ], className='stats-card', style={'background': 'rgba(255,255,255,0.08)', 'borderRadius': '12px', 'padding': '15px', 'textAlign': 'center', 'flex': '1', 'minWidth': '100px'}),
            
            # Total Return
            html.Div([
                html.Div("RETURN", style={'color': '#888', 'fontSize': '0.75em', 'marginBottom': '3px'}),
                html.Div(id='total-return-value', className='stats-card-value', style={'fontSize': '1.8em', 'fontWeight': 'bold'})
            ], className='stats-card', style={'background': 'rgba(255,255,255,0.08)', 'borderRadius': '12px', 'padding': '15px', 'textAlign': 'center', 'flex': '1', 'minWidth': '100px'}),
            
            # Win Rate
            html.Div([
                html.Div("WIN RATE", style={'color': '#888', 'fontSize': '0.75em', 'marginBottom': '3px'}),
                html.Div(id='win-rate-value', className='stats-card-value', style={'fontSize': '1.8em', 'fontWeight': 'bold', 'color': '#00d9ff'})
            ], className='stats-card', style={'background': 'rgba(255,255,255,0.08)', 'borderRadius': '12px', 'padding': '15px', 'textAlign': 'center', 'flex': '1', 'minWidth': '100px'}),
            
            # Max Drawdown
            html.Div([
                html.Div("DRAWDOWN", style={'color': '#888', 'fontSize': '0.75em', 'marginBottom': '3px'}),
                html.Div(id='max-drawdown-value', className='stats-card-value', style={'fontSize': '1.8em', 'fontWeight': 'bold', 'color': '#ff4757'})
            ], className='stats-card', style={'background': 'rgba(255,255,255,0.08)', 'borderRadius': '12px', 'padding': '15px', 'textAlign': 'center', 'flex': '1', 'minWidth': '100px'}),
            
        ], className='stats-row', style={'display': 'flex', 'gap': '10px', 'marginBottom': '15px', 'flexWrap': 'wrap'}),
        
        # Charts Row
        html.Div([
            html.Div([
                dcc.Graph(id='pnl-chart', config={'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'displaylogo': False}, style={'height': '350px'})
            ], className='chart-container', style={'flex': '1', 'background': 'rgba(255,255,255,0.05)', 'borderRadius': '12px', 'padding': '8px', 'minWidth': '400px'}),
        ], className='charts-row', style={'display': 'flex', 'gap': '10px', 'marginBottom': '15px', 'flexWrap': 'wrap'}),
        
        # IBKR Account Positions
        html.Div([
            html.Div([
                html.H2("IBKR Account Positions", style={
                    'color': '#eee',
                    'fontSize': '1.1em',
                    'marginBottom': '0px',
                    'borderLeft': '3px solid #ffa500',
                    'paddingLeft': '10px',
                    'display': 'inline-block'
                }),
                html.Span(id='positions-update-time', style={'color': '#666', 'fontSize': '0.7em', 'marginLeft': '15px'}),
            ], style={'marginBottom': '10px'}),
            dash_table.DataTable(
                id='positions-table',
                columns=[{'name': col, 'id': col} for col in ['Symbol', 'Type', 'Qty', 'Avg Cost', 'Price', 'Mkt Value', 'P&L']],
                data=[],
                style_header={
                    'backgroundColor': 'rgba(255,165,0,0.2)',
                    'color': '#eee',
                    'fontWeight': 'bold',
                    'border': 'none',
                    'borderBottom': '2px solid rgba(255,165,0,0.3)',
                    'fontSize': '0.8em',
                    'padding': '8px 5px'
                },
                style_cell={
                    'backgroundColor': 'transparent',
                    'color': '#eee',
                    'border': 'none',
                    'borderBottom': '1px solid rgba(255,255,255,0.1)',
                    'padding': '8px 5px',
                    'textAlign': 'left',
                    'fontSize': '0.8em',
                    'minWidth': '60px',
                    'maxWidth': '180px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis'
                },
                style_data_conditional=[
                    {'if': {'filter_query': '{P&L} contains "+"'}, 'color': '#00ff88', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{P&L} contains "-"'}, 'color': '#ff4757', 'fontWeight': 'bold'},
                ],
                style_table={'overflowX': 'auto'}
            )
        ], className='section-container', style={'background': 'rgba(255,255,255,0.05)', 'borderRadius': '12px', 'padding': '15px', 'marginBottom': '15px'}),
        
        # Trades Table - All Historical
        html.Div([
            html.H2("All Historical Trades", style={
                'color': '#eee',
                'fontSize': '1.1em',
                'marginBottom': '10px',
                'borderLeft': '3px solid #00d9ff',
                'paddingLeft': '10px'
            }),
            dash_table.DataTable(
                id='trades-table',
                columns=[{'name': col, 'id': col} for col in ['Date', 'Symbol', 'Type', 'Qty', 'Entry', 'Exit', 'P&L', 'Time', 'Status']],
                data=[],
                page_size=20,
                page_action='native',
                style_header={
                    'backgroundColor': 'rgba(0,217,255,0.2)',
                    'color': '#eee',
                    'fontWeight': 'bold',
                    'border': 'none',
                    'borderBottom': '2px solid rgba(0,217,255,0.3)',
                    'fontSize': '0.8em',
                    'padding': '8px 5px'
                },
                style_cell={
                    'backgroundColor': 'transparent',
                    'color': '#eee',
                    'border': 'none',
                    'borderBottom': '1px solid rgba(255,255,255,0.1)',
                    'padding': '8px 5px',
                    'textAlign': 'left',
                    'fontSize': '0.8em',
                    'minWidth': '50px',
                    'maxWidth': '150px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis'
                },
                style_data_conditional=[
                    {'if': {'filter_query': '{P&L} contains "+"'}, 'color': '#00ff88', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{P&L} contains "-"'}, 'color': '#ff4757', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Status} = "OPEN"'}, 'color': '#ffa500'},
                ],
                style_table={'overflowX': 'auto'}
            )
        ], className='section-container', style={'background': 'rgba(255,255,255,0.05)', 'borderRadius': '12px', 'padding': '15px', 'marginBottom': '15px'}),
        
        # Backend Terminal Output
        html.Div([
            html.H2("Backend Logs", style={
                'color': '#eee',
                'fontSize': '1.1em',
                'marginBottom': '10px',
                'borderLeft': '3px solid #ffa500',
                'paddingLeft': '10px'
            }),
            html.Div([
                html.Div(id='log-output')
            ], className='log-container', style={
                'background': 'rgba(0,0,0,0.3)',
                'borderRadius': '8px',
                'maxHeight': '300px',
                'overflowY': 'auto',
                'overflowX': 'hidden',
                'padding': '8px'
            })
        ], className='section-container', style={'background': 'rgba(255,255,255,0.05)', 'borderRadius': '12px', 'padding': '15px', 'marginBottom': '15px'}),
        
        # Timestamp
        html.Div(id='timestamp-display', 
                style={'textAlign': 'center', 'color': '#666', 'fontSize': '0.75em', 'padding': '10px'}),
        
        # Auto-refresh interval (3 seconds for real-time)
        dcc.Interval(id='interval-component', interval=3*1000, n_intervals=0),
        
        # Store for restart signal
        dcc.Store(id='restart-signal', data=0),

        # Store for IBKR status
        dcc.Store(id='ibkr-status-store', data='')
        
    ], className='main-container', style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '10px', 'color': '#eee'})


app.layout = serve_layout


# ============================================================
# CALLBACKS FOR REAL-TIME UPDATES
# ============================================================

@app.callback(
    [Output('spy-live-price', 'children'),
     Output('spy-live-change', 'children'),
     Output('spy-live-change', 'style'),
     Output('spy-live-time', 'children'),
     Output('today-pnl-value', 'children'),
     Output('today-pnl-value', 'style'),
     Output('today-pnl-dots', 'children'),
     Output('total-pnl-value', 'children'),
     Output('total-pnl-value', 'style'),
     Output('current-capital-value', 'children'),
     Output('broker-balance-time', 'children'),
     Output('total-return-value', 'children'),
     Output('total-return-value', 'style'),
     Output('win-rate-value', 'children'),
     Output('max-drawdown-value', 'children'),
     Output('timestamp-display', 'children'),
     Output('status-bar', 'children'),
     Output('log-output', 'children'),
     Output('trades-table', 'data'),
     Output('positions-table', 'data'),
     Output('positions-update-time', 'children'),
     Output('pnl-chart', 'figure'),
     Output('ibkr-status-store', 'data'),
     Output('ibkr-gateway-info', 'children'),
     Output('questrade-status-info', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    """Update all dashboard components in real-time."""
    state = get_state()
    df = get_trades_df()
    today_df = get_today_trades(df)

    # Fetch SPY real-time price from Questrade
    spy_data = get_spy_live_price()
    spy_price = spy_data.get("price")
    spy_change = spy_data.get("change")
    spy_change_pct = spy_data.get("change_pct")
    spy_time = spy_data.get("time")

    if spy_price is not None:
        spy_price_text = f"${spy_price:.2f}"
        spy_change = spy_change or 0
        spy_change_pct = spy_change_pct or 0
        color = '#00ff88' if spy_change >= 0 else '#ff4757'
        spy_change_text = f"{'+'if spy_change >= 0 else ''}{spy_change:.2f} ({spy_change_pct:+.2f}%)"
        spy_change_style = {'fontSize': '0.8em', 'marginTop': '2px', 'color': color}
        spy_time_text = f"Questrade {spy_time}" if spy_time else ""
    else:
        spy_price_text = "—"
        spy_change_text = "No data"
        spy_change_style = {'fontSize': '0.8em', 'marginTop': '2px', 'color': '#666'}
        spy_time_text = "Questrade disconnected"

    # Filter out phantom trades (closed with no real exit and $0 PnL)
    if not today_df.empty:
        phantom_mask = (today_df['status'] == 'closed') & (today_df['pnl'].fillna(0) == 0) & (today_df['exit_price'].fillna(0) == 0)
        today_df = today_df[~phantom_mask]
    
    # Calculate metrics
    today_pnl = today_df[today_df['status'] == 'closed']['pnl'].sum() if not today_df.empty else 0
    today_wins = len(today_df[(today_df['status'] == 'closed') & (today_df['pnl'] > 0)]) if not today_df.empty else 0
    today_losses = len(today_df[(today_df['status'] == 'closed') & (today_df['pnl'] <= 0)]) if not today_df.empty else 0
    
    initial_cap = state.get('initial_capital', _default_capital())
    current_cap = state.get('current_capital', _default_capital())
    total_pnl = state.get('total_pnl', 0)
    total_return = ((current_cap - initial_cap) / initial_cap) * 100 if initial_cap > 0 else 0
    
    # Today P&L
    today_pnl_text = f"{'+'if today_pnl >= 0 else ''}${today_pnl:.0f}"
    today_pnl_style = {'fontSize': '2em', 'fontWeight': 'bold', 'color': '#00ff88' if today_pnl >= 0 else '#ff4757'}
    today_dots = [
        html.Span("●" * today_wins, style={'color': '#00ff88', 'marginRight': '5px'}),
        html.Span("●" * today_losses, style={'color': '#ff4757'})
    ]
    
    # Total P&L
    total_pnl_text = f"{'+'if total_pnl >= 0 else ''}${total_pnl:.0f}"
    total_pnl_style = {'fontSize': '2em', 'fontWeight': 'bold', 'color': '#00ff88' if total_pnl >= 0 else '#ff4757'}
    
    # Current Capital — prefer real broker NLV, fall back to tracked capital
    broker_nlv = state.get('broker_nlv')
    broker_balance_time = state.get('broker_balance_time')
    if broker_nlv is not None:
        display_cap = broker_nlv
        current_cap_text = f"${broker_nlv:,.0f}"
        if broker_balance_time:
            try:
                bt = datetime.fromisoformat(broker_balance_time)
                broker_time_text = f"Updated {bt.strftime('%H:%M:%S')}"
            except Exception:
                broker_time_text = ""
        else:
            broker_time_text = ""
    else:
        display_cap = current_cap
        current_cap_text = f"${current_cap:,.0f}"
        broker_time_text = "No broker data"
    
    # Total Return
    total_return_text = f"{total_return:+.1f}%"
    total_return_style = {'fontSize': '2em', 'fontWeight': 'bold', 'color': '#00ff88' if total_return >= 0 else '#ff4757'}
    
    # Win Rate
    total_trades = state.get('total_trades', 0)
    total_wins = state.get('total_wins', 0)
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    win_rate_text = f"{total_wins}/{total_trades} ({win_rate:.0f}%)"
    
    # Max Drawdown
    max_dd_text = f"{state.get('max_drawdown', 0)*100:.1f}%"
    
    # Timestamp with ET and PST
    now_utc = datetime.now(TZ_UTC)
    now_et = now_utc.astimezone(TZ_ET)
    now_pst = now_utc.astimezone(TZ_PST)
    timestamp = f"ET: {now_et.strftime('%H:%M:%S')} | PST: {now_pst.strftime('%H:%M:%S')} | {now_et.strftime('%Y-%m-%d')}"
    
    # System Status Bar
    sys_status = get_system_status()
    def get_status_color(status_text):
        if status_text in ['Valid', 'Running', 'OK', 'Connected', 'live', 'starting'] or (isinstance(status_text, str) and status_text.startswith('OK')):
            return '#00ff88'
        elif status_text in ['Expired', 'Error', 'Missing', 'Stale', 'Unreachable', 'error', 'stopped']:
            return '#ff4757'
        elif status_text in ['Slow', 'Unknown', 'sleep', 'unknown']:
            return '#ffa500'
        return '#00d9ff'
    
    # Build engine status text with extra info
    engine_status = state.get('engine_status', 'unknown')
    engine_info = ""
    if engine_status == 'sleep':
        opens_in = state.get('engine_opens_in', '')
        if opens_in:
            engine_info = f" (opens in {opens_in})"
    elif engine_status == 'live':
        mode = state.get('engine_mode', '')
        strategy = state.get('engine_strategy', '')
        if mode or strategy:
            engine_info = f" ({mode}/{strategy})"
    elif engine_status == 'error':
        engine_info = f" ({state.get('engine_error_message', 'unknown')[:30]})"
    
    # Get strategy name from config
    strategy_config = load_strategy_config()
    tc = strategy_config.get('trade_config', {})
    active_strategy = tc.get('strategy', 'unknown').upper()
    pt_pct = tc.get('profit_target_pct', 0)
    sl_pct = tc.get('stop_loss_pct', 0)
    call_pt = tc.get('call_profit_target_pct')
    put_pt = tc.get('put_profit_target_pct')
    call_sl = tc.get('call_stop_loss_pct')
    put_sl = tc.get('put_stop_loss_pct')
    regime_on = tc.get('use_regime_detection', False)
    
    # Build exit info string
    if call_pt and put_pt and (call_pt != pt_pct or put_pt != pt_pct):
        exit_str = f"C:{int(call_pt*100)}/{int((call_sl or sl_pct)*100)} P:{int(put_pt*100)}/{int((put_sl or sl_pct)*100)}"
    else:
        exit_str = f"PT{int(pt_pct*100)}%/SL{int(sl_pct*100)}%"
    
    status_bar = html.Div([
        html.Span("ENGINE", style={'color': '#888', 'fontSize': '0.75em', 'marginRight': '10px'}),
        html.Span([
            html.Span(engine_status.upper(), style={'color': get_status_color(engine_status), 'fontWeight': 'bold'}),
            html.Span(engine_info, style={'color': '#888', 'fontSize': '0.85em'})
        ], style={'marginRight': '15px'}),
        html.Span([
            html.Span("Strategy: ", style={'color': '#888', 'fontSize': '0.85em'}),
            html.Span(f"{active_strategy} ({exit_str})", 
                      style={'color': '#00d9ff', 'fontWeight': 'bold', 'fontSize': '0.85em'})
        ], style={'marginRight': '15px'}),
        html.Span([
            html.Span("IBKR: ", style={'color': '#888', 'fontSize': '0.85em'}),
            html.Span(sys_status['ibkr_status'], style={'color': get_status_color(sys_status['ibkr_status']), 'fontWeight': 'bold', 'fontSize': '0.85em'})
        ], style={'marginRight': '15px'}),
        html.Span([
            html.Span("QT: ", style={'color': '#888', 'fontSize': '0.85em'}),
            html.Span(sys_status['questrade_status'], style={'color': '#00ff88' if sys_status['questrade_status'] == 'Connected' else '#ff4757', 'fontWeight': 'bold', 'fontSize': '0.85em'})
        ], style={'marginRight': '15px'}),
        html.Span([
            html.Span("Regime: ", style={'color': '#888', 'fontSize': '0.85em'}),
            html.Span("ON" if regime_on else "OFF", style={'color': '#00d9ff' if regime_on else '#666', 'fontWeight': 'bold', 'fontSize': '0.85em'})
        ], style={'marginRight': '15px'}),
        html.Span([
            html.Span("DB: ", style={'color': '#888', 'fontSize': '0.85em'}),
            html.Span(sys_status['db_status'].split('(')[0].strip(), style={'color': get_status_color(sys_status['db_status']), 'fontWeight': 'bold', 'fontSize': '0.85em'})
        ], style={'marginRight': '15px'}),
        html.Span([
            html.Span("Market: ", style={'color': '#888', 'fontSize': '0.85em'}),
            html.Span("OPEN" if (9 <= now_et.hour < 16 and now_et.weekday() < 5) else "CLOSED", 
                     style={'color': '#00ff88' if (9 <= now_et.hour < 16 and now_et.weekday() < 5) else '#ff4757', 
                            'fontWeight': 'bold', 'fontSize': '0.85em'})
        ])
    ], className='status-bar', style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '8px'})
    
    # Log Output
    log_lines = get_recent_logs(50)
    log_output = [
        html.Span(
            log_line,
            style={
                'display': 'block',
                'padding': '4px 10px',
                'borderBottom': '1px solid rgba(255,255,255,0.05)',
                'color': '#00ff88' if any(x in log_line for x in ['ENTRY', 'EXIT', 'Signal', 'Got']) else
                        '#ff4757' if any(x in log_line for x in ['ERROR', 'Exception', 'STOP LOSS', 'REJECTED']) else
                        '#ffa500' if any(x in log_line for x in ['WARNING', '[DASHBOARD]']) else
                        '#00d9ff' if any(x in log_line for x in ['Successfully', 'token', 'authenticated', '[ENGINE]']) else
                        '#888' if '[SYSTEM]' in log_line else '#aaa',
                'fontSize': '0.85em',
                'fontFamily': 'Consolas, Monaco, monospace',
                'whiteSpace': 'nowrap',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis'
            }
        ) for log_line in log_lines
    ]
    
    # Prepare table data — all historical trades
    table_data = []
    if not df.empty:
        for _, t in df.iterrows():
            pnl = t.get('pnl') or 0
            pnl_pct = t.get('pnl_percent') or 0
            status = t.get('status', 'open')
            symbol = t.get('symbol', '')
            # Get type from symbol if option_type is empty
            opt_type = str(t.get('option_type', '')).upper()
            if not opt_type:
                opt_type = get_option_type_from_symbol(symbol)
            if status == 'open':
                pnl_display = 'OPEN'
            elif pnl > 0:
                pnl_display = f'+${pnl:.0f} (+{pnl_pct:.1f}%)'
            else:
                pnl_display = f'-${abs(pnl):.0f} ({pnl_pct:.1f}%)'
            entry_time = t.get('entry_time', '')
            trade_date = str(entry_time)[:10] if entry_time else ''
            trade_time = str(entry_time)[11:16] if entry_time else ''
            table_data.append({
                'Date': trade_date,
                'Symbol': symbol[-15:],  # Truncate for mobile
                'Type': opt_type[:4],
                'Qty': t.get('quantity', 0),
                'Entry': f"${t.get('entry_price', 0):.2f}",
                'Exit': f"${t.get('exit_price', 0):.2f}" if t.get('exit_price') else '-',
                'P&L': pnl_display,
                'Time': trade_time,
                'Status': status.upper()[:4]
            })
    
    # Charts - P&L bars
    pnl_fig = create_pnl_bars(state)

    # IBKR status for the store
    ibkr_status = sys_status['ibkr_status']
    ibkr_store = f"{ibkr_status}|{sys_status.get('ibkr_host', '')}:{sys_status.get('ibkr_port', '')}"

    # IBKR Gateway info for config panel
    ibkr_color = '#00ff88' if ibkr_status == 'Connected' else '#ff4757'
    ibkr_gateway_info = html.Div([
        html.Span(f"Status: ", style={'color': '#888'}),
        html.Span(ibkr_status, style={'color': ibkr_color, 'fontWeight': 'bold'}),
        html.Span(f"  |  {sys_status.get('ibkr_host', '')}:{sys_status.get('ibkr_port', '')}", style={'color': '#aaa', 'marginLeft': '10px'}),
    ])

    # IBKR Account Positions (from DB) — skip zero-qty expired positions
    ibkr_positions, positions_time = get_ibkr_positions()
    positions_data = []
    active_positions = [p for p in ibkr_positions if p.get('quantity', 0) != 0]
    for p in active_positions:
        pnl = p.get('unrealized_pnl') or 0
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        is_opt = p.get('is_option', 0)
        opt_type = (p.get('option_type') or ('OPT' if is_opt else 'STK')).upper()
        positions_data.append({
            'Symbol': p.get('symbol', ''),
            'Type': opt_type[:4],
            'Qty': p.get('quantity', 0),
            'Avg Cost': f"${p.get('avg_cost', 0):.4f}",
            'Price': f"${p.get('current_price', 0):.4f}",
            'Mkt Value': f"${p.get('market_value', 0):,.2f}",
            'P&L': pnl_str,
        })
    if positions_time:
        try:
            pt = datetime.fromisoformat(positions_time)
            positions_time_text = f"Updated {pt.strftime('%H:%M:%S')}  |  {len(ibkr_positions)} positions"
        except Exception:
            positions_time_text = f"{len(ibkr_positions)} positions"
    else:
        positions_time_text = "Waiting for engine sync..."

    # Questrade status for config panel
    qt_status = sys_status['questrade_status']
    qt_detail = sys_status.get('questrade_detail', '')
    qt_color = '#00ff88' if qt_status == 'Connected' else '#ffa500' if qt_status == 'Token Expired' else '#ff4757'
    questrade_info = html.Div([
        html.Span(f"Status: ", style={'color': '#888'}),
        html.Span(qt_status, style={'color': qt_color, 'fontWeight': 'bold'}),
        html.Span(f"  |  {qt_detail}", style={'color': '#aaa', 'marginLeft': '10px'}) if qt_detail else None,
    ])

    return (spy_price_text, spy_change_text, spy_change_style, spy_time_text,
            today_pnl_text, today_pnl_style, today_dots, total_pnl_text, total_pnl_style,
            current_cap_text, broker_time_text, total_return_text, total_return_style, win_rate_text, max_dd_text,
            timestamp, status_bar, log_output, table_data, positions_data, positions_time_text,
            pnl_fig, ibkr_store, ibkr_gateway_info, questrade_info)


@app.callback(
    Output('restart-status', 'children'),
    [Input('restart-btn', 'n_clicks')],
    prevent_initial_call=True
)
def restart_engine(n_clicks):
    """Restart the trading engine."""
    if n_clicks:
        try:
            # Try systemd restart for VM deployment
            result = subprocess.run(
                "/usr/bin/sudo /usr/bin/systemctl restart trading-engine",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return html.Span("Engine restarted!", style={'color': '#00ff88'})
        except Exception:
            pass

        # Docker: exit this process — Docker restart=unless-stopped brings it back
        if os.path.exists("/.dockerenv"):
            _deferred_exit(1.5)
            return html.Span("Restarting container…", style={'color': '#00ff88'})

        # Local fallback: write restart signal file
        try:
            restart_file = PROJECT_DIR / "logs" / "restart_signal.txt"
            restart_file.parent.mkdir(exist_ok=True)
            with open(restart_file, 'w') as f:
                f.write(f"RESTART {datetime.now().isoformat()}")
            return html.Span("Restart signal sent!", style={'color': '#ffa500'})
        except Exception as e:
            return html.Span(f"Error: {str(e)}", style={'color': '#ff4757'})
    return ""


@app.callback(
    [Output('qt-token-result', 'children'),
     Output('qt-token-input', 'value')],
    [Input('qt-token-submit-btn', 'n_clicks')],
    [State('qt-token-input', 'value')],
    prevent_initial_call=True
)
def refresh_questrade_token(n_clicks, token_value):
    """Authenticate with a new Questrade refresh token."""
    if not n_clicks or not token_value or not token_value.strip():
        raise PreventUpdate

    token = token_value.strip()

    # Basic validation — Questrade tokens are alphanumeric with hyphens/underscores
    if len(token) < 10 or len(token) > 100:
        return html.Span("Invalid token length", style={'color': '#ff4757'}), token

    import time as _time

    try:
        from clients.questrade_client import QuestradeClient
        import os as _os

        # Determine token file path (same logic as create_questrade_client)
        token_file_env = _os.environ.get("QUESTRADE_TOKEN_FILE")
        token_file = Path(token_file_env) if token_file_env else None

        # Delete old token file so the client authenticates fresh
        target = token_file or QuestradeClient.TOKEN_FILE
        if target.exists():
            target.unlink()

        # Retry with backoff for transient 500 errors from Questrade
        last_err = None
        for attempt in range(3):
            try:
                client = QuestradeClient(
                    refresh_token=token,
                    token_file=token_file,
                    practice_mode=False,
                    auto_refresh=False,
                )
                # Validate by fetching accounts
                accounts = client.get_accounts()

                # Reset the dashboard's cached QT client so it picks up the new token
                global _qt_client, _qt_init_attempted
                _qt_client = client
                _qt_init_attempted = True

                acct_count = len(accounts) if accounts else 0
                return (
                    html.Span(f"Connected! {acct_count} account(s) found. Token saved.",
                               style={'color': '#00ff88'}),
                    "",  # Clear the input
                )
            except Exception as e:
                last_err = e
                err_str = str(e)
                # Only retry on 500 server errors (transient)
                if "500" in err_str and attempt < 2:
                    _time.sleep(2 * (attempt + 1))
                    # Re-delete token file in case partial state was written
                    if target.exists():
                        target.unlink()
                    continue
                break

        err = str(last_err) if last_err else "Unknown error"
        # Redact token from error message
        if token in err:
            err = err.replace(token, "<redacted>")
        # Add helpful hint for common errors
        if "500" in err:
            err += " — Token may have been consumed already. Generate a NEW token at Questrade App Hub."
        elif "400" in err:
            err += " — Token is expired or invalid. Generate a NEW token at Questrade App Hub."
        return html.Span(f"Failed: {err[:200]}", style={'color': '#ff4757'}), token
    except Exception as e:
        err = str(e)
        if token in err:
            err = err.replace(token, "<redacted>")
        return html.Span(f"Failed: {err[:200]}", style={'color': '#ff4757'}), token


@app.callback(
    Output('config-panel', 'style'),
    [Input('config-toggle-btn', 'n_clicks')],
    [State('config-panel', 'style')],
    prevent_initial_call=True
)
def toggle_config_panel(n_clicks, current_style):
    """Toggle visibility of config panel."""
    if n_clicks:
        if current_style.get('display') == 'none':
            return {**current_style, 'display': 'block'}
        else:
            return {**current_style, 'display': 'none'}
    return current_style


@app.callback(
    [Output('cfg-stop-after-first-loss', 'value'),
     Output('cfg-kelly-fraction', 'value'),
     Output('cfg-max-position-value', 'value'),
     Output('cfg-profit-target', 'value'),
     Output('cfg-stop-loss', 'value'),
     Output('cfg-call-pt', 'value'),
     Output('cfg-call-sl', 'value'),
     Output('cfg-put-pt', 'value'),
     Output('cfg-put-sl', 'value'),
     Output('cfg-max-hold', 'value'),
     Output('cfg-regime-detection', 'value'),
     Output('cfg-min-option-price', 'value'),
     Output('cfg-max-option-price', 'value'),
     Output('cfg-max-consec-losses', 'value'),
     Output('cfg-max-daily-loss', 'value')],
    [Input('config-toggle-btn', 'n_clicks')],
    prevent_initial_call=True
)
def load_config_values(n_clicks):
    """Load current config values when panel opens."""
    config = load_strategy_config()
    trade_cfg = config.get('trade_config', {})
    risk_cfg = config.get('risk_config', {})
    
    return (
        risk_cfg.get('max_daily_losses', 2),
        risk_cfg.get('kelly_fraction', 0.20),
        risk_cfg.get('max_position_value', 5000),
        trade_cfg.get('profit_target_pct', 0.50),
        trade_cfg.get('stop_loss_pct', 0.35),
        trade_cfg.get('call_profit_target_pct'),
        trade_cfg.get('call_stop_loss_pct'),
        trade_cfg.get('put_profit_target_pct'),
        trade_cfg.get('put_stop_loss_pct'),
        trade_cfg.get('max_hold_minutes', 80),
        trade_cfg.get('use_regime_detection', False),
        trade_cfg.get('min_option_price', 0.50),
        trade_cfg.get('max_option_price', 2.00),
        risk_cfg.get('max_consecutive_losses', 3),
        risk_cfg.get('max_daily_loss_pct', 0.008) * 100  # Convert 0.008 -> 0.8 for display
    )


@app.callback(
    Output('config-save-status', 'children'),
    [Input('save-config-btn', 'n_clicks')],
    [State('cfg-stop-after-first-loss', 'value'),
     State('cfg-kelly-fraction', 'value'),
     State('cfg-max-position-value', 'value'),
     State('cfg-profit-target', 'value'),
     State('cfg-stop-loss', 'value'),
     State('cfg-call-pt', 'value'),
     State('cfg-call-sl', 'value'),
     State('cfg-put-pt', 'value'),
     State('cfg-put-sl', 'value'),
     State('cfg-max-hold', 'value'),
     State('cfg-regime-detection', 'value'),
     State('cfg-min-option-price', 'value'),
     State('cfg-max-option-price', 'value'),
     State('cfg-max-consec-losses', 'value'),
     State('cfg-max-daily-loss', 'value')],
    prevent_initial_call=True
)
def save_config_and_restart(n_clicks, max_daily_losses, kelly_fraction, max_position_value,
                            profit_target, stop_loss, call_pt, call_sl, put_pt, put_sl,
                            max_hold, regime_detection, min_option_price, max_option_price,
                            max_consec_losses, max_daily_loss):
    """Save configuration and restart trading engine."""
    if not n_clicks:
        return ""
    
    try:
        # Prepare config updates
        trade_config = {
            'profit_target_pct': profit_target,
            'stop_loss_pct': stop_loss,
            'call_profit_target_pct': call_pt,
            'call_stop_loss_pct': call_sl,
            'put_profit_target_pct': put_pt,
            'put_stop_loss_pct': put_sl,
            'max_hold_minutes': max_hold,
            'use_regime_detection': regime_detection,
            'min_option_price': min_option_price,
            'max_option_price': max_option_price
        }
        risk_config = {
            'max_daily_losses': max_daily_losses,
            'kelly_fraction': kelly_fraction,
            'max_position_value': max_position_value,
            'max_consecutive_losses': max_consec_losses,
            'max_daily_loss_pct': max_daily_loss / 100 if max_daily_loss else 0.008  # Convert 0.8 -> 0.008 for storage
        }
        
        # Save config
        if save_strategy_config(trade_config, risk_config):
            # Restart engine using shell=True for proper sudo execution
            result = subprocess.run(
                "/usr/bin/sudo /usr/bin/systemctl restart trading-engine",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return html.Span("✅ Config saved & engine restarted!", style={'color': '#00ff88'})
            else:
                return html.Span(f"⚠️ Config saved, restart: {result.stderr or 'failed'}", style={'color': '#ffa500'})
        else:
            return html.Span("❌ Failed to save config", style={'color': '#ff4757'})
    except Exception as e:
        return html.Span(f"❌ Error: {str(e)}", style={'color': '#ff4757'})


def main():
    import argparse
    parser = argparse.ArgumentParser(description="0DTE Trading Dashboard")
    parser.add_argument("--port", type=int, default=None, help="Port number (default: from config)")
    args = parser.parse_args()
    port = args.port if args.port is not None else _default_dashboard_port()
    
    print(f"\n{'='*60}")
    print("  0DTE TRADING DASHBOARD (Plotly Dash)")
    print(f"{'='*60}")
    print(f"  URL: http://localhost:{port}")
    print(f"{'='*60}")
    print("  Press Ctrl+C to stop\n")

    # Sync historical trades from all sources (DB files, CSVs) at dashboard startup
    try:
        from live.trade_sync import TradeSync
        from live.trade_database import TradeDatabase
        _db_path = PROJECT_DIR / "data" / "live_0dte_trades.db"
        _tdb = TradeDatabase(str(_db_path))
        syncer = TradeSync(_tdb)
        syncer.sync_all()  # No IBKR client from dashboard; syncs DBs + CSVs
        _tdb.conn.close()
        print("  Trade sync complete")
    except Exception as e:
        print(f"  Trade sync skipped: {e}")
    
    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
    
    app.run(debug=False, port=port, host='0.0.0.0')


if __name__ == "__main__":
    main()
