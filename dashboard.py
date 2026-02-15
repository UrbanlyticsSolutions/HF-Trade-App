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
import subprocess
import signal
from flask import request, Response

# Timezone definitions
TZ_ET = ZoneInfo("America/New_York")
TZ_PST = ZoneInfo("America/Los_Angeles")
TZ_UTC = ZoneInfo("UTC")

# ============================================================
# AUTHENTICATION
# ============================================================
DASHBOARD_USERNAME = "realericzhu@gmail.com"
DASHBOARD_PASSWORD = "admin"


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

# ============================================================
# STRATEGY CONFIGURATION
# ============================================================

def load_strategy_config():
    """Load strategy configuration from JSON file."""
    config_path = Path(__file__).parent / "config" / "strategy.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            return {}
    return {}


def save_strategy_config(trade_config, risk_config):
    """Save strategy configuration to JSON file."""
    config_path = Path(__file__).parent / "config" / "strategy.json"
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

def get_system_status():
    """Get system status information."""
    status = {
        "token_status": "Unknown",
        "token_expires": None,
        "token_age_minutes": None,
        "engine_status": "Unknown",
        "db_status": "Unknown",
        "last_quote_time": None,
        "errors": []
    }
    
    # Check token file
    token_path = Path(__file__).parent / "clients" / ".questrade_token.json"
    if token_path.exists():
        try:
            with open(token_path) as f:
                token_data = json.load(f)
            
            expires_at = token_data.get("expires_at", 0)
            now = datetime.now().timestamp()
            
            if expires_at > now:
                remaining_seconds = expires_at - now
                remaining_minutes = remaining_seconds / 60
                status["token_status"] = "Valid"
                status["token_expires"] = datetime.fromtimestamp(expires_at).strftime("%H:%M:%S")
                status["token_age_minutes"] = round(remaining_minutes, 1)
            else:
                status["token_status"] = "Expired"
                status["errors"].append("Token has expired - needs refresh")
        except Exception as e:
            status["token_status"] = "Error"
            status["errors"].append(f"Token file error: {str(e)}")
    else:
        status["token_status"] = "Missing"
        status["errors"].append("No token file found")
    
    # Check database
    db_path = Path(__file__).parent / "data" / "live_0dte_trades.db"
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
    log_file = Path(__file__).parent / "logs" / f"live_0dte_{date.today().strftime('%Y%m%d')}.log"
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
        terminal_log = Path(__file__).parent / "logs" / "terminal_output.log"
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
        trading_log = Path(__file__).parent / "logs" / f"live_0dte_{date.today().strftime('%Y%m%d')}.log"
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
    db_path = Path(__file__).parent / "data" / "live_0dte_trades.db"
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
    """Load trading state from JSON file, synced with DB stats."""
    state_path = Path(__file__).parent / "trading_state.json"
    state = {
        "initial_capital": 10000,
        "current_capital": 10000,
        "total_pnl": 0,
        "total_trades": 0,
        "total_wins": 0,
        "total_losses": 0,
        "max_drawdown": 0,
        "equity_curve": [],
        "engine_status": "unknown"
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
    
    return state


def get_trades_df():
    """Load trades from database into DataFrame."""
    db_path = Path(__file__).parent / "data" / "live_0dte_trades.db"
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


def create_equity_curve(state, initial_capital=10000):
    """Create equity curve chart from state JSON or DB."""
    equity_curve = state.get('equity_curve', [])
    
    # If no equity curve in state, build from DB
    if not equity_curve or len(equity_curve) < 2:
        db_path = Path(__file__).parent / "data" / "live_0dte_trades.db"
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
        db_path = Path(__file__).parent / "data" / "live_0dte_trades.db"
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
    
    # Calculate metrics
    today_pnl = today_df[today_df['status'] == 'closed']['pnl'].sum() if not today_df.empty else 0
    today_wins = len(today_df[(today_df['status'] == 'closed') & (today_df['pnl'] > 0)]) if not today_df.empty else 0
    today_losses = len(today_df[(today_df['status'] == 'closed') & (today_df['pnl'] <= 0)]) if not today_df.empty else 0
    
    initial_cap = state.get('initial_capital', 10000)
    current_cap = state.get('current_capital', 10000)
    total_pnl = state.get('total_pnl', 0)
    total_return = ((current_cap - initial_cap) / initial_cap) * 100 if initial_cap > 0 else 0
    
    # Prepare table data
    table_data = []
    if not today_df.empty:
        for _, t in today_df.iterrows():
            pnl = t.get('pnl') or 0
            pnl_pct = t.get('pnl_percent') or 0
            status = t.get('status', 'open')
            
            if status == 'open':
                pnl_display = 'OPEN'
            elif pnl > 0:
                pnl_display = f'+${pnl:.0f} (+{pnl_pct:.1f}%)'
            else:
                pnl_display = f'-${abs(pnl):.0f} ({pnl_pct:.1f}%)'
            
            table_data.append({
                'Symbol': t.get('symbol', '')[-15:],  # Truncate for mobile
                'Type': str(t.get('option_type', '')).upper()[:4],
                'Qty': t.get('quantity', 0),
                'Entry': f"${t.get('entry_price', 0):.2f}",
                'Exit': f"${t.get('exit_price', 0):.2f}" if t.get('exit_price') else '-',
                'P&L': pnl_display,
                'Time': str(t.get('entry_time', ''))[11:16],  # Just HH:MM
                'Status': status.upper()[:4]
            })
    
    # Get system status
    sys_status = get_system_status()
    
    # Status colors
    def get_status_color(status_text):
        if status_text in ['Valid', 'Running', 'OK'] or status_text.startswith('OK'):
            return '#00ff88'
        elif status_text in ['Expired', 'Error', 'Missing', 'Stale']:
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
                        html.Label("Profit Target (50% = 0.50):", style={'display': 'block', 'marginBottom': '5px'}),
                        dcc.Input(id='cfg-profit-target', type='number', min=0.05, max=2.0, step=0.01,
                                  style={'width': '100px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff'})
                    ], style={'marginBottom': '15px'}),
                    html.Div([
                        html.Label("Stop Loss (35% = 0.35):", style={'display': 'block', 'marginBottom': '5px'}),
                        dcc.Input(id='cfg-stop-loss', type='number', min=0.05, max=1.0, step=0.01,
                                  style={'width': '100px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff'})
                    ], style={'marginBottom': '15px'}),
                    html.Div([
                        html.Label("Max Hold (minutes):", style={'display': 'block', 'marginBottom': '5px'}),
                        dcc.Input(id='cfg-max-hold', type='number', min=1, max=120, step=5,
                                  style={'width': '100px', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #444', 'background': '#333', 'color': '#fff'})
                    ], style={'marginBottom': '15px'}),
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
                html.Div("CAPITAL", style={'color': '#888', 'fontSize': '0.75em', 'marginBottom': '3px'}),
                html.Div(id='current-capital-value', className='stats-card-value', style={'fontSize': '1.8em', 'fontWeight': 'bold', 'color': '#00d9ff'})
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
                dcc.Graph(id='equity-chart', config={'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'displaylogo': False}, style={'height': '300px'})
            ], className='chart-container', style={'flex': '2', 'background': 'rgba(255,255,255,0.05)', 'borderRadius': '12px', 'padding': '8px', 'minWidth': '280px'}),
            
            html.Div([
                dcc.Graph(id='pnl-chart', config={'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'displaylogo': False}, style={'height': '350px'})
            ], className='chart-container', style={'flex': '1', 'background': 'rgba(255,255,255,0.05)', 'borderRadius': '12px', 'padding': '8px', 'minWidth': '400px'}),
        ], className='charts-row', style={'display': 'flex', 'gap': '10px', 'marginBottom': '15px', 'flexWrap': 'wrap'}),
        
        # Trades Table
        html.Div([
            html.H2(f"Today's Trades ({date.today().isoformat()})", style={
                'color': '#eee',
                'fontSize': '1.1em',
                'marginBottom': '10px',
                'borderLeft': '3px solid #00d9ff',
                'paddingLeft': '10px'
            }),
            dash_table.DataTable(
                id='trades-table',
                columns=[{'name': col, 'id': col} for col in ['Symbol', 'Type', 'Qty', 'Entry', 'Exit', 'P&L', 'Time', 'Status']],
                data=[],
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
        dcc.Store(id='restart-signal', data=0)
        
    ], className='main-container', style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '10px', 'color': '#eee'})


app.layout = serve_layout


# ============================================================
# CALLBACKS FOR REAL-TIME UPDATES
# ============================================================

@app.callback(
    [Output('today-pnl-value', 'children'),
     Output('today-pnl-value', 'style'),
     Output('today-pnl-dots', 'children'),
     Output('total-pnl-value', 'children'),
     Output('total-pnl-value', 'style'),
     Output('current-capital-value', 'children'),
     Output('total-return-value', 'children'),
     Output('total-return-value', 'style'),
     Output('win-rate-value', 'children'),
     Output('max-drawdown-value', 'children'),
     Output('timestamp-display', 'children'),
     Output('status-bar', 'children'),
     Output('log-output', 'children'),
     Output('trades-table', 'data'),
     Output('equity-chart', 'figure'),
     Output('pnl-chart', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    """Update all dashboard components in real-time."""
    state = get_state()
    df = get_trades_df()
    today_df = get_today_trades(df)
    
    # Calculate metrics
    today_pnl = today_df[today_df['status'] == 'closed']['pnl'].sum() if not today_df.empty else 0
    today_wins = len(today_df[(today_df['status'] == 'closed') & (today_df['pnl'] > 0)]) if not today_df.empty else 0
    today_losses = len(today_df[(today_df['status'] == 'closed') & (today_df['pnl'] <= 0)]) if not today_df.empty else 0
    
    initial_cap = state.get('initial_capital', 10000)
    current_cap = state.get('current_capital', 10000)
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
    
    # Current Capital
    current_cap_text = f"${current_cap:,.0f}"
    
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
        if status_text in ['Valid', 'Running', 'OK', 'live', 'starting'] or (isinstance(status_text, str) and status_text.startswith('OK')):
            return '#00ff88'
        elif status_text in ['Expired', 'Error', 'Missing', 'Stale', 'error', 'stopped']:
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
    active_strategy = strategy_config.get('trade_config', {}).get('strategy', 'unknown').upper()
    pt_pct = strategy_config.get('trade_config', {}).get('profit_target_pct', 0)
    sl_pct = strategy_config.get('trade_config', {}).get('stop_loss_pct', 0)
    
    status_bar = html.Div([
        html.Span("ENGINE", style={'color': '#888', 'fontSize': '0.75em', 'marginRight': '10px'}),
        html.Span([
            html.Span(engine_status.upper(), style={'color': get_status_color(engine_status), 'fontWeight': 'bold'}),
            html.Span(engine_info, style={'color': '#888', 'fontSize': '0.85em'})
        ], style={'marginRight': '15px'}),
        html.Span([
            html.Span("Strategy: ", style={'color': '#888', 'fontSize': '0.85em'}),
            html.Span(f"{active_strategy} (PT{int(pt_pct*100)}%/SL{int(sl_pct*100)}%)", 
                      style={'color': '#00d9ff', 'fontWeight': 'bold', 'fontSize': '0.85em'})
        ], style={'marginRight': '15px'}),
        html.Span([
            html.Span("Token: ", style={'color': '#888', 'fontSize': '0.85em'}),
            html.Span(sys_status['token_status'], style={'color': get_status_color(sys_status['token_status']), 'fontWeight': 'bold', 'fontSize': '0.85em'})
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
    
    # Prepare table data
    table_data = []
    if not today_df.empty:
        for _, t in today_df.iterrows():
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
            table_data.append({
                'Symbol': symbol[-15:],  # Truncate for mobile
                'Type': opt_type[:4],
                'Qty': t.get('quantity', 0),
                'Entry': f"${t.get('entry_price', 0):.2f}",
                'Exit': f"${t.get('exit_price', 0):.2f}" if t.get('exit_price') else '-',
                'P&L': pnl_display,
                'Time': str(t.get('entry_time', ''))[11:16],  # Just HH:MM
                'Status': status.upper()[:4]
            })
    
    # Charts - use state for equity curve and P&L bars
    equity_fig = create_equity_curve(state, initial_cap)
    pnl_fig = create_pnl_bars(state)
    
    return (today_pnl_text, today_pnl_style, today_dots, total_pnl_text, total_pnl_style,
            current_cap_text, total_return_text, total_return_style, win_rate_text, max_dd_text,
            timestamp, status_bar, log_output, table_data, equity_fig, pnl_fig)


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
            else:
                # Fallback: write restart signal file for local use
                restart_file = Path(__file__).parent / "logs" / "restart_signal.txt"
                restart_file.parent.mkdir(exist_ok=True)
                with open(restart_file, 'w') as f:
                    f.write(f"RESTART {datetime.now().isoformat()}")
                return html.Span("Restart signal sent!", style={'color': '#ffa500'})
        except Exception as e:
            return html.Span(f"Error: {str(e)}", style={'color': '#ff4757'})
    return ""


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
     Output('cfg-max-hold', 'value'),
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
        risk_cfg.get('stop_after_first_loss', True),
        risk_cfg.get('kelly_fraction', 0.20),
        risk_cfg.get('max_position_value', 5000),
        trade_cfg.get('profit_target_pct', 0.50),
        trade_cfg.get('stop_loss_pct', 0.35),
        trade_cfg.get('max_hold_minutes', 80),
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
     State('cfg-max-hold', 'value'),
     State('cfg-min-option-price', 'value'),
     State('cfg-max-option-price', 'value'),
     State('cfg-max-consec-losses', 'value'),
     State('cfg-max-daily-loss', 'value')],
    prevent_initial_call=True
)
def save_config_and_restart(n_clicks, stop_after_first_loss, kelly_fraction, max_position_value,
                            profit_target, stop_loss, max_hold, min_option_price, max_option_price,
                            max_consec_losses, max_daily_loss):
    """Save configuration and restart trading engine."""
    if not n_clicks:
        return ""
    
    try:
        # Prepare config updates
        trade_config = {
            'profit_target_pct': profit_target,
            'stop_loss_pct': stop_loss,
            'max_hold_minutes': max_hold,
            'min_option_price': min_option_price,
            'max_option_price': max_option_price
        }
        risk_config = {
            'stop_after_first_loss': stop_after_first_loss,
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
    parser.add_argument("--port", type=int, default=8050, help="Port number (default: 8050)")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("  0DTE TRADING DASHBOARD (Plotly Dash)")
    print(f"{'='*60}")
    print(f"  URL: http://localhost:{args.port}")
    print(f"{'='*60}")
    print("  Press Ctrl+C to stop\n")
    
    import webbrowser
    webbrowser.open(f"http://localhost:{args.port}")
    
    app.run(debug=False, port=args.port, host='0.0.0.0')


if __name__ == "__main__":
    main()
