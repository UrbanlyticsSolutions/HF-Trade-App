"""
Health Report with Email Notification

Gathers trading system health metrics and sends email alerts via Gmail SMTP.

Usage:
    # One-shot report
    python -m live.health_report
    
    # Scheduled loop (sends every N minutes)
    python -m live.health_report --loop --interval 30
    
    # Print report to console only (no email)
    python -m live.health_report --dry-run

Environment Variables (add to .env):
    SMTP_EMAIL           - Gmail address (sender)
    SMTP_APP_PASSWORD    - Gmail App Password (16 chars)
    ALERT_EMAIL          - Destination email address
"""
import os
import sys
import json
import time
import sqlite3
import socket
import smtplib
import logging
import argparse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path(__file__).parent.parent
logger = logging.getLogger(__name__)

# ============================================================
# HEALTH METRICS COLLECTION
# ============================================================

def check_ibkr_gateway() -> Dict[str, Any]:
    """Check IBKR Gateway TCP connectivity."""
    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    port = int(os.environ.get("IBKR_PAPER_PORT", 7497))
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()
        connected = result == 0
        return {"status": "Connected" if connected else "DOWN", "host": host, "port": port}
    except Exception as e:
        return {"status": "ERROR", "host": host, "port": port, "error": str(e)}


def check_engine_status() -> Dict[str, Any]:
    """Check engine status from today's log file."""
    log_file = PROJECT_DIR / "logs" / f"live_0dte_{date.today().strftime('%Y%m%d')}.log"
    if not log_file.exists():
        return {"status": "No logs today", "last_activity": None, "errors": []}

    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return {"status": "Log read error", "last_activity": None, "errors": [str(e)]}

    if not lines:
        return {"status": "Empty log", "last_activity": None, "errors": []}

    # Find last timestamp
    last_log_time = None
    for line in reversed(lines[-50:]):
        line = line.strip()
        if line:
            try:
                last_log_time = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f")
                break
            except (ValueError, IndexError):
                continue

    # Collect recent errors
    recent_errors = []
    for line in lines[-100:]:
        if "ERROR" in line or "CRITICAL" in line or "Exception" in line:
            recent_errors.append(line.strip()[:120])

    age_seconds = None
    status = "Unknown"
    if last_log_time:
        age_seconds = (datetime.now() - last_log_time).total_seconds()
        if age_seconds < 60:
            status = "Running"
        elif age_seconds < 300:
            status = f"Slow ({int(age_seconds)}s)"
        else:
            status = f"STALE ({int(age_seconds / 60)}m)"

    return {
        "status": status,
        "last_activity": last_log_time.strftime("%H:%M:%S") if last_log_time else None,
        "errors": recent_errors[-5:],  # Last 5 errors
    }


def check_database() -> Dict[str, Any]:
    """Check trade database status and today's metrics."""
    db_path = PROJECT_DIR / "data" / "live_0dte_trades.db"
    if not db_path.exists():
        return {"status": "No database", "today": {}}

    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        cursor = conn.cursor()

        # Total trade count
        cursor.execute("SELECT COUNT(*) FROM trades")
        total_trades = cursor.fetchone()[0]

        # Today's trades
        today_str = date.today().strftime("%Y-%m-%d")
        cursor.execute(
            "SELECT COUNT(*), "
            "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END), "
            "COALESCE(SUM(pnl), 0), "
            "COALESCE(SUM(commission), 0) "
            "FROM trades WHERE entry_time LIKE ? AND status = 'closed'",
            (f"{today_str}%",),
        )
        row = cursor.fetchone()
        today_trades = row[0] or 0
        today_wins = row[1] or 0
        today_losses = row[2] or 0
        today_pnl = row[3] or 0.0
        today_commissions = row[4] or 0.0

        # Open positions
        cursor.execute(
            "SELECT symbol, quantity, entry_price, entry_time "
            "FROM trades WHERE status = 'open'"
        )
        open_positions = []
        for r in cursor.fetchall():
            open_positions.append({
                "symbol": r[0],
                "qty": r[1],
                "entry_price": r[2],
                "entry_time": r[3][:19] if r[3] else "",
            })

        conn.close()

        return {
            "status": f"OK ({total_trades} total)",
            "today": {
                "trades": today_trades,
                "wins": today_wins,
                "losses": today_losses,
                "pnl": round(today_pnl, 2),
                "commissions": round(today_commissions, 2),
                "net_pnl": round(today_pnl - today_commissions, 2),
            },
            "open_positions": open_positions,
        }
    except Exception as e:
        return {"status": f"ERROR: {e}", "today": {}, "open_positions": []}


def check_trading_state() -> Dict[str, Any]:
    """Read trading_state.json for capital and equity info."""
    state_file = PROJECT_DIR / "trading_state.json"
    if not state_file.exists():
        return {"status": "No state file"}

    try:
        with open(state_file, "r") as f:
            state = json.load(f)
        return {
            "current_capital": state.get("current_capital", 0),
            "initial_capital": state.get("initial_capital", 0),
            "high_water_mark": state.get("high_water_mark", 0),
            "total_pnl": round(state.get("total_pnl", 0), 2),
            "max_drawdown": round(state.get("max_drawdown", 0), 2),
            "total_trades": state.get("total_trades", 0),
            "total_wins": state.get("total_wins", 0),
            "total_losses": state.get("total_losses", 0),
            "engine_status": state.get("engine_status", "unknown"),
            "last_updated": state.get("last_updated", ""),
        }
    except Exception as e:
        return {"status": f"ERROR: {e}"}


def collect_health_report() -> Dict[str, Any]:
    """Collect all health metrics into a single report."""
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ibkr": check_ibkr_gateway(),
        "engine": check_engine_status(),
        "database": check_database(),
        "state": check_trading_state(),
    }


# ============================================================
# REPORT FORMATTING
# ============================================================

def format_report_sms(report: Dict[str, Any]) -> str:
    """Format health report for SMS (concise, fits ~160-320 chars per segment)."""
    ts = report["timestamp"]
    ibkr = report["ibkr"]["status"]
    engine = report["engine"]["status"]

    db = report["database"]
    today = db.get("today", {})
    today_pnl = today.get("net_pnl", 0)
    today_trades = today.get("trades", 0)
    today_wins = today.get("wins", 0)
    today_losses = today.get("losses", 0)
    open_pos = db.get("open_positions", [])

    state = report["state"]
    capital = state.get("current_capital", 0)
    total_pnl = state.get("total_pnl", 0)

    pnl_sign = "+" if today_pnl >= 0 else ""
    total_sign = "+" if total_pnl >= 0 else ""

    lines = [
        f"TRADING HEALTH {ts[11:16]}",
        f"IBKR: {ibkr} | Engine: {engine}",
        f"Today: {today_trades}T {today_wins}W/{today_losses}L {pnl_sign}${today_pnl:.0f}",
        f"Capital: ${capital:,.0f} | Total: {total_sign}${total_pnl:,.0f}",
    ]

    if open_pos:
        pos_str = ", ".join(
            f"{p['symbol'][-15:]} x{p['qty']}" for p in open_pos[:3]
        )
        lines.append(f"Open: {pos_str}")

    errors = report["engine"].get("errors", [])
    if errors:
        lines.append(f"ERRORS({len(errors)}): {errors[-1][:60]}")

    # Flag critical issues
    alerts = []
    if ibkr != "Connected":
        alerts.append("IBKR DOWN")
    if "STALE" in engine:
        alerts.append("ENGINE STALE")
    if today_pnl < -500:
        alerts.append(f"BIG LOSS ${today_pnl:.0f}")
    if alerts:
        lines.insert(0, "⚠ " + " | ".join(alerts))

    return "\n".join(lines)


def format_report_console(report: Dict[str, Any]) -> str:
    """Format health report for console (detailed)."""
    lines = [
        "=" * 50,
        f"  TRADING SYSTEM HEALTH REPORT",
        f"  {report['timestamp']}",
        "=" * 50,
        "",
        "--- Connectivity ---",
        f"  IBKR Gateway:  {report['ibkr']['status']} ({report['ibkr']['host']}:{report['ibkr']['port']})",
        f"  Engine:        {report['engine']['status']}",
        f"  Last Activity: {report['engine'].get('last_activity', 'N/A')}",
        "",
        "--- Database ---",
        f"  Status:  {report['database']['status']}",
    ]

    today = report["database"].get("today", {})
    if today:
        lines.extend([
            "",
            "--- Today's Performance ---",
            f"  Trades:      {today.get('trades', 0)}",
            f"  Wins/Losses: {today.get('wins', 0)} / {today.get('losses', 0)}",
            f"  P&L:         ${today.get('pnl', 0):,.2f}",
            f"  Commissions: ${today.get('commissions', 0):,.2f}",
            f"  Net P&L:     ${today.get('net_pnl', 0):,.2f}",
        ])

    open_pos = report["database"].get("open_positions", [])
    if open_pos:
        lines.extend(["", "--- Open Positions ---"])
        for p in open_pos:
            lines.append(f"  {p['symbol']}  qty={p['qty']}  entry=${p['entry_price']:.2f}  at {p['entry_time']}")
    else:
        lines.append("\n  No open positions")

    state = report["state"]
    if "current_capital" in state:
        win_rate = 0
        total = state.get("total_wins", 0) + state.get("total_losses", 0)
        if total > 0:
            win_rate = state["total_wins"] / total * 100
        lines.extend([
            "",
            "--- Account ---",
            f"  Capital:       ${state['current_capital']:,.2f}",
            f"  High Water:    ${state.get('high_water_mark', 0):,.2f}",
            f"  Total P&L:     ${state.get('total_pnl', 0):,.2f}",
            f"  Max Drawdown:  ${state.get('max_drawdown', 0):,.2f}",
            f"  Win Rate:      {win_rate:.1f}% ({state.get('total_wins', 0)}W / {state.get('total_losses', 0)}L)",
        ])

    errors = report["engine"].get("errors", [])
    if errors:
        lines.extend(["", "--- Recent Errors ---"])
        for e in errors:
            lines.append(f"  {e}")

    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


# ============================================================
# EMAIL DELIVERY (Gmail SMTP)
# ============================================================

def send_alert(body: str, to_email: str = None) -> bool:
    """
    Send email alert via Gmail SMTP.

    Env vars:
        SMTP_EMAIL        - Gmail address (sender)
        SMTP_APP_PASSWORD - Gmail App Password (16 chars)
        ALERT_EMAIL       - Destination email address
    """
    smtp_email = os.environ.get("SMTP_EMAIL")
    smtp_password = os.environ.get("SMTP_APP_PASSWORD")
    to_email = to_email or os.environ.get("ALERT_EMAIL", "")

    if not all([smtp_email, smtp_password, to_email]):
        missing = []
        if not smtp_email:
            missing.append("SMTP_EMAIL")
        if not smtp_password:
            missing.append("SMTP_APP_PASSWORD")
        if not to_email:
            missing.append("ALERT_EMAIL")
        logger.error(f"Missing email config: {', '.join(missing)}")
        return False

    # Determine subject from body
    first_line = body.split("\n")[0][:80]
    subject = f"Trading Alert: {first_line}"

    msg = MIMEMultipart()
    msg["From"] = smtp_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, to_email, msg.as_string())
        logger.info(f"Alert email sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Email send failed: {e}")
        return False


# ============================================================
# ALERT CONDITIONS
# ============================================================

def check_alert_conditions(report: Dict[str, Any]) -> List[str]:
    """Check for conditions that warrant an immediate alert email."""
    alerts = []

    # IBKR down
    if report["ibkr"]["status"] != "Connected":
        alerts.append(f"IBKR Gateway {report['ibkr']['status']}")

    # Engine stale
    engine_status = report["engine"]["status"]
    if "STALE" in engine_status:
        alerts.append(f"Engine {engine_status}")

    # Big daily loss
    today_pnl = report["database"].get("today", {}).get("net_pnl", 0)
    if today_pnl < -500:
        alerts.append(f"Daily loss ${today_pnl:.0f}")

    # Critical errors
    errors = report["engine"].get("errors", [])
    critical = [e for e in errors if "CRITICAL" in e]
    if critical:
        alerts.append(f"{len(critical)} CRITICAL error(s)")

    return alerts


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_health_check(dry_run: bool = False, to_email: str = None) -> Dict[str, Any]:
    """Run a single health check, print report, and optionally send email."""
    report = collect_health_report()

    # Always print to console
    print(format_report_console(report))

    if dry_run:
        print("\n[DRY RUN] Email would be sent:")
        print(format_report_sms(report))
        return report

    # Send scheduled report email
    email_body = format_report_sms(report)
    sent = send_alert(email_body, to_email=to_email)
    if sent:
        print(f"\nEmail sent to {to_email or os.environ.get('ALERT_EMAIL', 'N/A')}")
    else:
        print("\nEmail send FAILED — check SMTP config")

    return report


def run_loop(interval_minutes: int = 30, to_email: str = None, alert_only: bool = False):
    """
    Run health checks in a loop.

    Args:
        interval_minutes: Minutes between scheduled reports
        to_email: Override destination email address
        alert_only: If True, only send email when alert conditions are detected
    """
    print(f"Health report loop started (every {interval_minutes}m, alert_only={alert_only})")
    last_alert_time = {}  # Deduplicate alerts by message

    while True:
        try:
            report = collect_health_report()
            print(format_report_console(report))

            alerts = check_alert_conditions(report)

            if alerts:
                # Deduplicate: don't re-send the same alert within 15 minutes
                now = time.time()
                new_alerts = []
                for a in alerts:
                    if now - last_alert_time.get(a, 0) > 900:
                        new_alerts.append(a)
                        last_alert_time[a] = now

                if new_alerts:
                    alert_body = "⚠ TRADING ALERT ⚠\n" + "\n".join(new_alerts) + "\n\n" + format_report_sms(report)
                    send_alert(alert_body, to_email=to_email)
                    print(f"[ALERT EMAIL] {new_alerts}")

            if not alert_only:
                email_body = format_report_sms(report)
                send_alert(email_body, to_email=to_email)

        except Exception as e:
            logger.error(f"Health check error: {e}")
            print(f"Health check error: {e}")

        time.sleep(interval_minutes * 60)


def main():
    parser = argparse.ArgumentParser(description="Trading System Health Report + Email")
    parser.add_argument("--dry-run", action="store_true", help="Print report without sending email")
    parser.add_argument("--loop", action="store_true", help="Run in continuous loop")
    parser.add_argument("--interval", type=int, default=30, help="Minutes between reports (default: 30)")
    parser.add_argument("--alert-only", action="store_true", help="Only send email on alert conditions")
    parser.add_argument("--email", type=str, default=None, help="Override destination email address")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.loop:
        run_loop(
            interval_minutes=args.interval,
            to_email=args.email,
            alert_only=args.alert_only,
        )
    else:
        run_health_check(dry_run=args.dry_run, to_email=args.email)


if __name__ == "__main__":
    main()
