#!/usr/bin/env python3
"""Collect system health + trading stats for deploy email notifications."""
import json
import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def trading_state():
    state_file = PROJECT_DIR / "trading_state.json"
    if not state_file.exists():
        print("No trading state file")
        return
    with open(state_file) as f:
        s = json.load(f)
    t = s.get("total_trades", 0)
    wr = (s.get("total_wins", 0) / t * 100) if t > 0 else 0
    print(f"Capital:      ${s.get('current_capital', 0):,.2f}")
    print(f"Total P&L:    ${s.get('total_pnl', 0):,.2f}")
    print(f"Max Drawdown: ${s.get('max_drawdown', 0):,.2f}")
    print(f"Trades:       {t} (W:{s.get('total_wins', 0)} L:{s.get('total_losses', 0)}) Win Rate: {wr:.1f}%")
    print(f"Engine:       {s.get('engine_status', 'unknown')}")
    print(f"Updated:      {s.get('last_updated', '?')}")


def today_stats(conn):
    c = conn.cursor()
    today_str = date.today().strftime("%Y-%m-%d")
    c.execute(
        "SELECT COUNT(*), "
        "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END), "
        "COALESCE(SUM(pnl), 0), "
        "COALESCE(SUM(commission), 0) "
        "FROM trades WHERE entry_time LIKE ? AND status = 'closed'",
        (today_str + "%",),
    )
    r = c.fetchone()
    trades, wins, losses = r[0] or 0, r[1] or 0, r[2] or 0
    pnl, comm = r[3] or 0, r[4] or 0
    wr = (wins / trades * 100) if trades > 0 else 0
    print(f"Trades: {trades} | Wins: {wins} | Losses: {losses} | Win Rate: {wr:.0f}%")
    print(f"Gross P&L: ${pnl:,.2f} | Commissions: ${comm:,.2f} | Net: ${pnl - comm:,.2f}")

    # Open positions
    c.execute("SELECT symbol, quantity, entry_price FROM trades WHERE status = 'open'")
    opens = c.fetchall()
    if opens:
        print(f"Open positions: {len(opens)}")
        for o in opens:
            print(f"  {o[0]} x{o[1]} @ ${o[2]:.2f}")
    else:
        print("No open positions")


def week_stats(conn):
    c = conn.cursor()
    log_dir = PROJECT_DIR / "logs"
    for i in range(7):
        d = (date.today() - timedelta(days=i)).strftime("%Y-%m-%d")
        c.execute(
            "SELECT COUNT(*), "
            "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), "
            "COALESCE(SUM(pnl), 0), "
            "COALESCE(SUM(commission), 0) "
            "FROM trades WHERE entry_time LIKE ? AND status = 'closed'",
            (d + "%",),
        )
        r = c.fetchone()
        t, w, p, cm = r[0] or 0, r[1] or 0, r[2] or 0, r[3] or 0
        if t > 0:
            wr = w / t * 100
            print(f"{d}: {t:>3} trades, {wr:>3.0f}% win, Net ${p - cm:>+10,.2f}")
        else:
            logf = log_dir / f"live_0dte_{d.replace('-', '')}.log"
            if logf.exists():
                print(f"{d}:   0 trades (engine ran, no fills)")
            else:
                day_name = (date.today() - timedelta(days=i)).strftime("%a")
                print(f"{d}:   - no activity ({day_name})")

    # 7-day totals
    week_ago = (date.today() - timedelta(days=6)).strftime("%Y-%m-%d")
    c.execute(
        "SELECT COUNT(*), "
        "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), "
        "COALESCE(SUM(pnl), 0), "
        "COALESCE(SUM(commission), 0) "
        "FROM trades WHERE entry_time >= ? AND status = 'closed'",
        (week_ago,),
    )
    r = c.fetchone()
    t7, w7, p7, c7 = r[0] or 0, r[1] or 0, r[2] or 0, r[3] or 0
    wr7 = (w7 / t7 * 100) if t7 > 0 else 0
    print(f"{'─' * 50}")
    print(f"7-Day Total: {t7} trades, {wr7:.0f}% win, Net P&L ${p7 - c7:,.2f}")


def engine_log():
    today_str = date.today().strftime("%Y%m%d")
    logf = PROJECT_DIR / "logs" / f"live_0dte_{today_str}.log"
    if not logf.exists():
        print("No engine log today")
        return
    with open(logf, "r", errors="ignore") as f:
        lines = f.readlines()
    if lines:
        print(f"Log lines today: {len(lines)}")
        errors = sum(1 for l in lines if "ERROR" in l or "CRITICAL" in l)
        print(f"Errors today: {errors}")
        print(f"Last entry: {lines[-1].strip()[:120]}")
    else:
        print("Engine log empty")


def main():
    print("=== TRADING STATE ===")
    try:
        trading_state()
    except Exception as e:
        print(f"Error: {e}")

    print()
    print("=== ENGINE LOG ===")
    try:
        engine_log()
    except Exception as e:
        print(f"Error: {e}")

    db_path = PROJECT_DIR / "data" / "live_0dte_trades.db"
    if not db_path.exists():
        print("\nNo trade database found")
        return

    conn = sqlite3.connect(str(db_path), timeout=5)

    print()
    print("=== TODAY ===")
    try:
        today_stats(conn)
    except Exception as e:
        print(f"Error: {e}")

    print()
    print("=== PAST 7 DAYS ===")
    try:
        week_stats(conn)
    except Exception as e:
        print(f"Error: {e}")

    conn.close()


if __name__ == "__main__":
    main()
