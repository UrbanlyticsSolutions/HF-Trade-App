"""
Reconcile Trading State with Database

This script:
1. Identifies orphaned/problematic trades in the DB
2. Fixes them WITHOUT touching actual open positions
3. Reconciles trading_state.json with the DB

Usage:
    python scripts/reconcile_db.py              # Dry run (show what would change)
    python scripts/reconcile_db.py --fix        # Actually apply fixes
    python scripts/reconcile_db.py --fix --close-orphans  # Also close orphaned trades
"""
import sqlite3
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_db_connection(db_path: str):
    """Get database connection"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def identify_orphans(conn) -> dict:
    """
    Identify orphaned/problematic trades.
    
    An orphan is a trade that:
    - Has status='open' but entry_time is old (> 24 hours for 0DTE)
    - Has action='BUY' but was never closed
    - Has status='cancelled' but has entry data
    """
    cursor = conn.cursor()
    
    results = {
        "stale_open": [],      # Open too long (0DTE should close same day)
        "cancelled_with_entry": [],  # Cancelled but has entry (should be closed or removed)
        "missing_pnl": [],     # Closed but no P&L calculated
        "duplicate_symbols": [],  # Multiple open trades for same symbol
        "actual_open": [],     # Legitimately open positions (DO NOT TOUCH)
        "orphaned_pairs": []   # BUY/SELL pairs that should be matched
    }
    
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    
    # 1. Find BUY/SELL pairs that should be matched (orphaned pairs)
    # These are BUYs with corresponding SELLs that were inserted separately
    cursor.execute("""
        SELECT id, symbol, action, quantity, entry_price, entry_time, status
        FROM trades 
        WHERE status = 'open'
        ORDER BY entry_time
    """)
    
    open_trades = [dict(row) for row in cursor.fetchall()]
    
    # Group by symbol and try to match BUY/SELL pairs
    buys = [t for t in open_trades if t['action'].upper() == 'BUY']
    sells = [t for t in open_trades if t['action'].upper() == 'SELL']
    
    matched_buy_ids = set()
    matched_sell_ids = set()
    
    for buy in buys:
        # Find matching SELL (same symbol, same quantity, after BUY)
        for sell in sells:
            if (sell['id'] not in matched_sell_ids and
                sell['symbol'] == buy['symbol'] and
                sell['quantity'] == buy['quantity'] and
                sell['entry_time'] > buy['entry_time']):
                
                results["orphaned_pairs"].append({
                    "buy": buy,
                    "sell": sell
                })
                matched_buy_ids.add(buy['id'])
                matched_sell_ids.add(sell['id'])
                break
    
    # 2. Find stale open trades (open from previous days - for 0DTE these are orphans)
    # Exclude already matched pairs
    cursor.execute("""
        SELECT id, symbol, action, quantity, entry_price, entry_time, status
        FROM trades 
        WHERE status = 'open' 
        AND DATE(entry_time) < ?
    """, (today,))
    
    for row in cursor.fetchall():
        trade = dict(row)
        if trade['id'] not in matched_buy_ids and trade['id'] not in matched_sell_ids:
            results["stale_open"].append(trade)
    
    # 3. Find today's open trades (these are LEGITIMATE - do not touch!)
    # Exclude matched pairs
    cursor.execute("""
        SELECT id, symbol, action, quantity, entry_price, entry_time, status
        FROM trades 
        WHERE status = 'open' 
        AND DATE(entry_time) = ?
    """, (today,))
    
    for row in cursor.fetchall():
        trade = dict(row)
        if trade['id'] not in matched_buy_ids and trade['id'] not in matched_sell_ids:
            results["actual_open"].append(trade)
    
    # 4. Find cancelled trades that have entry data
    cursor.execute("""
        SELECT id, symbol, action, quantity, entry_price, entry_time, status
        FROM trades 
        WHERE status = 'cancelled' 
        AND entry_price IS NOT NULL
        AND entry_price > 0
    """)
    
    for row in cursor.fetchall():
        results["cancelled_with_entry"].append(dict(row))
    
    # 5. Find closed trades missing P&L
    cursor.execute("""
        SELECT id, symbol, action, quantity, entry_price, exit_price, status
        FROM trades 
        WHERE status = 'closed' 
        AND (pnl IS NULL OR exit_price IS NULL)
    """)
    
    for row in cursor.fetchall():
        results["missing_pnl"].append(dict(row))
    
    # 6. Find duplicate open positions for same symbol (excluding matched pairs)
    remaining_open = [t for t in open_trades 
                     if t['id'] not in matched_buy_ids and t['id'] not in matched_sell_ids]
    symbol_counts = {}
    for t in remaining_open:
        symbol_counts[t['symbol']] = symbol_counts.get(t['symbol'], 0) + 1
    
    for symbol, count in symbol_counts.items():
        if count > 1:
            results["duplicate_symbols"].append({"symbol": symbol, "count": count})
    
    return results


def fix_orphaned_pairs(conn, pairs: list, dry_run: bool = True):
    """
    Fix orphaned BUY/SELL pairs by closing the BUY with the SELL's data
    and deleting the redundant SELL row.
    """
    cursor = conn.cursor()
    
    for pair in pairs:
        buy = pair["buy"]
        sell = pair["sell"]
        
        buy_id = buy["id"]
        sell_id = sell["id"]
        entry_price = buy["entry_price"]
        exit_price = sell["entry_price"]  # SELL's entry_price is the exit price
        quantity = buy["quantity"]
        
        # Calculate P&L
        pnl = (exit_price - entry_price) * 100 * quantity
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        
        print(f"  BUY #{buy_id} @ ${entry_price:.2f} + SELL #{sell_id} @ ${exit_price:.2f}")
        print(f"    -> Close BUY #{buy_id}, P&L: ${pnl:.2f} ({pnl_pct:.1f}%), delete SELL #{sell_id}")
        
        if not dry_run:
            # Update BUY to closed
            cursor.execute("""
                UPDATE trades 
                SET status='closed', 
                    exit_price=?, 
                    exit_time=?,
                    pnl=?, 
                    pnl_percent=?,
                    notes='Fixed by reconcile - matched with SELL row'
                WHERE id=?
            """, (exit_price, sell["entry_time"], pnl, pnl_pct, buy_id))
            
            # Delete SELL row (it's redundant)
            cursor.execute("DELETE FROM trades WHERE id=?", (sell_id,))
    
    if not dry_run:
        conn.commit()


def close_stale_trades(conn, trades: list, exit_price: float = None, dry_run: bool = True):
    """Close stale open trades at specified price (or $0.01 if expired)"""
    cursor = conn.cursor()
    
    for trade in trades:
        trade_id = trade["id"]
        entry_price = trade["entry_price"]
        quantity = trade["quantity"]
        action = trade["action"]
        
        # Use provided exit price or assume expired worthless
        price = exit_price if exit_price else 0.01
        
        # Calculate P&L
        if action.upper() == "BUY":
            pnl = (price - entry_price) * 100 * quantity
        else:
            pnl = (entry_price - price) * 100 * quantity
        
        pnl_pct = ((price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        
        print(f"  Trade #{trade_id}: {action} {quantity}x @ ${entry_price:.2f}")
        print(f"    -> Close @ ${price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:.1f}%)")
        
        if not dry_run:
            cursor.execute("""
                UPDATE trades 
                SET status='closed', 
                    exit_price=?, 
                    exit_time=?,
                    pnl=?, 
                    pnl_percent=?,
                    notes='Closed by reconcile script - stale 0DTE'
                WHERE id=?
            """, (price, datetime.now().isoformat(), pnl, pnl_pct, trade_id))
    
    if not dry_run:
        conn.commit()


def delete_cancelled_trades(conn, trades: list, dry_run: bool = True):
    """Delete cancelled trades that shouldn't exist"""
    cursor = conn.cursor()
    
    for trade in trades:
        trade_id = trade["id"]
        print(f"  Deleting cancelled trade #{trade_id}: {trade['symbol']}")
        
        if not dry_run:
            cursor.execute("DELETE FROM trades WHERE id=?", (trade_id,))
    
    if not dry_run:
        conn.commit()


def get_option_type(symbol: str) -> str:
    """Extract CALL or PUT from option symbol"""
    # OCC format: SPY28Jan26P694.00 - P=PUT, C=CALL
    import re
    match = re.search(r'[CP]\d', symbol)
    if match:
        return "PUT" if match.group()[0] == 'P' else "CALL"
    return "UNKNOWN"


def get_engine_status() -> str:
    """Determine engine status based on current time"""
    from datetime import datetime, time
    now = datetime.now()
    
    # Check if weekend
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return "sleep"
    
    # Trading hours: 9:30 AM - 4:00 PM ET
    market_open = time(9, 30)
    market_close = time(16, 0)
    current_time = now.time()
    
    if market_open <= current_time <= market_close:
        return "live"
    else:
        return "sleep"


def recalculate_state(conn, initial_capital: float = 10000.0):
    """Recalculate trading state from closed trades"""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) as wins,
            COALESCE(SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END), 0) as losses,
            COALESCE(SUM(pnl), 0) as total_pnl
        FROM trades 
        WHERE status = 'closed' AND pnl IS NOT NULL
    """)
    
    row = cursor.fetchone()
    
    total_trades = row["total"]
    wins = row["wins"]
    losses = row["losses"]
    total_pnl = row["total_pnl"]
    current_capital = initial_capital + total_pnl
    
    # Calculate max drawdown and equity curve from trade sequence
    # Use COALESCE to handle NULL exit_time (fall back to entry_time)
    cursor.execute("""
        SELECT id, symbol, pnl, COALESCE(exit_time, entry_time) as time FROM trades 
        WHERE status = 'closed' AND pnl IS NOT NULL 
        ORDER BY COALESCE(exit_time, entry_time)
    """)
    
    running = initial_capital
    hwm = initial_capital
    max_dd = 0.0
    equity_curve = [{"trade_id": 0, "type": "-", "equity": initial_capital, "pnl": 0}]
    
    for pnl_row in cursor.fetchall():
        running += pnl_row["pnl"]
        opt_type = get_option_type(pnl_row["symbol"])
        equity_curve.append({
            "trade_id": pnl_row["id"],
            "type": opt_type,
            "equity": round(running, 2),
            "pnl": round(pnl_row["pnl"], 2),
            "time": pnl_row["time"]
        })
        if running > hwm:
            hwm = running
        dd = (hwm - running) / hwm if hwm > 0 else 0
        if dd > max_dd:
            max_dd = dd
    
    return {
        "initial_capital": initial_capital,
        "current_capital": round(current_capital, 2),
        "high_water_mark": round(hwm, 2),
        "total_trades": total_trades,
        "total_wins": wins,
        "total_losses": losses,
        "total_pnl": round(total_pnl, 2),
        "max_drawdown": round(max_dd, 4),
        "equity_curve": equity_curve,
        "engine_status": get_engine_status()
    }


def update_state_file(state_file: str, new_state: dict, dry_run: bool = True):
    """Update trading_state.json with new values"""
    
    # Load existing state (handle corrupt files)
    existing = {}
    if Path(state_file).exists():
        try:
            with open(state_file, 'r') as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"  ⚠️ State file was corrupt, recreating from scratch")
            existing = {}
    
    # Merge new state
    existing.update(new_state)
    existing["last_updated"] = datetime.now().isoformat()
    
    print(f"\nUpdated state:")
    for k, v in new_state.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    if not dry_run:
        with open(state_file, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"\n✅ Saved to {state_file}")


def main():
    parser = argparse.ArgumentParser(description="Reconcile trading state with database")
    parser.add_argument("--fix", action="store_true", help="Apply fixes (default is dry run)")
    parser.add_argument("--close-orphans", action="store_true", help="Close stale open trades")
    parser.add_argument("--exit-price", type=float, default=0.01, help="Exit price for orphaned trades")
    parser.add_argument("--db", default="data/live_0dte_trades.db", help="Database path")
    parser.add_argument("--state", default="trading_state.json", help="State file path")
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Initial capital")
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent
    db_path = project_root / args.db
    state_path = project_root / args.state
    
    print("=" * 60)
    print("TRADING STATE RECONCILIATION")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"State file: {state_path}")
    print(f"Mode: {'FIX' if args.fix else 'DRY RUN (use --fix to apply)'}")
    print("=" * 60)
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1
    
    conn = get_db_connection(str(db_path))
    
    # Step 1: Identify orphans
    print("\n📊 ANALYZING DATABASE...")
    orphans = identify_orphans(conn)
    
    print(f"\n🔍 FINDINGS:")
    print(f"  ✓ Legitimate open trades (today): {len(orphans['actual_open'])}")
    for t in orphans["actual_open"]:
        print(f"      #{t['id']}: {t['symbol']} - DO NOT TOUCH")
    
    print(f"  ⚠️ Orphaned BUY/SELL pairs: {len(orphans['orphaned_pairs'])}")
    for pair in orphans["orphaned_pairs"]:
        buy = pair["buy"]
        sell = pair["sell"]
        pnl = (sell["entry_price"] - buy["entry_price"]) * 100 * buy["quantity"]
        print(f"      BUY #{buy['id']} + SELL #{sell['id']}: {buy['symbol']} -> P&L: ${pnl:.2f}")
    
    print(f"  ⚠️ Stale open trades (old 0DTE): {len(orphans['stale_open'])}")
    for t in orphans["stale_open"]:
        print(f"      #{t['id']}: {t['symbol']} from {t['entry_time'][:10]}")
    
    print(f"  ⚠️ Cancelled with entry data: {len(orphans['cancelled_with_entry'])}")
    print(f"  ⚠️ Closed missing P&L: {len(orphans['missing_pnl'])}")
    print(f"  ⚠️ Duplicate opens: {len(orphans['duplicate_symbols'])}")
    
    # Step 2a: Fix orphaned pairs (BUY/SELL that should be matched)
    if orphans["orphaned_pairs"]:
        print(f"\n🔧 FIXING ORPHANED BUY/SELL PAIRS...")
        fix_orphaned_pairs(conn, orphans["orphaned_pairs"], dry_run=not args.fix)
    
    # Step 2b: Close stale trades if requested
    if orphans["stale_open"] and args.close_orphans:
        print(f"\n🔧 CLOSING STALE TRADES @ ${args.exit_price:.2f}...")
        close_stale_trades(conn, orphans["stale_open"], args.exit_price, dry_run=not args.fix)
    
    # Step 3: Recalculate state from DB
    print("\n📈 RECALCULATING STATE FROM DB...")
    new_state = recalculate_state(conn, args.initial_capital)
    
    # Step 4: Update state file
    update_state_file(str(state_path), new_state, dry_run=not args.fix)
    
    conn.close()
    
    if not args.fix:
        print("\n⚠️  DRY RUN - No changes made. Use --fix to apply.")
    else:
        print("\n✅ RECONCILIATION COMPLETE")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
