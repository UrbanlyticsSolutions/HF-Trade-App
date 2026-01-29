"""
Live 0DTE Runner - Run the backtested 0DTE strategy live

Based on backtested results:
- Strategy: ORB (Opening Range Breakout)
- Win Rate: 91.2%
- Window: 10:00 - 11:00 AM ET
- Options: $0.50 - $1.00

Usage:
    python -m live.runner_0dte --capital 10000
    python -m live.runner_0dte --capital 10000 --strategy momentum
    python -m live.runner_0dte --capital 10000 --mode paper  # Paper trading
    python -m live.runner_0dte --capital 10000 --mode live   # Enable live trading
"""
import argparse
import logging
import sys
import os
import time
from datetime import datetime, time as dt_time, timedelta
import pytz

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# Trading hours (Eastern Time)
ET = pytz.timezone('America/New_York')
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)
PRE_MARKET_BUFFER_MINUTES = 5  # Wake up 5 minutes before market open


def get_eastern_time() -> datetime:
    """Get current time in Eastern timezone."""
    return datetime.now(ET)


def is_market_hours() -> bool:
    """Check if current time is within market hours (9:30 AM - 4:00 PM ET)."""
    now_et = get_eastern_time()
    current_time = now_et.time()
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    if now_et.weekday() >= 5:  # Saturday or Sunday
        return False
    
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


def is_trading_day() -> bool:
    """Check if today is a trading day (weekday, not holiday)."""
    now_et = get_eastern_time()
    # Basic check: weekday only (holidays would need a calendar)
    return now_et.weekday() < 5


def get_seconds_until_market_open() -> int:
    """Calculate seconds until market opens."""
    now_et = get_eastern_time()
    
    # If it's a weekend, calculate time to Monday
    days_until_monday = 0
    if now_et.weekday() == 5:  # Saturday
        days_until_monday = 2
    elif now_et.weekday() == 6:  # Sunday
        days_until_monday = 1
    
    # Target open time
    if days_until_monday > 0:
        target_date = now_et.date() + timedelta(days=days_until_monday)
    elif now_et.time() >= MARKET_CLOSE:
        # After market close, wait until next trading day
        if now_et.weekday() == 4:  # Friday
            target_date = now_et.date() + timedelta(days=3)  # Monday
        else:
            target_date = now_et.date() + timedelta(days=1)
    else:
        target_date = now_et.date()
    
    target_open = ET.localize(datetime.combine(target_date, MARKET_OPEN))
    
    # Subtract buffer to wake up early
    target_wakeup = target_open - timedelta(minutes=PRE_MARKET_BUFFER_MINUTES)
    
    seconds_until = (target_wakeup - now_et).total_seconds()
    return max(0, int(seconds_until))


def wait_for_market_open():
    """Sleep until market is about to open, with periodic status updates."""
    while not is_trading_day() or get_eastern_time().time() < MARKET_OPEN:
        now_et = get_eastern_time()
        seconds_until = get_seconds_until_market_open()
        
        if seconds_until <= 0:
            break
        
        # Format wait time nicely
        hours, remainder = divmod(seconds_until, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            wait_str = f"{int(hours)}h {int(minutes)}m"
        elif minutes > 0:
            wait_str = f"{int(minutes)}m {int(seconds)}s"
        else:
            wait_str = f"{int(seconds)}s"
        
        logger.info(f"Market closed. Current time: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}. "
                    f"Opens in: {wait_str}. Sleeping...")
        
        # Sleep in chunks (max 5 minutes) to allow for interrupts and status updates
        sleep_duration = min(seconds_until, 300)  # 5 minute max sleep
        time.sleep(sleep_duration)
    
    logger.info(f"Market is open! Time: {get_eastern_time().strftime('%H:%M:%S %Z')}")


def main():
    parser = argparse.ArgumentParser(description="Live 0DTE SPY Options Trading")
    parser.add_argument("--capital", type=float, default=10000, help="Account capital")
    parser.add_argument("--account", help="Questrade account ID (auto-detect if not provided)")
    parser.add_argument("--strategy", default="orb", choices=["orb", "momentum", "mean_reversion"],
                        help="Strategy type (default: orb)")
    parser.add_argument("--mode", default="monitor", choices=["monitor", "paper", "live"],
                        help="Trading mode: monitor (no orders), paper (simulated), live (real)")
    parser.add_argument("--target", type=float, default=0.22, help="Profit target %% (default: 22%%)")
    parser.add_argument("--stop", type=float, default=0.25, help="Stop loss %% (default: 25%%)")
    parser.add_argument("--max-contracts", type=int, default=5, help="Max contracts per trade")
    parser.add_argument("--no-stop-after-loss", action="store_true", help="Continue trading after first loss")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    mode = args.mode.upper()
    
    # Get project directory (parent of live/)
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_dir, 'logs')
    data_dir = os.path.join(project_dir, 'data')
    
    # Ensure directories exist
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = os.path.join(logs_dir, f"live_0dte_{datetime.now().strftime('%Y%m%d')}.log")
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    print("=" * 70)
    print("LIVE 0DTE SPY OPTIONS TRADING")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Capital: ${args.capital:,.2f}")
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Profit Target: {args.target:.0%}")
    print(f"Stop Loss: {args.stop:.0%}")
    print(f"Max Contracts: {args.max_contracts}")
    if mode == "LIVE":
        print(">>> WARNING: LIVE TRADING - REAL ORDERS WILL BE PLACED <<<")
    elif mode == "PAPER":
        print(">>> PAPER TRADING - Simulated fills, no real orders <<<")
    else:
        print(">>> MONITOR ONLY - No orders executed <<<")
    print(f"Stop After First Loss: {not args.no_stop_after_loss}")
    print("=" * 70)
    
    try:
        from clients.questrade_client import create_questrade_client
        from live.engine import create_engine
        from live.strategy_0dte import create_0dte_strategy
        
        # Connect to Questrade
        logger.info("Connecting to Questrade...")
        client = create_questrade_client()
        
        accounts = client.get_accounts()
        logger.info(f"Connected! Found {len(accounts)} accounts")
        
        # Get account ID
        if args.account:
            account_id = args.account
        else:
            # Use first margin account
            margin_accounts = [a for a in accounts if a.get('type') == 'Margin']
            if margin_accounts:
                account_id = str(margin_accounts[0]['number'])
            else:
                account_id = str(accounts[0]['number'])
        
        logger.info(f"Using account: {account_id}")
        
        # Create engine
        db_path = os.path.join(data_dir, 'live_0dte_trades.db')
        engine = create_engine(
            questrade_client=client,
            account_id=account_id,
            symbols=["SPY"],
            option_underlyings=["SPY"],
            mode=args.mode,
            db_path=db_path
        )
        
        # Create 0DTE strategy with client for ORB backfill
        strategy = create_0dte_strategy(
            account_capital=args.capital,
            strategy=args.strategy,
            profit_target_pct=args.target,
            stop_loss_pct=args.stop,
            max_contracts=args.max_contracts,
            stop_after_first_loss=not args.no_stop_after_loss,
            questrade_client=client
        )
        
        # Add strategy to engine
        engine.add_strategy(strategy)
        
        # Log quote updates
        def on_quote(symbol, quote):
            if symbol == "SPY":
                price = quote.get('lastTradePrice', 0)
                logger.debug(f"SPY: ${price:.2f}")
        
        engine.on_quote(on_quote)
        
        # Log signals
        def on_signal(signal):
            logger.info(f"SIGNAL: {signal.action} {signal.quantity} {signal.symbol} - {signal.reason}")
        
        engine.on_signal(on_signal)
        
        # Check market hours and wait if outside trading window
        now_et = get_eastern_time()
        logger.info(f"Current time (ET): {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        if not is_market_hours():
            if not is_trading_day():
                logger.info("Today is not a trading day (weekend).")
            elif now_et.time() < MARKET_OPEN:
                logger.info(f"Market not yet open. Opens at {MARKET_OPEN.strftime('%H:%M')} ET.")
            else:
                logger.info(f"Market is closed. Closed at {MARKET_CLOSE.strftime('%H:%M')} ET.")
            
            # Wait for market to open
            wait_for_market_open()
        
        # Print strategy status
        strategy.print_status()
        
        # Run engine
        logger.info("Starting engine... Press Ctrl+C to stop")
        engine.run()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


def quick_start(capital: float = 10000, strategy: str = "orb", mode: str = "monitor"):
    """
    Quick start function for Python usage.
    
    Usage:
        from live.runner_0dte import quick_start
        quick_start(capital=10000, strategy="orb", mode="paper")
    """
    from clients.questrade_client import create_questrade_client
    from live.engine import create_engine
    from live.strategy_0dte import create_0dte_strategy
    
    # Get project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Connect
    client = create_questrade_client()
    accounts = client.get_accounts()
    account_id = str(accounts[0]['number'])
    
    # Create engine
    db_path = os.path.join(data_dir, 'live_0dte_trades.db')
    engine = create_engine(
        questrade_client=client,
        account_id=account_id,
        symbols=["SPY"],
        option_underlyings=["SPY"],
        mode=mode,
        db_path=db_path
    )
    
    # Create and add strategy
    strategy = create_0dte_strategy(account_capital=capital, strategy=strategy)
    engine.add_strategy(strategy)
    
    # Run
    print(f"Running 0DTE {strategy} strategy with ${capital:,} capital")
    print(f"Mode: {mode.upper()}")
    engine.run()


if __name__ == "__main__":
    main()
