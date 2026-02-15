"""
Live 0DTE Runner - Phase 8 Momentum Strategy

Based on Phase 8 optimized results (Feb 2026):
- Strategy: Momentum (RSI > 70 -> CALL, RSI < 30 -> PUT)
- Win Rate: 75.6%
- Return: +187.3%
- Max DD: 5.8%
- Window: 10:00 - 11:00 AM ET
- Options: $0.50 - $2.00
- PT: 50%, SL: 35%, Hold: 80 min
- Risk: SFL + CL=3 + DLL=0.8%

Usage:
    python -m live.runner_0dte --capital 10000
    python -m live.runner_0dte --capital 10000 --mode paper
    python -m live.runner_0dte --capital 10000 --mode live
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


def update_engine_status(status: str, extra_info: dict = None):
    """Update engine status in trading_state.json for dashboard display."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    state_file = os.path.join(project_dir, 'trading_state.json')
    
    try:
        import json
        state = {}
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
        
        # Update engine status fields
        state['engine_status'] = status
        state['engine_last_update'] = datetime.now().isoformat()
        
        if extra_info:
            for k, v in extra_info.items():
                state[f'engine_{k}'] = v
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.debug(f"Failed to update engine status: {e}")


def wait_for_market_open():
    """Sleep until market is about to open, with periodic status updates."""
    update_engine_status("sleep", {"waiting_for": "market_open"})
    
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
        
        # Update status with wait time
        update_engine_status("sleep", {"opens_in": wait_str, "current_time_et": now_et.strftime('%H:%M:%S')})
        
        # Sleep in chunks (max 5 minutes) to allow for interrupts and status updates
        sleep_duration = min(seconds_until, 300)  # 5 minute max sleep
        time.sleep(sleep_duration)
    
    logger.info(f"Market is open! Time: {get_eastern_time().strftime('%H:%M:%S %Z')}")
    update_engine_status("starting", {"current_time_et": get_eastern_time().strftime('%H:%M:%S')})


def main():
    parser = argparse.ArgumentParser(description="Live 0DTE SPY Options Trading (Phase 8)")
    parser.add_argument("--capital", type=float, default=10000, help="Account capital")
    parser.add_argument("--account", help="Questrade account ID (auto-detect if not provided)")
    parser.add_argument("--strategy", default=None, choices=["orb", "momentum", "mean_reversion"],
                        help="Strategy type (default: from strategy.json)")
    parser.add_argument("--mode", default="monitor", choices=["monitor", "paper", "live"],
                        help="Trading mode: monitor (no orders), paper (simulated), live (real)")
    parser.add_argument("--target", type=float, default=None, help="Profit target %% (default: from config)")
    parser.add_argument("--stop", type=float, default=None, help="Stop loss %% (default: from config)")
    parser.add_argument("--max-contracts", type=int, default=50, help="Max contracts per trade")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    mode = args.mode.upper()
    
    # Get project directory (parent of live/)
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_dir, 'logs')
    data_dir = os.path.join(project_dir, 'data')
    config_dir = os.path.join(project_dir, 'config')
    
    # Load strategy config from JSON file
    import json
    config_file = os.path.join(config_dir, 'strategy.json')
    trade_config = {}
    risk_config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            trade_config = config.get('trade_config', {})
            risk_config = config.get('risk_config', {})
    
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
    
    # Resolve strategy: CLI > config > default
    strategy_name = args.strategy or trade_config.get('strategy', 'momentum')
    
    print("=" * 70)
    print("LIVE 0DTE SPY OPTIONS TRADING (PHASE 8)")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Capital: ${args.capital:,.2f}")
    print(f"Strategy: {strategy_name.upper()}")
    print(f"Profit Target: {trade_config.get('profit_target_pct', 0.50):.0%}")
    print(f"Stop Loss: {trade_config.get('stop_loss_pct', 0.35):.0%}")
    print(f"Max Contracts: {args.max_contracts}")
    print(f"Option Price Range: ${trade_config.get('min_option_price', 0.50):.2f} - ${trade_config.get('max_option_price', 2.00):.2f}")
    print(f"Trading Window: {trade_config.get('trade_start_hour', 10)}:{trade_config.get('trade_start_minute', 0):02d} - {trade_config.get('trade_end_hour', 11)}:{trade_config.get('trade_end_minute', 0):02d} ET")
    print(f"Max Hold: {trade_config.get('max_hold_minutes', 80)} min")
    print(f"SFL: {risk_config.get('stop_after_first_loss', True)} | CL: {risk_config.get('max_consecutive_losses', 3)} | DLL: {risk_config.get('max_daily_loss_pct', 0.008):.1%}")
    if mode == "LIVE":
        print(">>> WARNING: LIVE TRADING - REAL ORDERS WILL BE PLACED <<<")
    elif mode == "PAPER":
        print(">>> PAPER TRADING - Simulated fills, no real orders <<<")
    else:
        print(">>> MONITOR ONLY - No orders executed <<<")
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
        
        # Create 0DTE strategy with Phase 8 config from strategy.json
        strategy = create_0dte_strategy(
            account_capital=args.capital,
            strategy=strategy_name,
            profit_target_pct=trade_config.get('profit_target_pct', 0.50),
            stop_loss_pct=trade_config.get('stop_loss_pct', 0.35),
            max_contracts=args.max_contracts,
            stop_after_first_loss=risk_config.get('stop_after_first_loss', True),
            max_consecutive_losses=risk_config.get('max_consecutive_losses', 3),
            max_daily_loss_pct=risk_config.get('max_daily_loss_pct', 0.008),
            min_option_price=trade_config.get('min_option_price', 0.50),
            max_option_price=trade_config.get('max_option_price', 2.00),
            orb_minutes=trade_config.get('orb_minutes', 30),
            orb_buffer_pct=trade_config.get('orb_buffer_pct', 0.10),
            max_hold_minutes=trade_config.get('max_hold_minutes', 80),
            trade_start_hour=trade_config.get('trade_start_hour', 10),
            trade_start_minute=trade_config.get('trade_start_minute', 0),
            trade_end_hour=trade_config.get('trade_end_hour', 11),
            trade_end_minute=trade_config.get('trade_end_minute', 0),
            exit_hour=trade_config.get('exit_hour', 15),
            rsi_call_threshold=trade_config.get('rsi_call_threshold', 70),
            rsi_put_threshold=trade_config.get('rsi_put_threshold', 30),
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
        update_engine_status("live", {
            "mode": args.mode,
            "strategy": strategy_name,
            "capital": args.capital,
            "started_at": get_eastern_time().strftime('%H:%M:%S')
        })
        engine.run()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        update_engine_status("stopped", {"reason": "user_interrupt"})
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        update_engine_status("error", {"error_message": str(e)[:100]})
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
