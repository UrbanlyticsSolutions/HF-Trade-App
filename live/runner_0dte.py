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
    python -m live.runner_0dte                          # auto-detects capital from broker
    python -m live.runner_0dte --mode paper
    python -m live.runner_0dte --capital 25000 --mode live  # override capital
"""
import argparse
import logging
import sys
import os
import time
from datetime import datetime, time as dt_time, timedelta
import pytz
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from config.defaults import (
    initial_capital as _default_capital,
    max_contracts as _default_max_contracts,
    ibkr_live_port as _default_live_port,
    ibkr_paper_port as _default_paper_port,
    get_trade_config, get_risk_config,
)

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


def _sync_engine_positions(engine):
    """One-shot position sync from broker to DB for dashboard visibility."""
    try:
        engine.positions.sync_positions()
        engine.positions.update_quotes()
        if engine.db and hasattr(engine.db, 'update_current_positions'):
            positions = engine.positions.get_all_positions()
            engine.db.update_current_positions(positions)
            logger.info(f"Position sync: {len(positions)} positions written to DB")
    except Exception as e:
        logger.warning(f"Position sync failed: {e}")


def wait_for_market_open(engine=None):
    """Sleep until market is about to open, with periodic status updates."""
    update_engine_status("sleep", {"waiting_for": "market_open"})
    
    _last_sync = time.time()
    _SYNC_INTERVAL = 300  # re-sync positions every 5 minutes during sleep
    
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
        
        # Periodic position sync during sleep so dashboard stays current
        if engine and (time.time() - _last_sync) >= _SYNC_INTERVAL:
            _sync_engine_positions(engine)
            _last_sync = time.time()
    
    logger.info(f"Market is open! Time: {get_eastern_time().strftime('%H:%M:%S %Z')}")
    update_engine_status("starting", {"current_time_et": get_eastern_time().strftime('%H:%M:%S')})


def main():
    parser = argparse.ArgumentParser(description="Live 0DTE SPY Options Trading (Phase 8)")
    parser.add_argument("--capital", type=float, default=None, help="Account capital (default: from config)")
    parser.add_argument("--account", help="Questrade account ID (auto-detect if not provided)")
    parser.add_argument("--strategy", default=None, choices=["orb", "momentum", "mean_reversion"],
                        help="Strategy type (default: from strategy.json)")
    parser.add_argument("--mode", default="monitor", choices=["monitor", "paper", "live"],
                        help="Trading mode: monitor (no orders), paper (simulated), live (real)")
    parser.add_argument("--target", type=float, default=None, help="Profit target %% (default: from config)")
    parser.add_argument("--stop", type=float, default=None, help="Stop loss %% (default: from config)")
    parser.add_argument("--max-contracts", type=int, default=None, help="Max contracts per trade (default: from config)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Capital: CLI override > broker query > config fallback (resolved after broker connect)
    capital = args.capital  # None means "fetch from broker"
    max_contracts_val = args.max_contracts if args.max_contracts is not None else _default_max_contracts()
    
    mode = args.mode.upper()
    
    # Get project directory (parent of live/)
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_dir, 'logs')
    data_dir = os.path.join(project_dir, 'data')
    config_dir = os.path.join(project_dir, 'config')
    
    # Load strategy config from JSON file
    import json
    config_file = os.path.join(config_dir, 'strategy.json')
    with open(config_file) as f:
        full_config = json.load(f)
    trade_config = get_trade_config()
    risk_config = get_risk_config()
    
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
    print(f"Broker: IBKR (orders/positions/quotes) + Questrade (option chains)")
    print(f"Capital: {'$' + f'{capital:,.2f}' if capital else 'auto (from broker)'}")
    print(f"Strategy: {strategy_name.upper()}")
    print(f"Profit Target: {trade_config.get('profit_target_pct', 0.50):.0%}")
    print(f"Stop Loss: {trade_config.get('stop_loss_pct', 0.35):.0%}")
    print(f"Max Contracts: {max_contracts_val}")
    print(f"Option Price Range: ${trade_config.get('min_option_price', 0.50):.2f} - ${trade_config.get('max_option_price', 2.00):.2f}")
    print(f"Trading Window: {trade_config.get('trade_start_hour', 9)}:{trade_config.get('trade_start_minute', 35):02d} - {trade_config.get('trade_end_hour', 15)}:{trade_config.get('trade_end_minute', 0):02d} ET")
    print(f"Max Hold: {trade_config.get('max_hold_minutes', 80)} min")
    print(f"MDL: {risk_config.get('max_daily_losses', 2)} | CL: {risk_config.get('max_consecutive_losses', 3)} | DLL: {risk_config.get('max_daily_loss_pct', 0.008):.1%}")
    if mode == "LIVE":
        print(">>> WARNING: LIVE TRADING - REAL ORDERS WILL BE PLACED <<<")
    elif mode == "PAPER":
        print(">>> PAPER TRADING - Real orders on IBKR paper account <<<")
    else:
        print(">>> MONITOR ONLY - No orders executed <<<")
    print("=" * 70)
    
    try:
        from live.engine import create_engine
        from live.strategy_0dte import create_0dte_strategy
        
        # --- IBKR for orders/positions/quotes, Questrade ONLY for option chains ---
        from clients.ibkr_adapter import create_ibkr_client
        from clients.questrade_client import create_questrade_client
        
        qt_client = None
        ibkr_client = None
        
        # 1) Connect IBKR (primary — orders, positions, account, quotes)
        #    Retry with backoff in case gateway is still starting up (Docker)
        logger.info("Connecting to IBKR (primary broker)...")
        env_host = os.environ.get("IBKR_HOST")
        env_port = os.environ.get("IBKR_LIVE_PORT") if mode.upper() == "LIVE" else os.environ.get("IBKR_PAPER_PORT")
        ibkr_port = int(env_port) if env_port else (_default_live_port() if mode.upper() == "LIVE" else _default_paper_port())
        # In Docker the gateway TCP port opens before the API layer is ready;
        # wait a few seconds so IBC can finish its login dialogs.
        if env_host:
            logger.info("Docker detected — waiting 10s for gateway API layer...")
            time.sleep(10)
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                ibkr_client = create_ibkr_client(port=ibkr_port, host=env_host)
                ibkr_client.get_accounts()
                logger.info("IBKR connected!")
                break
            except Exception as ie:
                ibkr_client = None
                if attempt < max_retries:
                    delay = min(5 * attempt, 30)
                    logger.warning(f"IBKR connection attempt {attempt}/{max_retries} failed: {ie} — retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"IBKR connection failed after {max_retries} attempts: {ie}")
        
        if not ibkr_client:
            raise RuntimeError("IBKR could not connect. IBKR is required for orders/positions.")
        
        # 2) Try Questrade for option chain data only
        logger.info("Connecting to Questrade (option chains only)...")
        try:
            qt_client = create_questrade_client()
            qt_client.get_accounts()
            logger.info("Questrade connected (option chains)!")
        except Exception as qe:
            logger.warning(f"Questrade connection failed: {qe} — will use IBKR for option chains too")
            qt_client = None
        
        # IBKR is always the primary client for orders, positions, account, quotes
        client = ibkr_client
        quote_client = ibkr_client
        # Questrade is ONLY used for option chain discovery (get_atm_options)
        chains_client = qt_client or ibkr_client
        
        if qt_client:
            logger.info("DUAL BROKER: IBKR (orders/positions/quotes) + Questrade (option chains)")
        else:
            logger.info("SINGLE BROKER: IBKR for everything (including option chains)")
        
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
        
        # Resolve capital: CLI override > broker > config fallback
        if capital is None:
            try:
                summary = ibkr_client.get_account_balances(account_id)
                nlv = summary.get('NetLiquidation', {}).get('USD')
                if nlv is None:
                    nlv = summary.get('NetLiquidation', {}).get('CAD')
                if nlv is not None:
                    capital = float(nlv)
                    logger.info(f"Account equity from IBKR: ${capital:,.2f}")
                if capital is None:
                    live_cfg = full_config.get('live', {})
                    capital = live_cfg.get('fallback_capital', _default_capital())
                    logger.warning(f"Could not read equity from broker, using fallback: ${capital:,.2f}")
            except Exception as e:
                live_cfg = full_config.get('live', {})
                capital = live_cfg.get('fallback_capital', _default_capital())
                logger.warning(f"Failed to fetch account equity: {e}, using fallback: ${capital:,.2f}")
        
        # Create engine
        db_path = os.path.join(data_dir, 'live_0dte_trades.db')
        engine = create_engine(
            client=client,
            account_id=account_id,
            symbols=["SPY"],
            option_underlyings=["SPY"],
            mode=args.mode,
            db_path=db_path,
            quote_client=quote_client,
            chains_client=chains_client
        )
        
        # Create 0DTE strategy with all optimized params from strategy.json
        strategy = create_0dte_strategy(
            account_capital=capital,
            strategy=strategy_name,
            profit_target_pct=trade_config.get('profit_target_pct', 0.50),
            stop_loss_pct=trade_config.get('stop_loss_pct', 0.35),
            max_contracts=max_contracts_val,
            max_daily_losses=risk_config.get('max_daily_losses', 2),
            max_consecutive_losses=risk_config.get('max_consecutive_losses', 3),
            max_daily_loss_pct=risk_config.get('max_daily_loss_pct', 0.008),
            min_option_price=trade_config.get('min_option_price', 0.50),
            max_option_price=trade_config.get('max_option_price', 2.00),
            orb_minutes=trade_config.get('orb_minutes', 30),
            orb_buffer_pct=trade_config.get('orb_buffer_pct', 0.10),
            max_hold_minutes=trade_config.get('max_hold_minutes', 80),
            trade_start_hour=trade_config.get('trade_start_hour', 9),
            trade_start_minute=trade_config.get('trade_start_minute', 35),
            trade_end_hour=trade_config.get('trade_end_hour', 15),
            trade_end_minute=trade_config.get('trade_end_minute', 0),
            exit_hour=trade_config.get('exit_hour', 15),
            rsi_call_threshold=trade_config.get('rsi_call_threshold', 70),
            rsi_put_threshold=trade_config.get('rsi_put_threshold', 35),
            # Asymmetric CALL/PUT exits
            call_profit_target_pct=trade_config.get('call_profit_target_pct'),
            put_profit_target_pct=trade_config.get('put_profit_target_pct'),
            call_stop_loss_pct=trade_config.get('call_stop_loss_pct'),
            put_stop_loss_pct=trade_config.get('put_stop_loss_pct'),
            # Regime detection
            use_regime_detection=trade_config.get('use_regime_detection', False),
            regime_lookback_days=trade_config.get('regime_lookback_days', 5),
            regime_vol_percentile=trade_config.get('regime_vol_percentile', 0.30),
            regime_trend_percentile=trade_config.get('regime_trend_percentile', 0.25),
            regime_size_reduction=trade_config.get('regime_size_reduction', 0.50),
            regime_skip_first_bar=trade_config.get('regime_skip_first_bar', True),
            regime_rsi_buffer=trade_config.get('regime_rsi_buffer', 5),
            regime_tighter_stop_pct=trade_config.get('regime_tighter_stop_pct'),
            broker_client=ibkr_client
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
        
        # Sync historical trades from all sources before anything else
        try:
            from live.trade_sync import TradeSync
            if hasattr(engine, 'db') and engine.db is not None:
                syncer = TradeSync(engine.db)
                sync_results = syncer.sync_all(ibkr_client=ibkr_client)
                if any(v > 0 for v in sync_results.values()):
                    # Reconcile state after importing trades
                    if hasattr(strategy, 'persistence') and strategy.persistence:
                        strategy.persistence.reconcile_with_db()
                        logger.info("State reconciled after trade sync")
        except Exception as e:
            logger.warning(f"Trade sync at startup failed (non-fatal): {e}")

        # Initial position sync to DB so dashboard shows positions even during sleep
        _sync_engine_positions(engine)
        
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
            
            # Wait for market to open (with periodic position syncs)
            wait_for_market_open(engine=engine)
        
        # Print strategy status
        strategy.print_status()
        
        # Run engine
        logger.info("Starting engine... Press Ctrl+C to stop")
        update_engine_status("live", {
            "mode": args.mode,
            "strategy": strategy_name,
            "capital": capital,
            "started_at": get_eastern_time().strftime('%H:%M:%S')
        })
        engine.run()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        update_engine_status("stopped", {"reason": "user_interrupt"})
    except (ConnectionError, RuntimeError) as e:
        # Connection errors are retryable — don't sys.exit(1).
        # Docker/start.py auto-restart will reconnect, but logging the error
        # without exit(1) lets the process survive transient blips.
        logger.error(f"Connection error (will be retried by supervisor): {e}", exc_info=True)
        update_engine_status("error", {"error_message": str(e)[:100]})
        # Exit with code 2 to distinguish from fatal errors.
        # Docker restart: unless-stopped will restart on any non-zero exit.
        sys.exit(2)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        update_engine_status("error", {"error_message": str(e)[:100]})
        sys.exit(1)


def quick_start(capital: float = None, strategy: str = "orb", mode: str = "monitor"):
    """
    Quick start function for Python usage.
    
    Usage:
        from live.runner_0dte import quick_start
        quick_start(strategy="orb", mode="paper")  # auto-detects capital from broker
    """
    
    from live.engine import create_engine
    from live.strategy_0dte import create_0dte_strategy
    from clients.ibkr_adapter import create_ibkr_client
    from clients.questrade_client import create_questrade_client
    
    # Get project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # IBKR primary, Questrade only for option chains
    qt_client = None
    ibkr_client = None
    
    try:
        env_host = os.environ.get("IBKR_HOST")
        env_port = os.environ.get("IBKR_LIVE_PORT") if mode.upper() == "LIVE" else os.environ.get("IBKR_PAPER_PORT")
        ibkr_port = int(env_port) if env_port else (_default_live_port() if mode.upper() == "LIVE" else _default_paper_port())
        ibkr_client = create_ibkr_client(port=ibkr_port, host=env_host)
        ibkr_client.get_accounts()
    except Exception:
        ibkr_client = None
    
    if not ibkr_client:
        raise RuntimeError("IBKR could not connect. IBKR is required.")
    
    try:
        qt_client = create_questrade_client()
        qt_client.get_accounts()
    except Exception:
        qt_client = None
    
    client = ibkr_client
    quote_client = ibkr_client
    chains_client = qt_client or ibkr_client
    
    accounts = client.get_accounts()
    account_id = str(accounts[0]['number'])
    
    # Resolve capital
    if capital is None:
        capital = _default_capital()
    
    # Create engine
    db_path = os.path.join(data_dir, 'live_0dte_trades.db')
    engine = create_engine(
        client=client,
        account_id=account_id,
        symbols=["SPY"],
        option_underlyings=["SPY"],
        mode=mode,
        db_path=db_path,
        quote_client=quote_client,
        chains_client=chains_client
    )
    
    # Create and add strategy
    strategy_obj = create_0dte_strategy(account_capital=capital, strategy=strategy, broker_client=ibkr_client)
    engine.add_strategy(strategy_obj)
    
    # Run
    print(f"Running 0DTE {strategy} strategy with ${capital:,} capital (IBKR)")
    print(f"Mode: {mode.upper()}")
    engine.run()


if __name__ == "__main__":
    main()
