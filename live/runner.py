"""
Live Trading Runner - Main entry point for live trading

Usage:
    python -m live.runner --account YOUR_ACCOUNT_ID --symbols AAPL,MSFT --options SPY,QQQ
    python -m live.runner --account YOUR_ACCOUNT_ID --mode monitor  # Monitor only
    python -m live.runner --account YOUR_ACCOUNT_ID --mode paper    # Paper trading
    python -m live.runner --account YOUR_ACCOUNT_ID --mode live     # Live trading
"""
import argparse
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Live Trading System")
    parser.add_argument("--account", required=True, help="Questrade account ID")
    parser.add_argument("--symbols", default="", help="Comma-separated stock symbols")
    parser.add_argument("--options", default="", help="Comma-separated option underlyings")
    parser.add_argument("--mode", default="monitor", choices=["monitor", "paper", "live"],
                        help="Trading mode: monitor (no orders), paper (simulated), live (real)")
    parser.add_argument("--db", default="live_trades.db", help="Database file path")
    parser.add_argument("--interval", type=float, default=5.0, help="Quote interval in seconds")
    parser.add_argument("--max-loss", type=float, default=1000.0, help="Max daily loss before stopping")
    parser.add_argument("--refresh-token", help="Questrade refresh token (optional)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"live_trading_{datetime.now().strftime('%Y%m%d')}.log")
        ]
    )
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    option_underlyings = [s.strip().upper() for s in args.options.split(",") if s.strip()]
    
    logger.info("=" * 60)
    logger.info("LIVE TRADING SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Account: {args.account}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Option Underlyings: {option_underlyings}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Max Daily Loss: ${args.max_loss}")
    logger.info("=" * 60)
    
    try:
        # Import after path setup
        from clients.ibkr_adapter import create_ibkr_client
        from clients.questrade_client import create_questrade_client
        from live.engine import create_engine, EngineConfig
        from live.strategy import CoveredCallStrategy, PutCreditSpreadStrategy
        from config.defaults import ibkr_live_port, ibkr_paper_port
        
        # Connect IBKR (primary — orders, positions, account, quotes)
        logger.info("Connecting to IBKR (primary broker)...")
        env_host = os.environ.get("IBKR_HOST")
        env_port = os.environ.get("IBKR_LIVE_PORT") if args.mode == "live" else os.environ.get("IBKR_PAPER_PORT")
        ibkr_port = int(env_port) if env_port else (ibkr_live_port() if args.mode == "live" else ibkr_paper_port())
        client = create_ibkr_client(port=ibkr_port, host=env_host)
        
        # Questrade for option chains only
        chains_client = None
        try:
            qt = create_questrade_client(refresh_token=args.refresh_token)
            qt.get_accounts()
            chains_client = qt
            logger.info("Questrade connected (option chains only)")
        except Exception as qe:
            logger.warning(f"Questrade unavailable: {qe} — IBKR will handle option chains too")
        
        # Verify connection
        accounts = client.get_accounts()
        logger.info(f"Connected! Found {len(accounts)} accounts")
        
        # Verify account exists
        account_ids = [str(a.get('number')) for a in accounts]
        if args.account not in account_ids:
            logger.error(f"Account {args.account} not found. Available: {account_ids}")
            sys.exit(1)
        
        # Create engine
        engine = create_engine(
            client=client,
            account_id=args.account,
            symbols=symbols,
            option_underlyings=option_underlyings,
            mode=args.mode,
            db_path=args.db,
            chains_client=chains_client
        )
        
        # Update config
        engine.config.quote_interval = args.interval
        engine.config.max_daily_loss = args.max_loss
        
        # Add strategies if trading options
        if option_underlyings:
            # Add example strategies
            engine.add_strategy(CoveredCallStrategy(option_underlyings, target_delta=0.30))
            engine.add_strategy(PutCreditSpreadStrategy(option_underlyings, target_credit=0.30))
        
        # Register quote callback for logging
        def on_quote(symbol, quote):
            price = quote.get('lastTradePrice', 0)
            change = quote.get('lastTradeChange', 0)
            logger.debug(f"Quote: {symbol} ${price:.2f} ({change:+.2f})")
        
        engine.on_quote(on_quote)
        
        # Run engine
        logger.info("Starting engine... Press Ctrl+C to stop")
        engine.run()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


def run_monitor(account_id: str, symbols: List[str] = None, option_underlyings: List[str] = None):
    """
    Quick function to start monitoring (no trading).
    
    Usage from Python:
        from live.runner import run_monitor
        run_monitor("12345678", symbols=["AAPL", "MSFT"], option_underlyings=["SPY"])
    """
    from clients.ibkr_adapter import create_ibkr_client
    from live.engine import create_engine
    from config.defaults import ibkr_paper_port
    
    env_host = os.environ.get("IBKR_HOST")
    env_port = os.environ.get("IBKR_PAPER_PORT")
    port = int(env_port) if env_port else ibkr_paper_port()
    client = create_ibkr_client(port=port, host=env_host)
    engine = create_engine(
        client=client,
        account_id=account_id,
        symbols=symbols,
        option_underlyings=option_underlyings,
        mode="monitor"
    )
    
    engine.run()


def run_with_strategy(
    account_id: str,
    strategy,
    symbols: List[str] = None,
    option_underlyings: List[str] = None,
    mode: str = "monitor"
):
    """
    Run with a custom strategy.
    """
    from clients.ibkr_adapter import create_ibkr_client
    from live.engine import create_engine
    from config.defaults import ibkr_paper_port
    
    env_host = os.environ.get("IBKR_HOST")
    env_port = os.environ.get("IBKR_PAPER_PORT")
    port = int(env_port) if env_port else ibkr_paper_port()
    client = create_ibkr_client(port=port, host=env_host)
    engine = create_engine(
        client=client,
        account_id=account_id,
        symbols=symbols,
        option_underlyings=option_underlyings,
        mode=mode
    )
    
    engine.add_strategy(strategy)
    engine.run()


# Type hints for run_monitor
from typing import List


if __name__ == "__main__":
    main()
