import os
import sys
import argparse
from dotenv import load_dotenv

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clients.fmp_stable_client import FMPStableClient
from strategy.ml_trend_following_strategy import MLTrendFollowingStrategy
from strategy.risk_manager import RiskLimits
from strategy.strategy_engine import TradingEngine

def main():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        print("Error: FMP_API_KEY not found in environment variables or .env file.")
        return

    parser = argparse.ArgumentParser(description="Run HFT Strategy")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol to trade")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds")
    
    # Risk management parameters
    parser.add_argument("--max-position", type=int, default=1000, help="Maximum position size")
    parser.add_argument("--max-portfolio", type=float, default=100000.0, help="Maximum portfolio value")
    parser.add_argument("--stop-loss", type=float, default=0.02, help="Stop loss percentage (0.02 = 2%%)")
    parser.add_argument("--profit-target", type=float, default=0.03, help="Profit target percentage")
    parser.add_argument("--max-drawdown", type=float, default=0.10, help="Maximum drawdown before circuit breaker")
    
    args = parser.parse_args()

    print(f"Initializing Trend Following Strategy for {args.symbol}...")
    print(f"Settings: Interval={args.interval}s")

    # Initialize Client
    fmp_client = FMPStableClient(api_key=api_key)

    # Create risk limits
    risk_limits = RiskLimits(
        max_position_size=args.max_position,
        max_portfolio_value=args.max_portfolio,
        stop_loss_pct=args.stop_loss,
        profit_target_pct=args.profit_target,
        max_drawdown_pct=args.max_drawdown
    )
    
    strategy = MLTrendFollowingStrategy(
        symbol=args.symbol,
        risk_limits=risk_limits
    )
    
    print(f"Risk Limits: MaxPos={args.max_position}, StopLoss={args.stop_loss:.1%}, "
          f"ProfitTarget={args.profit_target:.1%}, MaxDD={args.max_drawdown:.1%}")

    # Initialize Engine
    engine = TradingEngine(client=fmp_client, strategy=strategy, interval=args.interval)

    # Run
    try:
        engine.run_live()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Cleanup - call shutdown if strategy has it
        if hasattr(strategy, 'shutdown'):
            strategy.shutdown()

if __name__ == "__main__":
    main()
