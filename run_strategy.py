"""
Run HFT Strategy with JSON Configuration
Loads parameters from strategy_config.json instead of command-line arguments
"""
import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clients.fmp_stable_client import FMPStableClient
from strategy.advanced_hft_momentum_strategy import AdvancedHFTMomentumStrategy
from strategy.risk_manager import RiskLimits
from strategy.strategy_engine import TradingEngine


def load_config(config_file="strategy_config.json"):
    """Load strategy configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        print("Error: FMP_API_KEY not found in environment variables or .env file.")
        return

    # Load configuration from JSON
    print("Loading strategy configuration from strategy_config.json...")
    config = load_config()
    
    print(f"\nStrategy: {config['strategy_name']}")
    print(f"Optimized for: {config['optimized_for']}")
    print(f"Expected Win Rate: {config['performance_targets']['expected_win_rate']:.1%}")
    print(f"Expected Profit Factor: {config['performance_targets']['expected_profit_factor']:.2f}")
    print()
    
    # Extract parameters
    momentum = config['momentum_parameters']
    risk = config['risk_management']
    
    # Create risk limits from config
    risk_limits = RiskLimits(
        max_position_size=risk['max_position_size'],
        max_portfolio_value=risk['max_portfolio_value'],
        max_position_pct=risk['max_position_pct'],
        max_drawdown_pct=risk['max_drawdown_pct'],
        stop_loss_pct=risk['stop_loss_pct'],
        profit_target_pct=risk['profit_target_pct'],
        trailing_stop_pct=risk['trailing_stop_pct'],
        max_daily_loss=risk['max_daily_loss']
    )
    
    # Initialize FMP Client
    fmp_client = FMPStableClient(api_key=api_key)
    
    # Create strategy with config parameters
    symbol = input("Enter symbol to trade (default: QQQ): ").strip() or "QQQ"
    
    strategy = AdvancedHFTMomentumStrategy(
        symbol=symbol,
        fast_window=momentum['fast_window'],
        medium_window=momentum['medium_window'],
        slow_window=momentum['slow_window'],
        volume_window=momentum['volume_window'],
        atr_window=momentum['atr_window'],
        base_threshold_pct=momentum['base_threshold_pct'],
        volume_threshold=momentum['volume_threshold'],
        risk_limits=risk_limits,
        output_dir="output/live"
    )
    
    print(f"\n{'='*70}")
    print(f"STRATEGY CONFIGURATION")
    print(f"{'='*70}")
    print(f"Symbol: {symbol}")
    print(f"\nMomentum Windows: {momentum['fast_window']}/{momentum['medium_window']}/{momentum['slow_window']}")
    print(f"Threshold: {momentum['base_threshold_pct']:.4%}")
    print(f"\nRisk Management:")
    print(f"  Stop-Loss: {risk['stop_loss_pct']:.1%}")
    print(f"  Profit-Target: {risk['profit_target_pct']:.1%}")
    print(f"  Trailing-Stop: {risk['trailing_stop_pct']:.1%}")
    print(f"  Max Position: {risk['max_position_size']} shares")
    print(f"  Max Drawdown: {risk['max_drawdown_pct']:.1%}")
    print(f"\nOrder Execution:")
    print(f"  Limit Orders: {'✓' if config['order_execution']['use_limit_orders'] else '✗'}")
    print(f"  Stop-Loss Orders: {'✓' if config['order_execution']['use_stop_loss_orders'] else '✗'}")
    print(f"  Profit-Target Orders: {'✓' if config['order_execution']['use_profit_target_orders'] else '✗'}")
    print(f"  Trailing Stops: {'✓' if config['order_execution']['use_trailing_stops'] else '✗'}")
    print(f"\nPosition Management:")
    print(f"  Multiple Positions: {'✓' if config['position_management']['allow_multiple_positions'] else '✗'}")
    print(f"  Max Simultaneous: {config['position_management']['max_simultaneous_positions']}")
    print(f"  Pyramiding: {'✓' if config['position_management']['pyramiding_enabled'] else '✗'}")
    print(f"{'='*70}\n")
    
    # Initialize Engine
    interval = float(input("Enter polling interval in seconds (default: 5.0): ").strip() or "5.0")
    engine = TradingEngine(client=fmp_client, strategy=strategy, interval=interval)
    
    # Run
    try:
        print(f"\nStarting live trading engine...")
        print(f"Press Ctrl+C to stop\n")
        engine.run_live()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        # Cleanup
        if hasattr(strategy, 'shutdown'):
            strategy.shutdown()
        print("\nStrategy stopped.")


if __name__ == "__main__":
    main()
