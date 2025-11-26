"""
Compare All Strategies
Runs all 3 strategies and compares results
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_mean_reversion import test_mean_reversion
from test_vwap_strategy import test_vwap_strategy
from test_recent_performance import run_recent_test

def compare_strategies(days=30):
    print("\n" + "="*80)
    print("STRATEGY COMPARISON - LAST 30 DAYS")
    print("="*80 + "\n")
    
    # Test 1: Mean Reversion
    print("\n1️⃣  MEAN REVERSION (Bollinger Bands + RSI)")
    print("-" * 80)
    mr_trades = test_mean_reversion(days)
    
    # Test 2: VWAP
    print("\n\n2️⃣  VWAP (Volume-Weighted Average Price)")
    print("-" * 80)
    vwap_trades = test_vwap_strategy(days)
    
    # Test 3: ML Momentum (current)
    print("\n\n3️⃣  ML MOMENTUM (Current Strategy)")
    print("-" * 80)
    ml_metrics = run_recent_test(days)
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY COMPARISON")
    print("="*80)
    
    print(f"\n{'Strategy':<20} {'Trades':<10} {'Trades/Day':<12} {'Win Rate':<12}")
    print("-" * 80)
    
    if mr_trades:
        import pandas as pd
        mr_df = pd.DataFrame(mr_trades)
        print(f"{'Mean Reversion':<20} {len(mr_trades):<10} {len(mr_trades)/days:<12.1f} {mr_df['profitable'].mean():<12.2%}")
    
    if vwap_trades:
        import pandas as pd
        vwap_df = pd.DataFrame(vwap_trades)
        print(f"{'VWAP':<20} {len(vwap_trades):<10} {len(vwap_trades)/days:<12.1f} {vwap_df['profitable'].mean():<12.2%}")
    
    if ml_metrics:
        print(f"{'ML Momentum':<20} {ml_metrics.get('total_trades', 0):<10} {ml_metrics.get('total_trades', 0)/days:<12.1f} {ml_metrics.get('win_rate', 0):<12.2%}")
    
    print("\n" + "="*80)
    print("💡 RECOMMENDATION:")
    
    # Determine best strategy
    strategies = []
    if mr_trades:
        strategies.append(('Mean Reversion', len(mr_trades)/days, mr_df['profitable'].mean()))
    if vwap_trades:
        strategies.append(('VWAP', len(vwap_trades)/days, vwap_df['profitable'].mean()))
    if ml_metrics:
        strategies.append(('ML Momentum', ml_metrics.get('total_trades', 0)/days, ml_metrics.get('win_rate', 0)))
    
    if strategies:
        # Sort by trades/day * win_rate (frequency * quality)
        strategies.sort(key=lambda x: x[1] * x[2], reverse=True)
        best = strategies[0]
        print(f"   Best: {best[0]} ({best[1]:.1f} trades/day, {best[2]:.1%} win rate)")
        print(f"   Consider combining top 2 strategies for 20-25 trades/day target")
    
    print("="*80)

if __name__ == "__main__":
    compare_strategies(days=30)
