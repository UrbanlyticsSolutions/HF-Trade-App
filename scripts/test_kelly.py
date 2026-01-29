"""Test Kelly sizing"""
from core.risk_manager import RiskManager, RiskConfig, KellyCalculator

capital = 10000
config = RiskConfig(
    kelly_fraction=0.20,
    min_kelly_pct=0.02,
    max_kelly_pct=0.20,
    max_risk_per_trade_pct=0.02,
    max_position_pct=0.07,
    max_position_value=5000,
    max_contracts=50
)
rm = RiskManager(capital, config)

# Calculate Kelly from backtest stats
kelly_calc = KellyCalculator(config)
kelly_pct = kelly_calc.calculate_from_winrate(
    win_rate=0.912,    # 91.2%
    avg_win=0.22,      # 22%
    avg_loss=0.25      # 25%
)
rm.set_kelly(kelly_pct)

print(f"Backtest Stats: 91.2% WR, 22% avg win, 25% avg loss")
print(f"Kelly Fraction: {kelly_pct:.1%}")
print(f"Capital: ${capital:,}")
print()

# Test with different option prices
for price in [0.50, 0.75, 1.00, 1.20]:
    contracts, pos_val = rm.get_position_size(price, stop_loss_pct=0.25)
    print(f"Option ${price:.2f}: {contracts} contracts (${pos_val:.0f})")
