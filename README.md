# 0DTE SPY Options Backtest

Backtest system for 0DTE SPY options using RSI momentum signals.

## Quick Start

```bash
python run.py backtest --year 2025    # Backtest 2025
python run.py backtest                 # Backtest configured dates
python run.py analyze                  # Analyze results
```

## Strategy

| Parameter | Value |
|-----------|-------|
| **Window** | 10:00 - 11:00 AM ET |
| **Options** | $0.50 - $1.00 |
| **Profit Target** | +22% |
| **Stop Loss** | -25% |
| **Signal** | RSI > 70 → CALL, RSI < 30 → PUT |

## Performance

| Period | Trades | Win Rate | P&L | DD |
|--------|--------|----------|-----|-----|
| Train 2024 H1 | 156 | 75.6% | $360K | 5% |
| Test 2024 H2 | 181 | 81.2% | $934K | 7% |
| **2025 OOS** | 392 | **84.4%** | **$2.3M** | 4% |

## Structure

```
HF-Trade/
├── run.py              # Main entry point
├── config/strategy.json # All settings
├── backtest/engine.py  # Backtest engine
├── indicators/         # RSI, ATR, etc.
├── core/risk_manager.py # Position sizing
├── clients/            # Data loaders
└── output/             # Results & charts
```

## Configuration

Edit `config/strategy.json` to modify parameters
