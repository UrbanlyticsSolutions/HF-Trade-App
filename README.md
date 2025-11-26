# HF-Trade: High-Frequency Trading System

ML-enhanced intraday trading system with trend-following strategy and optimized entry/exit points.

## 📁 Project Structure

```
HF-Trade/
├── strategy/              # Trading strategy implementations
│   ├── hft_momentum_strategy.py
│   ├── ml_trend_following_strategy.py
│   └── risk_manager.py
├── models/                # Trained ML models
│   ├── ml_ensemble.pkl
│   ├── ml_catboost.pkl
│   └── training_metrics.json
├── scripts/               # Data fetching and generation
│   ├── fetch_intraday_data.py
│   ├── generate_training_data.py
│   └── generate_trend_training_data.py
├── backtest/              # Backtesting and optimization
│   ├── backtest_hft.py
│   ├── backtest_trend_following.py
│   └── optimize_trend_parameters.py
├── data/                  # Training data (gitignored)
├── output/                # Backtest results (gitignored)
├── clients/               # API clients
├── ml_trade_classifier.py # ML model training
├── main_hft.py            # Main entry point
└── market_data.db         # SQLite database (gitignored)
```

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Fetch Data**
   ```bash
   python scripts/fetch_intraday_data.py
   ```

4. **Train ML Model**
   ```bash
   python ml_trade_classifier.py
   ```

5. **Run Backtest**
   ```bash
   python backtest/backtest_trend_following.py
   ```

6. **Stream Real-Time Quotes (optional)**
   ```bash
   python scripts/stream_realtime_quotes.py --symbol QQQ --interval 5
   ```
   This polls FMP's regular quote API during market hours and automatically
   switches to the after-market endpoint outside the session, persisting each
   snapshot to `market_data.db` (`realtime_quotes` table).

## 📊 Current Performance

### ML Model (Trend-Specific)
- **Test Accuracy**: 61.1%
- **Precision (Win)**: 56.6%
- **Recall (Win)**: 60.7%
- **Training Samples**: 4,674 (SMOTE balanced)

### Intraday Trend-Following Strategy
- **Win Rate**: 51.9%
- **Profit Factor**: 2.83
- **Trades/Day**: 4.4
- **Total P/L**: $123.66 (60 days)

## 🔧 Configuration

Edit `strategy_config.json` to adjust:
- Entry/exit thresholds
- Risk management parameters
- ML probability thresholds

## 📚 Documentation

- [Quick Start Guide](QUICK_START.md)
- [ML Workflow](ML_WORKFLOW.md)
- [Strategy Enhancements](STRATEGY_ENHANCEMENTS.md)
- [HFT Strategy Details](README_HFT_STRATEGY.md)

## 🎯 Key Features

- ✅ ML-enhanced entry/exit prediction (61% accuracy)
- ✅ Trend-following with MA20/MA50 filter
- ✅ SMOTE class balancing
- ✅ Optuna hyperparameter optimization
- ✅ Sharpe-based labeling for quality trades
- ✅ Real-time risk management
- ✅ Comprehensive backtesting framework

## 📈 Next Steps

1. Deploy with best configuration (ML=0.60, MA20/50)
2. Monitor live performance
3. Further optimization (feature selection, advanced labeling)
