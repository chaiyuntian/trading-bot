# Crypto Trading Bot

Automated cryptocurrency trading bot with multiple strategies, backtesting engine, walk-forward validation, and risk management.

> **WARNING**: Trading involves risk of financial loss. This project is for learning and research purposes only. Never invest more than you can afford to lose. Past backtest performance does not guarantee future results.

## Features

- **6 Trading Strategies**: RSI+MACD, Mean Reversion, Grid Trading, DCA Momentum, Ensemble (regime-adaptive), KAMA Trend
- **Adaptive Intelligence**: KAMA (Kaufman Adaptive MA), dynamic ensemble weighting, walk-forward optimization
- **Risk Management**: Position sizing, stop-loss/take-profit, max drawdown limits, daily loss limits
- **Backtesting Engine**: Historical data backtesting with fee/slippage modeling
- **Walk-Forward Validation**: Out-of-sample testing to detect overfitting
- **System Verification**: End-to-end smoke tests, integration tests, and validation suite
- **Paper Trading**: Simulated trading mode — no real money required
- **Multi-Exchange**: Supports Binance, Bybit, KuCoin, and 100+ exchanges via ccxt

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your settings

# 3. Backtest (recommended first step!)
python -m src.main --backtest

# 4. Backtest all strategies for comparison
python -m src.main --backtest-all

# 5. Walk-forward validation (detect overfitting)
python -m src.main --walk-forward

# 6. Paper trading (simulated)
python -m src.main

# 7. Live trading (use with caution!)
# Set mode: "live" in config.yaml and add your API keys
python -m src.main

# 8. Run system verification
python -m tests.verify
```

## Strategies

| Strategy | Best For | How It Works |
|----------|----------|--------------|
| `rsi_macd` | Trend + Momentum | RSI oversold recovery + MACD golden cross + EMA trend filter |
| `mean_reversion` | Ranging Markets | Price touches lower Bollinger Band + RSI oversold = buy |
| `grid` | Sideways Markets | Sets grid levels within a price range, buys low / sells high |
| `dca_momentum` | Long-term Accumulation | Dollar-cost averaging + momentum filter, buys only at favorable times |
| `ensemble` | All Conditions | Regime-adaptive multi-strategy voting with dynamic weighting |
| `kama_trend` | Adaptive Trending | Uses Kaufman Adaptive MA for fewer whipsaws in noisy markets |

## Risk Management Parameters

```yaml
risk:
  max_position_pct: 0.30    # Max position size per trade: 30%
  stop_loss_pct: 0.03       # Stop loss: 3%
  take_profit_pct: 0.06     # Take profit: 6% (2:1 reward-to-risk)
  max_daily_loss_pct: 0.05  # Daily max loss: 5% -> halt trading
  max_drawdown_pct: 0.25    # Max drawdown: 25% -> halt trading
  risk_per_trade_pct: 0.02  # Risk per trade: 2%
```

## Project Structure

```
├── config/
│   └── config.example.yaml      # Configuration template
├── src/
│   ├── main.py                   # Entry point (CLI)
│   ├── bot.py                    # Trading bot orchestrator
│   ├── exchange/
│   │   ├── base.py               # Exchange abstraction (ABC)
│   │   ├── ccxt_adapter.py       # Live exchange adapter (100+ exchanges)
│   │   └── paper_adapter.py      # Simulated trading adapter
│   ├── strategies/
│   │   ├── base.py               # Strategy interface
│   │   ├── rsi_macd.py           # RSI + MACD strategy
│   │   ├── mean_reversion.py     # Mean reversion strategy
│   │   ├── grid_trading.py       # Grid trading strategy
│   │   ├── dca_momentum.py       # DCA momentum strategy
│   │   ├── ensemble.py           # Regime-adaptive ensemble
│   │   ├── kama_trend.py         # KAMA adaptive trend strategy
│   │   └── regime.py             # Market regime detection
│   ├── risk/manager.py           # Risk management
│   ├── indicators/technical.py   # Technical indicators (RSI, MACD, KAMA, etc.)
│   ├── backtesting/
│   │   ├── engine.py             # Backtest engine
│   │   └── walk_forward.py       # Walk-forward optimization
│   └── utils/logger.py           # Logging
├── tests/
│   ├── test_indicators.py        # Indicator unit tests
│   ├── test_risk_manager.py      # Risk manager unit tests
│   ├── test_strategies.py        # Strategy unit tests
│   ├── test_integration.py       # Integration tests
│   └── verify.py                 # Full system verification suite
└── docs/
    └── QUANT_RESEARCH.md         # Academic research references
```

## System Verification

Run the full verification suite to confirm everything works:

```bash
# Quick verification (no network, uses synthetic data)
python -m tests.verify

# Run unit tests
python -m pytest tests/ -v

# Run integration tests
python -m pytest tests/test_integration.py -v
```

The verification suite checks:
- All indicators compute correctly
- All strategies produce valid signals
- Risk manager enforces limits properly
- Backtest engine produces consistent results
- Walk-forward validation detects overfitting
- Ensemble regime detection adapts correctly
- Full pipeline: data -> indicators -> strategy -> risk -> execution

## Usage Tips

1. **Backtest first** — Use `--backtest-all` to find the best strategy for current market conditions
2. **Validate** — Run `--walk-forward` to check for overfitting before deploying
3. **Paper trade** — Simulate for at least 1-2 weeks before going live
4. **Start small** — Use small capital when going live for the first time
5. **Monitor continuously** — Automated does not mean unattended; check and adjust regularly
