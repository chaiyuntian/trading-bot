# Trading Bot Evolution Roadmap

## Iteration Philosophy

This project follows a continuous evolution approach powered by quantitative research
and the EvoMap GEP (Genome Evolution Protocol). Every iteration must be:

1. **Data-driven**: Changes backed by backtest results, not intuition
2. **Risk-bounded**: No change may increase max drawdown beyond configured limits
3. **Measurable**: Every strategy change must show improvement in Sortino ratio (primary) or Sharpe ratio
4. **Reversible**: All changes tracked in version control with rollback capability
5. **Research-informed**: Integrate latest academic findings (see docs/QUANT_RESEARCH.md)

## Architecture Principles

- **Platform-agnostic**: All trading logic operates through `ExchangeAdapter` abstraction
- **Strategy-decoupled**: Strategies are pure signal generators, independent of execution
- **Regime-aware**: Ensemble strategy adapts to market conditions automatically
- **Performance-first**: OHLCV caching, numpy-accelerated calculations, minimal API calls
- **Verifiable**: Full system verification suite (run `python -m tests.verify`)

## Quality Targets (From 2024-2026 Research)

| Metric | Passive BTC | Our Target |
|--------|------------|------------|
| Sharpe Ratio | ~0.95 | > 1.0 |
| Sortino Ratio | ~1.93 | > 2.0 |
| Calmar Ratio | ~0.84 | > 1.0 |
| Max Drawdown | ~-70% | < -25% |
| Win Rate | - | > 55% |

## Evolution Stages

### Stage 1: Foundation (Complete)
- [x] Core exchange abstraction layer (ExchangeAdapter ABC)
- [x] CCXT adapter (100+ exchanges: Binance, Bybit, KuCoin, etc.)
- [x] Paper trading adapter (simulated execution with real market data)
- [x] 4 base strategies (RSI+MACD, Mean Reversion, Grid, DCA Momentum)
- [x] Risk management (Kelly position sizing, stop-loss, trailing stop, drawdown limits)
- [x] Backtesting engine with Sharpe/Sortino/profit factor metrics
- [x] Market regime detection (ADX + BB width + EMA slope + ATR spike)
- [x] Ensemble strategy with regime-weighted multi-strategy voting
- [x] Rich signal system (confidence scoring, metadata, entry/exit prices)
- [x] Evolver integration (GEP protocol, --loop mode, innovate strategy)

### Stage 2: Adaptive Intelligence (Complete)
- [x] **KAMA** (Kaufman Adaptive MA) indicator and strategy — fewer whipsaws in noisy markets
- [x] **Efficiency Ratio** indicator for trend quality measurement
- [x] **KAMA Trend Strategy** — adaptive trend following with ER-based confidence
- [x] **Dynamic ensemble weighting** (rolling Sharpe-weighted softmax voting)
- [x] **Walk-forward optimization** (rolling train/test window validation)
- [x] **System verification suite** (end-to-end testing without network)
- [x] **Integration tests** (full pipeline: data -> indicators -> strategy -> risk -> backtest)

### Stage 3: Advanced Adaptive (Next)
- [ ] **MAMA/FAMA** (Ehlers) for high-confidence regime change confirmation
- [ ] **Ehlers Sinewave** for cycle vs. trend mode detection
- [ ] **HMM regime detection** (3-state Hidden Markov Model, retrained monthly)
- [ ] **Multi-timeframe analysis** (15m signals confirmed by 1h/4h trend)
- [ ] **Regime smoothing** (prevent rapid regime flipping with hysteresis)

### Stage 4: Microstructure Edge
- [ ] **VWAP deviation Z-scores** for mean-reversion (Z > +2 or < -2)
- [ ] **Volume profile** (POC/VA for dynamic support/resistance)
- [ ] **Order flow imbalance** aggregated into hourly bars
- [ ] **Anti-martingale position scaling** (gradual 10-20% adjustments)
- [ ] Slippage-aware execution (market/limit order allocation)

### Stage 5: ML Layer
- [ ] **SAC** (Soft Actor-Critic) for portfolio management
- [ ] **FinGPT sentiment** as supplementary signal (10-20% weight)
- [ ] **Genetic algorithm** parameter optimization (30-day windows)
- [ ] Lightweight gradient boosting for signal filtering

### Stage 6: Production Scale
- [ ] WebSocket real-time data feed
- [ ] Multi-pair portfolio management
- [ ] Cross-exchange arbitrage detection
- [ ] Dashboard with real-time P&L visualization
- [ ] Circuit breakers for flash crashes
- [ ] Exchange failover (auto-switch between exchanges)
- [ ] Self-improving loop: paper trade -> analyze results -> evolve parameters

## Evolver Integration

The project uses [evolver](https://github.com/autogame-17/evolver) for automated
self-improvement cycles via the GEP protocol:

```bash
cd .evolver && node index.js --loop
```

Strategy: `innovate` — continuously search for improvements in:
- Strategy parameters (RSI thresholds, MACD periods, etc.)
- Risk parameters (position sizing, stop distances)
- New indicator combinations
- Regime detection accuracy
- Backtest performance metrics

## System Verification

Verify the entire system works before deploying:

```bash
# Full verification (no network required)
python -m tests.verify

# Unit tests
python -m pytest tests/ -v

# Walk-forward validation (requires network for data)
python -m src.main --walk-forward
```

## Critical Warnings

1. **No guarantee of profit** — this is a learning/research tool
2. **Always paper trade first** for at least 2 weeks
3. **Never invest more than you can afford to lose**
4. **Strategies that ignore fees overstate performance by 30-50%** (our backtest accounts for fees)
5. **Walk-forward test before deploying** — backtested != live performance
6. **Run verification suite** after any code change
