# Quantitative Trading Research Reference (2024-2026)

## Implementation Priority (Based on Research)

### P0: Foundation
1. **HMM-based regime detection** (3-state: bull/bear/sideways) using returns + volatility + Kaufman ER
2. **Strategy ensemble** with dynamic Sharpe-weighted softmax voting, recalibrated on regime changes
3. **Half-Kelly default**, quarter-Kelly in volatile regimes, 2% per-trade hard cap

### P1: Adaptive Intelligence
4. **KAMA** (Kaufman Adaptive MA) to replace fixed EMAs for trend signals
5. **Walk-forward testing** with 30-day rolling reoptimization windows
6. **Sortino optimization** (not Sharpe) for crypto — target Sortino > 2.0

### P2: Microstructure Edge
7. **VWAP deviation Z-scores** for mean-reversion entries (Z > +2 or < -2)
8. **Volume profile** (POC/VA) for dynamic S/R levels
9. **Order flow imbalance** aggregated into hourly bars

### P3: ML Layer
10. **SAC (Soft Actor-Critic)** for portfolio management and mean-reversion
11. **FinGPT sentiment** as supplementary signal (10-20% ensemble weight)
12. **Genetic algorithm parameter optimization** (30-day rolling windows)

## Key Metrics Targets

| Metric | Passive BTC | Our Target |
|--------|------------|------------|
| Sharpe Ratio | ~0.95 | > 1.0 |
| Sortino Ratio | ~1.93 | > 2.0 |
| Calmar Ratio | ~0.84 | > 1.0 |
| Max Drawdown | ~-70% | < -25% |

## Critical Warning: Overfitting

Strategies that ignore fees overstate performance by 30-50%.
Walk-forward testing with rolling windows is essential.
Use CSCV (combinatorial symmetric cross-validation) to reject overfitted agents.

## Sources

See full research document for 80+ academic paper references (2024-2026).
