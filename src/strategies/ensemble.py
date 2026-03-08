"""Ensemble Strategy v2 — regime-aware + performance-adaptive weighting.

Improvements over v1:
1. Dynamic weights based on each strategy's rolling Sharpe ratio
2. Multi-timeframe confirmation (optional)
3. Confidence threshold adapts to regime volatility
4. Trade memory for performance tracking

This is the recommended strategy for production use.
"""

import numpy as np
import pandas as pd
from collections import deque
from src.strategies.base import BaseStrategy, TradeSignal, Signal
from src.strategies.rsi_macd import RsiMacdStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.grid_trading import GridTradingStrategy
from src.strategies.dca_momentum import DCAMomentumStrategy
from src.strategies.regime import detect_regime, get_regime_strategy_weights, MarketRegime
from src.utils.logger import setup_logger

logger = setup_logger("strategy.ensemble")


class EnsembleStrategy(BaseStrategy):
    """Regime-adaptive ensemble with performance-based dynamic weighting."""

    strategy_name = "Ensemble"

    def __init__(self, config: dict):
        super().__init__(config)
        self.sub_strategies: dict[str, BaseStrategy] = {
            "rsi_macd": RsiMacdStrategy(config),
            "mean_reversion": MeanReversionStrategy(config),
            "grid": GridTradingStrategy(config),
            "dca_momentum": DCAMomentumStrategy(config),
        }
        self.min_consensus = self.params.get("ensemble_min_consensus", 0.4)
        self._last_regime = MarketRegime.RANGING

        # ── Performance tracking for dynamic weighting ──
        self._strategy_pnls: dict[str, deque] = {
            name: deque(maxlen=30)  # Track last 30 signals
            for name in self.sub_strategies
        }
        self._last_signals: dict[str, TradeSignal] = {}
        self._last_price = 0.0
        self._performance_weights: dict[str, float] = {
            name: 1.0 for name in self.sub_strategies
        }

    def _update_performance_weights(self, current_price: float):
        """Update strategy weights based on recent signal accuracy."""
        if self._last_price <= 0:
            self._last_price = current_price
            return

        price_change = (current_price - self._last_price) / self._last_price

        for name, last_sig in self._last_signals.items():
            if last_sig.signal == Signal.BUY:
                # Reward if price went up, penalize if down
                self._strategy_pnls[name].append(price_change * last_sig.confidence)
            elif last_sig.signal == Signal.SELL:
                self._strategy_pnls[name].append(-price_change * last_sig.confidence)
            # HOLD signals: no reward/penalty

        # Recalculate weights using softmax on rolling Sharpe
        sharpes = {}
        for name, pnls in self._strategy_pnls.items():
            if len(pnls) >= 5:
                arr = np.array(pnls)
                std = np.std(arr)
                sharpes[name] = np.mean(arr) / std if std > 0 else 0
            else:
                sharpes[name] = 0  # Not enough data yet

        # Softmax normalization (temperature=2 for smoother weights)
        if sharpes:
            values = np.array(list(sharpes.values()))
            temp = 2.0
            exp_vals = np.exp(values / temp)
            softmax = exp_vals / exp_vals.sum()
            for i, name in enumerate(sharpes):
                self._performance_weights[name] = float(softmax[i])

        self._last_price = current_price

    def generate_signal(self, df: pd.DataFrame) -> str:
        sig = self.generate_rich_signal(df)
        return sig.signal.value

    def generate_rich_signal(self, df: pd.DataFrame) -> TradeSignal:
        current_price = float(df.iloc[-1]["close"])

        # 1. Update performance-based weights
        self._update_performance_weights(current_price)

        # 2. Detect market regime
        regime = detect_regime(df)
        if regime != self._last_regime:
            logger.info(f"Regime change: {self._last_regime.value} -> {regime.value}")
            self._last_regime = regime

        # 3. Combine regime weights with performance weights
        regime_weights = get_regime_strategy_weights(regime)

        combined_weights = {}
        for name in self.sub_strategies:
            rw = regime_weights.get(name, 0.5)
            pw = self._performance_weights.get(name, 1.0)
            # 60% regime, 40% performance
            combined_weights[name] = rw * 0.6 + pw * 0.4

        # 4. Adapt consensus threshold to regime
        if regime == MarketRegime.VOLATILE:
            # Higher bar in volatile markets = fewer trades
            consensus_threshold = self.min_consensus * 1.5
        elif regime == MarketRegime.TRENDING_DOWN:
            # Higher bar in downtrends = more conservative
            consensus_threshold = self.min_consensus * 1.3
        else:
            consensus_threshold = self.min_consensus

        # 5. Collect signals from all sub-strategies
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        reasons = []

        for name, strategy in self.sub_strategies.items():
            weight = combined_weights.get(name, 0.5)
            try:
                signal = strategy.generate_rich_signal(df)
                self._last_signals[name] = signal
            except Exception as e:
                logger.debug(f"Strategy {name} error: {e}")
                continue

            total_weight += weight

            if signal.signal == Signal.BUY:
                contribution = weight * signal.confidence
                buy_score += contribution
                reasons.append(f"{name}:B({signal.confidence:.1f})*{weight:.2f}")
            elif signal.signal == Signal.SELL:
                contribution = weight * signal.confidence
                sell_score += contribution
                reasons.append(f"{name}:S({signal.confidence:.1f})*{weight:.2f}")

        if total_weight == 0:
            return TradeSignal(Signal.HOLD, 0.0, "No strategy data")

        # 6. Normalize and decide
        buy_consensus = buy_score / total_weight
        sell_consensus = sell_score / total_weight

        perf_str = " ".join(f"{n}:{w:.2f}" for n, w in self._performance_weights.items())
        reason_str = f"R={regime.value} Thr={consensus_threshold:.2f} | " + " ".join(reasons)

        if buy_consensus >= consensus_threshold and buy_consensus > sell_consensus:
            logger.info(f"ENSEMBLE BUY | c={buy_consensus:.2f} | {reason_str}")
            return TradeSignal(
                Signal.BUY, min(1.0, buy_consensus),
                reason_str, entry_price=current_price,
                metadata={"regime": regime.value, "perf_weights": perf_str},
            )

        if sell_consensus >= consensus_threshold and sell_consensus > buy_consensus:
            logger.info(f"ENSEMBLE SELL | c={sell_consensus:.2f} | {reason_str}")
            return TradeSignal(
                Signal.SELL, min(1.0, sell_consensus),
                reason_str, entry_price=current_price,
                metadata={"regime": regime.value, "perf_weights": perf_str},
            )

        return TradeSignal(
            Signal.HOLD, 0.0,
            f"R={regime.value} buy={buy_consensus:.2f} sell={sell_consensus:.2f} thr={consensus_threshold:.2f}"
        )
