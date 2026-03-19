"""Ensemble Strategy — combines multiple strategies with dynamic, regime-aware weighting.

Instead of relying on a single strategy, the ensemble:
1. Detects the current market regime (trending, ranging, volatile)
2. Runs all sub-strategies in parallel
3. Weights their signals by regime suitability AND rolling performance (Sharpe)
4. Produces a consensus signal

Dynamic weighting: strategies that have been performing well (higher rolling
Sharpe) get more weight. This is a softmax over recent Sharpe ratios, blended
with regime-based weights. This prevents a single bad strategy from dominating.

This is the recommended strategy for production use.
"""

import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy, TradeSignal, Signal
from src.strategies.rsi_macd import RsiMacdStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.grid_trading import GridTradingStrategy
from src.strategies.dca_momentum import DCAMomentumStrategy
from src.strategies.kama_trend import KamaTrendStrategy
from src.strategies.regime import detect_regime, get_regime_strategy_weights, MarketRegime
from src.utils.logger import setup_logger

logger = setup_logger("strategy.ensemble")


class EnsembleStrategy(BaseStrategy):
    """Regime-adaptive ensemble with dynamic Sharpe-weighted voting."""

    strategy_name = "Ensemble"

    def __init__(self, config: dict):
        super().__init__(config)
        self.sub_strategies: dict[str, BaseStrategy] = {
            "rsi_macd": RsiMacdStrategy(config),
            "mean_reversion": MeanReversionStrategy(config),
            "grid": GridTradingStrategy(config),
            "dca_momentum": DCAMomentumStrategy(config),
            "kama_trend": KamaTrendStrategy(config),
        }
        self.min_consensus = self.params.get("ensemble_min_consensus", 0.4)
        self._last_regime = MarketRegime.RANGING

        # Dynamic weighting state
        self._signal_history: dict[str, list[dict]] = {
            name: [] for name in self.sub_strategies
        }
        self._rolling_window = self.params.get("ensemble_rolling_window", 50)
        self._regime_weight_blend = self.params.get("ensemble_regime_blend", 0.6)

    def _record_signal(self, name: str, signal: TradeSignal, price: float):
        """Track signal history for dynamic weight calculation."""
        self._signal_history[name].append({
            "signal": signal.signal,
            "confidence": signal.confidence,
            "price": price,
        })
        # Keep only rolling window
        if len(self._signal_history[name]) > self._rolling_window:
            self._signal_history[name] = self._signal_history[name][-self._rolling_window:]

    def _compute_dynamic_weights(self) -> dict[str, float]:
        """Compute softmax weights based on rolling strategy performance.

        Simulates what would have happened if we followed each strategy's
        signals, and computes a pseudo-Sharpe from the result.
        """
        sharpes = {}
        for name, history in self._signal_history.items():
            if len(history) < 10:
                sharpes[name] = 0.0
                continue

            # Simulate returns: if a strategy said BUY and price went up, that's good
            returns = []
            for i in range(1, len(history)):
                prev = history[i - 1]
                curr = history[i]
                if prev["price"] == 0:
                    continue
                price_return = (curr["price"] - prev["price"]) / prev["price"]

                if prev["signal"] == Signal.BUY:
                    returns.append(price_return * prev["confidence"])
                elif prev["signal"] == Signal.SELL:
                    returns.append(-price_return * prev["confidence"])
                # HOLD contributes 0

            if len(returns) < 5:
                sharpes[name] = 0.0
                continue

            arr = np.array(returns)
            std = np.std(arr)
            sharpes[name] = np.mean(arr) / std if std > 0 else 0.0

        # Softmax over Sharpe ratios (temperature=1.0)
        values = np.array(list(sharpes.values()))
        if np.all(values == 0):
            return {name: 1.0 for name in sharpes}

        # Shift to avoid overflow
        values = values - np.max(values)
        exp_values = np.exp(values)
        softmax = exp_values / np.sum(exp_values)

        # Scale to [0.2, 2.0] range (don't completely zero out any strategy)
        result = {}
        for i, name in enumerate(sharpes.keys()):
            result[name] = float(0.2 + softmax[i] * 1.8)

        return result

    def generate_signal(self, df: pd.DataFrame) -> str:
        sig = self.generate_rich_signal(df)
        return sig.signal.value

    def generate_rich_signal(self, df: pd.DataFrame) -> TradeSignal:
        # 1. Detect market regime
        regime = detect_regime(df)
        if regime != self._last_regime:
            logger.info(f"Regime change: {self._last_regime.value} -> {regime.value}")
            self._last_regime = regime

        # 2. Get regime-based weights
        regime_weights = get_regime_strategy_weights(regime)

        # 3. Get dynamic performance-based weights
        dynamic_weights = self._compute_dynamic_weights()

        # 4. Blend: regime_weight * blend + dynamic_weight * (1 - blend)
        blend = self._regime_weight_blend
        combined_weights = {}
        for name in self.sub_strategies:
            rw = regime_weights.get(name, 0.5)
            dw = dynamic_weights.get(name, 1.0)
            combined_weights[name] = rw * blend + dw * (1 - blend)

        # 5. Collect signals from all sub-strategies
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        reasons = []
        current_price = float(df.iloc[-1]["close"])

        for name, strategy in self.sub_strategies.items():
            weight = combined_weights.get(name, 0.5)
            try:
                signal = strategy.generate_rich_signal(df)
            except Exception as e:
                logger.debug(f"Strategy {name} error: {e}")
                continue

            # Record for dynamic weighting
            self._record_signal(name, signal, current_price)

            total_weight += weight

            if signal.signal == Signal.BUY:
                contribution = weight * signal.confidence
                buy_score += contribution
                reasons.append(f"{name}:BUY({signal.confidence:.1f})*{weight:.2f}")
            elif signal.signal == Signal.SELL:
                contribution = weight * signal.confidence
                sell_score += contribution
                reasons.append(f"{name}:SELL({signal.confidence:.1f})*{weight:.2f}")

        if total_weight == 0:
            return TradeSignal(Signal.HOLD, 0.0, "No strategy data")

        # 6. Normalize scores
        buy_consensus = buy_score / total_weight
        sell_consensus = sell_score / total_weight

        reason_str = f"Regime={regime.value} | " + " ".join(reasons)

        # 7. Produce consensus signal
        if buy_consensus >= self.min_consensus and buy_consensus > sell_consensus:
            logger.info(f"ENSEMBLE BUY | consensus={buy_consensus:.2f} | {reason_str}")
            return TradeSignal(
                Signal.BUY, min(1.0, buy_consensus),
                reason_str, entry_price=current_price,
                metadata={"regime": regime.value, "consensus": buy_consensus},
            )

        if sell_consensus >= self.min_consensus and sell_consensus > buy_consensus:
            logger.info(f"ENSEMBLE SELL | consensus={sell_consensus:.2f} | {reason_str}")
            return TradeSignal(
                Signal.SELL, min(1.0, sell_consensus),
                reason_str, entry_price=current_price,
                metadata={"regime": regime.value, "consensus": sell_consensus},
            )

        return TradeSignal(
            Signal.HOLD, 0.0,
            f"Regime={regime.value} buy={buy_consensus:.2f} sell={sell_consensus:.2f}"
        )
