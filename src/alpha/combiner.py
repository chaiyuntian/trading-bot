"""Signal Combiner — aggregates many small alpha signals into one decision.

This is the central thesis: consistent returns come from combining many
small edges, not from one big idea. Each alpha signal contributes a small
directional score. The combiner:

1. Runs all signals against shared indicator state (no redundant computation)
2. Applies adaptive weights (regime + rolling performance)
3. Extracts volume/volatility modifiers
4. Produces a single TradeSignal with aggregated confidence

The combiner replaces monolithic strategy classes with a transparent,
decomposed, and adaptive signal aggregation layer.
"""

import numpy as np
import pandas as pd

from src.alpha.base import AlphaSignal, AlphaOutput
from src.alpha.momentum import (
    RsiOversoldRecovery, RsiOverboughtDrop, MacdCrossover, MacdMomentum,
    KamaCross, KamaSlope, EfficiencyRatioStrength,
)
from src.alpha.mean_reversion import BollingerBandTouch, BollingerPosition, BandwidthRegime
from src.alpha.trend import EmaTrendFilter, AdxTrendStrength, EmaSlopeDirection, DcaTimingFilter
from src.alpha.volatility import AtrSpike, PriceAcceleration
from src.alpha.volume import VolumeConfirmation, VolumeSurge
from src.alpha.structure import GridLevelProximity, StochRsiExtreme, HigherHighsLowerLows
from src.strategies.base import TradeSignal, Signal
from src.strategies.regime import detect_regime, MarketRegime
from src.utils.logger import setup_logger

logger = setup_logger("alpha.combiner")

# Regime preference for each signal category
REGIME_CATEGORY_WEIGHTS = {
    MarketRegime.TRENDING_UP: {
        "momentum": 1.2, "trend": 1.3, "mean_reversion": 0.4,
        "volatility": 0.8, "volume": 1.0, "structure": 0.7,
    },
    MarketRegime.TRENDING_DOWN: {
        "momentum": 1.0, "trend": 1.1, "mean_reversion": 0.5,
        "volatility": 0.8, "volume": 1.0, "structure": 0.7,
    },
    MarketRegime.RANGING: {
        "momentum": 0.6, "trend": 0.5, "mean_reversion": 1.4,
        "volatility": 0.8, "volume": 0.9, "structure": 1.2,
    },
    MarketRegime.VOLATILE: {
        "momentum": 0.5, "trend": 0.7, "mean_reversion": 0.6,
        "volatility": 1.5, "volume": 1.2, "structure": 0.8,
    },
}


class SignalCombiner:
    """Combines 21 atomic alpha signals with adaptive weighting."""

    def __init__(self, config: dict):
        params = config.get("strategy", {}).get("params", {})
        self.threshold = params.get("combiner_threshold", 0.15)
        self.regime_blend = params.get("combiner_regime_blend", 0.5)
        self.min_signals = params.get("combiner_min_signals", 3)

        self.signals: list[AlphaSignal] = [
            # Momentum (7)
            RsiOversoldRecovery(),
            RsiOverboughtDrop(),
            MacdCrossover(),
            MacdMomentum(),
            KamaCross(),
            KamaSlope(),
            EfficiencyRatioStrength(),
            # Mean Reversion (3)
            BollingerBandTouch(),
            BollingerPosition(),
            BandwidthRegime(),
            # Trend (4)
            EmaTrendFilter(),
            AdxTrendStrength(),
            EmaSlopeDirection(),
            DcaTimingFilter(),
            # Volatility (2)
            AtrSpike(),
            PriceAcceleration(),
            # Volume (2)
            VolumeConfirmation(),
            VolumeSurge(),
            # Structure (3)
            GridLevelProximity(),
            StochRsiExtreme(),
            HigherHighsLowerLows(),
        ]

        # Rolling performance tracking
        self._signal_scores: dict[str, list[float]] = {s.name: [] for s in self.signals}
        self._signal_outcomes: dict[str, list[float]] = {s.name: [] for s in self.signals}
        self._rolling_window = params.get("combiner_rolling_window", 100)
        self._last_regime = MarketRegime.RANGING
        self._prev_price = None

    def build_indicator_snapshot(self, df: pd.DataFrame) -> dict:
        """Build the shared indicator dict from a DataFrame.

        Computed once and shared across all 21 signals.
        """
        from src.indicators.technical import (
            add_rsi, add_macd, add_ema, add_bollinger_bands,
            add_atr, add_volume_sma, add_kama, add_kama_efficiency_ratio,
            add_stochastic_rsi,
        )
        import pandas_ta as ta

        df = add_rsi(df, 14)
        df = add_macd(df, 12, 26, 9)
        df = add_ema(df, 50)
        df = add_ema(df, 20)
        df = add_bollinger_bands(df)
        df = add_atr(df)
        df = add_volume_sma(df)
        df = add_kama(df, 10)
        df = add_kama_efficiency_ratio(df, 10)
        df = add_stochastic_rsi(df)

        # ADX
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        adx_val = dmp_val = dmn_val = None
        if adx_df is not None and not adx_df.empty:
            adx_val = float(adx_df["ADX_14"].iloc[-1]) if "ADX_14" in adx_df.columns else None
            dmp_val = float(adx_df["DMP_14"].iloc[-1]) if "DMP_14" in adx_df.columns else None
            dmn_val = float(adx_df["DMN_14"].iloc[-1]) if "DMN_14" in adx_df.columns else None

        df_clean = df.dropna()
        if len(df_clean) < 3:
            return {}

        curr = df_clean.iloc[-1]
        prev = df_clean.iloc[-2]

        # ATR 20-period average
        atr_col = df_clean["atr_14"]
        atr_avg20 = float(atr_col.rolling(20).mean().iloc[-1]) if len(atr_col) >= 20 else float(atr_col.mean())

        # EMA 20 from 5 candles ago
        ema_20_prev5 = None
        if "ema_20" in df_clean.columns and len(df_clean) >= 6:
            ema_20_prev5 = float(df_clean["ema_20"].iloc[-6])

        ind = {
            "close": float(curr["close"]),
            "close_prev": float(prev["close"]),
            "volume": float(curr["volume"]),
            "rsi_14": float(curr["rsi_14"]) if pd.notna(curr.get("rsi_14")) else None,
            "rsi_14_prev": float(prev["rsi_14"]) if pd.notna(prev.get("rsi_14")) else None,
            "macd_hist": float(curr["macd_hist"]) if pd.notna(curr.get("macd_hist")) else None,
            "macd_hist_prev": float(prev["macd_hist"]) if pd.notna(prev.get("macd_hist")) else None,
            "ema_50": float(curr.get("ema_50")) if pd.notna(curr.get("ema_50")) else None,
            "ema_20": float(curr.get("ema_20")) if pd.notna(curr.get("ema_20")) else None,
            "ema_20_prev5": ema_20_prev5,
            "bb_upper": float(curr["bb_upper"]) if pd.notna(curr.get("bb_upper")) else None,
            "bb_middle": float(curr["bb_middle"]) if pd.notna(curr.get("bb_middle")) else None,
            "bb_lower": float(curr["bb_lower"]) if pd.notna(curr.get("bb_lower")) else None,
            "bb_bandwidth": float(curr["bb_bandwidth"]) if pd.notna(curr.get("bb_bandwidth")) else None,
            "bb_bandwidth_prev": float(prev["bb_bandwidth"]) if pd.notna(prev.get("bb_bandwidth")) else None,
            "atr_14": float(curr["atr_14"]) if pd.notna(curr.get("atr_14")) else None,
            "atr_14_avg20": atr_avg20 if pd.notna(atr_avg20) else None,
            "vol_sma_20": float(curr["vol_sma_20"]) if pd.notna(curr.get("vol_sma_20")) else None,
            "kama_10": float(curr["kama_10"]) if pd.notna(curr.get("kama_10")) else None,
            "kama_10_prev": float(prev["kama_10"]) if pd.notna(prev.get("kama_10")) else None,
            "er_10": float(curr["er_10"]) if pd.notna(curr.get("er_10")) else None,
            "adx_14": adx_val,
            "dmp_14": dmp_val,
            "dmn_14": dmn_val,
            "stoch_rsi_k": float(curr.get("stoch_rsi_k")) if pd.notna(curr.get("stoch_rsi_k")) else None,
            "stoch_rsi_d": float(curr.get("stoch_rsi_d")) if pd.notna(curr.get("stoch_rsi_d")) else None,
        }
        return ind

    def _update_performance(self, outputs: list[AlphaOutput], price: float):
        """Track how well each signal predicted the next price move."""
        if self._prev_price is not None and self._prev_price > 0:
            price_return = (price - self._prev_price) / self._prev_price
            for output in outputs:
                if output.name in self._signal_outcomes:
                    self._signal_outcomes[output.name].append(price_return)
                    self._signal_scores[output.name].append(output.score)
                    # Trim to rolling window
                    if len(self._signal_outcomes[output.name]) > self._rolling_window:
                        self._signal_outcomes[output.name] = self._signal_outcomes[output.name][-self._rolling_window:]
                        self._signal_scores[output.name] = self._signal_scores[output.name][-self._rolling_window:]
        self._prev_price = price

    def _get_performance_weights(self) -> dict[str, float]:
        """Compute per-signal weight based on rolling prediction accuracy."""
        weights = {}
        for name in self._signal_scores:
            scores = self._signal_scores[name]
            outcomes = self._signal_outcomes[name]
            if len(scores) < 20:
                weights[name] = 1.0
                continue

            # Correlation between signal score and subsequent return
            s = np.array(scores)
            o = np.array(outcomes)
            # Avoid division by zero
            s_std = np.std(s)
            o_std = np.std(o)
            if s_std == 0 or o_std == 0:
                weights[name] = 1.0
                continue

            corr = np.corrcoef(s, o)[0, 1]
            if np.isnan(corr):
                weights[name] = 1.0
                continue

            # Map correlation [-1, 1] to weight [0.2, 2.0]
            # Positive correlation = signal is predictive = higher weight
            weights[name] = max(0.2, min(2.0, 1.0 + corr))

        return weights

    def combine(self, df: pd.DataFrame) -> TradeSignal:
        """Run all alpha signals and aggregate into one TradeSignal."""
        ind = self.build_indicator_snapshot(df)
        if not ind:
            return TradeSignal(Signal.HOLD, 0.0, "Insufficient data")

        price = ind["close"]

        # Detect regime
        regime = detect_regime(df)
        if regime != self._last_regime:
            logger.info(f"Regime: {self._last_regime.value} -> {regime.value}")
            self._last_regime = regime

        regime_cat_weights = REGIME_CATEGORY_WEIGHTS.get(regime, REGIME_CATEGORY_WEIGHTS[MarketRegime.RANGING])
        perf_weights = self._get_performance_weights()

        # Run all signals
        outputs: list[AlphaOutput] = []
        for signal in self.signals:
            try:
                output = signal.compute(df, ind)
                outputs.append(output)
            except Exception:
                continue

        # Update performance tracker
        self._update_performance(outputs, price)

        # Extract modifiers from volume/volatility signals
        vol_multiplier = 1.0
        vol_scale_factor = 1.0
        for o in outputs:
            if "vol_multiplier" in o.metadata:
                vol_multiplier = o.metadata["vol_multiplier"]
            if "scale_factor" in o.metadata:
                vol_scale_factor = o.metadata["scale_factor"]

        # Weighted aggregation
        total_score = 0.0
        total_weight = 0.0
        active_signals = 0
        signal_details = []

        for output in outputs:
            if output.score == 0 and output.confidence == 0:
                continue

            # Combine regime weight (by category) and performance weight (by signal)
            regime_w = regime_cat_weights.get(
                next((s.category for s in self.signals if s.name == output.name), "unknown"),
                1.0
            )
            perf_w = perf_weights.get(output.name, 1.0)
            w = regime_w * self.regime_blend + perf_w * (1 - self.regime_blend)

            contribution = output.score * output.confidence * w
            total_score += contribution
            total_weight += abs(w * output.confidence) if output.confidence > 0 else 0
            active_signals += 1

            if abs(output.score) > 0.1:
                signal_details.append(f"{output.name}:{output.score:+.2f}")

        if total_weight == 0 or active_signals < self.min_signals:
            return TradeSignal(Signal.HOLD, 0.0, f"Regime={regime.value} signals={active_signals}")

        normalized = total_score / total_weight

        # Apply volume and volatility modifiers
        normalized *= vol_multiplier * vol_scale_factor

        reason = f"Regime={regime.value} score={normalized:+.3f} n={active_signals} | " + " ".join(signal_details[:8])

        if normalized > self.threshold:
            confidence = min(1.0, abs(normalized))
            logger.info(f"ALPHA BUY | {reason}")
            return TradeSignal(
                Signal.BUY, confidence, reason,
                entry_price=price,
                metadata={"regime": regime.value, "alpha_score": normalized,
                           "active_signals": active_signals},
            )

        if normalized < -self.threshold:
            confidence = min(1.0, abs(normalized))
            logger.info(f"ALPHA SELL | {reason}")
            return TradeSignal(
                Signal.SELL, confidence, reason,
                entry_price=price,
                metadata={"regime": regime.value, "alpha_score": normalized,
                           "active_signals": active_signals},
            )

        return TradeSignal(Signal.HOLD, 0.0, reason)
