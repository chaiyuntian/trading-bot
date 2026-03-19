"""Volatility alpha signals — detect regime changes and adjust sizing."""

import pandas as pd
from src.alpha.base import AlphaSignal, AlphaOutput


class AtrSpike(AlphaSignal):
    """ATR spike relative to average = volatility regime shift.

    Not directional — dampens confidence when vol is extreme.
    """
    name = "atr_spike"
    category = "volatility"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        atr = ind.get("atr_14")
        atr_avg = ind.get("atr_14_avg20")
        if atr is None or atr_avg is None or atr_avg == 0:
            return AlphaOutput(self.name, 0.0, 0.0)

        ratio = atr / atr_avg
        if ratio > 1.8:
            # Extreme volatility — reduce all positions
            return AlphaOutput(self.name, 0.0, 0.0,
                               {"vol_regime": "extreme", "scale_factor": 0.3})
        if ratio > 1.3:
            return AlphaOutput(self.name, 0.0, 0.0,
                               {"vol_regime": "elevated", "scale_factor": 0.6})
        return AlphaOutput(self.name, 0.0, 0.0,
                           {"vol_regime": "normal", "scale_factor": 1.0})


class PriceAcceleration(AlphaSignal):
    """Second derivative of price — acceleration or deceleration."""
    name = "price_acceleration"
    category = "volatility"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        if len(df) < 5:
            return AlphaOutput(self.name, 0.0, 0.0)

        c = df["close"].values
        v1 = c[-1] - c[-2]  # velocity now
        v2 = c[-2] - c[-3]  # velocity prev
        accel = v1 - v2

        atr = ind.get("atr_14")
        if atr is None or atr == 0:
            return AlphaOutput(self.name, 0.0, 0.0)

        norm_accel = accel / atr
        score = max(-0.3, min(0.3, norm_accel * 0.15))
        return AlphaOutput(self.name, score, min(0.5, abs(norm_accel) * 0.2))
