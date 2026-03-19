"""Structure alpha signals — price structure and pattern detection."""

import pandas as pd
from src.alpha.base import AlphaSignal, AlphaOutput


class GridLevelProximity(AlphaSignal):
    """Dynamic grid levels based on ATR — buy near support, sell near resistance."""
    name = "grid_level"
    category = "structure"

    def __init__(self):
        self._levels: list[float] = []
        self._center = 0.0

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        price = ind.get("close")
        atr = ind.get("atr_14")
        bb_mid = ind.get("bb_middle")
        if None in (price, atr, bb_mid) or atr == 0:
            return AlphaOutput(self.name, 0.0, 0.0)

        # Rebuild grid if price moved significantly
        if not self._levels or abs(price - self._center) > atr * 3:
            spacing = atr * 0.5
            self._center = bb_mid
            self._levels = [bb_mid + i * spacing for i in range(-5, 6)]

        # Find nearest level
        for level in self._levels:
            dist = (price - level) / price
            if abs(dist) < 0.002:
                if level < self._center:
                    return AlphaOutput(self.name, 0.4, 0.6)  # buy zone
                elif level > self._center:
                    return AlphaOutput(self.name, -0.4, 0.6)  # sell zone

        return AlphaOutput(self.name, 0.0, 0.0)


class StochRsiExtreme(AlphaSignal):
    """Stochastic RSI at extremes = short-term exhaustion."""
    name = "stoch_rsi_extreme"
    category = "structure"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        stoch_k = ind.get("stoch_rsi_k")
        stoch_d = ind.get("stoch_rsi_d")
        if stoch_k is None:
            return AlphaOutput(self.name, 0.0, 0.0)

        if stoch_k < 15:
            return AlphaOutput(self.name, 0.35, 0.6)
        if stoch_k > 85:
            return AlphaOutput(self.name, -0.35, 0.6)
        return AlphaOutput(self.name, 0.0, 0.0)


class HigherHighsLowerLows(AlphaSignal):
    """Recent swing structure — trending or reversing."""
    name = "swing_structure"
    category = "structure"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        if len(df) < 20:
            return AlphaOutput(self.name, 0.0, 0.0)

        highs = df["high"].values[-20:]
        lows = df["low"].values[-20:]

        # Simple peak/trough detection using 5-bar windows
        peaks = []
        troughs = []
        for i in range(2, len(highs) - 2):
            if highs[i] == max(highs[i-2:i+3]):
                peaks.append(highs[i])
            if lows[i] == min(lows[i-2:i+3]):
                troughs.append(lows[i])

        if len(peaks) >= 2 and len(troughs) >= 2:
            higher_highs = peaks[-1] > peaks[-2]
            higher_lows = troughs[-1] > troughs[-2]
            lower_highs = peaks[-1] < peaks[-2]
            lower_lows = troughs[-1] < troughs[-2]

            if higher_highs and higher_lows:
                return AlphaOutput(self.name, 0.3, 0.5)  # uptrend structure
            if lower_highs and lower_lows:
                return AlphaOutput(self.name, -0.3, 0.5)  # downtrend structure

        return AlphaOutput(self.name, 0.0, 0.0)
