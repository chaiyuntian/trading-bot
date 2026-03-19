"""Multi-Timeframe Analysis — the single biggest signal quality improvement.

The principle: trade on your timeframe, confirm on the one above.
- Trade on 4h? Confirm trend on 1d.
- Trade on 1h? Confirm trend on 4h.

Only take longs when higher TF is bullish.
Only take shorts when higher TF is bearish.
This filters out 60-70% of false signals.

Since we can't fetch multiple timeframes during backtest (only one TF available),
we SIMULATE higher timeframe by resampling the data we have.
"""

import pandas as pd
import numpy as np
from src.alpha.base import AlphaSignal, AlphaOutput
from src.utils.logger import setup_logger

logger = setup_logger("alpha.mtf")

# Map: current TF -> simulated higher TF candle count
# e.g., if trading on 4h, resample 6 candles = 1 day
_RESAMPLE_MAP = {
    "1m": 15,    # -> 15m
    "5m": 12,    # -> 1h
    "15m": 4,    # -> 1h
    "1h": 4,     # -> 4h
    "4h": 6,     # -> 1d
    "1d": 7,     # -> 1w
}


class HigherTFTrend(AlphaSignal):
    """Higher timeframe trend direction — the most important filter.

    Computes trend on simulated higher TF by resampling.
    Returns strong directional bias: trades against HTF trend are suppressed.
    """
    name = "htf_trend"
    category = "trend"

    def __init__(self, timeframe: str = "4h"):
        self.resample_n = _RESAMPLE_MAP.get(timeframe, 6)

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        if len(df) < self.resample_n * 10:
            return AlphaOutput(self.name, 0.0, 0.0)

        close = df["close"].values

        # Resample to higher TF
        n = len(close)
        htf_closes = []
        for i in range(0, n - self.resample_n + 1, self.resample_n):
            htf_closes.append(close[i + self.resample_n - 1])

        if len(htf_closes) < 10:
            return AlphaOutput(self.name, 0.0, 0.0)

        htf = np.array(htf_closes)

        # EMA 10 on higher TF
        alpha = 2.0 / (10 + 1)
        ema = htf[0]
        for v in htf[1:]:
            ema = alpha * v + (1 - alpha) * ema

        # Slope of last 5 HTF candles
        recent = htf[-5:]
        slope = (recent[-1] - recent[0]) / recent[0] if recent[0] != 0 else 0

        # Current price vs HTF EMA
        price = close[-1]
        above_ema = price > ema

        # Score: strong directional bias
        if above_ema and slope > 0.01:
            score = min(0.6, 0.3 + slope * 5)
            return AlphaOutput(self.name, score, 0.8,
                               {"htf_trend": "bullish", "htf_slope": slope})
        elif not above_ema and slope < -0.01:
            score = max(-0.6, -0.3 + slope * 5)
            return AlphaOutput(self.name, score, 0.8,
                               {"htf_trend": "bearish", "htf_slope": slope})
        elif above_ema:
            return AlphaOutput(self.name, 0.15, 0.5,
                               {"htf_trend": "weak_bullish", "htf_slope": slope})
        else:
            return AlphaOutput(self.name, -0.15, 0.5,
                               {"htf_trend": "weak_bearish", "htf_slope": slope})


class HTFMomentum(AlphaSignal):
    """Higher TF momentum — rate of change on resampled data."""
    name = "htf_momentum"
    category = "trend"

    def __init__(self, timeframe: str = "4h"):
        self.resample_n = _RESAMPLE_MAP.get(timeframe, 6)

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        if len(df) < self.resample_n * 8:
            return AlphaOutput(self.name, 0.0, 0.0)

        close = df["close"].values
        htf_closes = []
        for i in range(0, len(close) - self.resample_n + 1, self.resample_n):
            htf_closes.append(close[i + self.resample_n - 1])

        if len(htf_closes) < 5:
            return AlphaOutput(self.name, 0.0, 0.0)

        htf = np.array(htf_closes)

        # Rate of change over 3 and 5 HTF periods
        roc3 = (htf[-1] - htf[-4]) / htf[-4] if len(htf) >= 4 else 0
        roc5 = (htf[-1] - htf[-6]) / htf[-6] if len(htf) >= 6 else roc3

        # Average momentum
        momentum = (roc3 + roc5) / 2
        score = max(-0.4, min(0.4, momentum * 5))
        conf = min(0.7, abs(momentum) * 10)

        return AlphaOutput(self.name, score, conf,
                           {"htf_roc3": roc3, "htf_roc5": roc5})


class TrendAlignment(AlphaSignal):
    """Checks if current TF and higher TF trends agree.

    Maximum edge when both timeframes point the same direction.
    Returns 0 when they disagree (conflicting signal = no trade).
    """
    name = "trend_alignment"
    category = "trend"

    def __init__(self, timeframe: str = "4h"):
        self._htf = HigherTFTrend(timeframe)

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        # Get higher TF direction
        htf_out = self._htf.compute(df, ind)
        htf_dir = htf_out.metadata.get("htf_trend", "neutral") if htf_out.metadata else "neutral"

        # Get current TF direction from indicators
        price = ind.get("close", 0)
        ema = ind.get("ema_50", 0)
        kama = ind.get("kama_10", 0)
        if not price or not ema:
            return AlphaOutput(self.name, 0.0, 0.0)

        current_bullish = price > ema and (not kama or price > kama)
        current_bearish = price < ema and (not kama or price < kama)

        # Alignment check
        if current_bullish and "bullish" in htf_dir:
            return AlphaOutput(self.name, 0.5, 0.9,
                               {"aligned": True, "direction": "bull"})
        elif current_bearish and "bearish" in htf_dir:
            return AlphaOutput(self.name, -0.5, 0.9,
                               {"aligned": True, "direction": "bear"})
        elif (current_bullish and "bearish" in htf_dir) or \
             (current_bearish and "bullish" in htf_dir):
            # Conflict — suppress trading
            return AlphaOutput(self.name, 0.0, 0.0,
                               {"aligned": False, "direction": "conflict"})
        else:
            return AlphaOutput(self.name, 0.0, 0.3,
                               {"aligned": False, "direction": "neutral"})
