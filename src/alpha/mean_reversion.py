"""Mean-reversion alpha signals — detect stretched prices likely to snap back."""

import pandas as pd
from src.alpha.base import AlphaSignal, AlphaOutput


class BollingerBandTouch(AlphaSignal):
    """Price at BB extremes = potential reversion."""
    name = "bb_touch"
    category = "mean_reversion"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        price = ind.get("close")
        bbu = ind.get("bb_upper")
        bbl = ind.get("bb_lower")
        rsi = ind.get("rsi_14")
        bw = ind.get("bb_bandwidth")
        bw_prev = ind.get("bb_bandwidth_prev")
        if None in (price, bbu, bbl, rsi, bw):
            return AlphaOutput(self.name, 0.0, 0.0)

        # Suppress during breakout (bandwidth expanding fast)
        if bw_prev and bw > bw_prev * 1.5:
            return AlphaOutput(self.name, 0.0, 0.0)

        if price <= bbl and rsi < 35:
            stretch = max(0, (bbl - price) / bbl * 100)
            conf = min(0.9, 0.5 + stretch * 0.1)
            return AlphaOutput(self.name, 0.6, conf)

        if price >= bbu and rsi > 65:
            stretch = max(0, (price - bbu) / bbu * 100)
            conf = min(0.9, 0.5 + stretch * 0.1)
            return AlphaOutput(self.name, -0.6, conf)

        return AlphaOutput(self.name, 0.0, 0.0)


class BollingerPosition(AlphaSignal):
    """Normalized position within BB — linear contrarian score."""
    name = "bb_position"
    category = "mean_reversion"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        price = ind.get("close")
        bbu = ind.get("bb_upper")
        bbl = ind.get("bb_lower")
        if None in (price, bbu, bbl):
            return AlphaOutput(self.name, 0.0, 0.0)

        bb_range = bbu - bbl
        if bb_range <= 0:
            return AlphaOutput(self.name, 0.0, 0.0)

        position = (price - bbl) / bb_range  # 0=lower, 1=upper
        # Contrarian: high position = sell, low = buy
        score = (0.5 - position) * 0.6  # -0.3 to +0.3
        return AlphaOutput(self.name, score, 0.4)


class BandwidthRegime(AlphaSignal):
    """Bollinger bandwidth expansion = breakout regime modifier.

    Doesn't give direction, but dampens mean-reversion signals when
    bandwidth is expanding (breakout happening).
    """
    name = "bandwidth_regime"
    category = "mean_reversion"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        bw = ind.get("bb_bandwidth")
        bw_prev = ind.get("bb_bandwidth_prev")
        if bw is None or bw_prev is None:
            return AlphaOutput(self.name, 0.0, 0.0)

        if bw > bw_prev * 1.5:
            # Breakout regime — this is a confidence dampener, not directional
            return AlphaOutput(self.name, 0.0, 0.0, {"breakout": True})

        return AlphaOutput(self.name, 0.0, 0.0, {"breakout": False})
