"""Volume alpha signals — confirm moves with participation."""

import pandas as pd
from src.alpha.base import AlphaSignal, AlphaOutput


class VolumeConfirmation(AlphaSignal):
    """Above-average volume = move has participation, more likely real."""
    name = "volume_confirm"
    category = "volume"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        vol = ind.get("volume")
        vol_avg = ind.get("vol_sma_20")
        if vol is None or vol_avg is None or vol_avg == 0:
            return AlphaOutput(self.name, 0.0, 0.0)

        ratio = vol / vol_avg
        if ratio > 0.8:
            # Volume confirms — boost confidence of other signals
            return AlphaOutput(self.name, 0.0, 0.0,
                               {"vol_multiplier": min(1.5, 0.8 + ratio * 0.3)})
        return AlphaOutput(self.name, 0.0, 0.0, {"vol_multiplier": 0.7})


class VolumeSurge(AlphaSignal):
    """Volume > 1.5x average = something significant happening."""
    name = "volume_surge"
    category = "volume"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        vol = ind.get("volume")
        vol_avg = ind.get("vol_sma_20")
        price = ind.get("close")
        price_prev = ind.get("close_prev")
        if None in (vol, vol_avg, price, price_prev) or vol_avg == 0:
            return AlphaOutput(self.name, 0.0, 0.0)

        ratio = vol / vol_avg
        if ratio < 1.5:
            return AlphaOutput(self.name, 0.0, 0.0)

        # Direction of the high-volume bar
        direction = 1.0 if price > price_prev else -1.0
        score = direction * 0.3
        conf = min(0.8, (ratio - 1.0) * 0.4)
        return AlphaOutput(self.name, score, conf, {"vol_ratio": ratio})
