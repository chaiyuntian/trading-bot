"""Momentum alpha signals — detect directional price momentum."""

import pandas as pd
from src.alpha.base import AlphaSignal, AlphaOutput


class RsiOversoldRecovery(AlphaSignal):
    """RSI dips below oversold and starts rising = buy pressure."""
    name = "rsi_oversold_recovery"
    category = "momentum"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        rsi = ind.get("rsi_14")
        rsi_prev = ind.get("rsi_14_prev")
        if rsi is None or rsi_prev is None:
            return AlphaOutput(self.name, 0.0, 0.0)

        if rsi_prev <= 30 and rsi > rsi_prev:
            depth = max(0, (30 - rsi_prev) / 30)  # deeper = stronger
            return AlphaOutput(self.name, 0.6 + 0.4 * depth, 0.8)
        if rsi < 45 and rsi > rsi_prev:
            return AlphaOutput(self.name, 0.2, 0.4)
        return AlphaOutput(self.name, 0.0, 0.0)


class RsiOverboughtDrop(AlphaSignal):
    """RSI rises above overbought and starts falling = sell pressure."""
    name = "rsi_overbought_drop"
    category = "momentum"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        rsi = ind.get("rsi_14")
        rsi_prev = ind.get("rsi_14_prev")
        if rsi is None or rsi_prev is None:
            return AlphaOutput(self.name, 0.0, 0.0)

        if rsi_prev >= 70 and rsi < rsi_prev:
            excess = max(0, (rsi_prev - 70) / 30)
            return AlphaOutput(self.name, -(0.6 + 0.4 * excess), 0.8)
        if rsi > 65 and rsi < rsi_prev:
            return AlphaOutput(self.name, -0.2, 0.4)
        return AlphaOutput(self.name, 0.0, 0.0)


class MacdCrossover(AlphaSignal):
    """MACD histogram sign change = momentum shift."""
    name = "macd_crossover"
    category = "momentum"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        hist = ind.get("macd_hist")
        hist_prev = ind.get("macd_hist_prev")
        if hist is None or hist_prev is None:
            return AlphaOutput(self.name, 0.0, 0.0)

        if hist_prev < 0 and hist >= 0:
            return AlphaOutput(self.name, 0.7, 0.8)
        if hist_prev > 0 and hist <= 0:
            return AlphaOutput(self.name, -0.7, 0.8)
        return AlphaOutput(self.name, 0.0, 0.0)


class MacdMomentum(AlphaSignal):
    """MACD histogram direction = momentum tendency."""
    name = "macd_momentum"
    category = "momentum"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        hist = ind.get("macd_hist")
        hist_prev = ind.get("macd_hist_prev")
        if hist is None or hist_prev is None:
            return AlphaOutput(self.name, 0.0, 0.0)

        if hist > hist_prev:
            return AlphaOutput(self.name, 0.3, 0.5)
        if hist < hist_prev:
            return AlphaOutput(self.name, -0.3, 0.5)
        return AlphaOutput(self.name, 0.0, 0.0)


class KamaCross(AlphaSignal):
    """Price crosses above/below KAMA = adaptive trend change."""
    name = "kama_cross"
    category = "momentum"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        price = ind.get("close")
        price_prev = ind.get("close_prev")
        kama = ind.get("kama_10")
        kama_prev = ind.get("kama_10_prev")
        if None in (price, price_prev, kama, kama_prev):
            return AlphaOutput(self.name, 0.0, 0.0)

        if price_prev <= kama_prev and price > kama:
            return AlphaOutput(self.name, 0.7, 0.7)
        if price_prev >= kama_prev and price < kama:
            return AlphaOutput(self.name, -0.7, 0.7)
        return AlphaOutput(self.name, 0.0, 0.0)


class KamaSlope(AlphaSignal):
    """KAMA slope normalized by ATR = adaptive trend direction."""
    name = "kama_slope"
    category = "momentum"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        kama = ind.get("kama_10")
        kama_prev = ind.get("kama_10_prev")
        atr = ind.get("atr_14")
        if None in (kama, kama_prev, atr) or atr == 0:
            return AlphaOutput(self.name, 0.0, 0.0)

        slope = (kama - kama_prev) / atr
        score = max(-0.5, min(0.5, slope * 0.5))
        conf = min(1.0, abs(slope) * 2)
        return AlphaOutput(self.name, score, conf)


class EfficiencyRatioStrength(AlphaSignal):
    """Kaufman ER confirms whether a move is trend or noise."""
    name = "er_strength"
    category = "momentum"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        er = ind.get("er_10")
        price = ind.get("close")
        kama = ind.get("kama_10")
        if er is None or price is None or kama is None:
            return AlphaOutput(self.name, 0.0, 0.0)

        if er < 0.2:
            return AlphaOutput(self.name, 0.0, 0.0)  # noise, no edge

        direction = 1.0 if price > kama else -1.0
        score = direction * 0.3 * min(er, 1.0)
        return AlphaOutput(self.name, score, er, {"er": er})
