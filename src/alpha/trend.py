"""Trend alpha signals — detect and confirm directional bias."""

import pandas as pd
from src.alpha.base import AlphaSignal, AlphaOutput


class EmaTrendFilter(AlphaSignal):
    """Price position relative to EMA = trend direction bias."""
    name = "ema_trend"
    category = "trend"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        price = ind.get("close")
        ema = ind.get("ema_50")
        if price is None or ema is None or ema == 0:
            return AlphaOutput(self.name, 0.0, 0.0)

        distance = (price - ema) / ema
        if price > ema:
            score = min(0.4, distance * 5)
            return AlphaOutput(self.name, score, 0.5)
        else:
            score = max(-0.4, distance * 5)
            return AlphaOutput(self.name, score, 0.5)


class AdxTrendStrength(AlphaSignal):
    """ADX value + DM direction = trend strength and direction."""
    name = "adx_strength"
    category = "trend"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        adx = ind.get("adx_14")
        dmp = ind.get("dmp_14")
        dmn = ind.get("dmn_14")
        if None in (adx, dmp, dmn):
            return AlphaOutput(self.name, 0.0, 0.0)

        if adx < 20:
            return AlphaOutput(self.name, 0.0, 0.0)  # no trend

        direction = 1.0 if dmp > dmn else -1.0
        strength = min(1.0, (adx - 20) / 30)  # 0 at adx=20, 1 at adx=50
        score = direction * 0.4 * strength
        return AlphaOutput(self.name, score, strength)


class EmaSlopeDirection(AlphaSignal):
    """EMA 20 slope = short-term trend direction."""
    name = "ema_slope"
    category = "trend"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        ema = ind.get("ema_20")
        ema_prev5 = ind.get("ema_20_prev5")
        if ema is None or ema_prev5 is None or ema_prev5 == 0:
            return AlphaOutput(self.name, 0.0, 0.0)

        slope = (ema - ema_prev5) / ema_prev5
        if abs(slope) < 0.001:
            return AlphaOutput(self.name, 0.0, 0.0)

        score = max(-0.4, min(0.4, slope * 50))
        conf = min(1.0, abs(slope) * 100)
        return AlphaOutput(self.name, score, conf)


class DcaTimingFilter(AlphaSignal):
    """Combined RSI + MACD + EMA proximity = favorable accumulation timing."""
    name = "dca_timing"
    category = "trend"

    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        rsi = ind.get("rsi_14")
        macd_hist = ind.get("macd_hist")
        macd_hist_prev = ind.get("macd_hist_prev")
        price = ind.get("close")
        ema = ind.get("ema_50")
        if None in (rsi, macd_hist, macd_hist_prev, price, ema) or ema == 0:
            return AlphaOutput(self.name, 0.0, 0.0)

        price_vs_ema = (price - ema) / ema
        momentum_improving = macd_hist > macd_hist_prev

        if rsi < 30 and price < ema and momentum_improving:
            return AlphaOutput(self.name, 0.5, 0.8)  # strong dip buy

        if rsi < 45 and momentum_improving and price_vs_ema < 0.02:
            return AlphaOutput(self.name, 0.25, 0.5)  # regular accumulation

        return AlphaOutput(self.name, 0.0, 0.0)
