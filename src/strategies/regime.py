"""Market Regime Detection.

Classifies the current market into one of:
  - TRENDING_UP: Strong uptrend (use momentum strategies)
  - TRENDING_DOWN: Strong downtrend (avoid longs, or short)
  - RANGING: Sideways market (use mean reversion / grid)
  - VOLATILE: High volatility breakout (reduce position size)

Uses a combination of:
  - ADX (Average Directional Index) for trend strength
  - Bollinger Band width for volatility regime
  - EMA slope for trend direction
  - ATR ratio (current vs average) for volatility spikes
"""

from enum import Enum
import pandas as pd
import numpy as np
import pandas_ta as ta


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


def detect_regime(df: pd.DataFrame, adx_period: int = 14,
                  bb_period: int = 20) -> MarketRegime:
    """Detect current market regime from OHLCV data.

    Returns the current regime classification.
    """
    if len(df) < 50:
        return MarketRegime.RANGING

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # 1. ADX for trend strength
    adx_df = ta.adx(high, low, close, length=adx_period)
    if adx_df is None or adx_df.empty:
        return MarketRegime.RANGING

    adx_col = f"ADX_{adx_period}"
    dmp_col = f"DMP_{adx_period}"
    dmn_col = f"DMN_{adx_period}"

    adx_val = float(adx_df[adx_col].iloc[-1]) if adx_col in adx_df.columns else 20
    dmp_val = float(adx_df[dmp_col].iloc[-1]) if dmp_col in adx_df.columns else 0
    dmn_val = float(adx_df[dmn_col].iloc[-1]) if dmn_col in adx_df.columns else 0

    if pd.isna(adx_val):
        adx_val = 20

    # 2. BB bandwidth for volatility
    bb = ta.bbands(close, length=bb_period, std=2.0)
    if bb is not None:
        bbu = bb.iloc[:, 0]
        bbm = bb.iloc[:, 1]
        bbl = bb.iloc[:, 2]
        bandwidth = ((bbu - bbl) / bbm).iloc[-1]
        avg_bandwidth = ((bbu - bbl) / bbm).rolling(50).mean().iloc[-1]
    else:
        bandwidth = 0.02
        avg_bandwidth = 0.02

    if pd.isna(bandwidth):
        bandwidth = 0.02
    if pd.isna(avg_bandwidth):
        avg_bandwidth = bandwidth

    # 3. EMA slope for direction
    ema_20 = ta.ema(close, length=20)
    if ema_20 is not None and len(ema_20) >= 5:
        slope = (float(ema_20.iloc[-1]) - float(ema_20.iloc[-5])) / float(ema_20.iloc[-5])
    else:
        slope = 0

    # 4. ATR spike detection
    atr = ta.atr(high, low, close, length=14)
    if atr is not None and len(atr) >= 20:
        atr_current = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0
        atr_avg = float(atr.rolling(20).mean().iloc[-1]) if not pd.isna(atr.rolling(20).mean().iloc[-1]) else atr_current
        atr_ratio = atr_current / atr_avg if atr_avg > 0 else 1.0
    else:
        atr_ratio = 1.0

    # ── Classification logic ──

    # High volatility regime: ATR spike + expanding BB
    if atr_ratio > 1.8 and bandwidth > avg_bandwidth * 1.5:
        return MarketRegime.VOLATILE

    # Strong trend: ADX > 25
    if adx_val > 25:
        if dmp_val > dmn_val and slope > 0.001:
            return MarketRegime.TRENDING_UP
        elif dmn_val > dmp_val and slope < -0.001:
            return MarketRegime.TRENDING_DOWN

    # Moderate trend with clear direction
    if adx_val > 20 and abs(slope) > 0.005:
        return MarketRegime.TRENDING_UP if slope > 0 else MarketRegime.TRENDING_DOWN

    # Default: ranging
    return MarketRegime.RANGING


def get_regime_strategy_weights(regime: MarketRegime) -> dict[str, float]:
    """Map market regime to strategy preference weights.

    Higher weight = more suitable for current conditions.
    """
    weights = {
        MarketRegime.TRENDING_UP: {
            "rsi_macd": 0.8,
            "dca_momentum": 0.9,
            "mean_reversion": 0.2,
            "grid": 0.3,
            "kama_trend": 0.9,
        },
        MarketRegime.TRENDING_DOWN: {
            "rsi_macd": 0.5,
            "dca_momentum": 0.3,
            "mean_reversion": 0.3,
            "grid": 0.2,
            "kama_trend": 0.7,
        },
        MarketRegime.RANGING: {
            "rsi_macd": 0.4,
            "dca_momentum": 0.5,
            "mean_reversion": 0.9,
            "grid": 0.8,
            "kama_trend": 0.3,
        },
        MarketRegime.VOLATILE: {
            "rsi_macd": 0.3,
            "dca_momentum": 0.2,
            "mean_reversion": 0.4,
            "grid": 0.5,
            "kama_trend": 0.6,
        },
    }
    return weights.get(regime, weights[MarketRegime.RANGING])
