"""KAMA Trend Strategy — adaptive trend following with Kaufman Adaptive MA.

Uses KAMA instead of fixed EMAs to reduce whipsaws in noisy markets.
KAMA adapts its smoothing based on the Efficiency Ratio:
  - Trending market: KAMA moves fast, signals are responsive
  - Choppy market: KAMA moves slowly, filters out noise

BUY when:
  - Price crosses above KAMA (trend reversal up)
  - KAMA slope is positive (confirming uptrend)
  - Efficiency Ratio > threshold (confirming real trend, not noise)
  - RSI not overbought (room to run)

SELL when:
  - Price crosses below KAMA (trend reversal down)
  - KAMA slope is negative
  - RSI overbought or momentum fading
"""

import pandas as pd
from src.strategies.base import BaseStrategy, TradeSignal, Signal
from src.indicators.technical import (
    add_kama, add_kama_efficiency_ratio, add_rsi, add_atr, add_volume_sma
)
from src.utils.logger import setup_logger

logger = setup_logger("strategy.kama_trend")


class KamaTrendStrategy(BaseStrategy):
    strategy_name = "KAMA Trend"

    def generate_signal(self, df: pd.DataFrame) -> str:
        sig = self.generate_rich_signal(df)
        return sig.signal.value

    def generate_rich_signal(self, df: pd.DataFrame) -> TradeSignal:
        kama_period = self.params.get("kama_period", 10)
        rsi_period = self.params.get("rsi_period", 14)
        er_threshold = self.params.get("er_threshold", 0.3)

        df = add_kama(df, kama_period)
        df = add_kama_efficiency_ratio(df, kama_period)
        df = add_rsi(df, rsi_period)
        df = add_atr(df)
        df = add_volume_sma(df)

        df = df.dropna()
        if len(df) < 5:
            return TradeSignal(Signal.HOLD, 0.0, "Insufficient data")

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]

        kama_col = f"kama_{kama_period}"
        rsi_col = f"rsi_{rsi_period}"
        er_col = f"er_{kama_period}"

        price = curr["close"]
        kama_now = curr[kama_col]
        kama_prev = prev[kama_col]
        kama_prev2 = prev2[kama_col]
        rsi = curr[rsi_col]
        er = curr[er_col] if not pd.isna(curr[er_col]) else 0.0
        atr = curr["atr_14"]
        vol = curr["volume"]
        vol_avg = curr["vol_sma_20"]

        # KAMA slope (normalized by ATR)
        kama_slope = (kama_now - kama_prev) / atr if atr > 0 else 0
        kama_accel = (kama_now - 2 * kama_prev + kama_prev2) / atr if atr > 0 else 0

        # Price position relative to KAMA
        price_vs_kama = (price - kama_now) / kama_now

        # Cross detection
        price_cross_above = prev["close"] <= kama_prev and price > kama_now
        price_cross_below = prev["close"] >= kama_prev and price < kama_now

        # ── BUY scoring ──
        buy_score = 0
        if price_cross_above:
            buy_score += 3
        if price > kama_now and kama_slope > 0:
            buy_score += 2
        if er > er_threshold:
            buy_score += 2
        if rsi < 65:  # not overbought
            buy_score += 1
        if kama_accel > 0:  # KAMA accelerating up
            buy_score += 1
        if vol > vol_avg * 0.8:
            buy_score += 1

        # ── SELL scoring ──
        sell_score = 0
        if price_cross_below:
            sell_score += 3
        if price < kama_now and kama_slope < 0:
            sell_score += 2
        if er > er_threshold:
            sell_score += 1  # Strong trend down confirmed
        if rsi > 70:
            sell_score += 2
        if kama_accel < 0:  # KAMA decelerating / turning down
            sell_score += 1

        if buy_score >= 5:
            confidence = min(1.0, buy_score / 10)
            confidence = max(confidence, er)  # boost by efficiency ratio
            confidence = min(1.0, confidence)
            stop_loss = price - 2.0 * atr
            take_profit = price + 3.0 * atr
            reason = (f"KAMA={kama_now:.2f} ER={er:.2f} slope={kama_slope:.3f} "
                      f"RSI={rsi:.1f} Score={buy_score}")
            logger.info(f"BUY signal | {reason}")
            return TradeSignal(Signal.BUY, confidence, reason,
                               entry_price=price, stop_loss=stop_loss,
                               take_profit=take_profit)

        if sell_score >= 4:
            confidence = min(1.0, sell_score / 9)
            reason = (f"KAMA={kama_now:.2f} ER={er:.2f} slope={kama_slope:.3f} "
                      f"RSI={rsi:.1f} Score={sell_score}")
            logger.info(f"SELL signal | {reason}")
            return TradeSignal(Signal.SELL, confidence, reason, entry_price=price)

        return TradeSignal(
            Signal.HOLD, 0.0,
            f"KAMA slope={kama_slope:.3f} ER={er:.2f} buy={buy_score} sell={sell_score}"
        )
