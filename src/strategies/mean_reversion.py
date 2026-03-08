"""Mean Reversion Strategy with Bollinger Bands + RSI.

BUY when price touches lower BB + RSI oversold + no breakout in progress.
SELL when price touches upper BB + RSI overbought, or price returns to mean.

Best for range-bound / sideways markets.
"""

import pandas as pd
from src.strategies.base import BaseStrategy, TradeSignal, Signal
from src.indicators.technical import add_rsi, add_bollinger_bands, add_atr, add_volume_sma
from src.utils.logger import setup_logger

logger = setup_logger("strategy.mean_reversion")


class MeanReversionStrategy(BaseStrategy):
    strategy_name = "Mean Reversion"

    def generate_signal(self, df: pd.DataFrame) -> str:
        sig = self.generate_rich_signal(df)
        return sig.signal.value

    def generate_rich_signal(self, df: pd.DataFrame) -> TradeSignal:
        rsi_period = self.params.get("rsi_period", 14)
        rsi_oversold = self.params.get("rsi_oversold", 30)
        rsi_overbought = self.params.get("rsi_overbought", 70)

        df = add_rsi(df, rsi_period)
        df = add_bollinger_bands(df)
        df = add_atr(df)
        df = add_volume_sma(df)

        df = df.dropna()
        if len(df) < 3:
            return TradeSignal(Signal.HOLD, 0.0, "Insufficient data")

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        rsi_col = f"rsi_{rsi_period}"
        price = curr["close"]
        rsi = curr[rsi_col]
        bb_lower = curr["bb_lower"]
        bb_upper = curr["bb_upper"]
        bb_middle = curr["bb_middle"]
        bandwidth = curr["bb_bandwidth"]
        prev_bandwidth = prev["bb_bandwidth"]
        vol = curr["volume"]
        vol_avg = curr["vol_sma_20"]

        bandwidth_expanding = bandwidth > prev_bandwidth * 1.5

        # Normalized position within bands
        bb_range = bb_upper - bb_lower
        bb_position = (price - bb_lower) / bb_range if bb_range > 0 else 0.5

        # ── BUY: Price at/below lower band + RSI oversold + no breakout ──
        if price <= bb_lower and rsi < rsi_oversold + 5 and not bandwidth_expanding:
            volume_spike = vol > vol_avg * 1.2
            if rsi < rsi_oversold or volume_spike:
                confidence = min(1.0, (1 - bb_position) * 0.6 + (rsi_oversold - rsi) / 100 * 0.4)
                confidence = max(0.4, confidence)
                reason = (f"Price={price:.2f} BB_lower={bb_lower:.2f} "
                          f"RSI={rsi:.1f} BW={bandwidth:.4f}")
                logger.info(f"BUY signal | {reason}")
                return TradeSignal(Signal.BUY, confidence, reason,
                                   entry_price=price, take_profit=bb_middle)

        # ── SELL: Price at/above upper band + RSI overbought ──
        if price >= bb_upper and rsi > rsi_overbought - 5:
            confidence = min(1.0, bb_position * 0.6 + (rsi - rsi_overbought) / 100 * 0.4)
            confidence = max(0.4, confidence)
            reason = f"Price={price:.2f} BB_upper={bb_upper:.2f} RSI={rsi:.1f}"
            logger.info(f"SELL signal | {reason}")
            return TradeSignal(Signal.SELL, confidence, reason, entry_price=price)

        return TradeSignal(
            Signal.HOLD, 0.0,
            f"BB_pos={bb_position:.0%} RSI={rsi:.1f}"
        )
