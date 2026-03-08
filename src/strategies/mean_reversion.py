import pandas as pd
from src.strategies.base import BaseStrategy
from src.indicators.technical import (
    add_rsi, add_bollinger_bands, add_atr, add_volume_sma
)
from src.utils.logger import setup_logger

logger = setup_logger("strategy.mean_reversion")


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy with Bollinger Bands
    ----------------------------------------------
    BUY when:
      - Price touches or crosses below lower Bollinger Band
      - RSI is oversold (< 30)
      - Volume spike confirms capitulation
      - Bandwidth is not expanding rapidly (not a trend breakout)

    SELL when:
      - Price touches or crosses above upper Bollinger Band
      - RSI is overbought (> 70)
      - OR price returns to middle band (mean)

    Best in: Range-bound, consolidating markets
    Risk: Fails during strong trend breakouts
    """

    def name(self) -> str:
        return "Mean Reversion"

    def generate_signal(self, df: pd.DataFrame) -> str:
        rsi_period = self.params.get("rsi_period", 14)
        rsi_oversold = self.params.get("rsi_oversold", 30)
        rsi_overbought = self.params.get("rsi_overbought", 70)

        df = add_rsi(df, rsi_period)
        df = add_bollinger_bands(df)
        df = add_atr(df)
        df = add_volume_sma(df)

        df = df.dropna()
        if len(df) < 3:
            return "hold"

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

        # BUY: Price at/below lower band + RSI oversold + no breakout
        if price <= bb_lower and rsi < rsi_oversold + 5 and not bandwidth_expanding:
            volume_spike = vol > vol_avg * 1.2
            if rsi < rsi_oversold or volume_spike:
                logger.info(
                    f"BUY signal | Price={price:.2f} BB_lower={bb_lower:.2f} "
                    f"RSI={rsi:.1f} BW={bandwidth:.4f}"
                )
                return "buy"

        # SELL: Price at/above upper band + RSI overbought
        if price >= bb_upper and rsi > rsi_overbought - 5:
            logger.info(
                f"SELL signal | Price={price:.2f} BB_upper={bb_upper:.2f} "
                f"RSI={rsi:.1f}"
            )
            return "sell"

        # SELL: Price returns to mean from above
        if price >= bb_middle and rsi > 55:
            # Only if we have an open long position context
            # The bot.py will check open trades
            pass

        return "hold"
