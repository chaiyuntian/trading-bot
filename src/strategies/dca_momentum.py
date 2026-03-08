"""DCA with Momentum Filter Strategy.

Dollar-Cost Averaging enhanced with momentum indicators.
Only buys during favorable momentum conditions.

BUY when (DCA interval + momentum confirmation):
  - RSI < 45 (not overbought)
  - MACD histogram improving
  - Price near or below EMA (not chasing)

SELL when:
  - RSI > 70 (overbought, take profits)
  - Price significantly above EMA
  - MACD showing strong bearish divergence

Best for gradual accumulation with better average entry price.
"""

import pandas as pd
from src.strategies.base import BaseStrategy, TradeSignal, Signal
from src.indicators.technical import add_rsi, add_ema, add_macd, add_atr, add_volume_sma
from src.utils.logger import setup_logger

logger = setup_logger("strategy.dca_momentum")


class DCAMomentumStrategy(BaseStrategy):
    strategy_name = "DCA Momentum"

    def __init__(self, config: dict):
        super().__init__(config)
        self.buy_count = 0
        self.dca_interval = self.params.get("dca_interval", 4)
        self.candle_count = 0

    def generate_signal(self, df: pd.DataFrame) -> str:
        sig = self.generate_rich_signal(df)
        return sig.signal.value

    def generate_rich_signal(self, df: pd.DataFrame) -> TradeSignal:
        rsi_period = self.params.get("rsi_period", 14)
        ema_period = self.params.get("ema_period", 50)

        df = add_rsi(df, rsi_period)
        df = add_ema(df, ema_period)
        df = add_macd(df, self.params.get("macd_fast", 12),
                      self.params.get("macd_slow", 26),
                      self.params.get("macd_signal", 9))
        df = add_atr(df)
        df = add_volume_sma(df)

        df = df.dropna()
        if len(df) < 3:
            return TradeSignal(Signal.HOLD, 0.0, "Insufficient data")

        self.candle_count += 1
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        rsi_col = f"rsi_{rsi_period}"
        ema_col = f"ema_{ema_period}"

        rsi = curr[rsi_col]
        price = curr["close"]
        ema = curr[ema_col]
        macd_hist = curr["macd_hist"]
        macd_hist_prev = prev["macd_hist"]
        price_vs_ema = (price - ema) / ema

        is_dca_interval = self.candle_count % self.dca_interval == 0

        # Strong buy: dip below EMA with RSI oversold
        strong_buy = (rsi < 30 and price < ema and macd_hist > macd_hist_prev)

        # DCA buy: interval + momentum filters
        momentum_ok = (rsi < 45 and macd_hist > macd_hist_prev and price_vs_ema < 0.02)

        if strong_buy:
            self.buy_count += 1
            reason = (f"STRONG DCA BUY #{self.buy_count} | RSI={rsi:.1f} "
                      f"Price={price:.2f} EMA={ema:.2f} ({price_vs_ema:+.2%})")
            logger.info(reason)
            return TradeSignal(Signal.BUY, 0.8, reason, entry_price=price)

        if is_dca_interval and momentum_ok:
            self.buy_count += 1
            confidence = 0.6 if price < ema else 0.4
            reason = (f"DCA BUY #{self.buy_count} | RSI={rsi:.1f} "
                      f"Price={price:.2f} EMA={ema:.2f} ({price_vs_ema:+.2%})")
            logger.info(reason)
            return TradeSignal(Signal.BUY, confidence, reason, entry_price=price)

        # SELL: Take profit conditions
        rsi_overbought = rsi > 70
        extended = price_vs_ema > 0.05
        macd_bearish = macd_hist < 0 and macd_hist < macd_hist_prev

        if rsi_overbought and (extended or macd_bearish):
            reason = (f"DCA SELL (profit) | RSI={rsi:.1f} "
                      f"EMA_dist={price_vs_ema:+.2%} MACD_h={macd_hist:.4f}")
            logger.info(reason)
            return TradeSignal(Signal.SELL, 0.7, reason, entry_price=price)

        return TradeSignal(
            Signal.HOLD, 0.0,
            f"DCA wait | candle={self.candle_count} RSI={rsi:.1f}"
        )
