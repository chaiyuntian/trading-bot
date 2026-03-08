import pandas as pd
from src.strategies.base import BaseStrategy
from src.indicators.technical import add_rsi, add_ema, add_macd, add_atr, add_volume_sma
from src.utils.logger import setup_logger

logger = setup_logger("strategy.dca_momentum")


class DCAMomentumStrategy(BaseStrategy):
    """
    DCA with Momentum Filter
    ---------------------------
    Dollar-Cost Averaging enhanced with momentum indicators.
    Only buys during favorable momentum conditions.

    BUY when (DCA intervals + momentum confirmation):
      - RSI < 45 (not overbought)
      - MACD histogram improving (momentum turning positive)
      - Price near or below EMA (not chasing)
      - Accumulates position over time in small increments

    SELL when:
      - RSI > 70 (overbought, take partial profits)
      - Price significantly above EMA (extended)
      - MACD showing strong bearish divergence

    Best for: Gradual accumulation with better average entry price
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.buy_count = 0
        self.dca_interval = self.params.get("dca_interval", 4)  # Buy every N candles
        self.candle_count = 0

    def name(self) -> str:
        return "DCA Momentum"

    def generate_signal(self, df: pd.DataFrame) -> str:
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
            return "hold"

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

        # DCA BUY conditions
        is_dca_interval = self.candle_count % self.dca_interval == 0

        # Momentum filters
        momentum_ok = (
            rsi < 45 and
            macd_hist > macd_hist_prev and  # Momentum improving
            price_vs_ema < 0.02  # Not more than 2% above EMA
        )

        # Strong buy: dip below EMA with RSI oversold
        strong_buy = (
            rsi < 30 and
            price < ema and
            macd_hist > macd_hist_prev
        )

        if strong_buy:
            self.buy_count += 1
            logger.info(
                f"STRONG DCA BUY #{self.buy_count} | RSI={rsi:.1f} "
                f"Price={price:.2f} EMA={ema:.2f} ({price_vs_ema:+.2%})"
            )
            return "buy"

        if is_dca_interval and momentum_ok:
            self.buy_count += 1
            logger.info(
                f"DCA BUY #{self.buy_count} | RSI={rsi:.1f} "
                f"Price={price:.2f} EMA={ema:.2f} ({price_vs_ema:+.2%})"
            )
            return "buy"

        # SELL: Take profit conditions
        rsi_overbought = rsi > 70
        extended = price_vs_ema > 0.05  # 5% above EMA
        macd_bearish = macd_hist < 0 and macd_hist < macd_hist_prev

        if rsi_overbought and (extended or macd_bearish):
            logger.info(
                f"DCA SELL (take profit) | RSI={rsi:.1f} "
                f"Price vs EMA={price_vs_ema:+.2%} MACD_hist={macd_hist:.4f}"
            )
            return "sell"

        return "hold"
