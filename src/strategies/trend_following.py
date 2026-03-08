"""Turtle/Donchian Trend Following — the most battle-tested systematic strategy.

Why this works for small accounts:
1. Doesn't use lagging oscillators (no RSI/MACD whipsaw)
2. Breakout entry — you're buying strength, not weakness
3. High reward/risk ratio (typically 3:1 to 10:1 on winners)
4. Low win rate (35-45%) but huge winners pay for many small losers
5. Works on 4h timeframe — fewer trades, lower fee burden
6. Works in both directions (long AND short on futures)

Entry: Price breaks above/below N-period high/low (Donchian channel)
Exit: Trailing stop at 2x ATR, or price breaks opposite N/2 channel
"""

import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy, TradeSignal, Signal
from src.utils.logger import setup_logger

logger = setup_logger("strategy.trend")


class TrendFollowingStrategy(BaseStrategy):
    """Donchian channel breakout with ATR-based position sizing.

    Adapted from the original Turtle Trading system for crypto:
    - Faster entry channel (20 periods instead of 55)
    - Tighter exit channel (10 periods instead of 20)
    - ATR trailing stop for volatile crypto markets
    - Volume confirmation to filter false breakouts
    """

    strategy_name = "TrendFollowing"

    def __init__(self, config: dict):
        super().__init__(config)
        self.entry_period = self.params.get("donchian_entry", 20)
        self.exit_period = self.params.get("donchian_exit", 10)
        self.atr_period = self.params.get("atr_period", 14)
        self.atr_stop_mult = self.params.get("atr_stop_mult", 2.0)
        self.volume_confirm = self.params.get("volume_confirm", True)
        self._position_side = None  # Track current position direction

    def _compute_donchian(self, df: pd.DataFrame) -> dict:
        """Compute Donchian channels and ATR."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Entry channel (longer period — slower to enter)
        entry_high = high.rolling(self.entry_period).max()
        entry_low = low.rolling(self.entry_period).min()

        # Exit channel (shorter period — faster to exit)
        exit_high = high.rolling(self.exit_period).max()
        exit_low = low.rolling(self.exit_period).min()

        # ATR for stop-loss sizing
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        # Volume MA for confirmation
        vol_ma = df["volume"].rolling(20).mean()

        return {
            "entry_high": entry_high,
            "entry_low": entry_low,
            "exit_high": exit_high,
            "exit_low": exit_low,
            "atr": atr,
            "vol_ma": vol_ma,
        }

    def generate_signal(self, df: pd.DataFrame) -> str:
        sig = self.generate_rich_signal(df)
        return sig.signal.value

    def generate_rich_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < self.entry_period + 5:
            return TradeSignal(Signal.HOLD, 0.0, "Insufficient data")

        ch = self._compute_donchian(df)
        current = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(current["close"])
        atr = float(ch["atr"].iloc[-1]) if not np.isnan(ch["atr"].iloc[-1]) else 0

        if atr == 0:
            return TradeSignal(Signal.HOLD, 0.0, "No ATR data")

        entry_high = float(ch["entry_high"].iloc[-2])  # Previous bar's channel
        entry_low = float(ch["entry_low"].iloc[-2])
        exit_high = float(ch["exit_high"].iloc[-2])
        exit_low = float(ch["exit_low"].iloc[-2])

        # Volume confirmation: current volume > 1.2x average
        vol_ok = True
        if self.volume_confirm:
            vol_ma = float(ch["vol_ma"].iloc[-1])
            vol_ok = float(current["volume"]) > vol_ma * 1.2 if vol_ma > 0 else True

        # ── Breakout detection ──
        # Long breakout: close above previous entry channel high
        if price > entry_high and float(prev["close"]) <= entry_high:
            # Breakout strength = how far above channel
            strength = (price - entry_high) / atr
            confidence = min(1.0, 0.5 + strength * 0.2)

            if not vol_ok:
                confidence *= 0.6  # Reduce confidence without volume

            stop = price - atr * self.atr_stop_mult
            tp = price + atr * self.atr_stop_mult * 3  # 3:1 R/R

            self._position_side = "long"
            return TradeSignal(
                Signal.BUY, confidence,
                f"Breakout HIGH ch={entry_high:.1f} atr={atr:.1f} str={strength:.2f}",
                entry_price=price, stop_loss=stop, take_profit=tp,
                metadata={"atr": atr, "channel": "high", "strength": strength},
            )

        # Short breakout: close below previous entry channel low
        if price < entry_low and float(prev["close"]) >= entry_low:
            strength = (entry_low - price) / atr
            confidence = min(1.0, 0.5 + strength * 0.2)

            if not vol_ok:
                confidence *= 0.6

            stop = price + atr * self.atr_stop_mult
            tp = price - atr * self.atr_stop_mult * 3

            self._position_side = "short"
            return TradeSignal(
                Signal.SELL, confidence,
                f"Breakout LOW ch={entry_low:.1f} atr={atr:.1f} str={strength:.2f}",
                entry_price=price, stop_loss=stop, take_profit=tp,
                metadata={"atr": atr, "channel": "low", "strength": strength},
            )

        # ── Exit signals (only if in position) ──
        # Exit long: price drops below exit channel low
        if self._position_side == "long" and price < exit_low:
            self._position_side = None
            return TradeSignal(
                Signal.SELL, 0.8,
                f"Exit long: below exit_ch={exit_low:.1f}",
                entry_price=price,
            )

        # Exit short: price rises above exit channel high
        if self._position_side == "short" and price > exit_high:
            self._position_side = None
            return TradeSignal(
                Signal.BUY, 0.8,
                f"Exit short: above exit_ch={exit_high:.1f}",
                entry_price=price,
            )

        return TradeSignal(
            Signal.HOLD, 0.0,
            f"Range: {entry_low:.1f} < {price:.1f} < {entry_high:.1f}"
        )
