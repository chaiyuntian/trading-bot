"""Strategy abstraction layer.

All strategies inherit from BaseStrategy and implement generate_signal().
Strategies are pure signal generators — they know nothing about exchanges
or order execution. This separation keeps them testable and portable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Signal(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(slots=True)
class TradeSignal:
    signal: Signal
    confidence: float       # 0.0 - 1.0
    reason: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[dict] = None

    def is_actionable(self, min_confidence: float = 0.3) -> bool:
        return self.signal != Signal.HOLD and self.confidence >= min_confidence


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    strategy_name: str = "base"

    def __init__(self, config: dict):
        self.config = config
        self.params = config.get("strategy", {}).get("params", {})

    def name(self) -> str:
        return self.strategy_name

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> str:
        """Analyze data and return a signal string: 'buy', 'sell', or 'hold'.

        This is the primary interface used by bot.py and backtest engine.
        """

    def generate_rich_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Generate a signal with confidence and metadata.

        Override this in subclasses for richer signal info.
        Falls back to wrapping generate_signal() result.
        """
        raw = self.generate_signal(df)
        signal_map = {"buy": Signal.BUY, "sell": Signal.SELL, "hold": Signal.HOLD}
        return TradeSignal(
            signal=signal_map.get(raw, Signal.HOLD),
            confidence=0.5 if raw != "hold" else 0.0,
            reason=raw,
        )
