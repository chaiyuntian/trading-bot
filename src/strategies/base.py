from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, config: dict):
        self.config = config
        self.params = config.get("strategy", {}).get("params", {})

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> str:
        """
        Analyze data and return a signal.
        Returns: "buy", "sell", or "hold"
        """
        pass

    @abstractmethod
    def name(self) -> str:
        pass
