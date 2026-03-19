"""Atomic Alpha Signal framework.

Each AlphaSignal detects one specific market condition and outputs a
directional score. Many small edges combined > one big idea.

Score convention:
  +1.0 = strong buy signal
   0.0 = no opinion
  -1.0 = strong sell signal
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class AlphaOutput:
    name: str
    score: float       # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    metadata: dict = field(default_factory=dict)


class AlphaSignal(ABC):
    """One atomic edge. Computes exactly one thing."""

    name: str = "base"
    category: str = "unknown"

    @abstractmethod
    def compute(self, df: pd.DataFrame, ind: dict) -> AlphaOutput:
        """Given OHLCV + pre-computed indicators, return directional score.

        `ind` contains pre-computed indicator values for the latest candle,
        shared across all signals to avoid redundant computation.
        """
