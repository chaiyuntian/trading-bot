"""Ring buffer for candle management — replaces repeated fetch_ohlcv calls.

Loaded once from REST, then updated incrementally from WebSocket events.
"""

from collections import deque
from typing import Optional

import pandas as pd


class CandleBuffer:
    """Efficient ring buffer of OHLCV candles."""

    def __init__(self, max_size: int = 500):
        self._buffer: deque[dict] = deque(maxlen=max_size)
        self._current: Optional[dict] = None

    def load(self, df: pd.DataFrame):
        """Bulk-load from a DataFrame (REST warmup)."""
        self._buffer.clear()
        for ts, row in df.iterrows():
            self._buffer.append({
                "timestamp": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })

    def update(self, candle: dict):
        """Update with a WebSocket kline event.

        If the candle is closed, append it to history.
        If still forming, update the live candle.
        """
        if candle.get("closed", False):
            self._buffer.append(candle)
            self._current = None
        else:
            self._current = candle

    def to_dataframe(self) -> pd.DataFrame:
        """Convert buffer to a DataFrame for signal computation."""
        data = list(self._buffer)
        if self._current:
            data.append(self._current)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df.set_index("timestamp", inplace=True)
        return df

    def __len__(self) -> int:
        return len(self._buffer) + (1 if self._current else 0)

    @property
    def last_price(self) -> Optional[float]:
        if self._current:
            return self._current["close"]
        if self._buffer:
            return self._buffer[-1]["close"]
        return None
