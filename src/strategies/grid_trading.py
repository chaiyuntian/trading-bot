"""Dynamic Grid Trading Strategy.

Sets grid levels based on ATR and Bollinger Bands.
Buys at lower grid levels, sells at upper grid levels.
Grid spacing adapts to current volatility.

Best for sideways / ranging markets.
"""

import pandas as pd
from src.strategies.base import BaseStrategy, TradeSignal, Signal
from src.indicators.technical import add_atr, add_bollinger_bands, add_rsi
from src.utils.logger import setup_logger

logger = setup_logger("strategy.grid")


class GridTradingStrategy(BaseStrategy):
    strategy_name = "Grid Trading"

    def __init__(self, config: dict):
        super().__init__(config)
        self.grid_levels = self.params.get("grid_levels", 10)
        self.grid_spacing_mult = self.params.get("grid_spacing_atr_mult", 0.5)
        self.grids: list[dict] = []
        self.last_grid_price = None

    def _build_grid(self, center_price: float, atr: float):
        spacing = atr * self.grid_spacing_mult
        half = self.grid_levels // 2

        self.grids = []
        for i in range(-half, half + 1):
            level_price = center_price + (i * spacing)
            self.grids.append({
                "price": level_price,
                "type": "buy" if i < 0 else ("sell" if i > 0 else "center"),
                "filled": False
            })

        logger.info(
            f"Grid built: {len(self.grids)} levels, "
            f"range {self.grids[0]['price']:.2f} - {self.grids[-1]['price']:.2f}, "
            f"spacing={spacing:.2f}"
        )

    def _should_rebuild(self, price: float) -> bool:
        if not self.grids:
            return True
        if price < self.grids[0]["price"] or price > self.grids[-1]["price"]:
            return True
        filled_count = sum(1 for g in self.grids if g["filled"])
        return filled_count > len(self.grids) * 0.7

    def generate_signal(self, df: pd.DataFrame) -> str:
        sig = self.generate_rich_signal(df)
        return sig.signal.value

    def generate_rich_signal(self, df: pd.DataFrame) -> TradeSignal:
        df = add_atr(df)
        df = add_bollinger_bands(df)
        df = add_rsi(df, 14)

        df = df.dropna()
        if len(df) < 3:
            return TradeSignal(Signal.HOLD, 0.0, "Insufficient data")

        curr = df.iloc[-1]
        price = curr["close"]
        atr = curr["atr_14"]
        bb_middle = curr["bb_middle"]

        if not self.grids or self._should_rebuild(price):
            self._build_grid(bb_middle, atr)
            self.last_grid_price = price
            return TradeSignal(Signal.HOLD, 0.0, "Grid rebuilt")

        for grid in self.grids:
            if grid["filled"]:
                continue

            distance = abs(price - grid["price"]) / price
            if distance < 0.002:  # Within 0.2% of grid level
                grid["filled"] = True
                if grid["type"] == "buy":
                    logger.info(f"Grid BUY at level {grid['price']:.2f}")
                    return TradeSignal(
                        Signal.BUY, 0.6,
                        f"Grid buy @ {grid['price']:.2f}",
                        entry_price=grid["price"],
                    )
                elif grid["type"] == "sell":
                    logger.info(f"Grid SELL at level {grid['price']:.2f}")
                    return TradeSignal(
                        Signal.SELL, 0.6,
                        f"Grid sell @ {grid['price']:.2f}",
                        entry_price=grid["price"],
                    )

        return TradeSignal(Signal.HOLD, 0.0, "No grid level hit")
