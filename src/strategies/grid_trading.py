import pandas as pd
from src.strategies.base import BaseStrategy
from src.indicators.technical import add_atr, add_bollinger_bands, add_rsi
from src.utils.logger import setup_logger

logger = setup_logger("strategy.grid")


class GridTradingStrategy(BaseStrategy):
    """
    Dynamic Grid Trading Strategy
    --------------------------------
    - Sets grid levels based on ATR and Bollinger Bands
    - Buys at lower grid levels, sells at upper grid levels
    - Grid spacing adapts to current volatility
    - Best for sideways/ranging markets

    Parameters:
      - grid_levels: Number of grid lines (default 10)
      - grid_spacing_atr_mult: ATR multiplier for grid spacing
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.grid_levels = self.params.get("grid_levels", 10)
        self.grid_spacing_mult = self.params.get("grid_spacing_atr_mult", 0.5)
        self.grids: list[dict] = []
        self.last_grid_price = None

    def name(self) -> str:
        return "Grid Trading"

    def _build_grid(self, center_price: float, atr: float):
        """Build grid levels around center price."""
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

    def generate_signal(self, df: pd.DataFrame) -> str:
        df = add_atr(df)
        df = add_bollinger_bands(df)
        df = add_rsi(df, 14)

        df = df.dropna()
        if len(df) < 3:
            return "hold"

        curr = df.iloc[-1]
        price = curr["close"]
        atr = curr["atr_14"]
        bb_middle = curr["bb_middle"]

        # Rebuild grid periodically or on first run
        if not self.grids or self._should_rebuild(price, atr):
            self._build_grid(bb_middle, atr)
            self.last_grid_price = price
            return "hold"

        # Find nearest unfilled grid level
        for grid in self.grids:
            if grid["filled"]:
                continue

            distance = abs(price - grid["price"]) / price

            if distance < 0.002:  # Within 0.2% of grid level
                grid["filled"] = True
                if grid["type"] == "buy":
                    logger.info(f"Grid BUY at level {grid['price']:.2f}")
                    return "buy"
                elif grid["type"] == "sell":
                    logger.info(f"Grid SELL at level {grid['price']:.2f}")
                    return "sell"

        return "hold"

    def _should_rebuild(self, price: float, atr: float) -> bool:
        if not self.grids:
            return True
        grid_range_low = self.grids[0]["price"]
        grid_range_high = self.grids[-1]["price"]
        if price < grid_range_low or price > grid_range_high:
            return True
        filled_count = sum(1 for g in self.grids if g["filled"])
        if filled_count > len(self.grids) * 0.7:
            return True
        return False
