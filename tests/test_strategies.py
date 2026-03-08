import pandas as pd
import numpy as np
from src.strategies.rsi_macd import RsiMacdStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.grid_trading import GridTradingStrategy
from src.strategies.dca_momentum import DCAMomentumStrategy


def make_config():
    return {
        "strategy": {
            "name": "rsi_macd",
            "params": {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "ema_period": 50,
                "grid_levels": 10,
                "grid_spacing_atr_mult": 0.5,
                "dca_interval": 4,
            }
        }
    }


def make_sample_df(n=200):
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 30
    volume = np.random.rand(n) * 1000 + 100

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="15min")
    return df


def test_rsi_macd_returns_valid_signal():
    config = make_config()
    strategy = RsiMacdStrategy(config)
    df = make_sample_df()
    signal = strategy.generate_signal(df)
    assert signal in ("buy", "sell", "hold")


def test_mean_reversion_returns_valid_signal():
    config = make_config()
    strategy = MeanReversionStrategy(config)
    df = make_sample_df()
    signal = strategy.generate_signal(df)
    assert signal in ("buy", "sell", "hold")


def test_grid_returns_valid_signal():
    config = make_config()
    strategy = GridTradingStrategy(config)
    df = make_sample_df()
    signal = strategy.generate_signal(df)
    assert signal in ("buy", "sell", "hold")


def test_dca_momentum_returns_valid_signal():
    config = make_config()
    strategy = DCAMomentumStrategy(config)
    df = make_sample_df()
    signal = strategy.generate_signal(df)
    assert signal in ("buy", "sell", "hold")


def test_strategy_handles_small_data():
    config = make_config()
    strategy = RsiMacdStrategy(config)
    # Only 5 candles - too little data
    df = make_sample_df(5)
    signal = strategy.generate_signal(df)
    assert signal == "hold"
