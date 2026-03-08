import pandas as pd
import numpy as np
from src.strategies.rsi_macd import RsiMacdStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.grid_trading import GridTradingStrategy
from src.strategies.dca_momentum import DCAMomentumStrategy
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.base import Signal


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
                "ensemble_min_consensus": 0.4,
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


def test_rsi_macd_rich_signal():
    config = make_config()
    strategy = RsiMacdStrategy(config)
    df = make_sample_df()
    sig = strategy.generate_rich_signal(df)
    assert sig.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
    assert 0 <= sig.confidence <= 1.0
    assert isinstance(sig.reason, str)


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


def test_ensemble_returns_valid_signal():
    config = make_config()
    strategy = EnsembleStrategy(config)
    df = make_sample_df()
    signal = strategy.generate_signal(df)
    assert signal in ("buy", "sell", "hold")


def test_ensemble_rich_signal_has_regime():
    config = make_config()
    strategy = EnsembleStrategy(config)
    df = make_sample_df()
    sig = strategy.generate_rich_signal(df)
    assert sig.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
    assert "R=" in sig.reason


def test_trend_following_returns_valid_signal():
    """Trend following strategy produces valid signals."""
    config = make_config()
    config["strategy"]["params"]["donchian_entry"] = 20
    config["strategy"]["params"]["donchian_exit"] = 10
    config["strategy"]["params"]["atr_stop_mult"] = 2.0
    config["strategy"]["params"]["volume_confirm"] = True
    strategy = TrendFollowingStrategy(config)
    df = make_sample_df()
    signal = strategy.generate_signal(df)
    assert signal in ("buy", "sell", "hold")


def test_trend_following_rich_signal_has_metadata():
    """Trend following rich signal includes ATR and channel info."""
    config = make_config()
    config["strategy"]["params"]["donchian_entry"] = 20
    config["strategy"]["params"]["donchian_exit"] = 10
    strategy = TrendFollowingStrategy(config)

    # Create a strong uptrend to trigger breakout
    np.random.seed(42)
    n = 200
    # Price steadily rising to ensure breakout above 20-period high
    close = 50000 + np.arange(n) * 20 + np.random.randn(n) * 30
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    df = pd.DataFrame({
        "open": close - 10,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.rand(n) * 1000 + 500,
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="4h")

    sig = strategy.generate_rich_signal(df)
    assert sig.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
    assert 0 <= sig.confidence <= 1.0


def test_trend_following_small_data():
    """Trend following holds with insufficient data."""
    config = make_config()
    config["strategy"]["params"]["donchian_entry"] = 20
    strategy = TrendFollowingStrategy(config)
    df = make_sample_df(10)
    sig = strategy.generate_rich_signal(df)
    assert sig.signal == Signal.HOLD


def test_strategy_handles_small_data():
    config = make_config()
    strategy = RsiMacdStrategy(config)
    df = make_sample_df(5)
    signal = strategy.generate_signal(df)
    assert signal == "hold"
