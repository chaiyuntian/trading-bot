import pandas as pd
import numpy as np
from src.indicators.technical import (
    add_rsi, add_macd, add_ema, add_bollinger_bands, add_atr
)


def make_sample_df(n=100):
    """Generate sample OHLCV data."""
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


def test_add_rsi():
    df = make_sample_df()
    df = add_rsi(df, 14)
    assert "rsi_14" in df.columns
    valid = df["rsi_14"].dropna()
    assert len(valid) > 0
    assert valid.min() >= 0
    assert valid.max() <= 100


def test_add_macd():
    df = make_sample_df()
    df = add_macd(df, 12, 26, 9)
    assert "macd" in df.columns
    assert "macd_signal" in df.columns
    assert "macd_hist" in df.columns


def test_add_ema():
    df = make_sample_df()
    df = add_ema(df, 50)
    assert "ema_50" in df.columns
    valid = df["ema_50"].dropna()
    assert len(valid) > 0


def test_add_bollinger_bands():
    df = make_sample_df()
    df = add_bollinger_bands(df)
    assert "bb_upper" in df.columns
    assert "bb_middle" in df.columns
    assert "bb_lower" in df.columns
    assert "bb_bandwidth" in df.columns


def test_add_atr():
    df = make_sample_df()
    df = add_atr(df)
    assert "atr_14" in df.columns
    valid = df["atr_14"].dropna()
    assert (valid >= 0).all()
