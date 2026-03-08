import pandas as pd
import numpy as np
import pandas_ta as ta


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
             signal: int = 9) -> pd.DataFrame:
    macd = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    if macd is None:
        df["macd"] = None
        df["macd_signal"] = None
        df["macd_hist"] = None
        return df
    df["macd"] = macd[f"MACD_{fast}_{slow}_{signal}"]
    df["macd_signal"] = macd[f"MACDs_{fast}_{slow}_{signal}"]
    df["macd_hist"] = macd[f"MACDh_{fast}_{slow}_{signal}"]
    return df


def add_ema(df: pd.DataFrame, period: int = 50) -> pd.DataFrame:
    df[f"ema_{period}"] = ta.ema(df["close"], length=period)
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20,
                        std: float = 2.0) -> pd.DataFrame:
    bb = ta.bbands(df["close"], length=period, std=std)
    if bb is None:
        df["bb_upper"] = None
        df["bb_middle"] = None
        df["bb_lower"] = None
        df["bb_bandwidth"] = None
        return df
    # pandas-ta column names vary by version; find them dynamically
    upper_col = [c for c in bb.columns if c.startswith("BBU_")][0]
    middle_col = [c for c in bb.columns if c.startswith("BBM_")][0]
    lower_col = [c for c in bb.columns if c.startswith("BBL_")][0]
    df["bb_upper"] = bb[upper_col]
    df["bb_middle"] = bb[middle_col]
    df["bb_lower"] = bb[lower_col]
    df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df[f"atr_{period}"] = ta.atr(df["high"], df["low"], df["close"], length=period)
    return df


def add_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df[f"vol_sma_{period}"] = ta.sma(df["volume"], length=period)
    return df


def add_stochastic_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    stoch = ta.stochrsi(df["close"], length=period)
    if stoch is not None:
        df["stoch_rsi_k"] = stoch.iloc[:, 0]
        df["stoch_rsi_d"] = stoch.iloc[:, 1]
    return df


def add_kama(df: pd.DataFrame, period: int = 10,
             fast_sc: int = 2, slow_sc: int = 30) -> pd.DataFrame:
    """Kaufman's Adaptive Moving Average (KAMA).

    Adapts speed based on Efficiency Ratio (ER):
    - High ER (trending): behaves like fast EMA
    - Low ER (noisy): behaves like slow EMA
    Far fewer whipsaw signals than SMA/EMA.
    """
    close = df["close"].values
    n = len(close)
    if n < period + 1:
        df[f"kama_{period}"] = float("nan")
        return df

    # Smoothing constants
    fast_alpha = 2.0 / (fast_sc + 1)
    slow_alpha = 2.0 / (slow_sc + 1)

    kama = np.full(n, float("nan"))
    kama[period - 1] = close[period - 1]

    for i in range(period, n):
        # Efficiency Ratio = direction / volatility
        direction = abs(close[i] - close[i - period])
        volatility = sum(abs(close[j] - close[j - 1]) for j in range(i - period + 1, i + 1))

        if volatility == 0:
            er = 0
        else:
            er = direction / volatility

        # Smoothing constant adapts between fast and slow
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])

    df[f"kama_{period}"] = kama
    return df


def add_vwap_deviation(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """VWAP Deviation Z-Score.

    Z > +2.0: Overbought relative to institutional fair value (mean-reversion short)
    Z < -2.0: Oversold relative to institutional fair value (mean-reversion long)
    Historical reversion rate at 2.0+ extremes: 60-75%.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumvol = df["volume"].rolling(period).sum()
    vwap = (typical_price * df["volume"]).rolling(period).sum() / cumvol

    deviation = df["close"] - vwap
    std = deviation.rolling(period).std()
    z_score = deviation / std

    df["vwap"] = vwap
    df["vwap_z"] = z_score
    return df


def add_efficiency_ratio(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """Kaufman's Efficiency Ratio (ER).

    ER = 1.0: perfect trend (all movement in one direction)
    ER = 0.0: pure noise (lots of movement, no net direction)

    Used for regime detection and adaptive indicator tuning.
    """
    close = df["close"]
    direction = abs(close - close.shift(period))
    volatility = abs(close - close.shift(1)).rolling(period).sum()
    er = direction / volatility
    er = er.fillna(0)
    df[f"er_{period}"] = er
    return df


def add_all_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add all indicators based on strategy config."""
    params = config.get("strategy", {}).get("params", {})

    df = add_rsi(df, params.get("rsi_period", 14))
    df = add_macd(df, params.get("macd_fast", 12),
                  params.get("macd_slow", 26),
                  params.get("macd_signal", 9))
    df = add_ema(df, params.get("ema_period", 50))
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volume_sma(df)

    return df
