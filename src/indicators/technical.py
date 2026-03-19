import pandas as pd
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
    """Kaufman Adaptive Moving Average (KAMA).

    Adapts smoothing speed based on the efficiency ratio (ER):
    - High ER (trending): KAMA moves fast like a short EMA
    - Low ER (noisy): KAMA moves slowly like a long EMA

    This reduces whipsaws in choppy markets while staying responsive in trends.
    """
    close = df["close"].values
    n = len(close)
    col = f"kama_{period}"

    if n < period + 1:
        df[col] = None
        return df

    fast_alpha = 2.0 / (fast_sc + 1)
    slow_alpha = 2.0 / (slow_sc + 1)

    kama = pd.Series(index=df.index, dtype=float)
    kama.iloc[period - 1] = close[period - 1]

    for i in range(period, n):
        direction = abs(close[i] - close[i - period])
        volatility = sum(abs(close[j] - close[j - 1]) for j in range(i - period + 1, i + 1))

        if volatility == 0:
            er = 0.0
        else:
            er = direction / volatility

        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        kama.iloc[i] = kama.iloc[i - 1] + sc * (close[i] - kama.iloc[i - 1])

    df[col] = kama
    return df


def add_kama_efficiency_ratio(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """Kaufman Efficiency Ratio — measures trend strength (0=noise, 1=perfect trend)."""
    close = df["close"]
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    df[f"er_{period}"] = direction / volatility.replace(0, float("nan"))
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
    df = add_kama(df, params.get("kama_period", 10))

    return df
