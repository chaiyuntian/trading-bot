import pandas as pd
import pandas_ta as ta


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
             signal: int = 9) -> pd.DataFrame:
    macd = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
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
    df["bb_upper"] = bb[f"BBU_{period}_{std}"]
    df["bb_middle"] = bb[f"BBM_{period}_{std}"]
    df["bb_lower"] = bb[f"BBL_{period}_{std}"]
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
