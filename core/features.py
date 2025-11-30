"""Shared feature engineering helpers for price prediction."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

ATR_PERIOD = 14

FEATURE_COLUMNS: List[str] = [
    "price_change_1",
    "price_change_3",
    "price_change_5",
    "price_volatility",
    "ma_5",
    "ma_10",
    "ma_20",
    "price_to_ma5",
    "price_to_ma20",
    "rsi",
    "macd",
    "macd_hist",
    "volume_ratio",
    "volume_change",
    "adx",
    "trend_strength",
    "hour",
    "minute",
]


def _ensure_timestamp(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["timestamp"], utc=False)
    return ts.dt.tz_localize(None)


def build_feature_frame(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Append standardized ML features to an OHLCV dataframe.

    Args:
        df: DataFrame sorted chronologically with columns timestamp/open/high/low/close/volume.
        dropna: Whether to drop rows that contain NaNs after rolling calculations.
    Returns:
        DataFrame with all original columns plus FEATURE_COLUMNS.
    """

    data = df.copy()
    data["timestamp"] = _ensure_timestamp(data)
    data.sort_values("timestamp", inplace=True)
    data.reset_index(drop=True, inplace=True)

    close = data["close"]
    volume = data["volume"]

    data["price_change_1"] = close.pct_change(1)
    data["price_change_3"] = close.pct_change(3)
    data["price_change_5"] = close.pct_change(5)
    data["price_volatility"] = close.rolling(window=20).std()

    data["ma_5"] = close.rolling(window=5).mean()
    data["ma_10"] = close.rolling(window=10).mean()
    data["ma_20"] = close.rolling(window=20).mean()
    data["price_to_ma5"] = close / data["ma_5"]
    data["price_to_ma20"] = close / data["ma_20"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss
    data["rsi"] = 100 - (100 / (1 + rs))

    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = exp1 - exp2
    macd_signal = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - macd_signal

    data["volume_ratio"] = volume / volume.rolling(window=20).mean()
    data["volume_change"] = volume.pct_change(1)

    high_low = data["high"] - data["low"]
    high_close = (data["high"] - close.shift()).abs()
    low_close = (data["low"] - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["atr_14"] = tr.rolling(window=ATR_PERIOD).mean()
    data["adx"] = (tr.rolling(window=14).mean() / close) * 100
    data["trend_strength"] = data["adx"]

    data["hour"] = data["timestamp"].dt.hour
    data["minute"] = data["timestamp"].dt.minute

    if dropna:
        data = data.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    return data


def extract_feature_vector(row: pd.Series | dict, feature_names: Iterable[str] | None = None) -> List[float]:
    """Return the ordered feature vector for the model."""
    names = list(feature_names) if feature_names is not None else FEATURE_COLUMNS
    vector = []
    getter = row.get if isinstance(row, dict) else row.__getitem__
    for name in names:
        value = getter(name) if name in row else np.nan
        if value is None or not np.isfinite(value):
            vector.append(0.0)
        else:
            vector.append(float(value))
    return vector
