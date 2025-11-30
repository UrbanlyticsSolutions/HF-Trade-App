"""Shared helpers for labeling directional price moves."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

DEFAULT_HORIZON = 5
DEFAULT_ATR_MULTIPLIER = 0.55  # tuned to ~0.055% when ATR ~= 0.1% of price
MIN_THRESHOLD_PCT = 0.0005


def compute_future_returns(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> Tuple[pd.Series, pd.Series]:
    """Return future close and pct move for the desired horizon."""
    future_close = df["close"].shift(-horizon)
    future_return = (future_close - df["close"]) / df["close"]
    return future_close, future_return


def compute_threshold_pct(
    df: pd.DataFrame,
    atr_column: str = "atr_14",
    multiplier: float = DEFAULT_ATR_MULTIPLIER,
    min_threshold: float = MIN_THRESHOLD_PCT,
) -> pd.Series:
    """Convert ATR into a symmetric percentage threshold."""
    atr = df.get(atr_column)
    if atr is None:
        raise ValueError(f"Column '{atr_column}' not found; ensure build_feature_frame populated it.")
    base = atr / df["close"]
    thresholds = (base * multiplier).clip(lower=min_threshold)
    return thresholds.fillna(min_threshold)


def assign_direction_label(future_return: float, threshold: float) -> int:
    if np.isnan(future_return) or np.isnan(threshold):
        return -1
    if future_return > threshold:
        return 2
    if future_return < -threshold:
        return 0
    return 1


def label_directions(future_returns: pd.Series, thresholds: pd.Series) -> pd.Series:
    """Vectorized wrapper for assign_direction_label."""
    values = [assign_direction_label(fr, th) for fr, th in zip(future_returns, thresholds)]
    return pd.Series(values, index=future_returns.index)


@dataclass(frozen=True)
class LabelMetadata:
    future_close_col: str = "future_close"
    future_return_col: str = "future_return"
    threshold_pct_col: str = "label_threshold_pct"
    atr_col: str = "atr_14"


LABEL_METADATA = LabelMetadata()
