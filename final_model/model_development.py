import logging
import numpy as np
import pandas as pd

from template.model_development_template import (
    _clean_array,
    allocate_sequential_stable,
)

# =============================================================================
# Constants
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL = "CapMVRVCur"

# Strategy parameters
MIN_W = 1e-6
DYNAMIC_STRENGTH = 2.0  # Softened from 3.5 to spread capital across the bear market
MVRV_ROLLING_WINDOW = 365
VOLATILITY_WINDOW = 30
MA_WINDOW = 200
MVRV_GRADIENT_WINDOW = 30

# =============================================================================
# Feature Engineering
# =============================================================================

def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MVRV, Momentum, Volatility, and Regime features for weight calculation.

    Features (all lagged 1 day to prevent look-ahead bias):
    - mvrv_zscore: 365-day rolling Z-score of MVRV.
    - roi_30d: 30-day return momentum.
    - roi_1yr: 365-day return momentum.
    - volatility_pct: 30-day rolling volatility, ranked as a percentile [0, 1].
    - price_vs_ma: Price relative to 200-day moving average.
    - mvrv_gradient: 30-day change in MVRV Z-score.

    Args:
        df: DataFrame with price and MVRV columns

    Returns:
        DataFrame with price and computed features
    """
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    # Filter to valid date range
    price = df[PRICE_COL].loc["2010-07-18":].copy()

    # 1. MVRV Z-Score (Primary Signal)
    if MVRV_COL in df.columns:
        mvrv = df[MVRV_COL].loc[price.index]
        mvrv_mean = mvrv.rolling(MVRV_ROLLING_WINDOW, min_periods=180).mean()
        mvrv_std = mvrv.rolling(MVRV_ROLLING_WINDOW, min_periods=180).std()
        mvrv_zscore = ((mvrv - mvrv_mean) / mvrv_std).fillna(0).clip(-4, 4)
    else:
        mvrv_zscore = pd.Series(0.0, index=price.index)

    # 2. Return Momentum (Secondary Signal)
    roi_30d = price.pct_change(30).fillna(0).clip(-1, 1)
    roi_1yr = price.pct_change(365).fillna(0).clip(-2, 5)

    # 3. Volatility Dampener (Tertiary Signal)
    daily_ret = price.pct_change().fillna(0)
    vol_30d = daily_ret.rolling(VOLATILITY_WINDOW, min_periods=15).std().fillna(0)
    
    # Normalize volatility to a rolling percentile [0, 1]
    volatility_pct = vol_30d.rolling(365, min_periods=180).apply(
        lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1) if len(x) > 1 else 0.5,
        raw=False
    ).fillna(0.5)

    # 4. Price vs 200-day MA (Regime Signal)
    ma_200 = price.rolling(MA_WINDOW, min_periods=100).mean()
    price_vs_ma = (price / ma_200) - 1.0
    price_vs_ma = price_vs_ma.fillna(0).clip(-0.8, 2.0)

    # 5. MVRV Gradient (Trajectory Signal)
    mvrv_gradient = mvrv_zscore.diff(MVRV_GRADIENT_WINDOW).fillna(0).clip(-3, 3)

    # Build DataFrame
    features = pd.DataFrame({
        PRICE_COL: price,
        "mvrv_zscore": mvrv_zscore,
        "roi_30d": roi_30d,
        "roi_1yr": roi_1yr,
        "volatility_pct": volatility_pct,
        "price_vs_ma": price_vs_ma,
        "mvrv_gradient": mvrv_gradient
    }, index=price.index)

    # Lag signals by 1 day to prevent look-ahead bias
    signal_cols = [
        "mvrv_zscore", "roi_30d", "roi_1yr", 
        "volatility_pct", "price_vs_ma", "mvrv_gradient"
    ]
    features[signal_cols] = features[signal_cols].shift(1).fillna(0)

    return features

# =============================================================================
# Dynamic Multiplier
# =============================================================================

def compute_dynamic_multiplier(
    mvrv_zscore: np.ndarray,
    roi_30d: np.ndarray,
    roi_1yr: np.ndarray,
    volatility_pct: np.ndarray,
    price_vs_ma: np.ndarray,
    mvrv_gradient: np.ndarray
) -> np.ndarray:
    """Compute weight multiplier using Regime-Conditional Blending.

    Args:
        mvrv_zscore: MVRV Z-score in [-4, 4]
        roi_30d: 30-day return
        roi_1yr: 1-year return
        volatility_pct: Volatility percentile [0, 1]
        price_vs_ma: Price relative to 200 DMA
        mvrv_gradient: 30-day change in MVRV Z-score

    Returns:
        Multipliers centred around 1.0
    """
    # 1. Value Score (MVRV)
    value_score = -mvrv_zscore
    
    # Non-linear boost for deep value + improving trajectory
    deep_value = mvrv_zscore < -1.0
    improving = mvrv_gradient > 0
    value_boost = np.where(deep_value & improving, np.abs(mvrv_zscore)**1.5, 0)
    value_score = value_score + value_boost

    # 2. Momentum Score
    mom_score = (roi_30d * 2.0) + (roi_1yr * 0.5)

    # 3. Regime-Conditional Blending
    is_bear = price_vs_ma < 0
    
    # In bear markets, value is the primary driver (ignore negative momentum).
    # In bull markets, momentum is more important to ride the trend.
    combined = np.where(
        is_bear,
        (value_score * 0.8) + (mom_score * 0.2),
        (value_score * 0.4) + (mom_score * 0.6)
    )

    # 4. Overvaluation Penalty
    # Regardless of momentum, if MVRV is dangerously high, suppress buying.
    overvalued_penalty = np.where(mvrv_zscore > 1.5, (mvrv_zscore - 1.5) * 2.0, 0)
    combined = combined - overvalued_penalty

    # 5. Regime Multiplier (Overall scaling)
    # Boost allocations in bear markets, gently suppress in bull markets
    regime_multiplier = np.where(
        is_bear,
        1.0 + np.abs(price_vs_ma) * 2.0,
        np.maximum(0.2, 1.0 - price_vs_ma * 0.75)
    )
    
    combined = combined * regime_multiplier

    # 6. Volatility Dampening
    # Extreme volatility (top 15%) reduces allocation to avoid catching falling knives blindly
    dampener = np.where(
        volatility_pct > 0.85,
        1.0 - 0.5 * ((volatility_pct - 0.85) / 0.15),
        1.0
    )
    combined = combined * dampener

    # Scale and clip to prevent extreme allocations
    adjustment = np.clip(combined * DYNAMIC_STRENGTH, -3, 5)
    multiplier = np.exp(adjustment)
    
    return np.where(np.isfinite(multiplier), multiplier, 1.0)

# =============================================================================
# Weight Computation API
# =============================================================================

def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date window using precomputed features."""
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    # Extract and clean features
    mvrv_zscore = _clean_array(df["mvrv_zscore"].values)
    roi_30d = _clean_array(df["roi_30d"].values)
    roi_1yr = _clean_array(df["roi_1yr"].values)
    volatility_pct = _clean_array(df["volatility_pct"].values)
    price_vs_ma = _clean_array(df["price_vs_ma"].values)
    mvrv_gradient = _clean_array(df["mvrv_gradient"].values)

    # Compute dynamic weights
    dyn = compute_dynamic_multiplier(
        mvrv_zscore, roi_30d, roi_1yr, volatility_pct, price_vs_ma, mvrv_gradient
    )
    raw = base * dyn

    # Allocate with stability
    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)

    return pd.Series(weights, index=df.index)

def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date range with lock-on-compute stability."""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Extend features for future dates
    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.0 for col in features_df.columns},
            index=missing,
        )
        if "volatility_pct" in placeholder.columns:
            placeholder["volatility_pct"] = 0.5
        if "price_vs_ma" in placeholder.columns:
            placeholder["price_vs_ma"] = 0.0
        if "mvrv_gradient" in placeholder.columns:
            placeholder["mvrv_gradient"] = 0.0
            
        features_df = pd.concat([features_df, placeholder]).sort_index()

    # Determine past/future split
    past_end = min(current_date, end_date)
    if start_date <= past_end:
        n_past = len(pd.date_range(start=start_date, end=past_end, freq="D"))
    else:
        n_past = 0

    weights = compute_weights_fast(
        features_df, start_date, end_date, n_past, locked_weights
    )
    return weights.reindex(full_range, fill_value=0.0)
