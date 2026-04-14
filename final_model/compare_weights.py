import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Import data loader and template functions
from template.prelude_template import load_data
from template.model_development_template import _clean_array, allocate_sequential_stable
import final_model.model_development as fm

# =============================================================================
# Model Iteration Definitions
# =============================================================================

def multiplier_v1(z, r30, r1y, v, pma, g, f):
    mvrv_signal = -z
    mvrv_boost = np.where(z < -1.5, (z + 1.5)**2, 0)
    mom_signal = (r30 * 1.5) + (r1y * 0.25)
    
    combined = ((mvrv_signal + mvrv_boost) * 0.75) + (mom_signal * 0.25)
    
    dampener = np.where(v > 0.8, 1.0 - 0.4 * ((v - 0.8) / 0.2), 1.0)
    
    adjustment = np.clip(combined * 3.5, -4, 10)
    multiplier = np.exp(adjustment) * dampener
    return np.where(np.isfinite(multiplier), multiplier, 1.0)

def multiplier_v2(z, r30, r1y, v, pma, g, f):
    mvrv_signal = -z
    deep_value = z < -1.5
    improving = g > 0
    
    mvrv_boost = np.where(deep_value, (z + 1.5)**2, 0)
    bottom_confirmation = np.where(deep_value & improving, mvrv_boost * 0.75, 0)
    
    # Multiplicative regime modifier
    regime_multiplier = np.where(pma < 0, 1.0 + np.abs(pma), np.maximum(0.05, 1.0 - pma))
    
    combined = ((mvrv_signal + mvrv_boost + bottom_confirmation) * 0.75) + ((r30 * 1.5 + r1y * 0.25) * 0.25)
    combined = combined * regime_multiplier
    
    adjustment = np.clip(combined * 3.5, -4, 10)
    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)

def multiplier_v3(z, r30, r1y, v, pma, g, f):
    value_score = -z + np.where((z < -1.0) & (g > 0), np.abs(z)**1.5, 0)
    mom_score = (r30 * 2.0) + (r1y * 0.5)
    
    is_bear = pma < 0
    combined = np.where(is_bear, (value_score * 0.8) + (mom_score * 0.2), (value_score * 0.4) + (mom_score * 0.6))
    
    # The Bug: Multiplicative scaling of a signal that crosses zero
    regime_multiplier = np.where(is_bear, 1.0 + np.abs(pma) * 2.0, np.maximum(0.2, 1.0 - pma * 0.75))
    combined = combined * regime_multiplier
    
    adjustment = np.clip(combined * 2.0, -3, 5)
    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)

def multiplier_v4(z, r30, r1y, v, pma, g, f):
    value_score = -z
    deep_value = z < -1.0
    improving = g > 0
    
    value_boost = np.where(deep_value, np.abs(z) - 1.0, 0)
    confirmation_boost = np.where(deep_value & improving, value_boost * 1.0, 0)
    value_score = value_score + value_boost + confirmation_boost

    mom_score = r30 * 2.0
    base_signal = (value_score * 0.8) + (mom_score * 0.2)

    # The Fix: Additive shift in log-space
    regime_shift = np.where(pma < 0, np.abs(pma) * 2.0, -pma * 3.0)
    combined = base_signal + regime_shift

    dampener = np.where(v > 0.85, 1.0 - 0.5 * ((v - 0.85) / 0.15), 1.0)
    
    adjustment = np.clip(combined * 2.5 * dampener, -4.0, 4.5)
    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)

# =============================================================================
# Plotting Engine
# =============================================================================

def generate_plot(features_df, multiplier_func, name, color, filename):
    """Generates and saves a static plot for a given model iteration."""
    print(f"Generating plot for {name}...")
    
    start_date = pd.Timestamp("2021-01-01")
    end_date = pd.Timestamp("2023-12-31")
    df_window = features_df.loc[start_date:end_date].copy()
    
    # Compute weights
    base = np.ones(len(df_window)) / len(df_window)
    z = _clean_array(df_window["mvrv_zscore"].values)
    r30 = _clean_array(df_window["roi_30d"].values)
    r1y = _clean_array(df_window["roi_1yr"].values)
    v = _clean_array(df_window["volatility_pct"].values)
    pma = _clean_array(df_window["price_vs_ma"].values)
    g = _clean_array(df_window["mvrv_gradient"].values)
    f = _clean_array(df_window["fed_uncertainty"].values)
    
    dyn = multiplier_func(z, r30, r1y, v, pma, g, f)
    raw = base * dyn
    weights = allocate_sequential_stable(raw, len(df_window), None)
    
    # Calculate multiplier vs uniform DCA
    uniform_weight = 1.0 / len(df_window)
    multiplier = weights / uniform_weight
    price = df_window["PriceUSD_coinmetrics"]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1.5]})
    
    # Top subplot: Price
    ax1.plot(price.index, price.values, color='black', linewidth=1.5, label='BTC Price (USD)')
    ax1.set_yscale('log')
    ax1.set_ylabel('Price (USD) - Log Scale')
    ax1.set_title(f'{name} - Allocation Weights (2021-2023)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Highlight events
    events = {'2021 ATH': '2021-11-10', 'Luna Crash': '2022-05-09', 'FTX Crash': '2022-11-08'}
    for ev_name, date in events.items():
        ts = pd.Timestamp(date)
        if ts in price.index:
            ax1.axvline(ts, color='red', linestyle='--', alpha=0.5)
            ax1.text(ts, price.max(), f' {ev_name}', rotation=90, va='top', alpha=0.7)
            ax2.axvline(ts, color='red', linestyle='--', alpha=0.5)
    
    # Bottom subplot: Multiplier
    ax2.plot(price.index, multiplier, color=color, linewidth=1.5, alpha=0.8, label=name)
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=2, label='Uniform DCA Baseline (1.0x)')
    ax2.set_ylabel('Allocation Multiplier (x Uniform)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("final_model/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved to {output_path}")

def main():
    print("Loading Bitcoin data...")
    df = load_data()
    
    print("Precomputing features...")
    features_df = fm.precompute_features(df)
    # Add dummy fed_uncertainty to match the notebook's 7-argument functions
    features_df["fed_uncertainty"] = 0.5
    
    # Generate plots for all iterations
    generate_plot(features_df, multiplier_v1, "V1: Baseline", "#3b82f6", "v1_baseline.png")
    generate_plot(features_df, multiplier_v2, "V2: The Sniper", "#8b5cf6", "v2_sniper.png")
    generate_plot(features_df, multiplier_v3, "V3: Multiplicative Bug", "#ef4444", "v3_bug.png")
    generate_plot(features_df, multiplier_v4, "V4: Log-Space Additivity (Final Model)", "#10b981", "v4_final.png")
    
    print("\nAll plots generated successfully! You can now display them in your notebook.")

if __name__ == "__main__":
    main()
