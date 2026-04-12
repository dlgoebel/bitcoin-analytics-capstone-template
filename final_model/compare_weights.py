import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Import data loader
from template.prelude_template import load_data

# Import both models
import example_1.model_development_example_1 as ex1
import final_model.model_development as fm

def main():
    print("Loading Bitcoin data...")
    df = load_data()
    
    # Define our comparison window (2021 bull market through 2022 bear market)
    start_date = pd.Timestamp("2021-01-01")
    end_date = pd.Timestamp("2023-12-31")
    
    print(f"Analysing window: {start_date.date()} to {end_date.date()}")
    
    # 1. Precompute features for both models
    print("Precomputing features for Example 1...")
    features_ex1 = ex1.precompute_features(df)
    
    print("Precomputing features for Final Model...")
    features_fm = fm.precompute_features(df)
    
    # 2. Compute weights for the window
    # We use end_date as the current_date to simulate the final weights for the period
    print("Computing weights...")
    weights_ex1 = ex1.compute_window_weights(
        features_ex1, start_date, end_date, end_date
    )
    
    weights_fm = fm.compute_window_weights(
        features_fm, start_date, end_date, end_date
    )
    
    # 3. Prepare data for plotting
    # Extract price for the window
    price = df.loc[start_date:end_date, "PriceUSD_coinmetrics"]
    
    # Calculate the uniform DCA baseline weight (1 / number of days)
    n_days = len(price)
    uniform_weight = 1.0 / n_days
    
    # 4. Create the visualisation
    print("Generating comparison chart...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.5]})
    
    # Top subplot: Bitcoin Price
    ax1.plot(price.index, price.values, color='black', linewidth=1.5, label='BTC Price (USD)')
    ax1.set_yscale('log')
    ax1.set_ylabel('Price (USD) - Log Scale', fontsize=11)
    ax1.set_title('Bitcoin Price vs Strategy Allocation Weights (2021-2023)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Highlight key events for context
    events = {
        '2021 ATH': '2021-11-10',
        'Luna Crash': '2022-05-09',
        'FTX Crash': '2022-11-08'
    }
    for name, date in events.items():
        ts = pd.Timestamp(date)
        if ts in price.index:
            ax1.axvline(ts, color='red', linestyle='--', alpha=0.5)
            ax1.text(ts, price.max(), f' {name}', rotation=90, verticalalignment='top', alpha=0.7)
            ax2.axvline(ts, color='red', linestyle='--', alpha=0.5)
    
    # Bottom subplot: Weights Comparison
    # Convert weights to a multiple of the uniform baseline for easier interpretation
    mult_ex1 = weights_ex1 / uniform_weight
    mult_fm = weights_fm / uniform_weight
    
    ax2.plot(mult_ex1.index, mult_ex1.values, color='#3b82f6', linewidth=1.5, alpha=0.8, 
             label='Example 1 (Includes 200 DMA & MVRV Gradient)')
    ax2.plot(mult_fm.index, mult_fm.values, color='#10b981', linewidth=1.5, alpha=0.8, 
             label='Final Model (MVRV + Momentum + 200 DMA + Gradient)')
    
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=2, label='Uniform DCA Baseline (1.0x)')
    
    ax2.set_ylabel('Allocation Multiplier (x times Uniform DCA)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Format x-axis
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("final_model/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "weight_comparison_2021_2023.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Chart saved successfully to: {output_path}")
    
    # Print some summary statistics
    print("\n--- Weight Distribution Summary (Multiplier of Uniform DCA) ---")
    print(f"Example 1   - Max: {mult_ex1.max():.2f}x, Min: {mult_ex1.min():.2f}x, Median: {mult_ex1.median():.2f}x")
    print(f"Final Model - Max: {mult_fm.max():.2f}x, Min: {mult_fm.min():.2f}x, Median: {mult_fm.median():.2f}x")

if __name__ == "__main__":
    main()
