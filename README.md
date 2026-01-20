# Bitcoin Analytics Capstone: Polymarket Integration

This capstone challenges you to build a Bitcoin DCA (Dollar Cost Averaging) strategy by integrating **Polymarket prediction data** with traditional on-chain metrics.

---

## Project (Trilemma Foundation): Improving Institutional Bitcoin Accumulation Strategies

* **Problem:** Institutions are accumulating Bitcoin; standard DCA may be suboptimal under competitive, high-volume conditions.
* **Goal:** Design **data-driven, long-only** accumulation strategies that keep DCA’s systematic discipline but **improve acquisition efficiency** within a fixed **budget + time horizon** (esp. institutional scale, e.g., $1M+/month).
* **Core workstreams:** BTC primer → EDA of BTC properties → feature-informed **daily purchase schedules** → **backtesting + tuning** (optional: slippage/execution risk) → evaluation tooling + visual benchmarks vs DCA → open-source dashboards/resources.
* **Data provided:** Market (price/OHLC/volume/BVOL), on-chain (UTXO/exchange flows/etc.), macro indicators, sentiment metrics; external data welcome after core exploration.
* **Tech expectations:** Python-first; time-series/statistical analysis + ML; API/data pipelines; interactive dashboards (Dash/Plotly or Streamlit).
* **Deliverables:** MIT-licensed open-source repo(s) with reproducible models, interactive dashboards, tutorials/notebooks/READMEs, and a final presentation.
* **IP/Licensing:** Code, analysis, and documentation are open-sourced under **MIT** (contributors retain attribution). The provided data is not covered by the MIT license and retains its original licensing terms.
* **Contacts:** Mohammad Ashkani (Project Lead) and Mateusz Faltyn (Technical Lead).

---

## The Capstone Objective

The goal of this project is to evolve a basic 200-day MA DCA model into a sophisticated, market-aware strategy.

**The Challenge:** Improve the integration of Polymarket data into the model to produce superior predictive signals.

### Your Tasks

1. **Fork the Template**: Create your own version of this repository to build your model.
2. **Integrate Polymarket Data**: Leverage the provided Polymarket datasets (Politics, Finance, Crypto) to extract predictive signals using `load_polymarket_data()`.
3. **Direct Integration**: Modify the core model logic to incorporate these signals alongside the 200-day Moving Average baseline.
4. **Outperform the Baseline**: Demonstrate through backtesting that your integrated model provides superior risk-adjusted outcomes compared to the foundation model.

---

### How to Run Backtests

Run the following commands from the root directory:

**Baseline Model (Template):**
```bash
python -m template.backtest_template
```

**Enhanced Model (Example 1):**
```bash
python -m example_1.run_backtest
```

Detailed documentation can be found in [Backtest Documentation](file:///Users/mattfaltyn/Desktop/hypertrial_trilemma/foundation/bitcoin-analytics-capstone-template/template/backtest_template.md).

---

---

## Getting Started

### 1. Repository Setup

1. **Fork this repository** to your own GitHub account.
2. Clone your fork locally:

   ```bash
   git clone <your-fork-url>
   cd bitcoin-analytics-capstone-template
   ```

3. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### 2. Data Acquisition

All necessary data is organized in the `data/` directory. You can download it automatically:

```bash
python data/download_data.py
```

**What's included:**

* **CoinMetrics BTC Data**: Historical price data (`data/Coin Metrics/`).
* **Polymarket Data**: 6 Parquet files containing trades, odds history, and market metadata (`data/Polymarket/`). Use `load_polymarket_data()` from `template/prelude_template.py` to load all raw data files.
* **Data Schemas**: Detailed documentation in `data/Polymarket/polymarket_btc_analytics_schema.md`.

---

## The Foundation (Baseline Model)

The repository provides a working **Foundation Model** located in `template/`. This simplified baseline model uses:

* **200-day Moving Average**: Buy more when price is below the 200-day MA, buy less when above.

This simple strategy serves as a baseline for comparison. Your task is to enhance it with Polymarket data and other signals.

### Running the Baseline

Before adding Polymarket info, run the current backtest to establish your baseline:

```bash
python -m template.backtest_template
```

---

## The Challenge (Polymarket Integration)

Your task is to modify `template/model_development_template.py` to integrate Polymarket signals.

> **Important:** While the primary goal is Polymarket integration, you are also encouraged to make changes to the base model logic (e.g., add MVRV, additional indicators) if it helps you better incorporate the prediction market data and improve the model.

### Data Loading

The template provides `load_polymarket_data()` in `template/prelude_template.py` to load all raw Polymarket parquet files:

```python
from template.prelude_template import load_polymarket_data

# Load all raw Polymarket data
polymarket_data = load_polymarket_data()
# Returns dict with keys: 'markets', 'tokens', 'trades', 'odds_history', 'event_stats', 'summary'

# Access specific datasets
markets_df = polymarket_data['markets']
trades_df = polymarket_data['trades']
```

For model-specific processing (e.g., extracting BTC sentiment), define your own functions in your model file. See `example_1/model_development_example_1.py` for an example.

### Potential Signal Leads

* **Election Probabilities**: How do presidential odds correlate with BTC volatility?

* **Economic Indicators**: Do prediction markets for Fed rate cuts lead BTC price movements?
* **Crypto Sentiment**: Use specific "Polymarket Crypto" markets as lead indicators for retail sentiment.

### Repository Workflow

```
.
├── template/                        # DIRECTORY TO FORK
│   ├── prelude_template.py          # Data loading utilities (includes load_polymarket_data())
│   ├── model_development_template.py # INTEGRATE POLYMARKET SIGNALS HERE
│   ├── backtest_template.py         # Evaluate your new strategy
│   ├── model_template.md            # Model logic documentation
│   └── backtest_template.md         # Backtest engine documentation
├── example_1/                       # REFERENCE IMPLEMENTATION
│   ├── model_development_example_1.py# Example Polymarket integration (imports from template)
│   └── model_example_1.md           # Documentation for updated logic
├── data/                            # Bitcoin & Polymarket source data
├── output/                          # Your strategy's performance visualizations
└── tests/                           # Ensure your model remains stable
```

---

## Example Implementation: `example_1`

To help you get started, we've provided `example_1/`. This demonstrates:

1. **Data Loading**: How to use `load_polymarket_data()` from template and create model-specific processing functions (e.g., `load_polymarket_btc_sentiment()`).
2. **Signal Generation**: A concrete example of mapping Polymarket odds to model modifiers.
3. **Import Pattern**: Shows how to import from `template/` modules and extend them with model-specific logic.

**Study `example_1/model_development_example_1.py` to understand the workflow before building your own model in a new folder.**

Note: Example implementations should import from `template/` rather than duplicating prelude and backtest files.

---

## Evaluation Metrics

Your integrated model will be evaluated on the following (automated via `backtest_template.py`):

* **Win Rate**: Must outperform uniform DCA in >50% of 1-year windows.
* **SPD Percentile**: Overall efficiency of satoshi accumulation.
* **Model Score**: A combination of win rate and reward-to-risk percentile.

---

## Documentation

* **Model Logic**: See `template/model_template.md` for model documentation.

* **Backtest Framework**: See `template/backtest_template.md` for scoring methodology.
* **Polymarket Schema**: See `data/Polymarket/polymarket_btc_analytics_schema.md`.

---
*Developed for the Bitcoin Analytics Capstone.*
