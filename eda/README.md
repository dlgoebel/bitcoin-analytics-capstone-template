# EDA Notebooks

# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data
python data/download_data.py

# 3. Open notebooks
jupyter lab

## Reading Order

| Notebook | Purpose | Time |
|---|---|---|
| `EDA_Executive.ipynb` | Key findings, narrative summary | ~5 min |
| `EDA.ipynb` | Full technical analysis and supporting work | Reference |

## Finding Specific Sections in EDA.ipynb

Use `Ctrl+F` to search for section titles:

| Topic | Search for |
|---|---|
| Stationarity | `"STATIONARITY TESTING"` |
| Granger causality | `"GRANGER CAUSALITY TESTS"` |
| Cointegration | `"COINTEGRATION TESTS"` |
| Mutual information | `"MUTUAL INFORMATION"` |
| Wavelet coherence | `"WAVELET COHERENCE"` |
| Polymarket audit | `"POLYMARKET DATA AUDIT"` |
| Cross-feature tests | `"CROSS-FEATURE ANALYSIS"` |

## Outputs

All plots saved to `plots/` — 19 figures total.
