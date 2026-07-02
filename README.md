# Tiered Cinnamon Forecast

A time-series forecasting system utilizing a tiered modeling approach to optimize forecast accuracy for items with distinct demand patterns.

## Tiered Forecasting Approach

This project segments product demand time series into two distinct tiers: high-volume items and sparse (intermittent) items. High-volume items, characterized by continuous and sufficient demand signals, are modeled using high-performance tree ensembles (LightGBM and XGBoost) to capture complex non-linear interactions, temporal patterns, and cross-features. In contrast, sparse items, characterized by intermittent and zero-heavy demand, are modeled using traditional statistical approaches designed for intermittency (such as Croston or TSB methods). This tiering strategy ensures that each product group receives the most appropriate modeling methodology, maximizing overall forecasting performance and reducing inventory planning errors.

## How to Run

1. **Prerequisites**
   - Python 3.11

2. **Create and Activate Virtual Environment**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
