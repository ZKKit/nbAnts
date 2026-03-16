# Crypto Algorithmic Trading with Machine Learning

This project provides a complete, configurable framework for developing and backtesting a cryptocurrency trading strategy using machine learning (XGBoost/LightGBM). It focuses on maximizing risk-adjusted returns (Sharpe, Sortino, Calmar ratios) through feature engineering, model-based signal generation, and portfolio optimization.

## Features

- **Modular design**: Separate modules for data fetching, feature engineering, modeling, portfolio construction, and backtesting.
- **Configurable**: All parameters are in `config.yaml` – no hardcoding.
- **Technical indicators**: RSI, MACD, Bollinger Bands, ATR, Garman-Klass volatility via `pandas_ta`.
- **Machine learning**: XGBoost or LightGBM for return prediction (classification or regression).
- **Walk‑forward validation**: Model retrained periodically to avoid look‑ahead bias.
- **Portfolio optimization**: Equal‑weight or mean‑variance (max Sharpe) with weight constraints.
- **Risk management**: Volatility targeting, transaction costs.
- **Performance metrics**: Sharpe, Sortino, Calmar, max drawdown, rolling statistics.
- **Visualization**: Equity curve, drawdown, rolling Sharpe, benchmark comparison.

## Requirements

- Python 3.8+
- Libraries listed in `requirements.txt`

## Installation

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt