# Crypto Algorithmic Trading with Machine Learning

This project implements a complete machine learning pipeline for algorithmic trading on cryptocurrency markets. It combines data fetching, feature engineering, model training (XGBoost/LightGBM ensemble), walk-forward validation, and a sophisticated backtesting engine with risk management features. The system is designed to predict future returns and construct a diversified portfolio with volatility targeting, stop-loss, and market trend filters.

## Features

- **Data Acquisition**: Fetches OHLCV data from Binance (or any CCXT exchange) for a configurable list of symbols.
- **Feature Engineering**: Computes a wide range of technical indicators (RSI, MACD, Bollinger Bands, ATR, Garman-Klass volatility, volume ratio, ADX, OBV, Stochastic), rolling returns and volatilities, correlations with a benchmark (e.g., BTC), and macro features (benchmark returns, volatility, SMA distances).
- **Liquidity Filter**: Ranks assets by dollar volume and keeps only the top N most liquid.
- **Machine Learning Models**: Supports XGBoost, LightGBM, or an ensemble of both. Models are trained in a walk‑forward manner (retrained periodically) to predict future returns (regression) or direction (classification). Feature selection and early stopping are available.
- **Portfolio Construction**: After obtaining predicted signals, the system selects the top assets and computes target weights using one of several methods:
  - Equal weighting
  - Signal‑weighted (proportional to predicted return)
  - Volatility targeting (scales weights by inverse volatility to achieve a target portfolio volatility)
  - Mean‑variance optimisation (via `PyPortfolioOpt`)
- **Risk Management**:
  - **Stop‑Loss**: Automatically sells a position if it drops more than a configurable percentage from its entry price.
  - **Volatility Scaling**: Dynamically reduces capital exposure when the market (benchmark) volatility exceeds a threshold.
  - **Trend Filter**: Cuts capital exposure when the benchmark price is below a long‑term simple moving average (SMA).
- **Backtesting Engine**: Simulates trading with daily NAV tracking, transaction costs, and rebalancing at a user‑defined frequency.
- **Performance Metrics**: Calculates total return, annualised return, volatility, Sharpe/Sortino ratios, max drawdown, Calmar ratio, alpha, beta, and more.
- **Live Prediction**: Uses a saved model to generate trading signals on recent data and outputs recommended portfolio weights.

## Prerequisites & Installation

### System Requirements
- Python
- pip package manager

### Installation

1. Clone the repository (or extract the provided code) into a directory.
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`, `lightgbm`
- `ccxt` (for data fetching)
- `pandas_ta` (technical indicators)
- `PyPortfolioOpt` (portfolio optimisation)
- `pyyaml`, `pyarrow` (for caching and configuration)

## Configuration

All settings are managed via YAML configuration files. The main file is `config.yaml`. A live‑trading variant (`live_config.yaml`) is also provided and should be kept consistent with the training configuration to avoid feature mismatches.

Below is an explanation of each section.

### `data`
- `symbols`: List of trading pairs (e.g., `BTC/USDT`, `ETH/USDT`).
- `start_date`, `end_date`: Date range for data fetching (format `YYYY-MM-DD`).
- `exchange`: Exchange name as per CCXT (e.g., `binance`).
- `timeframe`: Candle interval (`1d` for daily, `1h` for hourly, etc.).

### `features`
- `technical_indicators`: List of indicators to compute (RSI, MACD, BBANDS, ATR, Garman‑Klass vol, volume ratio, price position, ADX, OBV, stochastic).
- `rolling_returns`: List of lag periods for past returns (e.g., `[1,5,21,63]` days).
- `rolling_volatility`: List of windows for rolling volatility (e.g., `[21]`).
- `rolling_correlation`: List of windows for correlation with the benchmark.
- `macro_indicators`:
  - `benchmark`: Symbol used as market benchmark (e.g., `BTC/USDT`).
  - `benchmark_returns`, `benchmark_volatility`, `benchmark_sma`: Lags/windows for macro features.
- `dollar_volume_rank`: Whether to filter by liquidity.
- `top_n_liquid`: Number of most liquid assets to keep.

### `model`
- `type`: `xgboost`, `lightgbm`, or `ensemble`.
- `target_horizon`: Number of days ahead to predict (e.g., `5` for 5‑day forward return).
- `classification`: `false` for regression (predicts return), `true` for binary classification (up/down).
- `train_window`: Number of days of historical data used for each training (e.g., `730` ~2 years).
- `retrain_frequency`: How often to retrain the model (e.g., `21` days).
- `confidence_threshold`: (classification only) discard predictions below this probability.
- `use_feature_selection`: Apply median‑threshold feature selection after an initial model.
- `early_stopping_rounds`: Stop training if validation performance does not improve.
- `hyperparameters`: Nested dictionaries for XGBoost and LightGBM (see config for examples).

### `portfolio`
- `top_n`: Maximum number of assets to hold.
- `weight_method`: `equal`, `signal_weighted`, `volatility_targeting`, or `mean_variance`.
- `max_weight`, `min_weight`: Caps and floors for individual asset weights.
- `target_volatility`: Annualised volatility target (used in `volatility_targeting`).
- `stop_loss`: Fractional loss that triggers an automatic sale (e.g., `0.20` for 20%).
- `transaction_cost`: Round‑trip cost per trade (e.g., `0.0015` for 0.15%).
- `rebalance_frequency`: Pandas frequency string (e.g., `"1W"` for weekly).

### `risk`
- `volatility_scaling`: Enable dynamic capital adjustment based on market volatility.
- `benchmark_for_vol`: Symbol used to gauge market volatility.
- `max_market_volatility`: Threshold above which capital is reduced.
- `min_capital_factor`: Minimum fraction of capital to keep when scaling down.
- `trend_filter`:
  - `enabled`: Turn on/off.
  - `benchmark`: Symbol for trend check.
  - `sma_period`: Length of SMA (e.g., `300`).
  - `capital_multiplier_when_below` / `when_above`: Multipliers applied to capital when benchmark is below/above SMA.

### `backtest`
- `initial_capital`: Starting cash.
- `benchmark_symbol`: Symbol to compare against in performance reports.

### `logging` & `output`
- Logging level and file location.
- Directories for plots and results.

## Usage

### Running a Backtest

The main entry point for backtesting is `run.py`. It executes the entire pipeline:

1. Fetches data (caches to Parquet).
2. Computes features (caches to Parquet).
3. Trains models and generates signals via walk‑forward validation.
4. Runs the backtest.
5. Computes performance metrics.
6. Saves results (NAV series, metrics, model) and creates plots.

```bash
python run.py
```

You can specify a different config file:

```bash
python run.py --config my_config.yaml
```

After execution, you will find:
- `results/raw_data.parquet`, `results/featured_data.parquet` (cached data)
- `results/final_model.joblib` (saved model)
- `results/nav.csv` (daily NAV)
- `results/metrics.txt` (performance summary)
- `plots/performance.png` (equity curve, drawdown, rolling Sharpe)
- `plots/vs_benchmark.png` (strategy vs benchmark)

### Live Prediction

`live_predict.py` generates trading signals for the most recent data using a previously saved model (from `run.py`). It assumes the model exists in `results/final_model.joblib` and that the configuration file (`live_config.yaml`) is set up with the same features.

Steps:
1. Ensure `live_config.yaml` has the correct `end_date` (today) and any other adjustments.
2. Run:

```bash
python live_predict.py
```

The script will:
- Load the saved model.
- Fetch fresh data from the exchange.
- Compute features (dropping NaNs on the latest date).
- Predict signals for all symbols.
- Select top assets and compute target weights using the portfolio settings.
- Save the recommendations to `results/signals_YYYY-MM-DD.csv`.

You can also specify a custom config:

```bash
python live_predict.py --config my_live_config.yaml
```

## Outputs and Results

- **NAV Series**: The daily net asset value of the portfolio, accounting for trades, costs, and risk adjustments.
- **Metrics**: Standard performance measures including total return, annualised return, volatility, Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio, and (if benchmark provided) alpha and beta.
- **Plots**: Visualisation of equity curve, drawdown, rolling Sharpe, and strategy vs benchmark.
- **Trade Log**: (not saved by default but available in memory) Contains details of each rebalance.

## Customization

- **Add new features**: Extend `features.py` – add new technical indicators or macro variables. Ensure they are included in the feature list and that the model can handle them.
- **Modify portfolio logic**: Edit `portfolio.py` to implement new weighting methods or risk rules.
- **Change backtest behaviour**: Adjust `backtest.py` – for example, alter how stop‑loss is applied or add additional filters.
- **Integrate other exchanges**: CCXT supports many exchanges; simply change the `exchange` parameter in the config.
- **Use different timeframes**: The code works with any timeframe supported by the exchange, but ensure the rolling windows and target horizon make sense for that frequency.

## License

This project is provided for educational and research purposes only. Use it at your own risk. No warranty or guarantee of profitability is implied.

---

**Disclaimer**: Trading cryptocurrencies involves risk. This project is not financial advice. This project does not provide any investment advice. The providers are not responsible for any loss caused or potentially caused. Always backtest thoroughly and consult a professional before deploying real capital.
