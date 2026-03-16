#!/usr/bin/env python3
"""
live_predict.py – Generate live trading signals using a saved model.
Assumes you have run `run.py` at least once to produce a saved model.
"""

import os
import sys
import yaml
import pandas as pd
import logging
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from src.data_fetcher import DataFetcher
from src.features import FeatureEngineer
from src.model import TradingModel
from src.portfolio import PortfolioBuilder
from src.utils import setup_logging

def main(config_path='live_config.yaml'):
    if not os.path.exists(config_path):
        print(f"ERROR: {config_path} not found. Please create it from config.yaml")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    setup_logging(config)
    logger = logging.getLogger(__name__)

    # Determine results directory (fallback to 'results' if not in config)
    results_dir = config.get('output', {}).get('results_dir', 'results')
    model_path = os.path.join(results_dir, 'final_model.joblib')
    if not os.path.exists(model_path):
        logger.error(f"Saved model not found at {model_path}. Run the backtest first.")
        return

    model = TradingModel(config)
    model.load_model(model_path)
    logger.info("Model loaded successfully.")

    # Fetch recent data
    logger.info("Fetching recent market data...")
    fetcher = DataFetcher(config_path)
    raw_data = fetcher.fetch_all()
    if raw_data.empty:
        logger.error("No data fetched.")
        return

    logger.info(f"Raw data: {len(raw_data)} rows, from {raw_data.index.get_level_values(0).min()} to {raw_data.index.get_level_values(0).max()}")

    # Compute features (do not drop NaNs yet – we'll inspect later)
    logger.info("Computing features...")
    engineer = FeatureEngineer(config)
    featured_data = engineer.compute_features(raw_data, dropna=False)  # Keep NaNs for now
    if featured_data.empty:
        logger.error("Feature computation produced no data.")
        return

    # Check NaNs on the latest date
    latest_date = featured_data.index.get_level_values(0).max()
    latest_df = featured_data.loc[latest_date]
    nan_cols = latest_df.columns[latest_df.isnull().any()].tolist()
    if nan_cols:
        logger.warning(f"NaNs present in columns for latest date: {nan_cols}")
        # Optionally drop rows with NaNs on the latest date
        # For prediction, we need to drop them because model can't handle NaNs.
        # We'll drop rows where any NaN is present.
        featured_data = featured_data.dropna()
        logger.info(f"Dropped rows with NaNs, remaining rows: {len(featured_data)}")
        if featured_data.empty:
            logger.error("No data left after dropping NaNs on latest date.")
            return
        # Recompute latest date after dropping
        latest_date = featured_data.index.get_level_values(0).max()
        logger.info(f"New latest date after dropping NaNs: {latest_date}")

    # Get data for the latest date
    pred_data = featured_data.loc[latest_date:latest_date].copy()
    if pred_data.empty:
        logger.error("No data for latest date.")
        return

    # Prepare features for prediction
    X_pred = model.prepare_features(pred_data)
    if X_pred.empty:
        logger.error("Feature preparation failed.")
        return

    # Predict
    pred = model.predict(X_pred)
    signals = pd.Series(pred, index=pred_data.index, name='signal')
    # Convert to simple symbol index (drop timestamp level)
    signals = signals.droplevel(0)

    # Output top signals
    top_signals = signals.sort_values(ascending=False).head(10)
    logger.info(f"Top 10 signals for {latest_date.date()}:\n{top_signals}")

    # Build portfolio weights for execution (optional)
    if 'portfolio' not in config:
        logger.error("Portfolio configuration missing. Cannot compute weights.")
        return

    prices = featured_data['close'].unstack('symbol').ffill()
    latest_prices = prices.loc[latest_date]

    builder = PortfolioBuilder(config)
    selected = builder.select_top_assets(signals, latest_date)
    if selected:
        weights = builder.compute_weights(prices, selected, latest_date, signals=signals)
        logger.info(f"Recommended weights: {weights}")
        # Save signals and weights to a CSV
        output_df = pd.DataFrame({
            'symbol': list(weights.keys()),
            'weight': list(weights.values()),
            'signal': [signals.loc[sym] for sym in weights.keys()]
        })
        output_path = os.path.join(results_dir, f'signals_{latest_date.date()}.csv')
        output_df.to_csv(output_path, index=False)
        logger.info(f"Signals saved to {output_path}")
    else:
        logger.info("No assets selected – no trade recommendation.")

    logger.info("Live prediction completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='live_config.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)