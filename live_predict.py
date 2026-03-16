#!/usr/bin/env python3
"""
Live prediction using a saved model.
Fetches latest data, computes features, and outputs predicted returns.
Handles NaN features by forward-filling per symbol, then filling any remaining
NaNs on the latest date with the column median.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.data_fetcher import DataFetcher
from src.features import FeatureEngineer
from src.model import TradingModel
from src.portfolio import PortfolioBuilder
from src.utils import setup_logging

def main(config_path='live_config.yaml'):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    log_file = config.get('logging', {}).get('file', 'logs/live_predict.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging({'logging': config.get('logging', {'level': 'INFO', 'file': log_file})})
    logger = logging.getLogger(__name__)

    # Determine end_date: if null/None, use today
    data_cfg = config['data']
    if data_cfg.get('end_date') is None:
        data_cfg['end_date'] = datetime.today().strftime('%Y-%m-%d')
        logger.info(f"Using end_date = {data_cfg['end_date']}")

    # 1. Fetch historical data
    logger.info("Fetching data...")
    fetcher = DataFetcher(config)   # expects DataFetcher to accept dict
    raw_data = fetcher.fetch_all()
    if raw_data.empty:
        logger.error("No data fetched. Exiting.")
        return

    # 2. Feature engineering – keep NaNs for now (dropna=False)
    logger.info("Computing features...")
    engineer = FeatureEngineer(config)
    featured_data = engineer.compute_features(raw_data, dropna=False)
    if featured_data.empty:
        logger.error("No data after feature engineering. Check your date range and symbols.")
        return

    # 3. Forward-fill missing values per symbol to carry forward the last valid value.
    logger.info("Forward-filling NaNs per symbol...")
    featured_data = featured_data.groupby('symbol').ffill()

    # 4. Extract the latest date's data
    latest_date = featured_data.index.get_level_values('timestamp').max()
    logger.info(f"Latest date with features: {latest_date}")

    latest_df = featured_data.xs(latest_date, level='timestamp')  # index = symbols

    # 5. Prepare feature matrix for the latest date
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol', 'target']
    feature_cols = [c for c in latest_df.columns if c not in exclude_cols]
    X_live = latest_df[feature_cols]

    # 6. Handle any remaining NaNs on the latest date
    nan_counts = X_live.isnull().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"NaNs detected in features on latest date:\n{nan_counts[nan_counts > 0]}")
        # Fill NaNs with the median of that feature across all symbols
        for col in X_live.columns:
            if X_live[col].isnull().any():
                median_val = X_live[col].median()
                if pd.isna(median_val):   # entire column is NaN
                    median_val = 0.0       # fallback to zero (neutral value)
                X_live[col].fillna(median_val, inplace=True)
        logger.info("Filled remaining NaNs with column median (or zero).")

    # Final check
    if X_live.isnull().any().any():
        logger.error("Still have NaNs after filling. Cannot proceed.")
        return

    logger.info(f"Symbols retained: {len(X_live)}")

    # 7. Load model
    logger.info("Loading model...")
    model = TradingModel(config)  # dummy config, will be overwritten by load
    model.load_model(config['model_path'])

    # If feature selection was used, subset columns
    if model.selected_features is not None:
        missing = set(model.selected_features) - set(X_live.columns)
        if missing:
            logger.error(f"Missing features in live data: {missing}")
            return
        X_live = X_live[model.selected_features]

    # 8. Predict
    logger.info("Generating predictions...")
    predictions = model.predict(X_live)  # returns numpy array
    pred_series = pd.Series(predictions, index=X_live.index, name='pred_return')

    # Filter positive predictions (optional)
    pos = pred_series[pred_series > 0].sort_values(ascending=False)
    logger.info(f"Symbols with positive predicted return: {len(pos)}")

    # 9. Save predictions
    out_path = config['output'].get('predictions_file', 'live_predictions.csv')
    pos.to_csv(out_path, header=True)
    logger.info(f"Predictions saved to {out_path}")

    # 10. Optional: compute portfolio weights
    if 'portfolio' in config['output']:
        port_cfg = config['output']['portfolio']
        prices = featured_data['close'].unstack('symbol')
        prices = prices.ffill()
        # Use the latest available prices
        latest_prices = prices.loc[latest_date]

        signals = pred_series
        builder = PortfolioBuilder({'portfolio': port_cfg})

        selected = builder.select_top_assets(signals, latest_date)
        if not selected:
            logger.warning("No assets selected for portfolio.")
        else:
            weights = builder.compute_weights(prices, selected, latest_date, signals=signals)
            if weights:
                weights_df = pd.Series(weights).sort_values(ascending=False)
                weights_path = out_path.replace('.csv', '_weights.csv')
                weights_df.to_csv(weights_path, header=True)
                logger.info(f"Portfolio weights saved to {weights_path}")

    logger.info("Live prediction completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live prediction using trained model')
    parser.add_argument('--config', type=str, default='live_config.yaml',
                        help='Path to live configuration YAML')
    args = parser.parse_args()
    main(args.config)