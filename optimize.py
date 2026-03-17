#!/usr/bin/env python3
"""
optimize.py – Hyperparameter optimization for crypto trading config.
Uses Optuna to find numeric parameters that maximize the Sharpe ratio.
Now includes target_horizon, early_stopping_rounds, retrain_frequency, and rebalance_frequency.
"""

import os
import sys
import yaml
import shutil
import tempfile
import logging
import argparse
from copy import deepcopy

import optuna
from optuna.samplers import TPESampler
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the main pipeline functions
from src.data_fetcher import DataFetcher
from src.features import FeatureEngineer
from src.model import TradingModel
from src.backtest import Backtester
from src.utils import setup_logging, compute_metrics

def patch_config(trial, base_config):
    """
    Modify a copy of the base configuration with parameters suggested by the trial.
    Returns a new config dictionary.
    """
    config = deepcopy(base_config)

    # --- Model-level parameters ---
    model_top = config['model']
    # target_horizon: integer days, reasonable range 1 to 10 (or maybe 1-20)
    model_top['target_horizon'] = trial.suggest_int('target_horizon', 1, 10)
    # early_stopping_rounds: integer, 0 means None (disable)
    esr = trial.suggest_int('early_stopping_rounds', 0, 50)
    model_top['early_stopping_rounds'] = esr if esr > 0 else None
    # retrain_frequency: integer days, between 5 and 60
    model_top['retrain_frequency'] = trial.suggest_int('retrain_frequency', 5, 60)

    # --- Model hyperparameters (XGBoost & LightGBM) ---
    model_cfg = config['model']['hyperparameters']

    # XGBoost
    if 'xgboost' in model_cfg:
        xgb = model_cfg['xgboost']
        xgb['learning_rate'] = trial.suggest_float('xgb_learning_rate', 0.005, 0.2, log=True)
        xgb['max_depth'] = trial.suggest_int('xgb_max_depth', 3, 10)
        xgb['reg_alpha'] = trial.suggest_float('xgb_reg_alpha', 1e-5, 1.0, log=True)
        xgb['reg_lambda'] = trial.suggest_float('xgb_reg_lambda', 1e-5, 1.0, log=True)
        xgb['subsample'] = trial.suggest_float('xgb_subsample', 0.5, 1.0)
        xgb['colsample_bytree'] = trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0)
        xgb['n_estimators'] = trial.suggest_int('xgb_n_estimators', 500, 3000, step=100)

    # LightGBM
    if 'lightgbm' in model_cfg:
        lgb = model_cfg['lightgbm']
        lgb['learning_rate'] = trial.suggest_float('lgb_learning_rate', 0.005, 0.2, log=True)
        lgb['max_depth'] = trial.suggest_int('lgb_max_depth', 3, 10)
        lgb['reg_alpha'] = trial.suggest_float('lgb_reg_alpha', 1e-5, 1.0, log=True)
        lgb['reg_lambda'] = trial.suggest_float('lgb_reg_lambda', 1e-5, 1.0, log=True)
        lgb['subsample'] = trial.suggest_float('lgb_subsample', 0.5, 1.0)
        lgb['feature_fraction'] = trial.suggest_float('lgb_feature_fraction', 0.5, 1.0)
        lgb['n_estimators'] = trial.suggest_int('lgb_n_estimators', 500, 3000, step=100)

    # --- Portfolio parameters ---
    port_cfg = config['portfolio']
    port_cfg['max_weight'] = trial.suggest_float('max_weight', 0.1, 0.5)
    port_cfg['min_weight'] = trial.suggest_float('min_weight', 0.001, 0.05)
    port_cfg['target_volatility'] = trial.suggest_float('target_volatility', 0.10, 0.40)
    port_cfg['stop_loss'] = trial.suggest_float('stop_loss', 0.05, 0.30)
    port_cfg['top_n'] = trial.suggest_int('top_n', 5, 15)
    # rebalance_frequency: categorical choices (pandas frequency strings)
    freq_choices = ['2H', '4H', '8H', '12H', '1D', '2D', '4D', '1W']
    port_cfg['rebalance_frequency'] = trial.suggest_categorical('rebalance_frequency', freq_choices)

    # --- Risk parameters ---
    if 'risk' in config:
        risk_cfg = config['risk']
        if risk_cfg.get('volatility_scaling'):
            risk_cfg['max_market_volatility'] = trial.suggest_float('max_market_volatility', 0.5, 1.0)
            risk_cfg['min_capital_factor'] = trial.suggest_float('min_capital_factor', 0.05, 0.5)

        if risk_cfg.get('trend_filter', {}).get('enabled'):
            trend = risk_cfg['trend_filter']
            trend['sma_period'] = trial.suggest_int('sma_period', 10, 200, step=20)
            trend['capital_multiplier_when_below'] = trial.suggest_float('multiplier_below', 0.0, 0.5)

    return config

def objective(trial, base_config, raw_data, featured_data):
    """
    Optuna objective: run backtest with patched config and return Sharpe ratio.
    """
    # Patch config
    config = patch_config(trial, base_config)

    # Create a temporary directory for this trial's outputs
    temp_dir = tempfile.mkdtemp(prefix=f"trial_{trial.number}_")
    plots_dir = os.path.join(temp_dir, 'plots')
    results_dir = os.path.join(temp_dir, 'results')
    log_dir = os.path.join(temp_dir, 'logs')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Update config paths
    config['output']['plots_dir'] = plots_dir
    config['output']['results_dir'] = results_dir
    config['logging']['file'] = os.path.join(log_dir, 'trading.log')

    # Setup logging for this trial (to file only)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=config['logging']['file'],
        filemode='w'
    )
    logger = logging.getLogger(__name__)

    try:
        # --- Prepare prices ---
        prices = featured_data['close'].unstack('symbol')
        prices = prices.ffill()
        if prices.empty:
            raise ValueError("Prices DataFrame empty")

        # --- Model predictions (walk-forward) ---
        logger.info("Training model and generating signals...")
        model = TradingModel(config)
        dates = featured_data.index.get_level_values(0).unique().sort_values()
        signals = model.walk_forward_predict(featured_data, dates)
        signals.name = 'signal'

        # --- Backtest ---
        logger.info("Running backtest...")
        backtester = Backtester(config, featured_data, signals, prices)
        nav_df, trade_log = backtester.run()

        # --- Performance metrics ---
        logger.info("Computing performance metrics...")
        benchmark_prices = prices[config['backtest']['benchmark_symbol']].dropna()
        metrics = compute_metrics(nav_df, benchmark_prices)

        # Extract Sharpe ratio
        sharpe_str = metrics.get('Sharpe Ratio', '0.0').replace('%', '')
        sharpe = float(sharpe_str)

        return sharpe

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        return -1.0
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description='Optimize config for max Sharpe')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Base configuration file')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of optimization trials')
    parser.add_argument('--study-name', type=str, default='crypto_optimization',
                        help='Name for Optuna study (for persistence)')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna.db',
                        help='Database URL for study persistence')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching of raw/featured data (recompute each trial)')
    args = parser.parse_args()

    # Load base config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)

    # Prepare data once (unless no-cache)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, base_config['output']['results_dir'])
    os.makedirs(results_dir, exist_ok=True)

    # Fetch raw data if not cached
    raw_cache = os.path.join(results_dir, 'raw_data.parquet')
    if os.path.exists(raw_cache) and not args.no_cache:
        print("Loading cached raw data...")
        import pandas as pd
        raw_data = pd.read_parquet(raw_cache)
    else:
        print("Fetching raw data...")
        from src.data_fetcher import DataFetcher
        fetcher = DataFetcher(args.config)
        raw_data = fetcher.fetch_all()
        raw_data.to_parquet(raw_cache)

    # Compute features if not cached
    feat_cache = os.path.join(results_dir, 'featured_data.parquet')
    if os.path.exists(feat_cache) and not args.no_cache:
        print("Loading cached featured data...")
        import pandas as pd
        featured_data = pd.read_parquet(feat_cache)
    else:
        print("Computing features...")
        from src.features import FeatureEngineer
        engineer = FeatureEngineer(base_config)
        featured_data = engineer.compute_features(raw_data)
        featured_data.to_parquet(feat_cache)

    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction='maximize',
        sampler=TPESampler(seed=42)
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config, raw_data, featured_data),
        n_trials=args.trials,
        show_progress_bar=True
    )

    # Print best results
    print("\nBest trial:")
    print(f"  Value (Sharpe): {study.best_value}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Generate the best config and save it
    best_config = patch_config(optuna.trial.FixedTrial(study.best_params), base_config)
    best_config_path = os.path.join(results_dir, 'best_config.yaml')
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    print(f"Best configuration saved to {best_config_path}")

if __name__ == '__main__':
    main()