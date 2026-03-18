#!/usr/bin/env python3
"""
optimize.py – Multi‑objective hyperparameter optimization for crypto trading config.
Uses NSGA‑II (genetic algorithm) to maximize Sortino, Sharpe, and Calmar ratios simultaneously.
Saves all Pareto‑optimal configurations and prints up to three of them.
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
from optuna.samplers import NSGAIISampler
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.data_fetcher import DataFetcher
from src.features import FeatureEngineer
from src.model import TradingModel
from src.backtest import Backtester
from src.utils import setup_logging, compute_metrics

def patch_config(trial, base_config):
    """Modify a copy of the base configuration with parameters suggested by the trial."""
    config = deepcopy(base_config)

    # --- Model-level parameters ---
    model_top = config['model']
    model_top['target_horizon'] = trial.suggest_int('target_horizon', 1, 15)
    esr = trial.suggest_int('early_stopping_rounds', 0, 50)
    model_top['early_stopping_rounds'] = esr if esr > 0 else None
    model_top['retrain_frequency'] = trial.suggest_int('retrain_frequency', 1, 60)

    # --- Model hyperparameters (XGBoost & LightGBM) ---
    model_cfg = config['model']['hyperparameters']

    if 'xgboost' in model_cfg:
        xgb = model_cfg['xgboost']
        xgb['learning_rate'] = trial.suggest_float('xgb_learning_rate', 0.001, 0.3, log=True)
        xgb['max_depth'] = trial.suggest_int('xgb_max_depth', 3, 15)
        xgb['reg_alpha'] = trial.suggest_float('xgb_reg_alpha', 1e-5, 1.0, log=True)
        xgb['reg_lambda'] = trial.suggest_float('xgb_reg_lambda', 1e-5, 1.0, log=True)
        xgb['subsample'] = trial.suggest_float('xgb_subsample', 0.1, 1.0, step=0.001)
        xgb['colsample_bytree'] = trial.suggest_float('xgb_colsample_bytree', 0.1, 1.0, step=0.001)
        xgb['n_estimators'] = trial.suggest_int('xgb_n_estimators', 500, 3000, step=100)

    if 'lightgbm' in model_cfg:
        lgb = model_cfg['lightgbm']
        lgb['learning_rate'] = trial.suggest_float('lgb_learning_rate', 0.001, 0.3, log=True)
        lgb['max_depth'] = trial.suggest_int('lgb_max_depth', 3, 15)
        lgb['reg_alpha'] = trial.suggest_float('lgb_reg_alpha', 1e-5, 1.0, log=True)
        lgb['reg_lambda'] = trial.suggest_float('lgb_reg_lambda', 1e-5, 1.0, log=True)
        lgb['subsample'] = trial.suggest_float('lgb_subsample', 0.1, 1.0, step=0.001)
        lgb['feature_fraction'] = trial.suggest_float('lgb_feature_fraction', 0.1, 1.0)
        lgb['n_estimators'] = trial.suggest_int('lgb_n_estimators', 500, 3000, step=100)

    # --- Portfolio parameters ---
    port_cfg = config['portfolio']
    port_cfg['max_weight'] = trial.suggest_float('max_weight', 0.2, 0.5, step=0.0001)
    port_cfg['min_weight'] = trial.suggest_float('min_weight', 0.001, 0.2, step=0.0001)
    port_cfg['target_volatility'] = trial.suggest_float('target_volatility', 0.01, 0.67, step=0.0001)
    port_cfg['stop_loss'] = trial.suggest_float('stop_loss', 0.05, 0.50, step=0.0001)
    port_cfg['top_n'] = trial.suggest_int('top_n', 5, 20)
    freq_choices = ['2H', '3H', '4H', '5H', '6H', '8H', '10H', '12H', '1D']
    port_cfg['rebalance_frequency'] = trial.suggest_categorical('rebalance_frequency', freq_choices)

    # --- Risk parameters ---
    if 'risk' in config:
        risk_cfg = config['risk']
        if risk_cfg.get('volatility_scaling'):
            risk_cfg['max_market_volatility'] = trial.suggest_float('max_market_volatility', 0.3, 1.0, step=0.0001)
            risk_cfg['min_capital_factor'] = trial.suggest_float('min_capital_factor', 0.05, 0.5, step=0.0001)

        if risk_cfg.get('trend_filter', {}).get('enabled'):
            trend = risk_cfg['trend_filter']
            trend['sma_period'] = trial.suggest_int('sma_period', 10, 200, step=1)
            trend['capital_multiplier_when_below'] = trial.suggest_float('multiplier_below', 0.0, 0.67, step=0.0001)

    return config

def objective(trial, base_config, raw_data, featured_data):
    """
    Optuna objective: run backtest with patched config and return (Sortino, Sharpe, Calmar).
    All logging is directed to a per‑trial file to keep the console clean.
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

    # --- Set up root logger to write only to the trial log file ---
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(config['logging']['file'], mode='w')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(file_handler)
    root_logger.setLevel(getattr(logging, config['logging']['level']))

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

        # Extract individual metrics (remove '%' signs, convert to float)
        sortino = float(metrics.get('Sortino Ratio', '0.0').replace('%', ''))
        sharpe  = float(metrics.get('Sharpe Ratio', '0.0').replace('%', ''))
        calmar  = float(metrics.get('Calmar Ratio', '0.0').replace('%', ''))

        # Return a tuple of objectives for multi‑objective optimization
        return sortino, sharpe, calmar

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        # Return very poor values so the trial is dominated
        return -1.0, -1.0, -1.0
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description='Multi‑objective optimization of config (Sortino, Sharpe, Calmar)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Base configuration file')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of optimization trials')
    parser.add_argument('--study-name', type=str, default='crypto_multiobj',
                        help='Name for Optuna study (for persistence)')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_multi.db',
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
        raw_data = pd.read_parquet(raw_cache)
    else:
        print("Fetching raw data...")
        fetcher = DataFetcher(args.config)
        raw_data = fetcher.fetch_all()
        raw_data.to_parquet(raw_cache)

    # Compute features if not cached
    feat_cache = os.path.join(results_dir, 'featured_data.parquet')
    if os.path.exists(feat_cache) and not args.no_cache:
        print("Loading cached featured data...")
        featured_data = pd.read_parquet(feat_cache)
    else:
        print("Computing features...")
        engineer = FeatureEngineer(base_config)
        featured_data = engineer.compute_features(raw_data)
        featured_data.to_parquet(feat_cache)

    # Create multi‑objective study with NSGA‑II sampler
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        directions=['maximize', 'maximize', 'maximize'],   # Sortino, Sharpe, Calmar
        sampler=NSGAIISampler(seed=42)
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config, raw_data, featured_data),
        n_trials=args.trials,
        show_progress_bar=True
    )

    # Get Pareto front trials
    pareto_trials = study.best_trials   # best_trials returns non-dominated trials

    # Print Pareto front summary
    print("\n--- Pareto Front (non-dominated solutions) ---")
    for i, trial in enumerate(pareto_trials):
        print(f"  Trial {trial.number}: Sortino={trial.values[0]:.4f}, Sharpe={trial.values[1]:.4f}, Calmar={trial.values[2]:.4f}")

    # Save all Pareto‑optimal configurations
    best_configs_dir = os.path.join(results_dir, 'pareto_configs')
    os.makedirs(best_configs_dir, exist_ok=True)
    for i, trial in enumerate(pareto_trials):
        config = patch_config(optuna.trial.FixedTrial(trial.params), base_config)
        config_path = os.path.join(best_configs_dir, f'config_pareto_{i}_trial{trial.number}.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"  Saved Pareto config #{i} (trial {trial.number}) to {config_path}")

    # Print up to three Pareto configurations directly to console
    print("\n" + "="*60)
    print("BEST PARETO CONFIGURATIONS (up to 3)")
    print("="*60)
    for idx, trial in enumerate(pareto_trials[:3]):
        print(f"\n--- Pareto Configuration #{idx+1} (Trial {trial.number}) ---")
        print(f"Objectives: Sortino={trial.values[0]:.4f}, Sharpe={trial.values[1]:.4f}, Calmar={trial.values[2]:.4f}")
        config = patch_config(optuna.trial.FixedTrial(trial.params), base_config)
        print(yaml.dump(config, default_flow_style=False))

    if len(pareto_trials) > 3:
        print(f"\n(Note: Only the first 3 of {len(pareto_trials)} Pareto configurations are shown above; all are saved in {best_configs_dir})")

    print(f"\nAll Pareto configurations saved in {best_configs_dir}")

if __name__ == '__main__':
    main()