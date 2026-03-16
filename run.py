#!/usr/bin/env python3
"""
Main backtesting script – runs full pipeline: data → features → model → backtest → plots
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))

from src.data_fetcher import DataFetcher
from src.features import FeatureEngineer
from src.model import TradingModel
from src.backtest import Backtester
from src.utils import setup_logging, compute_metrics

def main(config_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if config_path is None:
        config_path = os.path.join(script_dir, 'config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    log_file = os.path.join(script_dir, config['logging']['file'])
    plots_dir = os.path.join(script_dir, config['output']['plots_dir'])
    results_dir = os.path.join(script_dir, config['output']['results_dir'])

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    config['logging']['file'] = log_file
    config['output']['plots_dir'] = plots_dir
    config['output']['results_dir'] = results_dir

    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info(f"Using config file: {config_path}")

    # 1. Fetch data (cache if exists)
    raw_cache = os.path.join(results_dir, 'raw_data.parquet')
    if os.path.exists(raw_cache):
        logger.info("Loading cached raw data...")
        raw_data = pd.read_parquet(raw_cache)
    else:
        logger.info("Fetching data...")
        fetcher = DataFetcher(config_path)
        raw_data = fetcher.fetch_all()
        raw_data.to_parquet(raw_cache)

    # 2. Feature engineering (cache if exists)
    feat_cache = os.path.join(results_dir, 'featured_data.parquet')
    if os.path.exists(feat_cache):
        logger.info("Loading cached featured data...")
        featured_data = pd.read_parquet(feat_cache)
    else:
        logger.info("Computing features...")
        engineer = FeatureEngineer(config)
        featured_data = engineer.compute_features(raw_data)
        if featured_data.empty:
            logger.error("No data after feature engineering. Check your date range and symbols.")
            return
        featured_data.to_parquet(feat_cache)

    # 3. Prepare prices for portfolio
    prices = featured_data['close'].unstack('symbol')
    prices = prices.ffill()
    if prices.empty:
        logger.error("Prices DataFrame is empty. Cannot run backtest.")
        return

    # 4. Model predictions (walk-forward)
    logger.info("Training model and generating signals...")
    model = TradingModel(config)
    dates = featured_data.index.get_level_values(0).unique().sort_values()
    signals = model.walk_forward_predict(featured_data, dates)
    signals.name = 'signal'

    model_save_path = os.path.join(results_dir, 'final_model.joblib')
    model.save_model(model_save_path)

    # 5. Backtest
    logger.info("Running backtest...")
    backtester = Backtester(config, featured_data, signals, prices)
    nav_df, trade_log = backtester.run()

    # 6. Performance metrics
    logger.info("Computing performance metrics...")
    benchmark_prices = prices[config['backtest']['benchmark_symbol']].dropna()
    metrics = compute_metrics(nav_df, benchmark_prices)

    total_trades = 0
    rebalance_days = 0
    for date, trades, nav in trade_log:
        if trades:
            total_trades += len(trades)
            rebalance_days += 1
    metrics['Total Trades'] = f'{total_trades}'
    metrics['Avg Trades per Rebalance'] = f'{total_trades/rebalance_days:.1f}' if rebalance_days>0 else '0'

    logger.info("Performance Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    # 7. Save results
    nav_df.to_csv(os.path.join(results_dir, 'nav.csv'))
    with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # 8. Plots
    logger.info("Generating plots...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(nav_df.index, nav_df['nav'], label='Strategy')
    axes[0].set_title('Portfolio Equity Curve')
    axes[0].set_ylabel('NAV')
    axes[0].legend()

    cum_nav = nav_df['nav']
    running_max = cum_nav.cummax()
    drawdown = (cum_nav - running_max) / running_max
    axes[1].fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
    axes[1].set_title('Drawdown')
    axes[1].set_ylabel('Drawdown %')

    rolling_sharpe = nav_df['returns'].rolling(126).mean() / nav_df['returns'].rolling(126).std() * np.sqrt(252)
    axes[2].plot(rolling_sharpe.index, rolling_sharpe, label='Rolling Sharpe (6m)')
    axes[2].axhline(0, color='black', linestyle='--')
    axes[2].set_title('Rolling Sharpe Ratio')
    axes[2].set_ylabel('Sharpe')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'performance.png'))
    plt.close()
    logger.info("performance.png created")

    # Benchmark comparison
    fig, ax = plt.subplots(figsize=(12,6))
    norm_strat = nav_df['nav'] / nav_df['nav'].iloc[0]
    bench = benchmark_prices.loc[nav_df.index[0]:nav_df.index[-1]]
    norm_bench = bench / bench.iloc[0]
    ax.plot(norm_strat.index, norm_strat, label='Strategy')
    ax.plot(norm_bench.index, norm_bench, label=config['backtest']['benchmark_symbol'])
    ax.set_title('Strategy vs Benchmark')
    ax.set_ylabel('Normalized Value')
    ax.legend()
    plt.savefig(os.path.join(plots_dir, 'vs_benchmark.png'))
    plt.close()
    logger.info("vs_benchmark.png created")

    logger.info("Done.")

if __name__ == '__main__':
    main()