# Helper functions and metrics

import numpy as np
import pandas as pd
import logging
import os

def setup_logging(config):
    log_file = config['logging']['file']
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'
    )
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, config['logging']['level']))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

class GarmanKlassVolatility:
    @staticmethod
    def calculate(high, low, open_, close):
        """Garman-Klass volatility estimator (daily)."""
        log_hl = (np.log(high) - np.log(low))**2
        log_co = (np.log(close) - np.log(open_))**2
        return 0.5 * log_hl - (2*np.log(2)-1) * log_co

def compute_metrics(nav_df, benchmark_prices=None, rf=0.0):
    """Compute standard performance metrics."""
    returns = nav_df['returns'].dropna()
    total_return = (nav_df['nav'].iloc[-1] / nav_df['nav'].iloc[0]) - 1
    ann_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = (ann_return - rf) / volatility if volatility != 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (ann_return - rf) / downside if downside != 0 else 0
    cum_nav = nav_df['nav']
    running_max = cum_nav.cummax()
    drawdown = (cum_nav - running_max) / running_max
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    metrics = {
        'Total Return': f'{total_return:.2%}',
        'Annualized Return': f'{ann_return:.2%}',
        'Volatility': f'{volatility:.2%}',
        'Sharpe Ratio': f'{sharpe:.3f}',
        'Sortino Ratio': f'{sortino:.3f}',
        'Max Drawdown': f'{max_dd:.2%}',
        'Calmar Ratio': f'{calmar:.3f}',
    }

    if benchmark_prices is not None:
        bench_ret = benchmark_prices.pct_change().dropna()
        common = returns.index.intersection(bench_ret.index)
        if len(common) > 0:
            strat = returns.loc[common]
            bench = bench_ret.loc[common]
            cov = np.cov(strat, bench)[0,1]
            var_bench = np.var(bench)
            beta = cov / var_bench if var_bench != 0 else np.nan
            alpha = (ann_return - rf) - beta * (bench.mean()*252 - rf)
            metrics['Beta'] = f'{beta:.3f}'
            metrics['Alpha (ann)'] = f'{alpha:.3f}'

    return metrics