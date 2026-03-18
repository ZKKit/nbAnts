# Backtesting engine – daily rebalance, stop loss, volatility targeting, market trend filter

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from src.portfolio import PortfolioBuilder

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, config, data, signals, prices):
        self.config = config
        self.data = data
        self.signals = signals
        self.prices = prices
        self.initial_capital = config['backtest']['initial_capital']
        self.rebalance_freq = config['portfolio']['rebalance_frequency']
        self.transaction_cost = config['portfolio']['transaction_cost']
        self.target_vol = config['portfolio']['target_volatility']
        self.builder = PortfolioBuilder(config)

        # Risk scaling parameters
        self.risk_config = config.get('risk', {})
        self.vol_scaling = self.risk_config.get('volatility_scaling', False)
        self.benchmark = self.risk_config.get('benchmark_for_vol', 'BTC/USDT')
        self.max_market_vol = self.risk_config.get('max_market_volatility', 0.8)
        self.min_cap_factor = self.risk_config.get('min_capital_factor', 0.2)

        # Trend filter
        self.trend_config = self.risk_config.get('trend_filter', {})
        self.trend_filter_enabled = self.trend_config.get('enabled', False)
        self.trend_benchmark = self.trend_config.get('benchmark', 'BTC/USDT')
        self.sma_period = self.trend_config.get('sma_period', 200)
        self.multiplier_below = self.trend_config.get('capital_multiplier_when_below', 0.1)
        self.multiplier_above = self.trend_config.get('capital_multiplier_when_above', 1.0)

    def run(self):
        all_dates = self.prices.index.sort_values()
        if len(all_dates) == 0:
            logger.error("No price data available")
            return pd.DataFrame(columns=['nav']), []

        rebalance_dates = pd.date_range(
            start=all_dates.min(),
            end=all_dates.max(),
            freq=self.rebalance_freq
        )
        rebalance_dates = [d for d in rebalance_dates if d in all_dates]

        cash = self.initial_capital
        holdings = {}
        entry_prices = {}
        daily_nav = []
        trade_log = []

        nav = cash
        daily_nav.append((all_dates[0], nav))

        for i, date in enumerate(all_dates[1:], start=1):
            # Stop-loss
            if holdings:
                holdings, stop_proceeds, sold = self.builder.apply_stop_loss(
                    holdings, self.prices, date, entry_prices
                )
                cash += stop_proceeds
                for sym in sold:
                    entry_prices.pop(sym, None)

            # Rebalance
            if date in rebalance_dates:
                try:
                    sigs = self.signals.loc[date]
                except KeyError:
                    nav = self._compute_nav(date, cash, holdings)
                    daily_nav.append((date, nav))
                    continue

                selected = self.builder.select_top_assets(sigs, date)
                if not selected:
                    nav = self._compute_nav(date, cash, holdings)
                    daily_nav.append((date, nav))
                    continue

                target_weights = self.builder.compute_weights(
                    self.prices, selected, date, signals=sigs
                )
                if not target_weights:
                    nav = self._compute_nav(date, cash, holdings)
                    daily_nav.append((date, nav))
                    continue

                current_value = self._compute_nav(date, cash, holdings)

                # --- Volatility scaling ---
                capital_multiplier = 1.0
                if self.vol_scaling and self.benchmark in self.prices.columns:
                    hist = self.prices[self.benchmark].loc[:date].iloc[-60:]
                    if len(hist) >= 21:
                        returns = hist.pct_change().dropna()
                        market_vol = returns.std() * np.sqrt(252)
                        if market_vol > self.max_market_vol:
                            capital_multiplier = max(self.min_cap_factor,
                                                     self.max_market_vol / market_vol)
                            logger.info(f"{date}: Market vol {market_vol:.2%} > {self.max_market_vol:.2%}, scaling capital to {capital_multiplier:.2f}×")

                # --- Trend filter ---
                if self.trend_filter_enabled and self.trend_benchmark in self.prices.columns:
                    # Compute SMA of benchmark
                    hist_bench = self.prices[self.trend_benchmark].loc[:date]
                    if len(hist_bench) >= self.sma_period:
                        sma = hist_bench.rolling(self.sma_period).mean().iloc[-1]
                        current_price = hist_bench.iloc[-1]
                        if current_price < sma:
                            capital_multiplier *= self.multiplier_below
                            logger.info(f"{date}: {self.trend_benchmark} below {self.sma_period}-day SMA, reducing multiplier to {capital_multiplier:.2f}")
                        else:
                            capital_multiplier *= self.multiplier_above
                    else:
                        capital_multiplier *= self.multiplier_above  # default if not enough data

                # Apply multiplier to target dollars
                target_dollars = {
                    sym: target_weights.get(sym, 0) * current_value * capital_multiplier
                    for sym in target_weights
                }

                # Execute trades (sell all, then buy)
                trades = []
                new_cash = cash
                for sym, shares in holdings.items():
                    if sym in self.prices.columns:
                        price = self.prices.loc[date, sym]
                        if pd.notna(price) and shares > 0:
                            proceeds = shares * price * (1 - self.transaction_cost)
                            new_cash += proceeds
                            trades.append(('SELL', sym, shares, price, proceeds))

                new_holdings = {}
                new_entry_prices = {}

                for sym, target_dollar in target_dollars.items():
                    if target_dollar <= 0:
                        continue
                    price = self.prices.loc[date, sym] if date in self.prices.index else np.nan
                    if pd.isna(price):
                        continue
                    cost_with_fee = price * (1 + self.transaction_cost)
                    shares_to_buy = target_dollar / cost_with_fee   # fractional
                    if shares_to_buy <= 0:
                        continue
                    actual_cost = shares_to_buy * price * (1 + self.transaction_cost)
                    if actual_cost > new_cash:
                        shares_to_buy = new_cash / cost_with_fee
                        if shares_to_buy <= 0:
                            continue
                        actual_cost = shares_to_buy * price * (1 + self.transaction_cost)
                    new_cash -= actual_cost
                    new_holdings[sym] = shares_to_buy
                    new_entry_prices[sym] = price
                    trades.append(('BUY', sym, shares_to_buy, price, actual_cost))

                holdings = new_holdings
                entry_prices = new_entry_prices
                cash = new_cash

                nav = self._compute_nav(date, cash, holdings)
                daily_nav.append((date, nav))
                trade_log.append((date, trades, nav))
                logger.info(f"{date}: Rebalanced, NAV={nav:.2f}, Cash={cash:.2f}, Holdings={holdings}")

            else:
                nav = self._compute_nav(date, cash, holdings)
                daily_nav.append((date, nav))

        nav_df = pd.DataFrame(daily_nav, columns=['date', 'nav']).set_index('date')
        nav_df['returns'] = nav_df['nav'].pct_change()
        return nav_df, trade_log

    def _compute_nav(self, date, cash, holdings):
        value = cash
        for sym, shares in holdings.items():
            if sym in self.prices.columns:
                price = self.prices.loc[date, sym]
                if pd.notna(price):
                    value += shares * price
        return value
