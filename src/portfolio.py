# Portfolio construction – signal weighting with volatility targeting (improved)

import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models, EfficientFrontier
import logging

logger = logging.getLogger(__name__)

class PortfolioBuilder:
    def __init__(self, config):
        self.config = config
        self.top_n = config['portfolio']['top_n']
        self.weight_method = config['portfolio']['weight_method']
        self.max_weight = config['portfolio']['max_weight']
        self.min_weight = config['portfolio']['min_weight']
        self.target_vol = config['portfolio']['target_volatility']
        self.stop_loss = config['portfolio'].get('stop_loss', None)
        self.transaction_cost = config['portfolio']['transaction_cost']

    def select_top_assets(self, signals, date):
        """Select top N assets by signal strength (predicted return)."""
        signals = signals.dropna()
        if len(signals) == 0:
            logger.warning(f"No signals on {date}")
            return []
        # For regression, signal is predicted return; for classification it's probability
        signals = signals[signals > 0]   # only positive predictions
        if len(signals) == 0:
            return []
        if len(signals) < self.top_n:
            logger.info(f"Only {len(signals)} positive signals on {date}, using all.")
            return signals.index.tolist()
        top = signals.nlargest(self.top_n).index.tolist()
        return top

    def compute_weights(self, prices, selected_symbols, date, signals=None):
        """
        Compute target weights for selected symbols.
        signals: Series with signal values (predicted returns) for all symbols.
        """
        if len(selected_symbols) == 0:
            return {}

        if self.weight_method == 'equal':
            w = 1.0 / len(selected_symbols)
            return {sym: w for sym in selected_symbols}

        elif self.weight_method == 'signal_weighted':
            if signals is None:
                logger.error("signal_weighted requires signals")
                return {}
            sigs = signals.loc[selected_symbols].copy()
            sigs = sigs.clip(lower=0)
            total = sigs.sum()
            if total <= 0:
                logger.warning(f"Sum of signals <= 0 on {date}, fallback to equal")
                w = 1.0 / len(selected_symbols)
                return {sym: w for sym in selected_symbols}
            raw_weights = sigs / total
            raw_weights = raw_weights.clip(lower=self.min_weight, upper=self.max_weight)
            raw_weights = raw_weights / raw_weights.sum()
            return raw_weights.to_dict()

        elif self.weight_method == 'volatility_targeting':
            if signals is None:
                logger.error("volatility_targeting requires signals")
                return {}
            # 1. Compute signal weights
            sigs = signals.loc[selected_symbols].copy()
            sigs = sigs.clip(lower=0)
            total_sig = sigs.sum()
            if total_sig <= 0:
                w = 1.0 / len(selected_symbols)
                return {sym: w for sym in selected_symbols}
            raw_weights = sigs / total_sig

            # 2. Estimate each asset's annualised volatility
            hist_prices = prices[selected_symbols].loc[:date].iloc[-60:]  # last 60 days
            if hist_prices.empty or len(hist_prices) < 20:
                logger.warning(f"Insufficient price history for vol estimation on {date}, fallback to equal")
                w = 1.0 / len(selected_symbols)
                return {sym: w for sym in selected_symbols}

            returns = hist_prices.pct_change().dropna()
            # Use exponential weighted volatility for more responsiveness
            vols = returns.ewm(span=21, adjust=False).std().iloc[-1] * np.sqrt(252)
            # Replace zero or NaN vols with median
            vols = vols.replace(0, np.nan).fillna(vols.median())

            # 3. Scale raw_weights by inverse volatility to achieve target portfolio volatility
            inv_vol_weights = raw_weights / vols
            weighted_vols = inv_vol_weights * vols
            port_vol_uncorr = np.sqrt((weighted_vols ** 2).sum())
            if port_vol_uncorr == 0:
                scale = 1.0
            else:
                scale = self.target_vol / port_vol_uncorr

            final_weights = inv_vol_weights * scale
            final_weights = final_weights.clip(lower=self.min_weight, upper=self.max_weight)
            final_weights = final_weights / final_weights.sum()

            if final_weights.isnull().any():
                logger.error("NaN weights detected in volatility_targeting, fallback to equal")
                w = 1.0 / len(selected_symbols)
                return {sym: w for sym in selected_symbols}

            return final_weights.to_dict()

        else:   # mean_variance (Ledoit-Wolf shrinkage)
            hist_prices = prices[selected_symbols].loc[:date].iloc[-252:]
            if hist_prices.empty or len(hist_prices) < 30:
                logger.warning(f"Insufficient price history for {date}, fallback to equal")
                w = 1.0 / len(selected_symbols)
                return {sym: w for sym in selected_symbols}
            try:
                mu = expected_returns.mean_historical_return(hist_prices)
                S = risk_models.CovarianceShrinkage(hist_prices).ledoit_wolf()
                ef = EfficientFrontier(mu, S, weight_bounds=(self.min_weight, self.max_weight))
                weights = ef.max_sharpe()
                cleaned = ef.clean_weights()
                cleaned = {sym: cleaned.get(sym, 0) for sym in selected_symbols}
                if any(np.isnan(v) for v in cleaned.values()):
                    logger.error("NaN weights from mean_variance, fallback to equal")
                    w = 1.0 / len(selected_symbols)
                    return {sym: w for sym in selected_symbols}
                return cleaned
            except Exception as e:
                logger.error(f"Optimization failed for {date}: {e}, using equal weight")
                w = 1.0 / len(selected_symbols)
                return {sym: w for sym in selected_symbols}

    def apply_stop_loss(self, holdings, prices, date, entry_prices):
        """Sell positions that have fallen more than stop_loss from entry."""
        if self.stop_loss is None:
            return holdings, 0.0, []

        sold = []
        proceeds = 0.0
        new_holdings = holdings.copy()
        for sym, shares in holdings.items():
            if sym not in prices.columns or pd.isna(prices.loc[date, sym]):
                continue
            current_price = prices.loc[date, sym]
            entry = entry_prices.get(sym)
            if entry is None:
                continue
            pnl_pct = (current_price - entry) / entry
            if pnl_pct < -self.stop_loss:
                sale_value = shares * current_price * (1 - self.transaction_cost)
                proceeds += sale_value
                del new_holdings[sym]
                sold.append(sym)
                logger.info(f"Stop-loss {sym} at {date}: loss {pnl_pct:.2%}")
        return new_holdings, proceeds, sold