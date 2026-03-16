# Feature engineering – technical indicators, rolling stats, and macro features

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from src.utils import GarmanKlassVolatility

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.tech_indicators = config['features']['technical_indicators']
        self.rolling_returns = config['features']['rolling_returns']
        self.rolling_volatility = config['features'].get('rolling_volatility', [])
        self.rolling_correlation = config['features'].get('rolling_correlation', [])
        self.macro_config = config['features'].get('macro_indicators', {})
        self.do_volume_rank = config['features']['dollar_volume_rank']
        self.top_n_liquid = config['features']['top_n_liquid']

    def compute_features(self, df, dropna=True):
        """
        df: MultiIndex (timestamp, symbol) with columns:
            open, high, low, close, volume
        dropna: if True, drop rows with any NaN (used in training).
                If False, return data with possible NaNs (used in live).
        Returns DataFrame with added features.
        """
        df = df.sort_index()
        df['dollar_volume'] = df['close'] * df['volume']

        # --- Technical indicators per symbol ---
        symbols = df.index.get_level_values('symbol').unique()
        logger.info(f"Computing features for {len(symbols)} symbols")

        dfs = []
        for symbol in symbols:
            sdf = df.xs(symbol, level='symbol').copy()
            sdf['symbol'] = symbol

            if 'rsi' in self.tech_indicators:
                sdf['rsi'] = ta.rsi(sdf['close'], length=14)
            if 'macd' in self.tech_indicators:
                macd = ta.macd(sdf['close'], fast=12, slow=26, signal=9)
                sdf['macd'] = macd['MACD_12_26_9'] if macd is not None else np.nan
            if 'bbands' in self.tech_indicators:
                bb = ta.bbands(sdf['close'], length=20, std=2)
                if bb is not None:
                    bb.columns = ['bb_lower', 'bb_mid', 'bb_upper', 'bb_bandwidth', 'bb_percent']
                    sdf = sdf.join(bb)
                    sdf['bb_width'] = (sdf['bb_upper'] - sdf['bb_lower']) / sdf['bb_mid']
                    sdf['bb_position'] = (sdf['close'] - sdf['bb_lower']) / (sdf['bb_upper'] - sdf['bb_lower'])
                    sdf.drop(columns=['bb_lower', 'bb_mid', 'bb_upper'], inplace=True, errors='ignore')
                else:
                    sdf['bb_width'] = np.nan
                    sdf['bb_position'] = np.nan
            if 'atr' in self.tech_indicators:
                sdf['atr'] = ta.atr(sdf['high'], sdf['low'], sdf['close'], length=14)
            if 'garman_klass_vol' in self.tech_indicators:
                sdf['garman_klass_vol'] = GarmanKlassVolatility.calculate(
                    sdf['high'], sdf['low'], sdf['open'], sdf['close']
                )
            if 'volume_ratio' in self.tech_indicators:
                vol_ma = sdf['volume'].rolling(20, min_periods=10).mean()
                sdf['volume_ratio'] = sdf['volume'] / vol_ma
            if 'price_position' in self.tech_indicators:
                high_20 = sdf['high'].rolling(20, min_periods=10).max()
                low_20  = sdf['low'].rolling(20, min_periods=10).min()
                sdf['price_position'] = (sdf['close'] - low_20) / (high_20 - low_20)
            if 'adx' in self.tech_indicators:
                adx = ta.adx(sdf['high'], sdf['low'], sdf['close'], length=14)
                sdf['adx'] = adx['ADX_14'] if adx is not None else np.nan
            if 'obv' in self.tech_indicators:
                sdf['obv'] = ta.obv(sdf['close'], sdf['volume'])
            if 'stoch_k' in self.tech_indicators:
                stoch = ta.stoch(sdf['high'], sdf['low'], sdf['close'], k=14, d=3)
                sdf['stoch_k'] = stoch['STOCHk_14_3_3'] if stoch is not None else np.nan

            for lag in self.rolling_returns:
                sdf[f'ret_{lag}d'] = sdf['close'].pct_change(lag)

            for window in self.rolling_volatility:
                sdf[f'volatility_{window}d'] = sdf['close'].pct_change().rolling(window).std()

            dfs.append(sdf)

        df = pd.concat(dfs)
        df.set_index('symbol', append=True, inplace=True)
        df = df.reorder_levels(['timestamp', 'symbol']).sort_index()
        logger.info(f"After per-symbol indicators: {len(df)} rows")

        # --- Rolling correlation with benchmark ---
        if self.rolling_correlation and self.macro_config:
            benchmark = self.macro_config['benchmark']
            if benchmark in symbols:
                bench_series = df.xs(benchmark, level='symbol')['close']
                for window in self.rolling_correlation:
                    def corr_with_btc(group):
                        ret = group['close'].pct_change()
                        common_idx = ret.index.intersection(bench_series.index)
                        if len(common_idx) < window:
                            return pd.Series(index=group.index, dtype=float)
                        ret_align = ret.loc[common_idx]
                        btc_ret = bench_series.loc[common_idx].pct_change()
                        corr = ret_align.rolling(window).corr(btc_ret)
                        return corr.reindex(group.index)
                    df[f'corr_btc_{window}d'] = df.groupby('symbol').apply(corr_with_btc).droplevel(0)
        logger.info(f"After correlation features: {len(df)} rows")

        # --- Macro features (common to all symbols) ---
        if self.macro_config:
            bench = self.macro_config['benchmark']
            if bench in symbols:
                bench_df = df.xs(bench, level='symbol')[['close']].copy()
                for lag in self.macro_config.get('benchmark_returns', []):
                    bench_df[f'bench_ret_{lag}d'] = bench_df['close'].pct_change(lag)
                for window in self.macro_config.get('benchmark_volatility', []):
                    bench_df[f'bench_vol_{window}d'] = bench_df['close'].pct_change().rolling(window).std()
                for period in self.macro_config.get('benchmark_sma', []):
                    sma = bench_df['close'].rolling(period).mean()
                    bench_df[f'bench_sma_{period}_dist'] = (bench_df['close'] - sma) / sma

                bench_df.drop(columns=['close'], inplace=True)
                df = df.join(bench_df, on='timestamp', how='left')
        logger.info(f"After macro features: {len(df)} rows")

        # --- Liquidity filter ---
        if self.do_volume_rank:
            df['dollar_vol_ma30'] = df.groupby('symbol')['dollar_volume'].transform(
                lambda x: x.rolling(30, min_periods=15).median()
            )
            df['dollar_vol_rank'] = df.groupby('timestamp')['dollar_vol_ma30'].rank(ascending=False)
            df = df[df['dollar_vol_rank'] <= self.top_n_liquid].drop(columns=['dollar_vol_rank'])
            logger.info(f"After liquidity filter (top {self.top_n_liquid}): {len(df)} rows")

        # --- Check NaNs on the latest date ---
        latest = df.index.get_level_values(0).max()
        latest_df = df.loc[latest]
        logger.info(f"Latest date: {latest}, symbols present: {len(latest_df)}")
        nan_counts = latest_df.isnull().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"NaNs on latest date:\n{nan_counts[nan_counts > 0]}")

        # Drop rows with any NaN if requested
        if dropna:
            df.dropna(inplace=True)
            logger.info(f"After dropping NaNs: {len(df)} rows")

        return df