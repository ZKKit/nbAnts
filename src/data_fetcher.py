# Download crypto data with robust pagination

import ccxt
import pandas as pd
import yaml
import logging
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, config_source=None):
        """
        config_source: either a path to a YAML file (str) or a dict containing config.
        """
        if config_source is None:
            config_source = 'config.yaml'
        if isinstance(config_source, dict):
            self.config = config_source
        else:
            with open(config_source, 'r') as f:
                self.config = yaml.safe_load(f)
        self.exchange_id = self.config['data']['exchange']
        self.exchange = getattr(ccxt, self.exchange_id)({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.symbols = self.config['data']['symbols']
        self.timeframe = self.config['data']['timeframe']
        self.start = self.config['data']['start_date']
        self.end = self.config['data']['end_date']
        self.logger = logging.getLogger(__name__)

    def fetch_ohlcv(self, symbol, since=None, limit=1000):
        """Fetch all OHLCV data for a symbol between start and end."""
        all_data = []
        if since is None:
            since = self.exchange.parse8601(self.start + 'T00:00:00Z')
        end_ts = self.exchange.parse8601(self.end + 'T00:00:00Z')

        while since < end_ts:
            try:
                data = self.exchange.fetch_ohlcv(symbol, self.timeframe,
                                                  since=since, limit=limit)
                if not data:
                    break
                all_data.extend(data)
                since = data[-1][0] + 1  # next candle after last
                self.logger.info(f"Fetched {len(data)} {symbol} candles, "
                                 f"up to {pd.to_datetime(data[-1][0], unit='ms')}")
            except Exception as e:
                self.logger.error(f"Error fetching {symbol}: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high',
                                             'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        # trim to exact date range
        df = df.loc[self.start:self.end]
        return df

    def fetch_all(self):
        """Fetch data for all symbols, return MultiIndex DataFrame."""
        dfs = []
        for symbol in self.symbols:
            df = self.fetch_ohlcv(symbol)
            if df.empty:
                self.logger.warning(f"No data for {symbol}")
                continue
            df['symbol'] = symbol
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs).reset_index()
        combined = combined.set_index(['timestamp', 'symbol']).sort_index()
        return combined

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fetcher = DataFetcher()
    data = fetcher.fetch_all()
    print(data.head())
    data.to_parquet('data.parquet')