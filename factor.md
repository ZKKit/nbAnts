# ==================== 特征工程（单币种，17个因子） ====================
def add_features_single(df):
    """单个币种的特征工程（17个核心因子）"""
    df = df.sort_values('date').reset_index(drop=True)

    df['returns_1'] = df['close'].pct_change(1)
    df['returns_24'] = df['close'].pct_change(24)

    for period in [12, 24]:
        df[f'volatility_{period}'] = df['returns_1'].rolling(period).std()
        df[f'volatility_ratio_{period}'] = (
            df[f'volatility_{period}'] /
            df[f'volatility_{period}'].rolling(period * 2).mean()
        )

    # OBV
    obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
    df['obv'] = obv.on_balance_volume()
    df['obv_ma_6'] = df['obv'].rolling(6).mean()

    # ATR
    df['atr_14'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14
    ).average_true_range()

    # MACD signal
    macd = ta.trend.MACD(df['close'])
    df['macd_signal'] = macd.macd_signal()

    # Realized price approx
    df['realized_price_approx'] = (
        df['quote_volume'].rolling(24).sum() /
        df['volume'].rolling(24).sum()
    )

    # Hour features
    df['hour'] = df['date'].dt.hour
    df['is_last_2h'] = df['hour'].isin([22, 23]).astype(int)
    df['volume_last_2h'] = df['volume'] * df['is_last_2h']
    df['volume_last2h_ratio_24h'] = (
        df['volume_last_2h'].rolling(24).sum() /
        df['volume'].rolling(24).sum()
    )
    df['is_first_hour'] = (df['hour'] == 0).astype(int)
    df['volume_first_hour'] = df['volume'] * df['is_first_hour']
    df['volume_first_hour_ratio_24h'] = (
        df['volume_first_hour'].rolling(24).sum() /
        df['volume'].rolling(24).sum()
    )

    # Date features
    df['dayofmonth'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # Interactions
    df['month_day'] = df['month'] * df['dayofmonth'] / 100
    df['daymonth_obv_ma'] = df['dayofmonth'] * df['obv_ma_6']
    df['month_last2h'] = df['month'] * df['volume_last2h_ratio_24h']
    df['last2h_ratio_dayofweek'] = df['volume_last2h_ratio_24h'] * df['dayofweek']
    df['ret24_volratio'] = df['returns_24'] * df['volatility_ratio_24']
    df['vol_12_24_product'] = df['volatility_12'] * df['volatility_24']

    keep_cols = [
        'month_day', 'daymonth_obv_ma', 'atr_14', 'obv_ma_6', 'obv',
        'month_last2h', 'last2h_ratio_dayofweek', 'volatility_ratio_12',
        'volatility_ratio_24', 'volume_first_hour_ratio_24h', 'volume_last2h_ratio_24h',
        'realized_price_approx', 'ret24_volratio', 'volatility_24', 'dayofmonth',
        'macd_signal', 'vol_12_24_product'
    ]
    available = [c for c in keep_cols if c in df.columns]
    base_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']
    df = df[available + base_cols]
    df.dropna(inplace=True)
    return df
