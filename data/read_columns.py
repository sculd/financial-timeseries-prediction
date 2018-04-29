import pandas as pd, numpy as np

FEATURE_COLS = ['Close', 'Volume']
_TARIN_RANGES = [(0.0, 0.2), (0.3, 0.5), (0.7, 0.9)]
NUM_FEATURES_RETURNS = 6
N_CHANNELS_HISTORY = 2 # price, volume, 2 bollingers
N_CHANNELS_RETURNS = 1


def bolinger_bands(stock_price, window_size = 20, num_of_std = 3):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std  = stock_price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return (rolling_mean - lower_band) / rolling_mean, (upper_band - lower_band) / rolling_mean

_RSI_COLUMN = 'rsi'
def rsi(stock_price, window_size = 20):
    d_up, d_down = stock_price.diff(), stock_price.diff()
    d_up[d_up < 0] = 0
    d_down[d_down > 0] = 0

    roll_up = pd.rolling_mean(d_up, window_size)
    roll_down = pd.rolling_mean(d_down, window_size).abs()

    rs = roll_up / roll_down
    return rs

def separate_train_valid(df_data, df_label, train_ranges, window_size = 22):
    train_data, train_labels = [], []
    valid_data, valid_labels = [], []
    dl = len(df_data)
    train_indices = []
    for i, train_range in enumerate(train_ranges):
        ts = int(dl *  train_range[0])
        if i > 0:
            ts += window_size
        te = int(dl * train_range[1])
        train_indices.append((ts,te))

    prev_train_range = None
    valid_indices = []
    for i, train_range in enumerate(train_ranges):
        if i > 0:
            vs = int(dl * prev_train_range[1])
            vs += window_size
            ve = int(dl * train_range[0])
            valid_indices.append((vs,ve))

        if i == len(train_ranges) - 1 and train_range[1] < 1.0:
            vs = int(dl * prev_train_range[1])
            vs += window_size
            ve = dl
            valid_indices.append((vs,ve))

        prev_train_range = train_range

    train_data = np.concatenate([df_data[pair[0]:pair[1]] for pair in train_indices], axis=0)
    train_labels = np.concatenate([df_label[pair[0]:pair[1]] for pair in train_indices], axis=0)

    valid_data = np.concatenate([df_data[pair[0]:pair[1]] for pair in valid_indices], axis=0)
    valid_labels = np.concatenate([df_label[pair[0]:pair[1]] for pair in valid_indices], axis=0)

    '''
    train_size = int(len(df_data) * 0.6)
    train_data, train_labels = np.array(df_data[:train_size]), df_label[:train_size]
    valid_data, valid_labels = np.array(df_data[train_size:]), df_label[train_size:]
    '''
    train_labels = np.eye(2)[((train_labels + 1.0) / 2.0).astype(int)]
    valid_labels = np.eye(2)[((valid_labels + 1.0) / 2.0).astype(int)]

    return train_data, train_labels, valid_data, valid_labels

def read_history(filename, feature_cols = FEATURE_COLS, vol_col = 'Volume', val_col = 'Close', window_size = 22, reshape_per_channel = True):
    cols = []# + feature_cols
    df = pd.read_csv(filename, index_col = 0)

    '''
    bollinger_cols = ['rolling', 'band']
    df[bollinger_cols[0]], df[bollinger_cols[1]] = bolinger_bands(df[val_col])

    df[_RSI_COLUMN] = rsi(df[val_col])
    '''

    df[val_col] = (df[val_col] - df[val_col].rolling(window_size).mean()) / df[val_col].rolling(window_size).std()
    df[vol_col] = (df[vol_col] - df[vol_col].rolling(window_size).mean()) / df[vol_col].rolling(window_size).mean()

    #df = (df - df.rolling(window_size).mean()) / df.rolling(window_size).std()
    all_feature_cols = feature_cols #+ bollinger_cols + [_RSI_COLUMN]
    for col in all_feature_cols:
        for i in range(0, window_size):
            c = col + ('%d' % (i))
            df[c] = df[col].shift(i)
            cols.append(c)
    df = df.dropna()

    df_data = df[cols]
    df_data = np.array(df_data)
    if reshape_per_channel:
        print("df_data shape before reshape", df_data.shape)
        df_data = df_data.reshape([len(df_data), -1, len(all_feature_cols)])

    #df_label = df[val_col].rolling(2).mean().shift(-1) > df[val_col]
    df_label = df[val_col].shift(-2) > df[val_col]
    df_label = np.array(df_label)

    train_data, train_labels, valid_data, valid_labels = separate_train_valid(df_data, df_label, _TARIN_RANGES, window_size=window_size)

    print("train_data shape", train_data.shape)
    if reshape_per_channel and len(train_data.shape) == 2:
        train_data = np.expand_dims(train_data, axis=-1)
        valid_data = np.expand_dims(valid_data, axis=-1)

    return train_data, train_labels, valid_data, valid_labels

def read_returns(filename, vol_col = 'Volume', val_col = 'Close', window_size = 10):
    cols = []# + feature_cols
    df = pd.read_csv(filename, index_col = 0)
    all_feature_cols = [vol_col]
    for i in [1, 5, 10, 20, 30]:
        c = val_col + ('%d' % (i))
        df[c] = df[val_col].shift(i) / df[val_col]
        cols.append(c)

    df[vol_col] = (df[vol_col] - df[vol_col].rolling(window_size).mean()) / df[vol_col].rolling(window_size).mean()
    cols.append(vol_col)

    df = df.dropna()
    df_data = df[cols]
    #df_label = df[val_col].rolling(2).mean().shift(-2) > df[val_col]
    df_label = df[val_col].shift(-2) > df[val_col]

    df_data = np.array(df_data)
    df_label = np.array(df_label)
    print("df_data shape before reshape", df_data.shape)
    df_data = df_data.reshape([len(df_data), -1, len(all_feature_cols)])

    train_data, train_labels, valid_data, valid_labels = separate_train_valid(df_data, df_label, _TARIN_RANGES, window_size=window_size)

    print("train_data shape", train_data.shape)
    if len(train_data.shape) == 2:
        train_data = np.expand_dims(train_data, axis=-1)
        valid_data = np.expand_dims(valid_data, axis=-1)

    return train_data, train_labels, valid_data, valid_labels

def read_sp500_close_history(feature_cols = FEATURE_COLS, vol_col = 'Volume', val_col = 'Close', window_size = 22, reshape_per_channel = True):
    return read_history('data/GSPC_ohlc.csv', feature_cols = feature_cols, vol_col = vol_col, val_col = val_col,
                        window_size = window_size, reshape_per_channel = reshape_per_channel)

def read_sp500_close_returns(vol_col = 'Volume', val_col = 'Close', window_size = 10):
    return read_returns('data/GSPC_ohlc.csv', vol_col = vol_col, val_col = val_col, window_size = window_size)

def read_kosdaq_close_history(feature_cols = FEATURE_COLS, vol_col = 'Volume', val_col = 'Close', window_size = 22, reshape_per_channel = True):
    return read_history('data/KQ11_ohlc.csv', feature_cols = feature_cols, vol_col = vol_col, val_col = val_col,
                        window_size = window_size, reshape_per_channel = reshape_per_channel)

def read_bitstamp_btcusd_2017_hourly_history(feature_cols = ['close', 'volume'], vol_col = 'volume', val_col = 'close', window_size = 22, reshape_per_channel = True):
    return read_history('data/BTCUSD_transactions_2017_hour.csv', feature_cols = feature_cols, vol_col = vol_col, val_col = val_col,
                        window_size = window_size, reshape_per_channel = reshape_per_channel)

def read_goog_close(feature_cols = FEATURE_COLS, vol_col = 'Volume', val_col = 'Close', window_size = 22, reshape_per_channel = True):
    return read_history('data/goog_ohlc.csv', feature_cols = feature_cols, vol_col = vol_col, val_col = val_col,
                        window_size = window_size, reshape_per_channel = reshape_per_channel)

def read_fb_close(feature_cols = FEATURE_COLS, vol_col = 'Volume', val_col = 'Close', window_size = 22, reshape_per_channel = True):
    return read_history('data/FB_ohlc.csv', feature_cols = feature_cols, vol_col = vol_col, val_col = val_col,
                        window_size = window_size, reshape_per_channel = reshape_per_channel)

