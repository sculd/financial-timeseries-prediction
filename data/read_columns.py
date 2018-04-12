import pandas as pd, numpy as np

FEATURE_COLS = ['Close', 'Volume']

def bolinger_bands(stock_price, window_size = 20, num_of_std = 3):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std  = stock_price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return (rolling_mean - lower_band) / rolling_mean, (upper_band - lower_band) / rolling_mean

def read(filename, feature_cols = FEATURE_COLS, vol_col = 'Volume', val_col = 'Close', window_size = 22):
    cols = []# + feature_cols
    df = pd.read_csv(filename, index_col = 0)
    bollinger_cols = [] #['rolling', 'band']
    #df[bollinger_cols[0]], df[bollinger_cols[1]] = bolinger_bands(df[val_col])
    df[val_col] = (df[val_col] - df[val_col].rolling(window_size).mean()) / df[val_col].rolling(window_size).std()
    df[vol_col] = (df[vol_col] - df[vol_col].rolling(window_size).mean()) / df[vol_col].rolling(window_size).mean()
    #df = (df - df.rolling(window_size).mean()) / df.rolling(window_size).std()
    all_feature_cols = feature_cols + bollinger_cols
    for i in range(0, window_size):
        for col in all_feature_cols:
            c = col + ('%d' % (i))
            df[c] = df[col].shift(i)
            cols.append(c)
    df = df.dropna()
    df_data = df[cols]
    df_label = df[val_col].shift(-1) > df['Close']
    df_data = np.array(df_data)
    df_label = np.array(df_label)
    print("df_data shape before reshape", df_data.shape)
    df_data = df_data.reshape([len(df_data), -1, len(all_feature_cols)])

    train_size = int(len(df_data) * 0.6)
    train_data, train_labels = np.array(df_data[:train_size]), df_label[:train_size]
    train_labels = np.eye(2)[((train_labels + 1.0) / 2.0).astype(int)]
    valid_data, valid_labels = np.array(df_data[train_size:]), df_label[train_size:]
    valid_labels = np.eye(2)[((valid_labels + 1.0) / 2.0).astype(int)]

    print("train_data shape", train_data.shape)
    if len(train_data.shape) == 2:
        train_data = np.expand_dims(train_data, axis=-1)
        valid_data = np.expand_dims(valid_data, axis=-1)

    return train_data, train_labels, valid_data, valid_labels

def read_sp500_close(feature_cols = FEATURE_COLS, vol_col = 'Volume', val_col = 'Close', window_size = 22):
    return read('data/sp500_ohlc.csv', feature_cols = feature_cols, vol_col = vol_col, val_col = val_col, window_size = window_size)

def read_goog_close(feature_cols = FEATURE_COLS, vol_col = 'Volume', val_col = 'Close', window_size = 22):
    return read('data/goog_ohlc.csv', feature_cols = feature_cols, vol_col = vol_col, val_col = val_col, window_size = window_size)

