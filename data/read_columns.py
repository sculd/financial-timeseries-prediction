import pandas as pd, numpy as np

def read(filename, feature_cols = ['Close'], val_col = 'Close', window_size = 22):
    cols = [] + feature_cols
    df = pd.read_csv(filename, index_col = 0)
    for i in range(1, window_size):
        for col in feature_cols:
            c = col + ('%d' % (i))
            df[c] = df[col].shift(i)
            cols.append(c)
    df = (df - df.rolling(window_size).mean()) / df.rolling(window_size).std()
    df = df.dropna()
    df_data = df[cols]
    df_label = df[val_col].shift(-1) > df['Close']

    train_size = int(len(df_data) * 0.6)
    train_data, train_labels = np.array(df_data[:train_size]), df_label[:train_size]
    train_labels = np.eye(2)[((train_labels + 1.0) / 2.0).astype(int)]
    valid_data, valid_labels = np.array(df_data[train_size:]), df_label[train_size:]
    valid_labels = np.eye(2)[((valid_labels + 1.0) / 2.0).astype(int)]

    print("train_data shape", train_data.shape)
    if len(train_data.shape) == 2:
        train_data = np.expand_dims(train_data, axis=-1)
        valid_labels = np.expand_dims(valid_labels, axis=-1)

    return train_data, train_labels, valid_data, valid_labels

def read_sp500_close(feature_cols = ['Close'], val_col = 'Close', window_size = 22):
    return read('data/sp500_ohlc.csv', feature_cols = feature_cols, val_col = val_col, window_size = window_size)

def read_goog_close(feature_cols = ['Close'], val_col = 'Close', window_size = 22):
    return read('data/goog_ohlc.csv', feature_cols = feature_cols, val_col = val_col, window_size = window_size)

