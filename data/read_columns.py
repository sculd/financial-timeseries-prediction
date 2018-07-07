import pandas as pd, numpy as np, random, datetime

VAL_ADJUSTED = 'close_adj'
VOLUME_ADJUSTED = 'volume_adj'
#_TARIN_RANGES = [(0.0, 0.2), (0.3, 0.5), (0.7, 0.9)]
_TARIN_RANGES = [(0.0, 0.7)]
WINDOW_STEP = 4
NUM_FEATURES_RETURNS = 6
N_CHANNELS_HISTORY = 8 # 2 prices, 2 volumes, 2 bollingers, rsi, return
N_CHANNELS_RETURNS = 1
PRED_LENGTH = 1
NUM_LABELS = 2
_RSI_COLUMN = 'rsi'
_RETURN_COLUMN = 'return'

def bolinger_bands(stock_price, window_size = 20, num_of_std = 3):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std  = stock_price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return 'rolling', 'band', (rolling_mean - lower_band) / rolling_mean, (upper_band - lower_band) / rolling_mean

def rsi(stock_price, window_size = 20):
    d_up, d_down = stock_price.diff(), stock_price.diff()
    d_up[d_up < 0] = 0
    d_down[d_down > 0] = 0

    roll_up = d_up.rolling(window=window_size).mean()
    roll_down = d_down.rolling(window=window_size).mean()

    rs = roll_up / roll_down - 1.0
    return _RSI_COLUMN, rs

def get_train_valid_indices(dl, train_ranges, window_size = 20):
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

        prev_train_range = train_range

    if train_ranges[-1][1] < 1.0:
        vs = int(dl * train_ranges[-1][1])
        vs += window_size
        ve = dl
        valid_indices.append((vs,ve))

    return train_indices, valid_indices

def separate_train_valid(data, train_ranges, window_size=22):
    if data is None:
        return None, None
    train_indices, valid_indices = get_train_valid_indices(len(data), train_ranges, window_size=window_size)

    train = np.concatenate([data[pair[0]:pair[1]] for pair in train_indices], axis=0)
    valid = np.concatenate([data[pair[0]:pair[1]] for pair in valid_indices], axis=0)

    return train, valid

def df_to_np_tensors(df, feature_cols = [], vol_col = 'Volume', val_col = 'Close', window_size = 20, reshape_per_channel = True):
    """returns the trainable np tensors.

    Args:
      emb_dim: The dimension of the Embedding layer.
      num_words: The number of the most frequent tokens
        to be used from the corpus.
      sentence_length: The number of words in each sentence.
        Longer sentences get cut, shorter ones padded.
      hid_dim: The dimension of the Embedding layer.
      class_dim: The number of the CNN layer filters.
      dropout_rate: The portion of kept value in the Dropout layer.
    Returns:
      tf.keras.models.Model: A model.
    """

    cols = []
    features = []
    bbc1, bbc2, bb1, bb2 = bolinger_bands(df[val_col])
    df[bbc1], df[bbc2] = bb1, bb2
    bb_cols = [bbc1, bbc2]
    cols += bb_cols

    rsic, r = rsi(df[val_col])
    df[rsic] = r
    cols += [rsic]

    df[VAL_ADJUSTED] = (df[val_col] - df[val_col].rolling(window_size).mean()) / df[val_col].rolling(window_size).std() * 10
    df[VAL_ADJUSTED + "_mean"] = (df[val_col] - df[val_col].rolling(window_size).mean()) / df[val_col].rolling(window_size).mean() * 10
    df[VOLUME_ADJUSTED] = (df[vol_col] - df[vol_col].rolling(window_size).mean()) / df[vol_col].rolling(window_size).std() * 10
    df[VOLUME_ADJUSTED + "_mean"] = (df[vol_col] - df[vol_col].rolling(window_size).mean()) / df[vol_col].rolling(window_size).mean() * 10
    val_vol_cols = [VAL_ADJUSTED, VAL_ADJUSTED + "_mean", VOLUME_ADJUSTED, VOLUME_ADJUSTED + "_mean"]
    cols += val_vol_cols

    for col in feature_cols:
        c = col + '_adjusted'
        df[c] = (df[col] - df[col].rolling(window_size).mean()) / df[col].rolling(window_size).std() * 10
        cols.append(c)

    #df = (df - df.rolling(window_size).mean()) / df.rolling(window_size).std()
    for col in cols:
        for i in range(0, window_size, WINDOW_STEP):
            c = col + ('%d' % (i))
            df[c] = df[col].shift(i)
            features.append(c)

    cols += [_RETURN_COLUMN]
    for i in range(0, window_size, WINDOW_STEP):
        rc = _RETURN_COLUMN + ('%d' % (i))
        df[rc] = (df[val_col].shift(i) / df[val_col] - 1.0) * 10
        features.append(rc)

    df = df.dropna()
    if len(df) == 0:
        return None, None, None

    data = df[features]
    data = np.array(data)
    if reshape_per_channel:
        print("df_data shape before reshape", data.shape)
        data = data.reshape([len(data), -1, len(cols)])

    #df_label = df[val_col].rolling(2).mean().shift(-1) > df[val_col]
    labels = df[val_col].shift(-PRED_LENGTH) > df[val_col]
    labels = np.array(labels)
    labels = np.eye(2)[((labels + 1.0) / 2.0).astype(int)]

    target = (df[val_col].shift(-PRED_LENGTH) / df[val_col] - 1.0) * 10
    target = np.array(target)
    target = target.reshape([len(target), 1])

    print("train_data shape", data.shape)
    if reshape_per_channel and len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)

    return data, labels, target

def history_from_df(df, feature_cols = [], vol_col = 'Volume', val_col = 'Close', window_size = 20, reshape_per_channel = True):
    data, labels, target = df_to_np_tensors(df, feature_cols = feature_cols, vol_col = vol_col, val_col = val_col, window_size = window_size, reshape_per_channel = reshape_per_channel)

    train_data, valid_data = separate_train_valid(data, _TARIN_RANGES, window_size=window_size)
    train_labels, valid_labels = separate_train_valid(labels, _TARIN_RANGES, window_size=window_size)
    train_targets, valid_targets = separate_train_valid(target, _TARIN_RANGES, window_size=window_size)

    return df, data, labels, target, train_data, train_labels, train_targets, valid_data, valid_labels, valid_targets

def read_history(filename, feature_cols = [], vol_col = 'Volume', val_col = 'Close', window_size = 20, reshape_per_channel = True):
    df = pd.read_csv(filename, index_col = 0)
    return history_from_df(df, feature_cols = feature_cols, vol_col=vol_col, val_col=val_col, window_size=window_size, reshape_per_channel=reshape_per_channel)

def read_returns(filename, vol_col = 'Volume', val_col = 'Close', window_size = 10):
    cols = []# + feature_cols
    df = pd.read_csv(filename, index_col = 0)
    all_feature_cols = [vol_col]
    for i in [i for i in range(1,window_size)]:
        c = val_col + ('return_%d' % (i))
        df[c] = df[val_col].shift(i) / df[val_col]
        cols.append(c)

    df[vol_col] = (df[vol_col] - df[vol_col].rolling(window_size).mean()) / df[vol_col].rolling(window_size).mean()
    cols.append(vol_col)

    df = df.dropna()
    df_data = df[cols]
    #df_label = df[val_col].rolling(2).mean().shift(-2) > df[val_col]
    df_label = df[val_col].shift(-PRED_LENGTH) > df[val_col]

    data = np.array(df_data)
    labels = np.array(df_label)
    labels = np.eye(2)[((labels + 1.0) / 2.0).astype(int)]
    print("df_data shape before reshape", df_data.shape)
    df_data = np.reshape(data, [len(data), -1, len(all_feature_cols)])

    train_data, valid_data = separate_train_valid(df_data, _TARIN_RANGES, window_size=window_size)
    train_labels, valid_labels = separate_train_valid(labels, _TARIN_RANGES, window_size=window_size)

    print("train_data shape", train_data.shape)
    if len(train_data.shape) == 2:
        train_data = np.expand_dims(train_data, axis=-1)
        valid_data = np.expand_dims(valid_data, axis=-1)

    return df, data, labels, train_data, train_labels, valid_data, valid_labels

def generate_random_wak(n_walks, window_size = 20, reshape_per_channel = True):
    random.seed(datetime.datetime.now())
    rw1, rw2 = list(), list()
    rw1.append(-1 if random.random() < 0.5 else 1)
    rw2.append(-1 if random.random() < 0.5 else 1)
    for i in range(1, n_walks):
        movement = -1 if random.random() < 0.5 else 1
        value = rw1[i - 1] + movement
        rw1.append(value)
        movement = -1 if random.random() < 0.5 else 1
        value = rw1[i - 1] + movement
        rw2.append(value)
    dfr = pd.DataFrame({'walk1': rw1, 'walk2': rw2})

    return history_from_df(dfr, feature_cols=['walk1', 'walk2'], vol_col='walk1', val_col='walk2', window_size = window_size,
                    reshape_per_channel = reshape_per_channel)

def read_sp500_close_history(feature_cols = [], vol_col = 'Volume', val_col = 'Close', window_size = 20, reshape_per_channel = True):
    return read_history('data/GSPC_ohlc.csv', feature_cols = feature_cols, vol_col = vol_col, val_col = val_col,
                        window_size = window_size, reshape_per_channel = reshape_per_channel)

def read_sp500_ohlc_history(feature_cols = ['Open', 'High', 'Low'], vol_col = 'Volume', val_col = 'Close', window_size = 20, reshape_per_channel = True):
    return read_history('data/GSPC_ohlc.csv', feature_cols = feature_cols, vol_col = vol_col, val_col = val_col,
                        window_size = window_size, reshape_per_channel = reshape_per_channel)

def read_sp500_close_returns(vol_col = 'Volume', val_col = 'Close', window_size = 20):
    return read_returns('data/GSPC_ohlc.csv', vol_col = vol_col, val_col = val_col, window_size = window_size)

def read_kosdaq_close_history(feature_cols = [], vol_col = 'Volume', val_col = 'Close', window_size = 20, reshape_per_channel = True):
    return read_history('data/KQ11_ohlc.csv', feature_cols = feature_cols, vol_col = vol_col, val_col = val_col,
                        window_size = window_size, reshape_per_channel = reshape_per_channel)

def read_kosdaq_ohlc_history(feature_cols = ['Open', 'High', 'Low'], vol_col = 'Volume', val_col = 'Close', window_size = 20, reshape_per_channel = True):
    return read_history('data/KQ11_ohlc.csv', feature_cols = feature_cols, vol_col = vol_col, val_col = val_col,
                        window_size = window_size, reshape_per_channel = reshape_per_channel)

def read_bitstamp_btcusd_2017_hourly_history(feature_cols = ['close', 'volume'], vol_col = 'volume', val_col = 'close', window_size = 20, reshape_per_channel = True):
    return read_history('data/BTCUSD_transactions_2017_hour.csv', feature_cols = feature_cols, vol_col = vol_col, val_col = val_col,
                        window_size = window_size, reshape_per_channel = reshape_per_channel)

def read_bitstamp_btcusd_full_hourly_history(feature_cols = ['close', 'volume'], vol_col = 'volume', val_col = 'close', window_size = 20, reshape_per_channel = True):
    return read_history('data/BTCUSD_transactions_full_hour.csv', feature_cols = feature_cols, vol_col = vol_col, val_col = val_col,
                        window_size = window_size, reshape_per_channel = reshape_per_channel)

def read_goog_close(feature_cols = [], vol_col = 'Volume', val_col = 'Close', window_size = 20, reshape_per_channel = True):
    return read_history('data/goog_ohlc.csv', feature_cols = feature_cols, vol_col = vol_col, val_col = val_col,
                        window_size = window_size, reshape_per_channel = reshape_per_channel)

def read_fb_close(feature_cols = [], vol_col = 'Volume', val_col = 'Close', window_size = 20, reshape_per_channel = True):
    return read_history('data/FB_ohlc.csv', feature_cols = feature_cols, vol_col = vol_col, val_col = val_col,
                        window_size = window_size, reshape_per_channel = reshape_per_channel)

