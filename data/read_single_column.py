import time
import warnings
import numpy as np, pandas as pd
from numpy import newaxis

warnings.filterwarnings("ignore")

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def load_data(csvfilename, column_name, seq_len, if_normalise_window, if_binary_classification):
    df = pd.read_csv(csvfilename, index_col=0)
    data = np.array(df[column_name])

    sequence_length = seq_len + 2
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    if if_normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    train = result[:, :]
    np.random.shuffle(train)
    x_train = train[:, :-2]
    y_train = train[:, -2]
    if if_binary_classification:
        y_train = y_train > 0
        y_train = np.eye(2)[((y_train + 1.0) / 2.0).astype(int)]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return [x_train, y_train]

if __name__ == "__main__":
    x_train, y_train = load_data('AAPL_ohlc.csv', 'Close', 50, True, True)

