import os, pandas as pd, numpy as np, pickle
import data.read_single_column as read_single_column

CLOSE_COLUMN = 'Close'
VOLUME_COLUMN = 'Volume'
SEQ_LEN = 50

def load_data(column_name, seq_len, if_normalise_window, if_binary_classification):
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'csvs')
    fs = [f for f in os.listdir(dir) if '.py' not in f and '.csv in f']

    x_train, y_train = [], []
    x_test, y_test = [], []

    for i, f in enumerate(fs):
        x_t, y_t = read_single_column.load_data(os.path.join(dir, f), column_name, seq_len, if_normalise_window, if_binary_classification)
        if i % 2 == 0:
            x_train.append(x_t)
            y_train.append(y_t)
        else:
            x_test.append(x_t)
            y_test.append(y_t)

        if i > 30:
            break

    x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)
    x_test, y_test = np.concatenate(x_test), np.concatenate(y_test)
    return [x_train, y_train, x_test, y_test]

if __name__ == '__main__':
    xt, yt, xv, yv = load_data('Close', SEQ_LEN, True, False)
    pickle.dump([xt, yt, xv, yv], open('close_50_seq_regression.data', 'w'))

    xt, yt, xv, yv = load_data('Close', SEQ_LEN, True, True)
    pickle.dump([xt, yt, xv, yv], open('close_50_seq_classification.data', 'w'))
    xt, yt, xv, yv = pickle.load(open('close_50_seq.data', 'r'))
    pass
