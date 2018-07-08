import os, pandas as pd, numpy as np, pickle
import data.read_single_column as read_single_column

CLOSE_COLUMN = ' value'
FILENAME = 'sp500_close.csv'
SEQ_LEN = 50
TEST_SPLIT = 0.65

def load_from_file(if_normalise_window, if_binary_classification):
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'csvs')
    fn = os.path.join(dir, FILENAME)
    x_t, y_t = read_single_column.load_data(fn, CLOSE_COLUMN, SEQ_LEN, if_normalise_window, if_binary_classification)

    idx_split = int(len(x_t) * TEST_SPLIT)
    x_train, y_train = x_t[:idx_split], y_t[:idx_split]
    # add SEQ_LEN to remove the correlation between train and test
    x_test, y_test = x_t[idx_split+SEQ_LEN:], y_t[idx_split+SEQ_LEN:]
    return [x_train, y_train, x_test, y_test]

if __name__ == '__main__':
    xt, yt, xv, yv = load_from_file(True, False)
    pass
