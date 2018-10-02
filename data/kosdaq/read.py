import os, pandas as pd, numpy as np, pickle
import data.read_columns as read_columns

CLOSE_COLUMN = 'Close'
VOLUME_COLUMN = 'Volume'
WINDOW_SIZE = 32

def _get_processed_file_name(window_size = 32, reshape_per_channel = True):
    return ""

def process(window_size = WINDOW_SIZE, reshape_per_channel = True):
    dir = dir_path = os.path.dirname(os.path.realpath(__file__))
    fs = [f for f in os.listdir(dir) if '.py' not in f and '.csv in f']

    train_data_list, train_labels_list, train_targets_list, valid_data_list, valid_labels_list, valid_targets_list = [], [], [], [], [], []
    for i, f in enumerate(fs):
        print('processing %s, %d out of %d' % (f, i, len(fs)))
        df = pd.read_csv(os.path.join(dir, f), index_col=0)
        if len(df) == 0:
            print('the data size is 0')
            continue

        df, data, labels, target, train_data_, train_labels_, train_targets_, valid_data_, valid_labels_, valid_targets_ = \
            read_columns.history_from_df(df, feature_cols=[], vol_col=VOLUME_COLUMN, val_col=CLOSE_COLUMN,
                                     window_size=window_size, reshape_per_channel=reshape_per_channel)

        if train_data_ is None or valid_data_ is None:
            print('the train / valid data is None')
            continue

        train_data_list.append(train_data_)
        train_labels_list.append(train_labels_)
        train_targets_list.append(train_targets_)
        valid_data_list.append(valid_data_)
        valid_labels_list.append(valid_labels_)
        valid_targets_list.append(valid_targets_)

    train_data = np.concatenate(train_data_list)
    train_labels = np.concatenate(train_labels_list)
    train_targets = np.concatenate(train_targets_list)
    valid_data = np.concatenate(valid_data_list)
    valid_labels = np.concatenate(valid_labels_list)
    valid_targets = np.concatenate(valid_targets_list)

    train_data.dump("train.data")
    train_labels.dump("train.labels.data")
    train_targets.dump("train.targets.data")
    valid_data.dump("test.data")
    valid_labels.dump("test.labels.data")
    valid_targets.dump("test.targets.data")

    return train_data, train_labels, train_targets, valid_data, valid_labels, valid_targets

def load():
    dir = dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(dir, "train.data")):
        process()
    train_data = np.load(os.path.join(dir, "train.data"))
    train_labels = np.load(os.path.join(dir, "train.labels.data"))
    train_targets = np.load(os.path.join(dir, "train.targets.data"))
    valid_data = np.load(os.path.join(dir, "test.data"))
    valid_labels = np.load(os.path.join(dir, "test.labels.data"))
    valid_targets = np.load(os.path.join(dir, "test.targets.data"))

    return train_data, train_labels, train_targets, valid_data, valid_labels, valid_targets

if __name__ == '__main__':
    process()
    train_data, train_labels, train_targets, valid_data, valid_labels, valid_targets = load()
    pass
