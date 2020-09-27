import numpy as np


def add_binary_target_column(df, step_forward, value_column_name = 'close', target_column_name = 'target'):
    '''
    Adds a binary (1 for larger or -1 for smaller) target column with given `target_column_name`.

    The target value is decided by looking forward by `step_forward` and see if the value is larger or smaller.

    :param df:
    :param value_column_name:
    :param step_forward:
    :return:
    '''
    df[target_column_name] = -df[value_column_name].diff(-step_forward) > 0
    df.target = df.target.astype(np.int64) * 2 - 1.0
    df = df[:-step_forward]
    return df








