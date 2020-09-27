from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_HIHDDEN_1 = 50
_HIHDDEN_2 = 100
_DROPOUT_RATE = 0.5

class LSTM(tf.keras.models.Model):
  """CNN for sentimental analysis."""

  def __init__(self, seq_len, n_columns, if_binary_classification):
    input = tf.keras.layers.Input(shape=(seq_len, n_columns,), dtype=tf.float32)
    layer = input
    layer = tf.keras.layers.LSTM(_HIHDDEN_1, return_sequences=True)(layer)
    layer = tf.keras.layers.Dropout(_DROPOUT_RATE)(layer)

    layer = tf.keras.layers.LSTM(_HIHDDEN_2, return_sequences=False)(layer)
    layer = tf.keras.layers.Dropout(_DROPOUT_RATE)(layer)

    output_dim = 1
    if if_binary_classification:
        output_dim = 2
    output = tf.keras.layers.Dense(output_dim, activation="linear")(layer)

    super(LSTM, self).__init__(inputs=[input], outputs=output)

