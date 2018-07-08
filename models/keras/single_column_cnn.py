from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_DROPOUT_RATE = 0.5

class CNN(tf.keras.models.Model):
  """CNN for sentimental analysis."""

  def __init__(self, seq_len, if_binary_classification):
    input = tf.keras.layers.Input(shape=(seq_len, 1,), dtype=tf.float32)
    layer = input
    layer = tf.keras.layers.BatchNormalization()(layer)

    layer_conv3 = tf.keras.layers.Conv1D(128, 3, activation="relu")(layer)
    layer_conv3 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv3)
    layer_conv3 = tf.keras.layers.Flatten()(layer_conv3)

    layer_conv4 = tf.keras.layers.Conv1D(128, 4, activation="relu")(layer)
    layer_conv4 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv4)
    layer_conv4 = tf.keras.layers.Flatten()(layer_conv4)

    layer_conv5 = tf.keras.layers.Conv1D(128, 5, activation="relu")(layer)
    layer_conv5 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv5)
    layer_conv5 = tf.keras.layers.Flatten()(layer_conv5)

    layer = tf.keras.layers.concatenate([layer_conv3, layer_conv4, layer_conv5], axis=1)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Dropout(_DROPOUT_RATE)(layer)

    output_dim = 1
    if if_binary_classification:
        output_dim = 2
    output = tf.keras.layers.Dense(output_dim, activation="softmax")(layer)

    super(CNN, self).__init__(inputs=[input], outputs=output)
