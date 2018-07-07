from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _dynamic_pooling(w_embs):
  """Dynamic Pooling layer.

  Given the variable-sized output of the convolution layer,
  the pooling with a fixed pooling kernel size and stride would
  produce variable-sized output, whereas the following fully-connected
  layer expects fixes input layer size.
  Thus we fix the number of pooling units (to 2) and dynamically
  determine the pooling region size on each data point.

  Args:
    w_embs: a input tensor with dimensionality of 1.
  Returns:
    A tensor of size 2.
  """
  # a Lambda layer maintain separate context, so that tf should be imported
  # here.
  import tensorflow as tf
  t = tf.expand_dims(w_embs, 2)
  pool_size = w_embs.shape[1].value / 2
  pooled = tf.keras.backend.pool2d(t, (pool_size, 1), strides=(pool_size, 1), data_format='channels_last')
  return tf.squeeze(pooled, 2)


def _dynamic_pooling_output_shape(input_shape):
  """Output shape for the dynamic pooling layer.

  This function is used for keras Lambda layer to indicate
  the output shape of the dynamic poolic layer.

  Args:
    input_shape: A tuple for the input shape.
  Returns:
    output shape for the dynamic pooling layer.
  """
  shape = list(input_shape)
  assert len(shape) == 2  # only valid for 2D tensors
  shape[1] = 2
  return tuple(shape)


class CNN(tf.keras.models.Model):
  """CNN for sentimental analysis."""

  def __init__(self, channels, winsow_size, hid_dim, class_dim, dropout_rate):
    input = tf.keras.layers.Input(shape=(channels, winsow_size,), dtype=tf.float32)
    layer = input
    layer = tf.keras.layers.BatchNormalization()(layer)

    layer_conv3 = tf.keras.layers.Conv1D(hid_dim, 3, activation="relu")(layer)
    layer_conv3 = tf.keras.layers.Lambda(_dynamic_pooling,
        output_shape=_dynamic_pooling_output_shape)(layer_conv3)
    #layer_conv3 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv3)
    #layer_conv3 = tf.keras.layers.Flatten()(layer_conv3)

    layer_conv4 = tf.keras.layers.Conv1D(hid_dim, 4, activation="relu")(layer)
    layer_conv4 = tf.keras.layers.Lambda(_dynamic_pooling,
        output_shape=_dynamic_pooling_output_shape)(layer_conv4)
    #layer_conv4 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv4)
    #layer_conv2 = tf.keras.layers.Flatten()(layer_conv2)

    layer = tf.keras.layers.concatenate([layer_conv4, layer_conv3], axis=1)
    layer = tf.keras.layers.Flatten()(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Dropout(dropout_rate)(layer)

    output = tf.keras.layers.Dense(class_dim, activation="softmax")(layer)

    super(CNN, self).__init__(inputs=[input], outputs=output)
