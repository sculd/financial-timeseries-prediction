# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf, numpy as np, math
from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModeKeys as Modes

tf.logging.set_verbosity(tf.logging.INFO)
_LOOK_BACK = 9
_H1_SIZE = 200
_H2_SIZE = 2

def read_and_decode(filename):
  dataset = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=filename, target_dtype=np.float32, features_dtype=np.float32)
  return dataset

def input_fn(filename, batch_size=100):
  dataset = read_and_decode(filename)
  #dataset_batch = tf.train.batch([dataset], batch_size=batch_size)      
  #return {'inputs': price_histories}, labels
  return dataset

def get_input_fn(filename, batch_size=100):
  return lambda: input_fn(filename, batch_size)

def _cnn_model_fn(features, labels, mode):
  # Input Layer

  input_layer = features.data

  wsto1 = tf.Variable(tf.truncated_normal([_LOOK_BACK + 0, _H1_SIZE], stddev = 1.0 / math.sqrt(_H1_SIZE)))
  b1 = tf.Variable(tf.zeros([_H1_SIZE]))
  ws1to2 = tf.Variable(tf.truncated_normal([_H1_SIZE, _H2_SIZE], stddev = 1.0 / math.sqrt(_H2_SIZE)))
  b2 = tf.Variable(tf.zeros([_H2_SIZE]))

  h1 = tf.matmul(input_layer, wsto1) + b1
  h1 = tf.nn.relu(h1)
  h2 = tf.matmul(h1, ws1to2) + b2
  h2 = tf.nn.relu(h2)
  output_layer = h2

  # Logits Layer
  logits = output_layer

  # Define operations
  if mode in (Modes.INFER, Modes.EVAL):
    predicted_indices = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits, name='softmax_tensor')

  if mode in (Modes.TRAIN, Modes.EVAL):
    global_step = tf.contrib.framework.get_or_create_global_step()
    label_indices = features.target
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=tf.one_hot(label_indices, depth=10), logits=logits)
    tf.summary.scalar('OptimizeLoss', loss)

  if mode == Modes.INFER:
    predictions = {
        'classes': predicted_indices,
        'probabilities': probabilities
    }
    export_outputs = {
        'prediction': tf.estimator.export.PredictOutput(predictions)
    }
    return tf.estimator.EstimatorSpec(
        mode, predictions=predictions, export_outputs=export_outputs)

  if mode == Modes.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  if mode == Modes.EVAL:
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
    }
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops)


def build_estimator(model_dir):
  return tf.estimator.Estimator(
      model_fn=_cnn_model_fn,
      model_dir=model_dir,
      config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180))


def serving_input_fn():
  inputs = {'inputs': tf.placeholder(tf.float32, [None, 12])}
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)
