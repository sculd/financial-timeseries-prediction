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
from tensorflow.python import debug as tf_debug

tf.logging.set_verbosity(tf.logging.INFO)
h1_size = 250
h2_size = 100
h3_size = 50
# up, down
num_labels = 2
reg_lambda = 0.001

def read_and_decode(filename):
  dataset = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=filename, target_dtype=np.float32, features_dtype=np.float32)
  data, target = dataset.data, dataset.target
  target = np.eye(2)[((np.sign(target) + 1.0)/2.0).astype(int)]
  return data, target

def input_fn(filename, batch_size=100):
  data, target = read_and_decode(filename)
  #dataset_batch = tf.train.batch([dataset], batch_size=batch_size)      
  #return {'inputs': price_histories}, labels
  return {'inputs': data}, target

def get_input_fn(filename, batch_size=100):
  return lambda: input_fn(filename, batch_size)

def _model_fn(features, labels, mode):
  # Input Layer
  input_layer = features['inputs']
  numof_i = input_layer.shape[0]
  i_size = input_layer.shape[1]

  wsdatato1 = tf.Variable(tf.truncated_normal([i_size, h1_size], stddev = 1.0 / math.sqrt(h1_size)), name = 'wsdatato1')
  h1bs = tf.Variable(tf.zeros([h1_size]), name = 'h1bs')
  ws1to2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev = 1.0 / math.sqrt(h2_size)), name = 'ws1to2')
  h2bs = tf.Variable(tf.zeros([h2_size]), name = 'h2bs')
  ws2to3 = tf.Variable(tf.truncated_normal([h2_size, h3_size], stddev = 1.0 / math.sqrt(h3_size)), name = 'ws2to3')
  h3bs = tf.Variable(tf.zeros([h3_size]), name = 'h3bs')
  ws3to4 = tf.Variable(tf.truncated_normal([h3_size, num_labels], stddev = 1.0 / math.sqrt(num_labels)), name = 'ws3to4')
  h4bs = tf.Variable(tf.zeros([num_labels]), name = 'h4bs')

  def model(data):
      h1 = tf.matmul(data, wsdatato1) + h1bs
      h1 = tf.nn.relu(h1)
      h2 = tf.matmul(h1, ws1to2) + h2bs
      h2 = tf.nn.relu(h2)
      h3 = tf.matmul(h2, ws2to3) + h3bs
      h3 = tf.nn.relu(h3)
      h4 = tf.matmul(h3, ws3to4) + h4bs
      return h4

  logits = model(input_layer)

  # Define operations
  if mode in (Modes.INFER, Modes.EVAL):
    probabilities = tf.nn.softmax(logits, name='prediction_probability')
    predicted_indices = tf.argmax(input=logits, axis=1, name='predicted_indices')

  if mode in (Modes.TRAIN, Modes.EVAL):
    global_step = tf.contrib.framework.get_or_create_global_step()
    label_indices = tf.argmax(input=labels, axis=1, name='label_indices')
    loss = 0
    prediction_loss = tf.losses.absolute_difference(labels, predictions=tf.one_hot(tf.argmax(input=logits, axis=1), 2))

    res_loss = 0
    res_loss += reg_lambda * (tf.nn.l2_loss(wsdatato1) + tf.nn.l2_loss(h1bs))
    res_loss += reg_lambda * (tf.nn.l2_loss(ws1to2) + tf.nn.l2_loss(h2bs))
    res_loss += reg_lambda * (tf.nn.l2_loss(ws2to3) + tf.nn.l2_loss(h3bs))
    res_loss += reg_lambda * (tf.nn.l2_loss(ws3to4) + tf.nn.l2_loss(h4bs))
    loss = prediction_loss + res_loss    
    tf.summary.scalar('OptimizeLoss', loss)
    tf.summary.scalar('PredictionLoss', prediction_loss)

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
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.95)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  if mode == Modes.EVAL:
    '''
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
    }
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)      
    '''
    return tf.estimator.EstimatorSpec(mode, loss=prediction_loss)

def build_estimator(model_dir):
  return tf.estimator.Estimator(
      model_fn=_model_fn,
      model_dir=model_dir,
      config=tf.contrib.learn.RunConfig(save_checkpoints_secs=10))

def serving_input_fn():
  inputs = {'inputs': tf.placeholder(tf.float32, [None, 10])}
  print(inputs)
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)
