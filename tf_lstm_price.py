import tensorflow as tf, math, pandas as pd, numpy as np
from collections import deque
import data.read_columns as read_columns

##########################################################

_WINDOW_SIZE = 4
_NUM_FEATURES = _WINDOW_SIZE
N_CHANNELS = read_columns.N_CHANNELS_HISTORY
_LSTM_CELL_SIZE = 100
NUM_LABELS = 2 # up or down
reg_lambda = 0.0005

DEVICE_NAME = "/gpu:0"

graph = tf.Graph()
with tf.device(DEVICE_NAME):
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, [None, _NUM_FEATURES, N_CHANNELS], name='inputs')
        labels = tf.placeholder(tf.float32, [None, 2], name='labels')
        keep_prob_ = tf.placeholder(tf.float32, name='keep')
        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate_ = tf.train.exponential_decay(0.003, global_step, 1, 0.999, staircase=True)

        def model(data):
            # lstm element
            cell = tf.contrib.rnn.LSTMCell(_LSTM_CELL_SIZE, state_is_tuple=True)
            val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
            val = tf.transpose(val, [1, 0, 2])
            layer = tf.gather(val, _WINDOW_SIZE - 1)
            layer = tf.nn.dropout(layer, keep_prob=keep_prob_)

            return layer

        layer = model(inputs)
        # Predictions
        logits = tf.layers.dense(layer, NUM_LABELS)
        pred = tf.argmax(logits, 1)

        # Cost function and optimizer
        with tf.name_scope('cost'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            reg_cost = tf.losses.get_regularization_loss()
            cost += reg_cost
        tf.summary.scalar('cost', cost)
        tf.summary.scalar('reg_cost', reg_cost)

        optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost, global_step=global_step)

        # Accuracy
        correct_pred = tf.equal(pred, tf.argmax(labels, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

################################################################################################

train_data, train_labels, valid_data, valid_labels = read_columns.read_bitstamp_btcusd_2017_hourly_history(window_size = _WINDOW_SIZE)

num_batch_steps = 1 * 3000 + 1
batch_size = 100
keep_prob = 0.5

with tf.Session(graph=graph) as session:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_log', session.graph)
    test_writer = tf.summary.FileWriter('test_log')

    tf.global_variables_initializer().run()
    prev_weights = None

    for step in range(num_batch_steps):
        offset = (step * batch_size) % (train_data.shape[0] - batch_size)

        batch_data = train_data[offset:(offset + batch_size)]
        batch_labels = train_labels[offset:(offset + batch_size)]

        feed_dict = {inputs: batch_data, labels: batch_labels, keep_prob_: keep_prob}
        summary, opt, acc = session.run([merged, optimizer, accuracy], feed_dict=feed_dict)

        if (step % 10 == 0 or step == num_batch_steps - 1):
            summary, acc = session.run([merged, accuracy],
                                            feed_dict={inputs: train_data, labels: train_labels, keep_prob_: keep_prob})
            train_writer.add_summary(summary, step)

            test_summary, test_acc = session.run([merged,  accuracy],
                                            feed_dict = {inputs: valid_data, labels: valid_labels, keep_prob_: keep_prob})
            test_writer.add_summary(test_summary, step)


        if (step % 100 == 0 or step == num_batch_steps - 1):
            print('step %d' % (step))
            print('train accuracy: %.2f' % (acc))
            print('test accuracy: %.2f' % (test_acc))
            pr = session.run([pred], feed_dict={inputs: batch_data, labels: batch_labels, keep_prob_: keep_prob})
            print('mean prediction in a batch', np.mean(pr))
            print()

    session.close()
    del session

