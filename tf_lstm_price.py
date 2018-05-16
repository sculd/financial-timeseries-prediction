import tensorflow as tf, math, pandas as pd, numpy as np
from collections import deque
import data.read_columns as read_columns
import optimize_model

##########################################################

_WINDOW_SIZE = 16
_NUM_FEATURES = _WINDOW_SIZE / read_columns.WINDOW_STEP
N_CHANNELS = read_columns.N_CHANNELS_HISTORY + 3
_LSTM_CELL_SIZE = 100

DEVICE_NAME = "/gpu:0"

graph = tf.Graph()
with tf.device(DEVICE_NAME):
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, [None, _NUM_FEATURES, N_CHANNELS], name='inputs')
        labels = tf.placeholder(tf.float32, [None, read_columns.NUM_LABELS], name='labels')
        keep_prob_ = tf.placeholder(tf.float32, name='keep')
        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate_ = tf.train.exponential_decay(0.003, global_step, 1, 0.999, staircase=True)

        def model(data):
            # lstm element
            cell = tf.contrib.rnn.LSTMCell(_LSTM_CELL_SIZE, state_is_tuple=True)
            val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
            val = tf.transpose(val, [1, 0, 2])
            layer = tf.gather(val, _NUM_FEATURES - 1)
            layer = tf.nn.dropout(layer, keep_prob=keep_prob_)

            return layer

        layer = model(inputs)
        pred, logits, total_cost, accuracy = optimize_model.optimize_classifier(layer, labels, read_columns.NUM_LABELS)
        #optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(total_cost, global_step=global_step)
        optimizer = tf.contrib.opt.PowerSignOptimizer().minimize(total_cost, global_step=global_step)

################################################################################################

df, data_all, labels_all, target_all, train_data, train_labels, train_targets, valid_data, valid_labels, valid_targets = read_columns.read_sp500_ohlc_history(window_size = _WINDOW_SIZE)
#train_data, train_labels, train_targets, valid_data, valid_labels, valid_targets = read_columns.generate_random_wak(10000, window_size = _WINDOW_SIZE)

num_batch_steps = 15 * 100 + 1
batch_size = 100
keep_prob = 0.4

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
        opt = session.run(optimizer, feed_dict=feed_dict)

        if (step % 10 == 0 or step == num_batch_steps - 1):
            summary = session.run(merged, feed_dict={inputs: train_data, labels: train_labels, keep_prob_: keep_prob})
            train_writer.add_summary(summary, step)

            test_summary = session.run(merged, feed_dict = {inputs: valid_data, labels: valid_labels, keep_prob_: keep_prob})
            test_writer.add_summary(test_summary, step)

        if (step % 100 == 0 or step == num_batch_steps - 1):
            acc, pr, lgt = session.run([accuracy, pred, logits], feed_dict={inputs: train_data, labels: train_labels, keep_prob_: keep_prob})
            test_acc, pr_test, test_lgt = session.run([accuracy, pred, logits], feed_dict = {inputs: valid_data, labels: valid_labels, keep_prob_: keep_prob})

            print('step %d' % (step))
            print('train accuracy vs benchmark: %.3f vs %.3f' % (acc, np.mean(train_labels[:, 1])))
            print('test accuracy vs benchmark: %.3f vs %.3f' % (test_acc, np.mean(valid_labels[:, 1])))
            print('mean prediction in a batch %.2f' % (np.mean(pr)))
            print('mean prediction in the test set %.2f' % (np.mean(pr_test)))

            lgt_exp = np.exp(lgt)
            sft_max = lgt_exp / lgt_exp.sum(axis=1)[:, np.newaxis]
            print('correlation between softmax and future return in train %.4f' % (np.corrcoef(sft_max[:, 1][:3000], train_targets[:3000,0])[0, 1]))
            test_lgt_exp = np.exp(test_lgt)
            test_sft_max = test_lgt_exp / test_lgt_exp.sum(axis=1)[:, np.newaxis]
            print('correlation between softmax and future return in test %.4f' % (np.corrcoef(test_sft_max[:,1][:3000], valid_targets[:3000,0])[0,1]))

            print()

    session.close()
    del session

