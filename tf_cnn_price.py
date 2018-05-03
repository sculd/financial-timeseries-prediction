import tensorflow as tf, math, pandas as pd, numpy as np
from collections import deque
import data.read_columns as read_columns
import optimize_model

##########################################################

_WINDOW_SIZE = 8
_NUM_FEATURES = _WINDOW_SIZE
N_CHANNELS = read_columns.N_CHANNELS_HISTORY
_NUM_LABELS = 2 # up or down
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

        def cnn(inputs, filters):
            layer = tf.layers.conv1d(
                inputs,
                filters,
                kernel_size = 2,
                activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0030),
                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002)
                #activity_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001)
                )

            layer = tf.layers.max_pooling1d(
                layer,
                pool_size = 2,
                strides = 2,
                padding='same')

            return layer

        def model(data):
            # (batch, WINDDOW_SIZE, N_CHANNELS) -> (batch, WINDDOW_SIZE/2, 12)
            layer = cnn(data, 12)
            # (batch, WINDDOW_SIZE/2, 12) -> (batch, WINDDOW_SIZE/4, 24)
            layer = cnn(layer, 24)
            # (batch, WINDDOW_SIZE/4, 24) -> (batch, WINDDOW_SIZE/8, 48)
            #layer = cnn(layer, 48)

            # Flatten and add dropout
            # 1/8 * 48 = 6
            layer = tf.reshape(layer, (-1, _WINDOW_SIZE * 6))
            layer = tf.nn.dropout(layer, keep_prob=keep_prob_)

            return layer

        layer = model(inputs)
        pred, logits, cost, accuracy = optimize_model.optimize_classifier(layer, labels, _NUM_LABELS)
        optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost, global_step=global_step)
        #optimizer = tf.contrib.opt.PowerSignOptimizer().minimize(cost, global_step=global_step)

################################################################################################

df, data_all, labels_all, target_all, train_data, train_labels, train_targets, valid_data, valid_labels, valid_targets = read_columns.read_sp500_close_history(window_size = _WINDOW_SIZE)
#train_data, train_labels, train_targets, valid_data, valid_labels, valid_targets = read_columns.generate_random_wak(10000, window_size = _WINDOW_SIZE)

num_batch_steps = 1500 + 1
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
            print('train accuracy: %.3f' % (acc))
            print('test accuracy: %.3f' % (test_acc))
            print('mean prediction in a batch %.2f' % (np.mean(pr)))
            print('mean prediction in the test set %.2f' % (np.mean(pr_test)))

            lgt_exp = np.exp(lgt)
            sft_max = lgt_exp / lgt_exp.sum(axis=1)[:, np.newaxis]
            print('correlation between softmax and future return in train %.4f' % (np.corrcoef(sft_max[:, 1][:3000], train_targets[:3000])[0, 1]))
            test_lgt_exp = np.exp(test_lgt)
            test_sft_max = test_lgt_exp / test_lgt_exp.sum(axis=1)[:, np.newaxis]
            print('correlation between softmax and future return in test %.4f' % (np.corrcoef(test_sft_max[:,1][:3000], valid_targets[:3000])[0,1]))
            print()

    n_na = len(df) - len(data_all)
    acc, pr, lgt = session.run([accuracy, pred, logits], feed_dict={inputs: data_all, labels: labels_all, keep_prob_: keep_prob})

    df_dropped = df.dropna()
    df_dropped['pred'] = pr
    session.close()
    del session

