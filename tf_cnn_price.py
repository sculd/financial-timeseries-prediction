import tensorflow as tf, math, pandas as pd, numpy as np
import data.read_columns as read_columns
import optimize_model
from tensorflow.python import debug as tf_debug

##########################################################

_WINDOW_SIZE = 64
_NUM_FEATURES = _WINDOW_SIZE / read_columns.WINDOW_STEP
N_CHANNELS = read_columns.N_CHANNELS_HISTORY# + 3

DEVICE_NAME = "/gpu:0"

graph = tf.Graph()
with tf.device(DEVICE_NAME):
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, [None, _NUM_FEATURES, N_CHANNELS], name='inputs')
        labels = tf.placeholder(tf.float32, [None, read_columns.NUM_LABELS], name='labels')
        targets = tf.placeholder(tf.float32, [None, 1], name='targets')
        keep_prob_ = tf.placeholder(tf.float32, name='keep')
        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate_ = tf.train.exponential_decay(0.0008, global_step, 1, 1.0 - 0.0003, staircase=True)
        learning_rate_regr_ = tf.train.exponential_decay(0.001, global_step, 1, 1.0 - 0.0003, staircase=True)
        with tf.name_scope('optimizer'):
            tf.summary.scalar('learning_rate', learning_rate_)
            tf.summary.scalar('learning_rate_regression', learning_rate_regr_)

        n_cnn = 0
        def cnn(inputs, filters, kernel_size, pool_size):
            global n_cnn
            n_cnn += 1
            layer = tf.layers.conv1d(
                inputs,
                filters,
                kernel_size = kernel_size,
                padding = "same",
                name = "cnn_%d" % (n_cnn),
                activation=tf.nn.leaky_relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.002),
                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.002)
                #activity_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005)
                )

            if pool_size > 1:
                layer = tf.layers.max_pooling1d(
                    layer,
                    pool_size = pool_size,
                    strides = pool_size,
                    padding='same',
                    name = "pooling_%d" % (n_cnn))

            layer = tf.layers.batch_normalization(layer, name = "batch_norm_%d" % (n_cnn))
            layer = tf.nn.dropout(layer, keep_prob=keep_prob_, name = "dropout_%d" % (n_cnn))

            return layer

        def model(data):
            '''
            '''
            # (batch, _NUM_FEATURES, N_CHANNELS) -> (batch, _NUM_FEATURES/2, 12)
            layer = cnn(data, 96, 8, 2)
            # -> (batch, _NUM_FEATURES/4, 24)
            layer = cnn(layer, 96, 8, 2)
            # -> (batch, _NUM_FEATURES/8, 48)
            layer = cnn(layer, 96, 8, 2)
            # -> (batch, _NUM_FEATURES/16, 96)
            #layer = cnn(layer, 96, 8, 2)

            '''
            # (batch, _NUM_FEATURES, N_CHANNELS) -> (batch, _NUM_FEATURES, 12)
            layer = cnn(data, 12, 8, 1)
            # -> (batch, _NUM_FEATURES, 24)
            layer = cnn(layer, 24, 8, 1)
            # -> (batch, _NUM_FEATURES, 48)
            layer = cnn(layer, 48, 8, 1)
            '''

            # Flatten and add dropout
            # 1/8 * 48 = 6
            # 1/16 * 64 = 4
            layer = tf.reshape(layer, (-1, 12 * _NUM_FEATURES))

            return layer

        layer = model(inputs)
        pred, logits, total_cost, accuracy = optimize_model.optimize_classifier(layer, labels, read_columns.NUM_LABELS)
        pred_regressor, total_cost_regressor = optimize_model.optimize_regressor(layer, targets)

        optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(total_cost, global_step=global_step)
        optimizer_regression = tf.train.AdamOptimizer(learning_rate_regr_).minimize(total_cost_regressor, global_step=global_step)

################################################################################################

#df, data_all, labels_all, target_all, train_data, train_labels, train_targets, valid_data, valid_labels, valid_targets = read_columns.read_sp500_ohlc_history(window_size = _WINDOW_SIZE)
df, data_all, labels_all, target_all, train_data, train_labels, train_targets, valid_data, valid_labels, valid_targets = read_columns.read_goog_close(window_size = _WINDOW_SIZE)
#train_data, train_labels, train_targets, valid_data, valid_labels, valid_targets = read_columns.generate_random_wak(10000, window_size = _WINDOW_SIZE)

num_batch_steps = 5000 + 1
batch_size = 90
keep_prob = 0.50

with tf.Session(graph=graph) as session:
    session = tf_debug.LocalCLIDebugWrapperSession(session)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_log', session.graph)
    test_writer = tf.summary.FileWriter('test_log')

    tf.global_variables_initializer().run()
    prev_weights = None

    for step in range(num_batch_steps):
        offset = (step * batch_size) % (train_data.shape[0] - batch_size)

        batch_data = train_data[offset:(offset + batch_size)]
        batch_labels = train_labels[offset:(offset + batch_size)]
        batch_targets = train_targets[offset:(offset + batch_size)]

        feed_dict = {inputs: batch_data, labels: batch_labels, targets: batch_targets, keep_prob_: keep_prob}
        opt = session.run(optimizer, feed_dict=feed_dict)

        if (step % 20 == 0 or step == num_batch_steps - 1):
            summary = session.run(merged, feed_dict={inputs: train_data, labels: train_labels, targets: train_targets, keep_prob_: keep_prob})
            train_writer.add_summary(summary, step)

            test_summary = session.run(merged, feed_dict = {inputs: valid_data, labels: valid_labels, targets: valid_targets, keep_prob_: keep_prob})
            test_writer.add_summary(test_summary, step)

        if (step % 500 == 0 or step == num_batch_steps - 1):
            acc, pr, lgt = session.run([accuracy, pred, logits], feed_dict={inputs: train_data, labels: train_labels, keep_prob_: keep_prob})
            test_acc, pr_test, test_lgt = session.run([accuracy, pred, logits], feed_dict = {inputs: valid_data, labels: valid_labels, keep_prob_: keep_prob})
            all_acc, pr_all, all_lgt = session.run([accuracy, pred, logits], feed_dict = {inputs: data_all, labels: labels_all, keep_prob_: keep_prob})

            print('step %d' % (step))
            print('train accuracy vs benchmark: %.3f vs %.3f' % (acc, np.mean(train_labels[:, 1])))
            print('test accuracy vs benchmark: %.3f vs %.3f' % (test_acc, np.mean(valid_labels[:, 1])))
            print('whole data accuracy: %.3f' % (all_acc))
            print('mean prediction in a batch %.2f' % (np.mean(pr)))
            print('mean prediction in the test set %.2f' % (np.mean(pr_test)))

            lgt_exp = np.exp(lgt)
            sft_max = lgt_exp / lgt_exp.sum(axis=1)[:, np.newaxis]
            print('correlation between softmax and future return in train %.4f' % (np.corrcoef(sft_max[:, 1][:3000], train_targets[:3000,0])[0, 1]))
            test_lgt_exp = np.exp(test_lgt)
            test_sft_max = test_lgt_exp / test_lgt_exp.sum(axis=1)[:, np.newaxis]
            print('correlation between softmax and future return in test %.4f' % (np.corrcoef(test_sft_max[:,1][:3000], valid_targets[:3000,0])[0,1]))

            pr_regr, tcst_regr = session.run([pred_regressor, total_cost_regressor], feed_dict={inputs: train_data, labels: train_labels, targets: train_targets, keep_prob_: keep_prob})
            test_pr_regr, test_tcst_regr = session.run([pred_regressor, total_cost_regressor], feed_dict = {inputs: valid_data, labels: valid_labels, targets: valid_targets, keep_prob_: keep_prob})
            print('for regression')
            print('mean cost in the training set %.2f' % (np.mean(tcst_regr)))
            print('mean cost in the test set %.2f' % (np.mean(test_tcst_regr)))
            print('mean prediction in the training set %.2f' % (np.mean(pr_regr)))
            print('mean prediction in the test set %.2f' % (np.mean(test_pr_regr)))

            print('classification accuracy from the regression on training %.2f' % (np.mean((train_labels[:, 1] > 0) == (pr_regr[:, 0] > 0))))
            print('classification accuracy from the regression on test %.2f' % (np.mean((valid_labels[:, 1] > 0) == (test_pr_regr[:, 0] > 0))))

            print()

    '''
    n_na = len(df) - len(data_all)
    acc, pr, lgt = session.run([accuracy, pred, logits], feed_dict={inputs: data_all, labels: labels_all, keep_prob_: keep_prob})

    df_dropped = df.dropna()
    df_dropped['pred'] = pr
    '''
    session.close()
    del session

