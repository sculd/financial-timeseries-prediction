import tensorflow as tf, math, pandas as pd, numpy as np
from collections import deque
import data.read_columns as read_columns

##########################################################

_WINDOW_SIZE = 10
_NUM_FEATURES = _WINDOW_SIZE * read_columns.N_CHANNELS_HISTORY
N_CHANNELS = read_columns.N_CHANNELS_HISTORY

NUM_LABELS = 2 # up or down
H1_SIZE = 100
H2_SIZE = 50
H3_SIZE = 20
DEVICE_NAME = "/gpu:0"

graph = tf.Graph()
with tf.device(DEVICE_NAME):
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, [None, _NUM_FEATURES], name='data')
        labels = tf.placeholder(tf.float32, [None, NUM_LABELS], name='labels')
        keep_prob_ = tf.placeholder(tf.float32, name='keep')
        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate_ = tf.train.exponential_decay(0.003, global_step, 1, 0.999, staircase=True)

        def fully_connect(layer, out_size):
            layer = tf.layers.dense(layer, out_size,
                                    activation=tf.nn.relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0040),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002)
                                    )
            layer = tf.nn.dropout(layer, keep_prob=keep_prob_)
            return layer

        # Model.
        def model(layer):
            layer = fully_connect(layer, H1_SIZE)
            layer = fully_connect(layer, H2_SIZE)
            layer = fully_connect(layer, H3_SIZE)
            layer = fully_connect(layer, NUM_LABELS)
            return layer

        logits = model(inputs)
        pred = tf.argmax(logits, 1)

        # Cost function and optimizer
        with tf.name_scope('cost'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            reg_cost = tf.losses.get_regularization_loss()
            cost += reg_cost
        tf.summary.scalar('cost', cost)
        tf.summary.scalar('reg_cost', reg_cost)

        predicted_indices = tf.argmax(input=logits, axis=1)
        predictions = tf.one_hot(predicted_indices, 2)
        label_indices = tf.argmax(input=labels, axis=1)

        #learning_rate = tf.train.exponential_decay(0.001, global_step, 1, 0.99995, staircase=True)
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate_).minimize(cost, global_step=global_step)

        # Accuracy
        correct_pred = tf.equal(pred, tf.argmax(labels, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

################################################################################################


train_data, train_labels, valid_data, valid_labels = read_columns.read_sp500_close_history(window_size = _WINDOW_SIZE, reshape_per_channel = False)

num_batch_steps = 1 * 3000 + 1
batch_size = 100
keep_prob = 0.5

with tf.Session(graph=graph) as session:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_log', session.graph)
    test_writer = tf.summary.FileWriter('test_log')

    tf.global_variables_initializer().run()

    for step in range(num_batch_steps):
        offset = (step * batch_size) % (train_data.shape[0] - batch_size)

        # Generate a minibatch.
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






