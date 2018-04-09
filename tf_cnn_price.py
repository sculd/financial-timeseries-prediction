import tensorflow as tf, math, pandas as pd, numpy as np
from collections import deque
import data.read_columns as read_columns

##########################################################

WINDDOW_SIZE = 64
N_CHANNELS = 1
NUM_LABELS = 2 # up or down
H1_SIZE = 128 * N_CHANNELS
reg_lambda = 0.0005
DEVICE_NAME = "/gpu:0"

graph = tf.Graph()
with tf.device(DEVICE_NAME):
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, [None, WINDDOW_SIZE, N_CHANNELS], name='inputs')
        labels = tf.placeholder(tf.float32, [None, 2], name='labels')
        keep_prob_ = tf.placeholder(tf.float32, name='keep')
        learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

        wdatato1 = tf.Variable(tf.truncated_normal([WINDDOW_SIZE, N_CHANNELS, N_CHANNELS * H1_SIZE], stddev=1.0 / math.sqrt(H1_SIZE)))
        b1 = tf.Variable(tf.zeros([N_CHANNELS * H1_SIZE]))

        def fully_connect(inputs, w, b):
            h = tf.matmul(inputs, w) + b
            h = tf.nn.leaky_relu(h)
            h = tf.nn.dropout(h, 0.5)
            return h

        def cnn(inputs, filters):
            layer = tf.layers.conv1d(
                inputs,
                filters,
                kernel_size = 2,
                activation=tf.nn.relu)

            layer = tf.layers.max_pooling1d(
                layer,
                pool_size = 2,
                strides = 2,
                padding='same')

            return layer

        def model(data):
            # (batch, WINDDOW_SIZE, N_CHANNELS) -> (batch, H1_SIZE)
            layer = fully_connect(data, wdatato1, b1)
            # (batch, H1_SIZE) -> (batch, H1_SIZE/2, 12)
            layer = cnn(layer, 12)
            # (batch, H1_SIZE/2, 12) -> (batch, H1_SIZE/4, 24)
            layer = cnn(layer, 24)
            # (batch, H1_SIZE/4, 24) -> (batch, H1_SIZE/8, 48)
            layer = cnn(layer, 48)

            # Flatten and add dropout
            flat = tf.reshape(layer, (-1, H1_SIZE * 6))
            flat = tf.nn.dropout(flat, keep_prob=keep_prob_)

            return flat

        layer = model(inputs)
        # Predictions
        logits = tf.layers.dense(layer, NUM_LABELS)

        # Cost function and optimizer
        with tf.name_scope('cost'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        tf.summary.scalar('cost', cost)

        optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

################################################################################################

train_data, train_labels, valid_data, valid_labels = read_columns.read_goog_close(window_size = WINDDOW_SIZE)

num_steps = 1 * 2000 + 1
batch_size = 100

batch_scores_size = 160
with tf.Session(graph=graph) as session:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_log', session.graph)
    test_writer = tf.summary.FileWriter('test_log')

    tf.global_variables_initializer().run()
    prev_weights = None
    bench_scores = deque([], batch_scores_size)

    for step in range(num_steps):
        offset = (step * batch_size) % (train_data.shape[0] - batch_size)

        batch_data = train_data[offset:(offset + batch_size)]
        batch_labels = train_labels[offset:(offset + batch_size)]

        feed_dict = {inputs: batch_data, labels: batch_labels}
        summary, opt, acc = session.run([merged, optimizer, accuracy], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        if (step % 100 == 0 or step == num_steps - 1):
            print('step %d' % (step))
            print('train accuracy: %.2f' % (acc))
            summary, opt, acc = session.run([merged, optimizer, accuracy],
                                                feed_dict = {inputs: valid_data, labels: valid_labels})
            print('test accuracy: %.2f' % (acc))
            test_writer.add_summary(summary, step)
            print()

    session.close()
    del session

