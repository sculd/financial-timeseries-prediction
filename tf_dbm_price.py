import tensorflow as tf, math, pandas as pd, numpy as np
from collections import deque
import data.read_columns as read_columns

##########################################################

WINDDOW_SIZE = 30
NUM_LABELS = 2 # up or down
H1_SIZE = 200
H2_SIZE = 100
H3_SIZE = 50
H4_SIZE = 20
reg_lambda = 0.0001
DEVICE_NAME = "/gpu:0"

graph = tf.Graph()
with tf.device(DEVICE_NAME):
    with graph.as_default():
        input_layer = tf.placeholder(tf.float32, [None, WINDDOW_SIZE], name='data')
        labels = tf.placeholder(tf.float32, [None, NUM_LABELS], name='labels')

        wdatato1 = tf.Variable(tf.truncated_normal([WINDDOW_SIZE, H1_SIZE], stddev=1.0 / math.sqrt(H1_SIZE)))
        bh1 = tf.Variable(tf.zeros([H1_SIZE]))
        w1to2 = tf.Variable(tf.truncated_normal([H1_SIZE, H2_SIZE], stddev=1.0 / math.sqrt(H2_SIZE)))
        bh2 = tf.Variable(tf.zeros([H2_SIZE]))
        w2to3 = tf.Variable(tf.truncated_normal([H2_SIZE, H3_SIZE], stddev=1.0 / math.sqrt(H3_SIZE)))
        bh3 = tf.Variable(tf.zeros([H3_SIZE]))
        w3to4 = tf.Variable(tf.truncated_normal([H3_SIZE, H4_SIZE], stddev=1.0 / math.sqrt(H4_SIZE)))
        bh4 = tf.Variable(tf.zeros([H4_SIZE]))
        w4to5 = tf.Variable(tf.truncated_normal([H4_SIZE, NUM_LABELS], stddev=1.0 / math.sqrt(NUM_LABELS)))
        bh5 = tf.Variable(tf.zeros([NUM_LABELS]))

        def fully_connect(inputs, w, b):
            h = tf.matmul(inputs, w) + b
            h = tf.nn.leaky_relu(h)
            h = tf.nn.dropout(h, 0.5)
            return h

        # Model.
        def model(data):
            h1 = fully_connect(data, wdatato1, bh1)
            h2 = fully_connect(h1, w1to2, bh2)
            h3 = fully_connect(h2, w2to3, bh3)
            h4 = fully_connect(h3, w3to4, bh4)
            h5 = fully_connect(h4, w4to5, bh5)
            return h5

        logits = model(input_layer)
        predicted_indices = tf.argmax(input=logits, axis=1)
        predictions = tf.one_hot(predicted_indices, 2)
        label_indices = tf.argmax(input=labels, axis=1)
        with tf.name_scope('loss'):
            loss = tf.losses.absolute_difference(labels, predictions=predictions)
            reg_loss = 0
            ts = [wdatato1, bh1, w1to2, bh2, w2to3, bh3, w3to4, bh4, w4to5, bh5]
            for t in ts:
                reg_loss += reg_lambda * (tf.nn.l2_loss(t))

        tf.summary.scalar('prediction_loss', loss)
        tf.summary.scalar('reg_loss', reg_loss)

        with tf.name_scope('loss'):
            loss += reg_loss
        tf.summary.scalar('total_loss', loss)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        global_step = tf.Variable(0)  # count the number of steps taken.
        #learning_rate = tf.train.exponential_decay(0.001, global_step, 1, 0.99995, staircase=True)
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(loss, global_step=global_step)

################################################################################################


train_data, train_labels, valid_data, valid_labels = read_columns.read_goog_close(window_size = WINDDOW_SIZE)

num_steps = 1 * 6000 + 1
batch_size = 50

with tf.Session(graph=graph) as session:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_log', session.graph)
    test_writer = tf.summary.FileWriter('test_log')

    tf.global_variables_initializer().run()

    for step in range(num_steps):
        offset = (step * batch_size) % (train_data.shape[0] - batch_size)

        # Generate a minibatch.
        batch_data = train_data[offset:(offset + batch_size)]
        batch_labels = train_labels[offset:(offset + batch_size)]

        feed_dict = {input_layer: batch_data, labels: batch_labels}
        summary, opt, lgt, rl, l, acc = session.run([merged, optimizer, logits, reg_loss, loss, accuracy], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        if (step % 100 == 0 or step == num_steps - 1):
            print('step %d' % (step))
            print('train accuracy: %.2f' % (acc))
            summary, opt, lgt, rl, l, acc = session.run([merged, optimizer, logits, reg_loss, loss, accuracy],
                                                        feed_dict = {input_layer: valid_data, labels: valid_labels})
            print('test accuracy: %.2f' % (acc))
            test_writer.add_summary(summary, step)
            print()

    session.close()
    del session




