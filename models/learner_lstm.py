import tensorflow as tf, math, pandas as pd, numpy as np
from data.read import get_data, look_back, TEST_SIZE_LSTM, TAIL_VALID_SIZE_LSTM, TRAIN_SZIE_LSTM, HEAD_VALID_SIZE_LSTM
from models.learner_common import batchmark_accuracy, accuracy, print_message

# part of the source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/

##########################################################

test_data, test_labels, test_index, tail_valid_data, tail_valid_labels, tv_index, head_valid_data, head_valid_labels, \
    hv_index, train_data, train_labels, train_index = get_data(TEST_SIZE_LSTM, TAIL_VALID_SIZE_LSTM, TRAIN_SZIE_LSTM, HEAD_VALID_SIZE_LSTM)

test_data, test_labels, tail_valid_data, tail_valid_labels, head_valid_data, head_valid_labels, train_data, train_labels = \
    test_data.as_matrix(), test_labels.as_matrix(), tail_valid_data.as_matrix(), \
    tail_valid_labels.as_matrix(), head_valid_data.as_matrix(), head_valid_labels.as_matrix(), train_data.as_matrix(), train_labels.as_matrix()

##########################################################

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
batch_size = 500
reg_lambda = 0.01
num_steps = 10001

NODE_SIZE = 100
# up, down
num_labels = 2

device_name = "/gpu:0"

graph = tf.Graph()
with tf.device(device_name):
    with graph.as_default():
        data = tf.placeholder(tf.float32, [None, look_back + 0, 1])
        target = tf.placeholder(tf.float32, [None, num_labels])

        # lstm element
        cell = tf.contrib.rnn.LSTMCell(NODE_SIZE, state_is_tuple=True)
        val, _ = tf.nn.dynamic_rnn(cell, data, dtype = tf.float32)
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, look_back - 1)

        # lstm layout
        weights = tf.Variable(tf.truncated_normal([NODE_SIZE, num_labels], stddev = 1.0 / math.sqrt(num_labels)))
        biases = tf.Variable(tf.zeros([num_labels]))

        prediction = tf.matmul(last, weights) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=prediction))
        loss += reg_lambda * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases))

        # optimize
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)
        #global_step = tf.Variable(0)  # count the number of steps taken.
        #learning_rate = tf.train.exponential_decay(0.05, global_step, 1, 0.99995, staircase=True)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)


        ########################################################################

        def reshape(data):
            return data.reshape(tuple(list(data.shape) + [1]))

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            for step in range(num_steps):
                offset = (step * batch_size) % (train_data.shape[0] - batch_size)
                
                # Generate a minibatch.
                batch_data = train_data[offset : (offset + batch_size)]
                batch_labels = train_labels[offset : (offset + batch_size)]
                
                _, predictions = session.run([optimizer, prediction], feed_dict = {data: reshape(batch_data), target: batch_labels})

                if (step % 500 == 0):
                    print("at step %d" % step)
                    pred_tail_valid = session.run(prediction, feed_dict = {data: reshape(tail_valid_data), target: tail_valid_labels})
                    pred_head_valid = session.run(prediction, feed_dict = {data: reshape(head_valid_data), target: head_valid_labels})
                    pred_test = session.run(prediction, feed_dict = {data: reshape(test_data), target: test_labels})

                    print_message('batch', accuracy(predictions, batch_labels), batch_labels)
                    print_message('validation (tail)', accuracy(pred_tail_valid, tail_valid_labels), tail_valid_labels)
                    print_message('validation (head)', accuracy(pred_head_valid, head_valid_labels), head_valid_labels)
                    print_message('test', accuracy(pred_test, test_labels), test_labels)
                    print()


            def pred_save(mark, prices, labels, index):
                pred = session.run(prediction, feed_dict = {data: prices, target: labels})
                pred_df = pd.DataFrame(data=pred, index=index, columns = ['up', 'down'])
                pred_df.to_csv('predictions/pred_' + mark + '.csv')

            pred_save('train', train_data, train_labels, train_index)
            pred_save('valid_tail', tail_valid_data, tail_valid_labels, tv_index)
            pred_save('valid_head', head_valid_data, head_valid_labels, hv_index)
            pred_save('test', test_data, test_labels, test_index)

            session.close()
            del session



