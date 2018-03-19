import tensorflow as tf, math, pandas as pd, numpy as np, os, boto3
from collections import deque
from trainer import model

##########################################################

look_back = 10
h1_size = 250
h2_size = 100
h3_size = 50
# up, down
num_labels = 2
reg_lambda = 0.0005

device_name = "/gpu:0"

graph = tf.Graph()
with tf.device(device_name):
    with graph.as_default():
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        input_layer = tf.placeholder(tf.float32, [None, look_back + 0], name = 'data')
        labels = tf.placeholder(tf.float32, [None, num_labels], name = 'labels')

        # Variables.
        # These are the parameters that we are going to be training.
        # past prices and volume
        wsdatato1 = tf.Variable(tf.truncated_normal([look_back + 0, h1_size], stddev = 1.0 / math.sqrt(h1_size)))
        h1bs = tf.Variable(tf.zeros([h1_size]))
        ws1to2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev = 1.0 / math.sqrt(h2_size)))
        h2bs = tf.Variable(tf.zeros([h2_size]))
        ws2to3 = tf.Variable(tf.truncated_normal([h2_size, h3_size], stddev = 1.0 / math.sqrt(h3_size)))
        h3bs = tf.Variable(tf.zeros([h3_size]))
        ws3to4 = tf.Variable(tf.truncated_normal([h3_size, num_labels], stddev = 1.0 / math.sqrt(num_labels)))
        h4bs = tf.Variable(tf.zeros([num_labels]))

        # Model.
        def model(data):
            h1 = tf.matmul(data, wsdatato1) + h1bs
            h1 = tf.nn.relu(h1)
            h2 = tf.matmul(h1, ws1to2) + h2bs
            h2 = tf.nn.relu(h2)
            h3 = tf.matmul(h2, ws2to3) + h3bs
            h3 = tf.nn.relu(h3)
            h4 = tf.matmul(h3, ws3to4) + h4bs
            return h4

        # Training computation.
        logits = model(input_layer)
        predicted_indices = tf.argmax(input=logits, axis=1)
        predictions = tf.one_hot(predicted_indices, 2)
        label_indices = tf.argmax(input=labels, axis=1)
        loss = tf.losses.absolute_difference(labels, predictions=predictions)

        res_loss = 0
        res_loss += reg_lambda * (tf.nn.l2_loss(wsdatato1) + tf.nn.l2_loss(h1bs))
        res_loss += reg_lambda * (tf.nn.l2_loss(ws1to2) + tf.nn.l2_loss(h2bs))
        res_loss += reg_lambda * (tf.nn.l2_loss(ws2to3) + tf.nn.l2_loss(h3bs))
        res_loss += reg_lambda * (tf.nn.l2_loss(ws3to4) + tf.nn.l2_loss(h4bs))
        loss += res_loss

        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        #optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(0.001, global_step, 1, 0.99995, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.

################################################################################################

def read_and_decode(filename):
  dataset = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=filename, target_dtype=np.float32, features_dtype=np.float32)
  return dataset

def print_message(text_mark, acc, target):
    print(text_mark + " accuracy: %.1f%%, benchmark: %.1f%%" % (acc, batchmark_accuracy(target)))

train_dataset = read_and_decode('./GOOG_series_train.csv')
valid_dataset = read_and_decode('./GOOG_series_validation.csv')
train_data, train_labels = train_dataset.data, train_dataset.target
train_labels = np.eye(2)[((np.sign(train_labels) + 1.0)/2.0).astype(int)]
valid_data, valid_labels = valid_dataset.data, valid_dataset.target
valid_labels = np.eye(2)[((np.sign(valid_labels) + 1.0)/2.0).astype(int)]

num_steps = 1 * 2000 + 1
# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
batch_size = 50

batch_scores_size = 160
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    prev_weights = None
    bench_scores = deque([], batch_scores_size)

    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_data.shape[0] - batch_size)
        
        # Generate a minibatch.
        batch_data = train_data[offset:(offset + batch_size)]
        batch_labels = train_labels[offset:(offset + batch_size)]
        
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph
        feed_dict = {input_layer : batch_data, labels : batch_labels}
        _, lgt, rl, l = session.run([optimizer, logits, res_loss, loss], 
          feed_dict=feed_dict)

        if (step % 100 == 0 or step == num_steps - 1):
            print("at step %d" % step)
            print("Minibatch loss: %f" % l)
            print("loss from prediction error: %f" % (l - rl))

            print('')

    session.close()
    del session




