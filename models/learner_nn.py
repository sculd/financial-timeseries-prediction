import tensorflow as tf, math, pandas as pd, numpy as np, os, boto3
from collections import deque
from data.read import get_data, look_back, TEST_SIZE_NN, TAIL_VALID_SIZE_NN, TRAIN_SZIE_NN, HEAD_VALID_SIZE_NN, get_decorated
from models.learner_common import batchmark_accuracy, accuracy, print_message
from models.manager import MODEL_SAVE_PATH_BASE, BUCKET
from models.nn import MODEL_SAVE_NAME

decorated = get_decorated()
MODEL_SAVE_PATH = MODEL_SAVE_PATH_BASE + MODEL_SAVE_NAME + '/'

##########################################################

test_data, test_labels, test_index, tail_valid_data, tail_valid_labels, tv_index, head_valid_data, head_valid_labels, \
    hv_index, train_data, train_labels, train_index = get_data(TEST_SIZE_NN, TAIL_VALID_SIZE_NN, HEAD_VALID_SIZE_NN, TRAIN_SZIE_NN)

test_data, test_labels, tail_valid_data, tail_valid_labels, head_valid_data, head_valid_labels, train_data, train_labels = \
    test_data.as_matrix(), test_labels.as_matrix(), tail_valid_data.as_matrix(), \
    tail_valid_labels.as_matrix(), head_valid_data.as_matrix(), head_valid_labels.as_matrix(), train_data.as_matrix(), train_labels.as_matrix()

##########################################################

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
batch_size = 500
reg_lambda = 0.02

h1_size = 250
h2_size = 100
h3_size = 50
# up, down
num_labels = 2

device_name = "/gpu:0"

graph = tf.Graph()
with tf.device(device_name):
    with graph.as_default():
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        data = tf.placeholder(tf.float32, [None, look_back + 0], name = 'data')
        target = tf.placeholder(tf.float32, [None, num_labels], name = 'target')

        # Variables.
        # These are the parameters that we are going to be training.
        # past prices and volume
        weights = tf.Variable(tf.truncated_normal([look_back + 0, h1_size], stddev = 1.0 / math.sqrt(h1_size)))
        biases = tf.Variable(tf.zeros([h1_size]))
        ws1to2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev = 1.0 / math.sqrt(h2_size)))
        h2bs = tf.Variable(tf.zeros([h2_size]))
        ws2to3 = tf.Variable(tf.truncated_normal([h2_size, h3_size], stddev = 1.0 / math.sqrt(h3_size)))
        h3bs = tf.Variable(tf.zeros([h3_size]))
        ws3to4 = tf.Variable(tf.truncated_normal([h3_size, num_labels], stddev = 1.0 / math.sqrt(num_labels)))
        h4bs = tf.Variable(tf.zeros([num_labels]))

        # Model.
        def model(data):
            h1 = tf.matmul(data, weights) + biases
            h1 = tf.nn.relu(h1)
            h2 = tf.matmul(h1, ws1to2) + h2bs
            h2 = tf.nn.relu(h2)
            h3 = tf.matmul(h2, ws2to3) + h3bs
            h3 = tf.nn.relu(h3)
            h4 = tf.matmul(h3, ws3to4) + h4bs
            return h4

        # Training computation.
        logits = model(data)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits), name = 'loss')

        loss += reg_lambda * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases))
        loss += reg_lambda * (tf.nn.l2_loss(ws1to2) + tf.nn.l2_loss(h2bs))
        loss += reg_lambda * (tf.nn.l2_loss(ws2to3) + tf.nn.l2_loss(h3bs))
        loss += reg_lambda * (tf.nn.l2_loss(ws3to4) + tf.nn.l2_loss(h4bs))

        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        #optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(0.05, global_step, 1, 0.99995, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        prediction = tf.nn.softmax(logits, name = 'prediction')


################################################################################################

num_steps = 1 * 1000 + 1

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
        feed_dict = {data : batch_data, target : batch_labels}
        _, l, predictions = session.run([optimizer, loss, prediction], feed_dict=feed_dict)
        acc = accuracy(predictions, batch_labels)
        bench_scores.appendleft(acc)

        if (step % 4000 == 0 or step == num_steps - 1):
            pred_tail_valid = session.run(prediction, feed_dict = {data: tail_valid_data, target: tail_valid_labels})
            pred_head_valid = session.run(prediction, feed_dict = {data: head_valid_data, target: head_valid_labels})
            pred_test = session.run(prediction, feed_dict = {data: test_data, target: test_labels})

            print("at step %d" % step)
            print("Minibatch loss: %f" % l)
            print_message('batch', np.mean(bench_scores), batch_labels)
            print_message('validation (Tail)', accuracy(pred_tail_valid, tail_valid_labels), tail_valid_labels)
            print_message('validation (Head)', accuracy(pred_head_valid, head_valid_labels), head_valid_labels)
            print_message('test', accuracy(pred_test, test_labels), test_labels)

            print("Test prediction")
            print(np.argmax(pred_test, 1))
            print("Test label")
            print(np.argmax(test_labels, 1))

            ws = weights.eval()
            if prev_weights is not None:
                print("weights change: %f" % (np.sum(abs(ws - prev_weights))))

            print()
            prev_weights = ws

    def pred_save(mark, prices, labels, index):
        pred = session.run(prediction, feed_dict = {data: prices, target: labels})
        pred_df = pd.DataFrame(data=pred, index=index, columns = ['up', 'down'])
        pred_df.to_csv('predictions/pred_' + mark + '.csv')

    pred_save('train', train_data, train_labels, train_index)
    pred_save('valid_tail', tail_valid_data, tail_valid_labels, tv_index)
    pred_save('valid_head', head_valid_data, head_valid_labels, hv_index)
    pred_save('test', test_data, test_labels, test_index)

    # Now, save the graph
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    tf.train.Saver().save(session, MODEL_SAVE_PATH + MODEL_SAVE_NAME)

    # upload the saved graph to S3    
    client = boto3.client('s3')
    for file in os.listdir(MODEL_SAVE_PATH):
        print('Uploading %s...' % file)
        path = os.path.join(MODEL_SAVE_PATH, file)    
        client.upload_file(path, BUCKET, MODEL_SAVE_NAME + '/' + file)
        print('Uploading %s is done.' % file)

    session.close()
    del session




