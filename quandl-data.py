import abc, tensorflow as tf, math, pandas as pd, numpy as np
from deeplearning_timeseries_prediction.model import TimeSeriesPredictionModel

class NN(TimeSeriesPredictionModel):
    DEVICE_NAME = "/gpu:0"

    REG_LAMBDA_DEFAULT = 0.01
    LEARNING_RATE_INIT_DEFAULT = 0.005
    LEARNING_RATE_DECAY_RATE = 0.99995

    variables = {}
    prev_weights = None
    learning_rate = None

    @classmethod
    def print_training_hook(cls):
        ws = cls.variables['ws_0'].eval()
        print('weights size: %f' % (np.linalg.norm(ws)))        
        if cls.prev_weights is not None:
            print("weights change: %f" % (np.linalg.norm(abs(ws - cls.prev_weights))))
        cls.prev_weights = ws
        print('learning_rate: %f' % cls.learning_rate.eval())

    @classmethod
    def _get_graph(cls, data_size, predict_dimension, graph_config):
        """
        return graph, data, target, optimizer, loss, prediction
        """
        if 'n_layers' not in graph_config:
            graph_config = {
                'n_layers': 4, # 1 + hidden
                'layer_size_1': 250,
                'layer_size_2': 100,
                'layer_size_3': 50,
            }
        if 'reg_lambda' not in graph_config:
            graph_config['reg_lambda'] = cls.REG_LAMBDA_DEFAULT

        if 'learning_rate_init' not in graph_config:
            graph_config['learning_rate_init'] = cls.LEARNING_RATE_INIT_DEFAULT

        if 'learning_rate_decay_rate' not in graph_config:
            graph_config['learning_rate_decay_rate'] = cls.LEARNING_RATE_DECAY_RATE

        graph = tf.Graph()
        with tf.device(cls.DEVICE_NAME):
            with graph.as_default():
                # Input data.
                # Load the training, validation and test data into constants that are
                # attached to the graph.
                data = tf.placeholder(tf.float32, [None, data_size + 0], name = cls.TF_NAME_DATA)
                target = tf.placeholder(tf.float32, [None, predict_dimension], name = 'target')

                n_layers = graph_config['n_layers']
                variables = cls.variables

                # Variables.
                # These are the parameters that we are going to be training.
                # past prices and volume
                for i in range(n_layers):
                    s1 = data_size if i == 0 else graph_config['layer_size_' + str(i)]
                    s2 = predict_dimension if i == n_layers - 1 else graph_config['layer_size_' + str(i + 1)]
                    variables['ws_' + str(i)] = tf.Variable(tf.truncated_normal([s1, s2], stddev = 1.0 / math.sqrt(s2)))
                    variables['bias_' + str(i)] = tf.Variable(tf.zeros([s2]))

                # Model.
                def model(data):
                    logits = tf.matmul(data, variables['ws_0']) + variables['bias_0']
                    for i in range(1, n_layers):
                        logits = tf.nn.relu(logits)
                        logits = tf.matmul(logits, variables['ws_' + str(i)]) + variables['bias_' + str(i)]
                    return logits

                # Training computation.
                logits = model(data)
                if predict_dimension == 1:
                    loss = tf.sqrt(tf.reduce_mean(tf.pow(target - logits, 2)), name = 'loss')
                else:
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits), name = 'loss')                
                for i in range(1, n_layers):
                    loss += graph_config['reg_lambda'] * (tf.nn.l2_loss(variables['ws_' + str(i)]) + tf.nn.l2_loss(variables['bias_' + str(i)]))

                # Optimizer.
                # We are going to find the minimum of this loss using gradient descent.
                #optimizer = tf.train.GradientDescentOptimizer(graph_config['learning_rate_init']).minimize(loss)

                global_step = tf.Variable(0)  # count the number of steps taken.
                cls.learning_rate = tf.train.exponential_decay(graph_config['learning_rate_init'], global_step, 1, graph_config['learning_rate_decay_rate'], staircase=True)
                optimizer = tf.train.GradientDescentOptimizer(cls.learning_rate).minimize(loss, global_step=global_step)

                # Predictions for the training, validation, and test data.
                # These are not part of training, but merely here so that we can report
                # accuracy figures as we train.
                if predict_dimension == 1:
                    prediction = tf.identity(logits, name = cls.TF_NAME_PREDICTION)
                else:
                    prediction = tf.nn.softmax(logits, name = cls.TF_NAME_PREDICTION)

        return graph, data, target, optimizer, loss, prediction
