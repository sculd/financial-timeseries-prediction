import tensorflow as tf

device_name = "/gpu:0"

graph = tf.Graph()
with tf.device(device_name):
    with graph.as_default():
        f = tf.placeholder(tf.float32, [None], name='feature')

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'wall street see key economic datum': tf.FixedLenFeature([], tf.float32),
        })

    print(features)
    return features


sess = tf.InteractiveSession()

filename_queue = tf.train.string_input_producer(['data/embeddings_reuters.1.sst'], num_epochs=10)
features = read_and_decode(filename_queue)
ft = features['wall street see key economic datum']
#tf.Print(ft, [ft, tf.shape(ft)],summarize=20,message="ft:")
#print(eval("ft"))

iterator = ft.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    #iterator = features['wall street see key economic datum'].make_initializable_iterator()
    #next_element = iterator.get_next()
    #print(session.run(next_element))
    #print(session.run([features['wall street see key economic datum']]))
    sess.run(next_element)

    session.close()
    del session
