import tensorflow as tf

def _parse_function(example_proto):
    features = {"wall street see key economic datum": tf.FixedLenFeature((), tf.float32, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["wall street see key economic datum"]

# Creates a dataset that reads all of the examples from two files, and extracts
# the image and label features.
filenames = ['data/embeddings_reuters.1.sst']
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)

batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    while True:
        try:
            sess.run(next_element)
        except tf.errors.OutOfRangeError:
            break