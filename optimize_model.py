import tensorflow as tf

def optimize_classifier(layer, labels, num_labels):
    # Predictions
    logits = tf.layers.dense(layer, num_labels)
    pred = tf.argmax(logits, 1)

    # Cost function and optimizer
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        reg_cost = tf.losses.get_regularization_loss()
        cost += reg_cost
    tf.summary.scalar('cost', cost)
    tf.summary.scalar('reg_cost', reg_cost)

    # Accuracy
    correct_pred = tf.equal(pred, tf.argmax(labels, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    tf.summary.scalar('accuracy', accuracy)

    return pred, cost, accuracy
