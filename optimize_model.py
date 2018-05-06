import tensorflow as tf

def optimize_classifier(layer, labels, num_labels):
    # Predictions
    logits = tf.layers.dense(layer, num_labels)
    soft_max = tf.nn.softmax(logits)

    # Cost function and optimizer
    with tf.name_scope('cost'):
        pred_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        reg_cost = tf.losses.get_regularization_loss()
        total_cost = pred_cost + reg_cost
        tf.summary.scalar('total_cost', total_cost)
        tf.summary.scalar('prediction cost', pred_cost)
        tf.summary.scalar('reg. cost', reg_cost)

    # Accuracy
    pred = tf.argmax(logits, 1)
    pred_hit = tf.equal(pred, tf.argmax(labels, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(pred_hit, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

    return pred, logits, total_cost, accuracy
