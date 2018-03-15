import numpy as np

def batchmark_accuracy(labels):
    acc = sum(np.argmax(labels, axis=1)) * 1.0 / len(labels)
    acc = acc if acc >= 0.5 else 1.0 - acc
    return 100.0 * acc

def accuracy(predictions, labels):
    return 100.0 - 100.0 * np.sum(abs(np.argmax(predictions, 1) - np.argmax(labels, 1))) / predictions.shape[0]
    #return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def print_message(text_mark, acc, target):
    print(text_mark + " accuracy: %.1f%%, benchmark: %.1f%%" % (acc, batchmark_accuracy(target)))
