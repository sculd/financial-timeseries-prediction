from time import time

import tensorflow as tf

Sequential = tf.keras.models.Sequential
Dense, Dropout = tf.keras.layers.Dense, tf.keras.layers.Dropout
Embedding = tf.keras.layers.Embedding
Conv1D, MaxPooling1D = tf.keras.layers.Conv1D, tf.keras.layers.MaxPooling1D
BatchNormalization = tf.keras.layers.BatchNormalization
Flatten = tf.keras.layers.Flatten
TensorBoard = tf.keras.callbacks.TensorBoard

import data.read_columns as read_columns
import data.kosdaq.read as kosdaq_read
from models.keras import keras_cnn

##########################################################

_WINDOW_SIZE = kosdaq_read.WINDOW_SIZE
_NUM_FEATURES = _WINDOW_SIZE / read_columns.WINDOW_STEP
N_CHANNELS = read_columns.N_CHANNELS_HISTORY + 3

class_dim=2
hid_dim=128
reg = 0.000

'''
model = Sequential()
model.add(Conv1D(hid_dim, 2, activation='relu', batch_input_shape=(None, _NUM_FEATURES, N_CHANNELS)))
model.add(MaxPooling1D(2))
model.add(BatchNormalization())
model.add(Conv1D(hid_dim, 2, activation='relu'))
model.add(MaxPooling1D(2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(class_dim, activation='softmax'))
model.summary()
'''

model = keras_cnn.CNN(8, 11, hid_dim, class_dim, 0.8)
model.summary()

sgd = tf.keras.optimizers.SGD(lr=5e-4, decay=1e-2, momentum=0.3, nesterov=True)
# rmsprop
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print('loading the data')
df, data_all, labels_all, target_all, x_train, y_train, train_targets, x_test, y_test, valid_targets = read_columns.read_sp500_ohlc_history(window_size = _WINDOW_SIZE)
print('done loading the data')

tensorboard = TensorBoard(log_dir="logs".format(time()))
model.fit(x_train, y_train, batch_size=60, validation_split=0.4, epochs=100, callbacks=[tensorboard])
score = model.evaluate(x_test, y_test, batch_size=40)
print('score', score)

