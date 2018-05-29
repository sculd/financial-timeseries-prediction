import tensorflow as tf
from time import time

Sequential = tf.keras.models.Sequential
Dense, Dropout = tf.keras.layers.Dense, tf.keras.layers.Dropout
Embedding = tf.keras.layers.Embedding
Conv1D, MaxPooling1D = tf.keras.layers.Conv1D, tf.keras.layers.MaxPooling1D
BatchNormalization = tf.keras.layers.BatchNormalization
Flatten = tf.keras.layers.Flatten
TensorBoard = tf.keras.callbacks.TensorBoard

import data.read_columns as read_columns
import data.kosdaq.read as kosdaq_read

##########################################################

_WINDOW_SIZE = 20
_NUM_FEATURES = _WINDOW_SIZE / read_columns.WINDOW_STEP
N_CHANNELS = read_columns.N_CHANNELS_HISTORY + 3

batch_size = 300
class_dim=2
reg = 0.000

model = Sequential()
model.add(BatchNormalization(batch_input_shape=(None, _WINDOW_SIZE, 1)))
model.add(Conv1D(128, 2, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(33, activation='relu'))
model.add(Dropout(0.7))
model.add(BatchNormalization())
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.7))
model.add(BatchNormalization())
model.add(Dense(4, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.7))
model.add(BatchNormalization())
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print('loading the data')
#df, data_all, labels_all, target_all, x_train, y_train, train_targets, x_test, y_test, valid_targets = read_columns.read_sp500_ohlc_history(window_size = _WINDOW_SIZE)
df, data, labels, x_train, y_train, x_test, y_test = read_columns.read_sp500_close_returns(window_size = _WINDOW_SIZE)
print('done loading the data')

tensorboard = TensorBoard(log_dir="logs")
model.fit(x_train, y_train, batch_size=batch_size, validation_split=0.4, epochs=40)
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('score', score)

