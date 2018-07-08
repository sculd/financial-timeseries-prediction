import pickle
import models.keras.single_column_lstm as single_column_lstm
import models.keras.single_column_cnn as single_column_cnn

import data.read_columns as read_columns

_SEQ_LEN = 50

##########################################################

model = single_column_cnn.CNN(_SEQ_LEN, True)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='rmsprop')

X_train, y_train, X_test, y_test = pickle.load(open('data/kosdaq/close_50_seq_classification.data', 'r'))

model.fit(X_train, y_train, batch_size=512, nb_epoch=3, validation_split=0.25)

score = model.evaluate(X_test, y_test, batch_size=512)
print('score', score)
