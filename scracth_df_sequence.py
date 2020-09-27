import pandas as pd

#df = pd.read_csv('data/stock_minute_ta_window60_JPM_2020-09-14 06:30:00_to_2020-09-14 13:00:00.csv').set_index('datetime')
df = pd.read_csv('data/binance_ta_minute_window60_ETHUSDT_2020-08-12 17:00:00_to_2020-09-20 16:00:00.csv').set_index('datetime')

import data.util.df.target

df_t = data.util.df.target.add_binary_target_column(df, 10, value_column_name = 'close', target_column_name = 'target')

columns = ['target']
columns += ['volatility_bbm', 'volatility_bbl', 'volatility_kcc', 'volatility_kch', 'volatility_kcl', 'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 'trend_ichimoku_conv', 'trend_ichimoku_base', 'trend_ichimoku_a', 'momentum_kama', 'others_cr']
df_t = df_t[columns]


import keras.preprocessing
from keras.utils import to_categorical
_SEQ_LEN = 50

x = df_t.drop('target', axis=1).values
y = to_categorical(df_t['target'].values.reshape(-1, 1))


batches = keras.preprocessing.timeseries_dataset_from_array(x, y, _SEQ_LEN)

for batch in batches:
    data_batch, targets_batch = batch
    print(data_batch.shape)
    print(targets_batch.shape)


##########################################################

_NUM_VARS = 14

import models.keras.multi_column_lstm
model = models.keras.multi_column_lstm.LSTM(_SEQ_LEN, _NUM_VARS, True)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='rmsprop')

history = model.fit(batches, epochs=20)

print(history.history)


print('done')

