This repository is my personal project for predicting snp500, kosdaq daily price move direction prediction.

## Files

`keras_*_price.py` are the main files built with Keras. 

`keras_cnn_price.py` is the model inspired by [this paper](https://arxiv.org/abs/1412.1058). 
Its Keras/Tensorflow implementation is commited to tensorflow's [repository](https://github.com/tensorflow/models/tree/master/research/sentiment_analysis).


`keras_nn_encoder.py` is the model inspired by [this paper](https://arxiv.org/abs/1508.01993).

`keras_lstm.py` is the model with two layered LSTM.

`tf_*_price.py` are the legacy main files built with the native Tensorflow.


## Performance

`keras_nn_encoder.py` - s&p500 close price binary classification using nn
* train: 0.5217
* test: 0.5331

`keras_cnn_price.py` - s&p500 close price binary classification using cnn 
* train: 0.5139
* test: 0.5079

`keras_lstm.py` - kosdaq close price binary classification using lstm
* train: 0.5729
* test: 0.5312
 
