import tensorflow as tf, math, pandas as pd, numpy as np, boto3, os, threading, time
#from data.read import get_data, look_back, TEST_SIZE_NN, TAIL_VALID_SIZE_NN, TRAIN_SZIE_NN, HEAD_VALID_SIZE_NN, decorated
from models.learner_common import batchmark_accuracy, accuracy, print_message
from models.nn import MODEL_SAVE_NAME
from collections import deque
from data.read import decorate, look_back, format as doformat

model_name = MODEL_SAVE_NAME
MODEL_SAVE_PATH_BASE = 'models/saves/'
MODEL_SAVE_PATH = MODEL_SAVE_PATH_BASE + model_name + '/'
BUCKET = 'bitcoin-predition-model-headgecoast'
RELOAD_MODEL_CYCLE_SECONDS = 60 * 60 * 6 # every 6 hours

def s3_to_file():
    ''
    client = boto3.client('s3')
    bucket_objects_res = client.list_objects(Bucket = BUCKET, Prefix = model_name)
    print(bucket_objects_res)
    if not 'Contents' in bucket_objects_res: return

    bucket_objects_contents = bucket_objects_res['Contents']
    for bucket_object in bucket_objects_contents:
        key = bucket_object['Key']
        print('Downloading %s...' % key)
        print(key)
        client.download_file(BUCKET, key, MODEL_SAVE_PATH_BASE + key)
        print('Downloading %s is done.' % key)

session, data, prediction = None, None, None

def load():
    global session, data, prediction
    session = tf.Session()    
    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH + model_name + '.meta')
    saver.restore(session, tf.train.latest_checkpoint(MODEL_SAVE_PATH))
    graph = tf.get_default_graph()
    data = graph.get_tensor_by_name("data:0")
    prediction = graph.get_tensor_by_name("prediction:0")

def start_updating_thread():
    def worker():
        print('Starting a thread to cycle the model update.')
        while True:
            time.sleep(RELOAD_MODEL_CYCLE_SECONDS)
            print('Downloading and reloading the model.')
            s3_to_file()
            load()

    t = threading.Thread(target=worker)
    t.start()

prices = deque([], look_back + 1)
def on_tick(price, volume = None, type = None):
    global prices
    prices.append(price)

def predict(ps = None):
    '''
    @param prices 
    '''
    if ps is None:
        formatted = doformat(decorate(pd.DataFrame({'price': prices}), price_column_name = 'price', cut_tail_expect_window = False))
        price_formatted = formatted[0].iloc[-1]
        ps = price_formatted.values.reshape(1, price_formatted.shape[0])
    if session is None: return None
    pred = session.run(prediction, {data: ps})
    return pred



