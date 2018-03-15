import tensorflow as tf, math, pandas as pd, numpy as np, boto3, os, threading, time
#from data.read import get_data, look_back, TEST_SIZE_NN, TAIL_VALID_SIZE_NN, TRAIN_SZIE_NN, HEAD_VALID_SIZE_NN, decorated
from models.learner_common import batchmark_accuracy, accuracy, print_message
from models.nn import MODEL_SAVE_NAME

model_name = MODEL_SAVE_NAME
MODEL_SAVE_PATH = 'models/saves/' + model_name + '/'
BUCKET_MODEL = 'bitcoin-predition-model-headgecoast'
RELOAD_MODEL_CYCLE_SECONDS = 60 * 60 * 6 # every 6 hours

def s3_to_file(folder):
    ''
    client = boto3.client('s3')
    bucket_objects_res = client.list_objects(Bucket = BUCKET_MODEL, Prefix = model_name)
    print(bucket_objects_res)
    if not 'Contents' in bucket_objects_res: return

    bucket_objects_contents = bucket_objects_res['Contents']
    for bucket_object in bucket_objects_contents:
        key = bucket_object['Key']
        print('Downloading %s...' % key)
        print(key)
        client.download_file(BUCKET_MODEL, key, MODEL_SAVE_PATH + key)
        print('Downloading %s is done.' % key)

def load():
    session = tf.Session()    
    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH + '.meta')
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

def predict(prices):
    '''
    @param prices 
    '''
    pred = session.run(prediction, {data: prices})
    return pred
