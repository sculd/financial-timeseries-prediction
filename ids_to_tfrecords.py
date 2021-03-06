import data.news.news_with_series_and_word_ids as news_with_ids
import sys
import tensorflow as tf


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  return tf.train.Feature(int64_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def to_tfrecord(infile, outfile):
    writer = tf.python_io.TFRecordWriter(outfile)

    dy = news_with_ids.Daily(infile)
    dit = news_with_ids.DailyIter(dy)
    while dit.has_next():
        day = dit.next()

        feature = {'series': _float_feature(day.series),
                   'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

if __name__ == '__main__':
    #to_tfrecord(news_with_ids.FILENAME_BLOOMBERG_PADDED)
    #to_tfrecord(news_with_ids.FILENAME_REUTERS_PADDED)
    pass

