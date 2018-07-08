from data.news import news_with_series_and_word_ids as news_with_ids


def to_tfrecord(infile):
    dy = news_with_ids.Days(infile)
    dy.pad_ids()
    outfile = infile.split('.txt')[0] + '_padded.txt'
    dy.save(outfile)

if __name__ == '__main__':
    to_tfrecord(news_with_ids.FILENAME_BLOOMBERG)
    to_tfrecord(news_with_ids.FILENAME_REUTERS)
