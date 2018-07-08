import tensorflow as tf, tensorflow_hub as hub, re
from nltk.corpus import stopwords

_STOP_WORDS = set(stopwords.words())
_DATE_REGEXP = re.compile('\d{4}-\d{2}-\d{2}')

with tf.Graph().as_default():
    embed_nnlm = hub.Module("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1")

    with tf.Session() as session:
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())

        sentenses = []
        def ss_to_es(infile, outfile):
            with open(outfile, 'w') as with_embedding:
                for line in open(infile, 'r'):
                    if _DATE_REGEXP.match(line):
                        with_embedding.write(line)
                    elif 'series:' in line:
                        with_embedding.write(line)
                    else:
                        ws = ' , '.join(line.split(',')[1:]).split()
                        ws = [w for w in ws if w not in _STOP_WORDS]
                        sentence = ' '.join(ws)
                        sentenses.append(sentence)

            embeddings = embed_nnlm(sentenses)
            embeddings_values = session.run(embeddings)
            for e in embeddings_values:
                with_embedding.write(' '.join(list(map(lambda v: str(v), e))) + '\n')

        ss_to_es('sp500_bloomberg_combined.txt', 'sp500_bloomberg_combined_with_embeddings.txt')
