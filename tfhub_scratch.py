import tensorflow as tf
import tensorflow_hub as hub

graph = tf.Graph()
with graph.as_default():
    global session, embed_nnlm
    embed_nnlm = hub.Module("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1")
    embeddings_nnlm = embed_nnlm(["cat is on the mat"])

    '''
    embed_wiki = hub.Module("https://tfhub.dev/google/Wiki-words-250-with-normalization/1")
    embeddings_wiki = embed_wiki(["cat is on the mat"])

    embed_use = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
    embeddings_use = embed_use([
        "The quick brown fox jumps over the lazy dog.",
        "I am a sentence for which I would like to get its embedding"])
    '''

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())

    #print(session.run(embeddings_nnlm))

    # print(sess.run(embeddings_wiki))
    # print(sess.run(embeddings_use))

def get_sentence_embedding(sentence):
    global session, embed_nnlm, graph
    with graph.as_default():
        e = embed_nnlm([sentence])
        return session.run(e)
    #manager = graph.as_default()
    #manager.__enter__()

if __name__ == '__main__':
    print(get_sentence_embedding("this is so ambitous"))