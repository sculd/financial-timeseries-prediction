import numpy as np

GLOVE_FILE = '/Users/hjunlim/Documents/projects/newsfeed_openie/glove/glove.6B/glove.6B.100d.txt'

def get():
    global words, vectors, vocab, ivocab, W_norm
    if 'words' in globals(): return

    with open(GLOVE_FILE, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(GLOVE_FILE, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit length
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
