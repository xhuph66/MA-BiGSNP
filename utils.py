import os
import numpy as np
import pickle

def create_emb_mat(emb_path, word_idx, emb_dim):
    embeddings_index = {}
    f = open(os.path.join(emb_path), encoding='utf-8')
    for line in f:
        values = line.split()
        i = len(values)-300
        if i > 1:
            word = str(values[:i])
            vec = np.array(values[i:], dtype=np.float64)
        else:
            word = values[0]
            vec = np.array(values[1:], dtype=np.float64)
        embeddings_index[word] = vec
    f.close()

    counter = 0
    emb_matrix = np.random.random((len(word_idx) + 1, emb_dim))
    for word, i in word_idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            emb_matrix[i] = embedding_vector
        else:
            counter += 1
    print('invalid word embedding: ', counter)
    return emb_matrix


def pickle_load(file_path):
    return pickle.load(open(file_path, 'rb'))


def pickle_dump(obj, file_path):
    pickle.dump(obj, open(file_path, 'wb'))

