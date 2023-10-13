from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk import tokenize
import re
import os
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from config import *
from utils import pickle_dump


def clean_str(string):
    try:
        string = re.sub(r"@\w*", "", string)
        string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
        string = re.sub(r'(.)\1+', r'\1\1', string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = re.sub(r"\\", "", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"\"", "", string)

    except Exception as e:
        print(type(string))
        print(string)
        print(e)
    return string.strip().lower()

def process_spilt_data(all_path, folder, len_train, len_dev):
    reviews = []
    tokenized_text = []
    ps = PorterStemmer()
    df = pd.read_csv(all_path, encoding='utf-8')
    # stopword = stopwords.words('english')
    X = df['reviewText']
    X = [str(x) for x in X]
    X = [ps.stem(word) for word in X]

    Y = np.asarray(df['label'], dtype=np.int)
    Y = to_categorical(np.asarray(Y))
    print(Y)

    for txt in list(X):

        sentences = tokenize.sent_tokenize(txt)
        reviews.append(sentences)

    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(list(X))
    word_idx = tokenizer.word_index

    for sentences in reviews:
        tokenized_sentences = tokenizer.texts_to_sequences(sentences)
        tokenized_sentences = tokenized_sentences[:max_senten]
        tokenized_text.append(tokenized_sentences)

    X = np.zeros((len(tokenized_text), max_senten, max_length), dtype='int32')
    for i in range(len(tokenized_text)):
        sentences = tokenized_text[i]
        seq_sequences = pad_sequences(sentences, maxlen=max_length)
        for j in range(len(seq_sequences)):
            X[i, j] = seq_sequences[j]


    x_train = X[:len_train]
    y_train = Y[:len_train]
    x_dev = X[len_train:(len_train+len_dev)]
    y_dev = Y[len_train:(len_train+len_dev)]
    x_test = X[(len_train+len_dev):]
    y_test = Y[(len_train+len_dev):]
    print(x_test)
    pickle_dump(word_idx, os.path.join(folder, 'word_idx.pkl'))
    pickle_dump((x_train, y_train, x_dev, y_dev, x_test, y_test), os.path.join(folder, 'data.pkl'))


if __name__ == '__main__':
    all_data_path = 'data/'+file_name +'/all_data.csv'

    len_train, len_dev = 6919, 871
    process_spilt_data(all_data_path, folder, len_train, len_dev)
