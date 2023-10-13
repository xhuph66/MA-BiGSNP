from __future__ import print_function
from keras import models
from keras import layers
from keras.layers import Embedding,Dropout
from keras import backend as K
import pickle
import time
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2

from multi_attention import Word_Attention, Sentence_Attention
from utils import create_emb_mat, pickle_load
from config import *

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='max')
save_best_model = ModelCheckpoint(filepath='saves/'+file_name+'/{epoch:02d}e-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}.hdf5',
                                  monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)


x_train, y_train, x_dev, y_dev, x_test, y_test = pickle_load(os.path.join(folder, 'data.pkl'))
word_idx = pickle_load(os.path.join(folder, 'word_idx.pkl'))

embedding_matrix = create_emb_mat(em_path, word_idx, em_dim)
length = len(word_idx)

def build_model(emb_matrix, len, max_len=max_length, max_sent_len=max_senten, embedding_dim=em_dim):
    embedding_layer = Embedding(len+1,
                                embedding_dim,
                                weights=[emb_matrix],
                                mask_zero=False,
                                input_length=max_len,
                                trainable=True)
    inputs = layers.Input(shape=(max_len,), dtype='int32')
    em_inputs = embedding_layer(inputs)
    GSNP_inputs = layers.Bidirectional(layers.GSNP(100, return_sequences=True, kernel_regularizer=l2(L2),
                                                   recurrent_dropout=0.5, dropout=0.5))(em_inputs)
    GSNP_inputs = Word_Attention(100)(GSNP_inputs)
    s_model = models.Model(inputs=inputs, outputs=GSNP_inputs)
    s_model.summary()

    text = layers.Input(shape=(max_sent_len, max_len), dtype='int32')
    en_text = layers.TimeDistributed(s_model)(text)
    SA = Sentence_Attention(1, 32)([en_text, en_text, en_text])
    GAP = layers.GlobalAveragePooling1D()(SA)
    DROP = Dropout(0.5)(GAP)
    preds = layers.Dense(class_num, activation='sigmoid',)(DROP)
    model = models.Model(text, preds)
    return model

K.set_learning_phase(1)
model = build_model(embedding_matrix, length)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.summary()
start = time.process_time()

hist = model.fit(x_train, y_train, batch_size=Batch_size, shuffle=True,
                 validation_data=(x_dev, y_dev), epochs=Epochs,
                 callbacks=[save_best_model, early_stopping])

print("Evaluating...")
score = model.evaluate(x_test, y_test, batch_size=Batch_size)
print("Test score: ", score)
print(model.metrics_names)

total_time = time.process_time() - start
with open('saves/'+file_name+'/model_fitting_log', 'a') as f:
    f.write('Total training time of {}_{} is {}\n'.format(file_name, max_senten, total_time))
model.save(filepath='saves/'+file_name+'/{}_{}_{}.h5'.format(file_name, max_senten, time.time()))

hist_dict = hist.history
with open('saves/'+file_name+'/{}_{}_{}.dic'.format(file_name, max_senten, time.time()), "wb") as f:
    pickle.dump(hist_dict, f)
del model


