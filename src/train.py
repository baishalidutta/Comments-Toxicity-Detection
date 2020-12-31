# -*- coding: utf-8 -*-
"""
@author: baishalidutta
"""
############### USER INPUT ####################
# user input
import argparse
parser = argparse.ArgumentParser(description='Toxic comments prediction using Bidirectional LSTM')
parser.add_argument('--max_sequence_length', default=100, type=int, help='max number of words in a sentence')
parser.add_argument('--max_vocab_size', default=20000, type=int, help='max number of words in vocabulary')
parser.add_argument('--embedding_dim', default=100, choices=[50,100,200,300], type=int, help='dimension of the embedding matrix')
parser.add_argument('--validation_split', default=0.2, type=float, help='size of validation set')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--clean_text', default=1, choices=[0,1], help='cleaning text during preprocessing step or not')
#parser.add_argument('--use_gpu', default=1, choices=[0,1], help='use GPU or not')
args = parser.parse_args()

# configuration with editor
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 256
EPOCHS = 10
CLEAN_TEXT = 1
USE_GPU = 0

'''
# configuration
MAX_SEQUENCE_LENGTH = args.max_sequence_length
MAX_VOCAB_SIZE = args.max_vocab_size
EMBEDDING_DIM = args.embedding_dim
VALIDATION_SPLIT = args.validation_split
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
CLEAN_TEXT = args.clean_text
#USE_GPU = args.use_gpu
'''

############### IMPORT LIBRARIES ##################
print('Importing libaries...')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, SpatialDropout1D, GlobalAveragePooling1D, GRU
from keras.layers import MaxPooling1D, Embedding, LSTM, Bidirectional, Dropout, concatenate, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, constraints, optimizers, layers
import keras.backend as K
from preprocessing import *

#import nltk
#from nltk import WordNetLemmatizer
import re
from textblob import TextBlob
import pickle

import keras.backend as K
#if (len(K.tensorflow_backend._get_available_gpus()) > 0) and USE_GPU==1:
  #from keras.layers import CuDNNLSTM as LSTM

# Set fitting condition
model_dir = './models'
if not os.path.exists(''):
    os.makedirs(model_dir, exist_ok=True)
    
# Load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('pretrained/glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM)) as f:
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))
    

################ PREPROCESSING DATA ####################
# ---------- Load and clean text --------------#
# Load train data
print('Loading in comments...')

train_orig = pd.read_csv("../data_train/train.csv")
possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets = train_orig[possible_labels].values

train = pd.DataFrame()
if CLEAN_TEXT == 1:
    # Before processing
    print('Before processing...')
    vocab = build_vocab(train_orig['comment_text'])
    oov_glove = check_coverage(vocab, word2vec)
    
    print('Cleaning text...')
    train['comment_text'] = train_orig['comment_text'].apply(lambda x: clean_text(x))

    print('After processing...')
    vocab = build_vocab(train['comment_text'])
    oov_glove = check_coverage(vocab, word2vec)   
    
else:
    train['comment_text'] = train_orig['comment_text']
sentences = train["comment_text"].fillna("DUMMY_VALUE").values

#---------Tokenizing--------#

# Tokenize
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# Print status
print('Max sequence length: %i' % max(len(s) for s in sequences))
print('Min sequence length: %i' % min(len(s) for s in sequences))
seq_length = [len(s) for s in sequences]
print('Median sequence length: %i' %np.median(seq_length))

# Word -> integer mapping
word2idx = tokenizer.word_index
print('Found %i unique tokens' %len(word2idx))

# Save the tokenizer
print('Saving tokens ...')
with open(model_dir+'/tokenizer.pickle', 'wb') as handle:
     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#--------- Padding and embedding -------------#
# Padding
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
print('Shape of data tensor: ', data.shape)

# Prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector
      
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False
)

################# BUILDING MODEL ###################
'''
# Attention layer
class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
'''

# Building model
print('Building model...')
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = SpatialDropout1D(0.1)(x)
x = Bidirectional(GRU(64, return_sequences=True))(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
#atten = AttentionWeightedAverage()(x)

conc = concatenate([avg_pool, max_pool])
conc = BatchNormalization()(conc)
conc = Dense(64, activation='relu')(conc)
conc = Dropout(0.2)(conc)
out = Dense(6, activation='sigmoid')(conc)

model = Model(inputs = input_, outputs=out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model summary
trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
print('Trainable params: %i' %trainable_count)
print('Non-trainable params: %i' %non_trainable_count)

# Fitting model
print('Fitting model...')
#model_path = os.path.join(model_dir,'WEIGHTS_len_%i_vocab_%ik_cleantxt_%i_embedding_%i_.best.hdf5'%(MAX_SEQUENCE_LENGTH, int(MAX_VOCAB_SIZE/1000), CLEAN_TEXT, EMBEDDING_DIM))
model_path = os.path.join(model_dir,'weights_cpu.best.hdf5')
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min')

callbacks_list = [checkpoint, early_stopping]
history = model.fit(data, targets, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT, callbacks=callbacks_list)

# Print result
train_acc = max(history.history['acc'])
val_acc = max(history.history['val_acc'])
print('Final training accuracy: %.4f' %train_acc)
print('Final validating accuracy: %.4f' %val_acc)

'''
model.predict(data[1:2,:])
data[1:2,:].shape
'''
