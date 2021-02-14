__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

################## IMPORTING LIBRARIES AND MODELS ####################
print('Importing libraries...')
from flask import Flask, url_for
from flask import request
from flask import jsonify, Flask,render_template,url_for,request

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, SpatialDropout1D, GlobalAveragePooling1D, GRU
from tensorflow.keras.layers import MaxPooling1D, Embedding, LSTM, Bidirectional, Dropout, concatenate, BatchNormalization
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.compat.v1.keras import backend as K
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from preprocessing import *

import os
import pickle
import re
import itertools


model_dir = './models'


    
with open(os.path.join(model_dir,'tokenizer.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)
    

global graph
#graph = tf.get_default_graph()
graph = tf.compat.v1.get_default_graph()

print('Loading model...')
#BD 


with graph.as_default():
    model = load_model(os.path.join(model_dir, 'weights_cpu.best.hdf5'))

MAX_SEQUENCE_LENGTH = model.input_shape[1]

################## MAKING PREDICTION ####################

# Prediction
def rate_toxic(text):
    text_clean = clean_text(text)
    text_split = text_clean.split(' ')
    
    # Tokenizer
    sequences = tokenizer.texts_to_sequences(text_split)
    sequences = [[item for sublist in sequences for item in sublist]]
    
    # Padding
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    # Prediction
    with graph.as_default():
        predict = model.predict(data).reshape(-1,1)
    return predict
'''
toxic, severe_toxic, obsence, threat, insult, identity_hate = rate_toxic(text)
print('Prediction succesful!')
print(rate_toxic(text))
'''
################### BUILD THE APP ###################
app = Flask(__name__,static_url_path='/static')

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    if request.method == 'POST':
        txt_input = request.form['comment']
        toxic, severe_toxic, obsence, threat, insult, identity_hate = rate_toxic(txt_input)
        '''
        response = {}
        response['Toxic score'] = '%.4f'%toxic
        response['Severe toxic score'] = '%.4f'%severe_toxic
        response['Obsence score'] = '%.4f'%obsence
        response['Threat score'] = '%.4f'%threat
        response['Insult score'] = '%.4f'%insult
        response['Identity hate score'] = '%.4f'%identity_hate
        '''
        return render_template('home.html', Score1='%.4f'%toxic, Score2='%.4f'%severe_toxic, Score3='%.4f'%obsence, Score4='%.4f'%threat, Score5='%.4f'%insult, Score6='%.4f'%identity_hate)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)

















