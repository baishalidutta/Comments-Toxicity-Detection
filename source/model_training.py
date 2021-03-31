__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Dropout, \
    GlobalMaxPooling1D, LSTM, Bidirectional
from keras.models import Model
from sklearn.metrics import roc_auc_score

from config import *
from data_preprocessing import DataPreprocess


# -------------------------------------------------------------------------
#                   Build and Train the RNN Model Architecture
# -------------------------------------------------------------------------
def build_rnn_model(data, target_classes, embedding_layer):
    """
    Build and Train the RNN architecture (Bidirectional LSTM)
    :param data: the preprocessed padded data
    :param target_classes: Assigned target labels for the comments
    :param embedding_layer: Embedding layer comprising preprocessed comments
    :return: the trained model
    """
    # create an LSTM network with a single LSTM
    input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = embedding_layer(input_)
    x = Bidirectional(LSTM(units=64,
                           return_sequences=True,
                           recurrent_dropout=0.2))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    # x = GlobalMaxPooling1D()(x)

    #  Sigmoid Classifier
    output = Dense(len(DETECTION_CLASSES), activation="sigmoid")(x)

    model = Model(input_, output)

    # Display the model
    model.summary()

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Defining the callbacks
    # TODO Check whether to use the restore_best_weights
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=5,
                               mode='min',
                               restore_best_weights=True)

    checkpoint = ModelCheckpoint(filepath=MODEL_LOC,  # saves the 'best' model
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min')

    # Fit the model
    history = model.fit(data,
                        target_classes,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_split=VALIDATION_SPLIT,
                        callbacks=[early_stop, checkpoint],
                        verbose=1)

    # Return model training history
    return model, history


# -------------------------------------------------------------------------
#                   Plotting the training history
# -------------------------------------------------------------------------
def plot_training_history(rnn_model, history, data, target_classes):
    """
    Generates plots for accuracy and loss
    :param rnn_model: the trained model
    :param history: the model history
    :param data: preprocessed data
    :param target_classes: target classes for every comment
    :return: None
    """
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("../plots/accuracy.jpeg")
    plt.show()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("../plots/loss.jpeg")
    plt.show()

    # Print Average ROC_AUC_Score
    p = rnn_model.predict(data)
    aucs = []
    for j in range(len(DETECTION_CLASSES)):
        auc = roc_auc_score(target_classes[:, j], p[:, j])
        aucs.append(auc)
    print(f'Average ROC_AUC Score: {np.mean(aucs)}')


def execute():
    """
    Import the training data csv file and save it into a dataframe
    """
    training_data = pd.read_csv(TRAINING_DATA_LOC)

    preprocessing = DataPreprocess(training_data)
    rnn_model, history = build_rnn_model(preprocessing.padded_data,
                                         preprocessing.target_classes,
                                         preprocessing.embedding_layer)
    plot_training_history(rnn_model,
                          history,
                          preprocessing.padded_data,
                          preprocessing.target_classes)


# -------------------------------------------------------------------------
#                               Main Execution
# -------------------------------------------------------------------------
if __name__ == '__main__':
    execute()
