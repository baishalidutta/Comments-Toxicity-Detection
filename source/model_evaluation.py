__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------

import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import roc_auc_score, accuracy_score

from config import *
from data_preprocessing import DataPreprocess


# -------------------------------------------------------------------------
#                   Prepare Test Data
# -------------------------------------------------------------------------
def prepare_test_data():
    """
    Loads the test dataset and prepare it for further evaluation
    :return: prepared test data
    """
    test_label = pd.read_csv('../data/test_labels.csv')
    test_comments = pd.read_csv('../data/test.csv')

    # Merge the two dataframes into one for better handling
    test_data = pd.merge(test_comments, test_label, on='id')

    # Dropping the rows where correct label are not assigned
    # In such rows, the all labels are filled with -1
    test_data = test_data[test_data['toxic'] != -1]

    return test_data


# -------------------------------------------------------------------------
#                   Evaluate Preprocessed Test Data
# -------------------------------------------------------------------------
def make_prediction(preprocessing):
    """
    Loads the RNN model
    :param preprocessing: prepared DataPreprocess instance
    :return: the loaded model instance
    """
    rnn_model = load_model(MODEL_LOC)

    prediction = rnn_model.predict(preprocessing.padded_data,
                                   steps=len(preprocessing.padded_data) / BATCH_SIZE,
                                   verbose=1)
    return prediction


def evaluate_roc_auc(preprocessing, prediction_binary):
    """
    Evaluates the model
    :param preprocessing: prepared DataPreprocess instance
    :param prediction_binary: boolean expression for the predicted classes
    """
    aucs = []
    for j in range(len(DETECTION_CLASSES)):
        auc = roc_auc_score(preprocessing.target_classes[:, j], prediction_binary[:, j])
        aucs.append(auc)

    return np.mean(aucs)


def evaluate_accuracy_score(preprocessing, prediction_binary):
    """
    Evaluates the accuracy score
    :param preprocessing: prepared DataPreprocess instance
    :param prediction_binary: boolean expression for the predicted classes
    """
    accuracy = []
    for j in range(len(DETECTION_CLASSES)):
        acc = accuracy_score(preprocessing.target_classes[:, j], prediction_binary[:, j])
        accuracy.append(acc)

    return np.mean(accuracy)


# -------------------------------------------------------------------------
#                               Main Execution
# -------------------------------------------------------------------------
def execute():
    test_data = prepare_test_data()
    preprocessing = DataPreprocess(test_data, do_load_existing_tokenizer=True)
    prediction = make_prediction(preprocessing)
    roc_auc = evaluate_roc_auc(preprocessing, prediction > 0.5)
    accuracy = evaluate_accuracy_score(preprocessing, prediction > 0.5)

    print(f'Average ROC_AUC Score on Test Data: {roc_auc}')
    print(f'Average Accuracy Score on Test Data: {accuracy}')


if __name__ == '__main__':
    execute()
