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
#                   Prepare the Test Data
# -------------------------------------------------------------------------
test_label = pd.read_csv('../data/test_labels.csv')
test_comments = pd.read_csv('../data/test.csv')

# Merge the two dataframes into one for better handling
test_data = pd.merge(test_comments, test_label, on='id')

# Dropping the rows where correct label are not assigned
# In such rows, the all labels are filled with -1
test_data = test_data[test_data['toxic'] != -1]

# -------------------------------------------------------------------------
#                   Preprocess Test Data
# -------------------------------------------------------------------------
preprocessing = DataPreprocess(test_data)

# -------------------------------------------------------------------------
#                   Evaluate Preprocessed Test Data
# -------------------------------------------------------------------------
# load the trained CNN model
rnn_model = load_model(MODEL_LOC)
pred = rnn_model.predict(preprocessing.padded_data, steps=len(preprocessing.padded_data) / BATCH_SIZE, verbose=1)

pred_binary = pred > 0.5

aucs = []
for j in range(6):
    auc = roc_auc_score(preprocessing.target_classes[:, j], pred_binary[:, j])
    aucs.append(auc)
print(f'Average ROC_AUC Score on Test Data: {np.mean(aucs)}')

accuracy = []
for j in range(6):
    acc = accuracy_score(preprocessing.target_classes[:, j], pred_binary[:, j])
    accuracy.append(acc)
print(f'Average Accuracy Score on Test Data: {np.mean(accuracy)}')
