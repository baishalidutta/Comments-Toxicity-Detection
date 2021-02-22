__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import pickle

import gradio as gr
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from data_cleaning import clean_text

# -------------------------------------------------------------------------
#                     Load Existing Model and Tokenizer
# -------------------------------------------------------------------------

# load the trained model
rnn_model = load_model("comments_toxicity.h5")

# load the tokenizer
with open("tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)


# -------------------------------------------------------------------------
#                           Main Application
# -------------------------------------------------------------------------

def make_prediction(input_comment):
    """
    Predicts the toxicity of the specified comment
    :param input_comment: the comment to be verified
    """
    input_comment = clean_text(input_comment)
    input_comment = input_comment.split(" ")

    sequences = tokenizer.texts_to_sequences(input_comment)
    sequences = [[item for sublist in sequences for item in sublist]]

    padded_data = pad_sequences(sequences, maxlen=100)
    result = rnn_model.predict(padded_data, len(padded_data), verbose=1)

    return \
        {
            "Toxic": str(result[0][0]),
            "Very Toxic": str(result[0][1]),
            "Obscene": str(result[0][2]),
            "Threat": str(result[0][3]),
            "Insult": str(result[0][4]),
            "Hate": str(result[0][5]),
            "Neutral": str(result[0][6])
        }


comment = gr.inputs.Textbox(lines=17, placeholder="Enter your comment here")

title = "Comments Toxicity Detection"
description = "This application uses a Bidirectional Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) " \
              "model to predict the inappropriateness of a comment"

gr.Interface(fn=make_prediction,
             inputs=comment,
             outputs="label",
             title=title,
             description=description,
             article="http://raw.githubusercontent.com/baishalidutta/Comments-Toxicity-Detection/gradio/README.md") \
    .launch(share=True)
