__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Importing the libraries
# -------------------------------------------------------------------------
import gradio as gr
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from source.config import *

# load the trained model
rnn_model = load_model(MODEL_LOC)


def make_prediction(test_comment):
    """
    Predicts the toxicity of the specified comment
    """
    test_comment = [test_comment]
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(test_comment)
    sequences = tokenizer.texts_to_sequences(test_comment)

    padded_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    result = rnn_model.predict(padded_data, len(padded_data), verbose=1)
    print(result)

    return \
        {
            "Toxic": str(result[0][0]),
            "Severe": str(result[0][1]),
            "Obscene": str(result[0][2]),
            "Threat": str(result[0][3]),
            "Insult": str(result[0][4]),
            "Hate": str(result[0][5])
        }


input_comment = gr.inputs.Textbox(lines=15, placeholder="Enter your comment here")

title = "Comments Toxicity Detection"
description = "This application uses a Bidirectional Long short-term memory (LSTM) Recurrent Neural Network (RNN) " \
              "model to predict whether a comment is classified as toxic in nature"

gr.Interface(fn=make_prediction,
             inputs=input_comment,
             outputs="label",
             title=title,
             description=description) \
    .launch()
