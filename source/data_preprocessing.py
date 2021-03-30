__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import numpy as np
import pickle
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from config import *
from data_cleaning import clean_text_column


# -------------------------------------------------------------------------
#                           Utility Functions
# -------------------------------------------------------------------------
def sum_of_columns(dataframe, columns):
    """
    Calculates the sum of the specified columns from the specified dataframe
    :param dataframe: the dataframe
    :param columns: the columns of the dataframe
    :return: the sum of the specified columns
    """
    temp = 0
    for col in columns:
        temp += dataframe[col]
    return temp


# -------------------------------------------------------------------------
#                           Data Preprocessing
# -------------------------------------------------------------------------
class DataPreprocess:
    """
    Preprocesses the data
    """

    def __init__(self, data, do_load_existing_tokenizer=False):
        """
        Initializes and prepares the data with necessary steps either to be trained
        or evaluated by the RNN model
        :param data: the dataframe extracted from the .csv file
        :param do_load_existing_tokenizer: True if existing tokenizer should be loaded or False instead
        """
        self.data = data
        self.doLoadExistingTokenizer = do_load_existing_tokenizer

        # The pre-trained word vectors used (http://nlp.stanford.edu/data/glove.6B.zip)
        word_to_vector = {}
        with open(EMBEDDING_FILE_LOC) as file:
            # A space-separated text file in the format
            # word vec[0] vec[1] vec[2] ...
            for line in file:
                word = line.split()[0]
                word_vec = line.split()[1:]
                # converting word_vec into numpy array
                # adding it in the word_to_vector dictionary
                word_to_vector[word] = np.asarray(word_vec, dtype='float32')

        # Print the total words found
        print(f'Total of {len(word_to_vector)} word vectors are found.')

        # Add a new target class (label) 'neutral' to the dataframe
        cols = DETECTION_CLASSES.copy()
        cols.remove('neutral')
        data['neutral'] = np.where(sum_of_columns(data, cols) > 0, 0, 1)

        # Clean the comment texts
        data['comment_text'] = clean_text_column(data['comment_text'])

        # Split the data into feature and target labels
        comments = data['comment_text'].values
        self.target_classes = data[DETECTION_CLASSES].values

        if not do_load_existing_tokenizer:
            # Convert the comments (strings) into integers
            tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
            tokenizer.fit_on_texts(comments)
        else:
            with open(TOKENIZER_LOC, 'rb') as handle:
                tokenizer = pickle.load(handle)

        sequences = tokenizer.texts_to_sequences(comments)

        # Word to integer mapping
        word_to_index = tokenizer.word_index
        print(f'Found {len(word_to_index)} unique tokens')

        if not do_load_existing_tokenizer:
            # Save tokenizer
            print('Saving tokens ...')
            with open(TOKENIZER_LOC, 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Pad sequences so that we get a N x T matrix
        # TODO: check whether to choose post or pre padding
        self.padded_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        print('Shape of Data Tensor:', self.padded_data.shape)

        # Construct and Prepare Embedding matrix
        num_words = min(MAX_VOCAB_SIZE, len(word_to_index) + 1)
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIMENSION))
        for word, i in word_to_index.items():
            if i < MAX_VOCAB_SIZE:
                embedding_vector = word_to_vector.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all zeros.
                    embedding_matrix[i] = embedding_vector

        # Load pre-trained word embeddings into an embedding layer
        # Set trainable = False to keep the embeddings fixed
        self.embedding_layer = Embedding(num_words,
                                         EMBEDDING_DIMENSION,
                                         weights=[embedding_matrix],
                                         input_length=MAX_SEQUENCE_LENGTH,
                                         trainable=False)
