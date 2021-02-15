__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Importing the libraries
# -------------------------------------------------------------------------
import numpy as np
import pandas as pd
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# -------------------------------------------------------------------------
#                               Configurations
# -------------------------------------------------------------------------
EMBEDDING_DIMENSION = 100
EMBEDDING_FILE_LOC = '../glove/glove.6B.' + str(EMBEDDING_DIMENSION) + 'd.txt'
TRAINING_DATA_LOC = '../data/train.csv'
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100
BATCH_SIZE = 128
EPOCHS = 10
VALIDATION_SPLIT = 0.2
DETECTION_CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat',
                     'insult', 'identity_hate']


# -------------------------------------------------------------------------
#                           Data Preprocessing
# -------------------------------------------------------------------------
class DataPreprocess():
    """
    Preprocess the data
    """

    def __init__(self, data_location):
        """
        Initializes and prepares the data with necessary steps either to be trained
        or evaluated by the RNN model
        :param data_location: the location of the data
        """
        self.data_location = data_location

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

        # print the total words found
        print(f'Total of {len(word_to_vector)} word vectors are found.')

        # Import the training data
        training_data = pd.read_csv(data_location)

        # Split the data into feature and target labels
        comments = training_data['comment_text'].values
        self.target_classes = training_data[DETECTION_CLASSES].values

        # Convert the comments (strings) into integers
        tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
        tokenizer.fit_on_texts(comments)
        sequences = tokenizer.texts_to_sequences(comments)

        # Word to integer mapping
        word_to_index = tokenizer.word_index
        print(f'Found {len(word_to_index)} unique tokens')

        # pad sequences so that we get a N x T matrix
        self.padded_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        print('Shape of data tensor:', self.padded_data.shape)

        # Constructing the Embedding matrix

        # Prepare the embedding matrix
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
