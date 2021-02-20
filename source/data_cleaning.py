__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import re

import nltk
import spacy
from nltk.corpus import stopwords
from textblob import TextBlob

# -------------------------------------------------------------------------
#                        One-shot Instance Creation
# -------------------------------------------------------------------------
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nltk.download('stopwords')
stop_words = stopwords.words('english')


# -------------------------------------------------------------------------
#                           Data Cleaning
# -------------------------------------------------------------------------

def convert_to_lower_case_on_string(text):
    """
    Coverts the specified text to lower case
    :param text: the text to convert
    :return: the lower cased text
    """
    return " ".join(text.lower() for text in text.split())


def convert_to_lower_case(text_column):
    """
    Coverts the text in the specified column to lower case
    :param text_column: the text column whose context needs to be converted
    :return: the text column containing the lower cased text
    """
    return text_column.apply(
        lambda x: convert_to_lower_case_on_string(x))


def apply_contraction_mapping_on_string(text):
    """
    Applies the contraction mapping to the specified text
    :param text: the text on which the contraction will be mapped
    :return: the text after the application of contraction mapping
    """
    contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                           "could've": "could have", "couldn't": "could not",
                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                           "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
                           "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                           "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                           "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                           "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                           "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
                           "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                           "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                           "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                           "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                           "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                           "she'll've": "she will have", "she's": "she is", "should've": "should have",
                           "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                           "so's": "so as",
                           "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                           "that's": "that is",
                           "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                           "here's": "here is",
                           "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                           "they'll've": "they will have",
                           "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                           "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                           "we'll've": "we will have",
                           "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                           "what'll've": "what will have",
                           "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                           "when've": "when have",
                           "where'd": "where did", "where's": "where is", "where've": "where have",
                           "who'll": "who will",
                           "who'll've": "who will have",
                           "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                           "will've": "will have",
                           "won't": "will not", "won't've": "will not have", "would've": "would have",
                           "wouldn't": "would not",
                           "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                           "y'all'd've": "you all would have",
                           "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                           "you'd've": "you would have", "you'll": "you will",
                           "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])
    return text


def apply_contraction_mapping(text_column):
    """
    Applies the contraction mapping to the text in the specified column
    :param text_column: the text column on which the contraction will be mapped
    :return: the text column after the application of contraction mapping
    """
    return text_column.apply(lambda x: apply_contraction_mapping_on_string(x))


def fix_misspelled_words_on_string2(text):
    """
    Fixes the misspelled words on the specified text (uses predefined misspelled dictionary)
    :param text: The text to be fixed
    :return: the fixed text
    """
    mispelled_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                      'counselling': 'counseling',
                      'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization',
                      'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora',
                      'sallary': 'salary',
                      'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are',
                      'howcan': 'how can',
                      'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I',
                      'theBest': 'the best',
                      'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate',
                      "mastrubating": 'masturbating',
                      'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data',
                      '2k17': '2017', '2k18': '2018',
                      'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what',
                      'watsapp': 'whatsapp',
                      'demonitisation': 'demonetization', 'demonitization': 'demonetization',
                      'demonetisation': 'demonetization', ' ur ': 'your', ' u r ': 'you are'}
    for word in mispelled_dict.keys():
        text = text.replace(word, mispelled_dict[word])
    return text


def fix_misspelled_words_on_string(text):
    """
    Fixes the misspelled words on the specified text (uses TextBlob model)
    :param text: The text to be fixed
    :return: the fixed text
    """
    b = TextBlob(text)
    return str(b.correct())


def fix_misspelled_words(text_column):
    """
    Fixes the misspelled words on the text column
    :param text_column: The text column to be fixed
    :return: the text column containing the text
    """
    return text_column.apply(lambda x: fix_misspelled_words_on_string2(x))


def remove_punctuations_on_string(text):
    """
    Removes all punctuations from the specified text
    :param text: the text whose punctuations to be removed
    :return: the text after removing the punctuations
    """
    return text.replace('[^\w\s]', '')


def remove_punctuations(text_column):
    """
    Removes all punctuations from the text of the specified text column
    :param text_column: the text column whose punctuations to be removed
    :return: the text column after removing the punctuations
    """
    return remove_punctuations_on_string(text_column.str)


def remove_emojis_on_string(text):
    """
    Removes emojis from the specified text
    :param text: the text whose emojis need to be removed
    :return: the text after removing the emojis
    """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_emojis(text_column):
    """
    Removes emojis from the text of the specified column
    :param text_column: the text column whose emojis need to be removed
    :return: the text column after removing the emojis
    """
    return text_column.apply(lambda x: remove_emojis_on_string(x))


def remove_stopwords_on_string(text):
    """
    Removes all stop words from the specified text
    :param text: the text whose stop words need to be removed
    :return: the text after removing the stop words
    """
    return " ".join(x for x in text.split() if x not in stop_words)


def remove_stopwords(text_column):
    """
    Removes all stop words from the text of the specified column
    :param text_column: the text column whose stop words need to be removed
    :return: the text column after removing the stop words
    """
    return text_column.apply(
        lambda x: remove_stopwords_on_string(x))


def lemmatize_on_string(text):
    """
    Lemmatizes the specified text
    :param text: the text which needs to be lemmatized
    :return: the lemmatized text
    """
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


def lemmatize(text_column):
    """
    Lemmatizes the text of the specified text column
    :param text_column: the text column which needs to be lemmatized
    :return: the lemmatized text column
    """
    return text_column.apply(lemmatize_on_string)


def clean_text_column(text_column):
    """
    Cleans the data specified in the text column
    The cleaning procedure is as follows:
    1. Convert the context to lower case
    2. Apply contraction mapping in which we fix shorter usages of english sentences
    3. Fixe misspelled words
    4. Remove all punctuations
    5. Remove all emojis
    6. Remove all stop words
    7. Apply lemmatisation
    :return: the text column with the cleaned data
    """
    text_column = convert_to_lower_case(text_column)
    text_column = apply_contraction_mapping(text_column)
    text_column = fix_misspelled_words(text_column)
    text_column = remove_punctuations(text_column)
    text_column = remove_emojis(text_column)
    text_column = remove_stopwords(text_column)
    text_column = lemmatize(text_column)

    return text_column


def clean_text(text):
    """
    Cleans the specified text
    The cleaning procedure is as follows:
    1. Convert the context to lower case
    2. Apply contraction mapping in which we fix shorter usages of english sentences
    3. Fixe misspelled words
    4. Remove all punctuations
    5. Remove all emojis
    6. Remove all stop words
    7. Apply lemmatization
    :return: the cleaned text
    """
    text = convert_to_lower_case_on_string(text)
    text = apply_contraction_mapping_on_string(text)
    text = fix_misspelled_words_on_string(text)
    text = remove_punctuations_on_string(text)
    text = remove_emojis_on_string(text)
    text = remove_stopwords_on_string(text)
    text = lemmatize_on_string(text)

    return text
