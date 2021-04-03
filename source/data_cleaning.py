__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import nltk
import re
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

def convert_to_lower_case(text):
    """
    Coverts the specified text to lower case
    :param text: the text to convert
    :return: the lower cased text
    """
    return " ".join(text.lower() for text in text.split())


def apply_contraction_mapping(text):
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


def fix_misspelled_words2(text):
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


def fix_misspelled_words(text):
    """
    Fixes the misspelled words on the specified text (uses TextBlob model)
    :param text: The text to be fixed
    :return: the fixed text
    """
    b = TextBlob(text)
    return str(b.correct())


def remove_punctuations(text):
    """
    Removes all punctuations from the specified text
    :param text: the text whose punctuations to be removed
    :return: the text after removing the punctuations
    """
    return text.replace(r'[^\w\s]', '')


def remove_emojis(text):
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


def remove_stopwords(text):
    """
    Removes all stop words from the specified text
    :param text: the text whose stop words need to be removed
    :return: the text after removing the stop words
    """
    return " ".join(x for x in text.split() if x not in stop_words)


def lemmatise(text):
    """
    Lemmatises the specified text
    :param text: the text which needs to be lemmatised
    :return: the lemmatised text
    """
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


def clean_text_column(text_column):
    """
    Cleans the text specified in the text column
    The cleaning procedure is as follows:
    1. Convert the context to lower case
    2. Apply contraction mapping in which we fix shorter usages of english sentences
    3. Fix misspelled words
    4. Remove all punctuations
    5. Remove all emojis
    6. Remove all stop words
    7. Apply Lemmatisation
    :param text_column: the text column which needs to be cleaned
    :return: the text column with the cleaned data
    """
    return text_column.apply(lambda x: clean_text(x))


def clean_text(text):
    """
    Cleans the specified text
    The cleaning procedure is as follows:
    1. Convert the context to lower case
    2. Apply contraction mapping in which we fix shorter usages of english sentences
    3. Fix misspelled words
    4. Remove all punctuations
    5. Remove all emojis
    6. Remove all stop words
    7. Apply Lemmatisation
    :param text: the text which needs to be cleaned
    :return: the cleaned text
    """
    text = convert_to_lower_case(text)
    text = apply_contraction_mapping(text)
    text = fix_misspelled_words2(text)
    text = remove_punctuations(text)
    text = remove_emojis(text)
    text = remove_stopwords(text)
    text = lemmatise(text)

    return text
