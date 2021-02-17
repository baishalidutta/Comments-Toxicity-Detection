__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Importing Libraries
# -------------------------------------------------------------------------
import operator


# -------------------------------------------------------------------------
#                           Data Cleaning
# -------------------------------------------------------------------------

class DataClean:
    """
    Cleans text from the input source

    * Lowers all text
    * Removes uncommon punctuations
    * Expands abbreviations
    * Corrects misspelled words
    """

    def __init__(self, text):
        self.text = text
        self.vocab = {}
        self.known_words = {}
        self.unknown_words = {}
        self.nb_known_words = 0
        self.nb_unknown_words = 0
        # Contraction mapping
        self.contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                                    "could've": "could have", "couldn't": "could not",
                                    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                                    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
                                    "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                                    "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                                    "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
                                    "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                                    "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                                    "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
                                    "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                                    "mightn't": "might not",
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
                                    "they're": "they are", "they've": "they have", "to've": "to have",
                                    "wasn't": "was not",
                                    "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                                    "we'll've": "we will have",
                                    "we're": "we are", "we've": "we have", "weren't": "were not",
                                    "what'll": "what will",
                                    "what'll've": "what will have",
                                    "what're": "what are", "what's": "what is", "what've": "what have",
                                    "when's": "when is",
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
        # Clean special characters
        self.punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "euro", "™": "tm", "√": " square root ",
                              "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'",
                              '“': '"', '”': '"', "£": "pound", '∞': 'infinity', 'θ': 'theta',
                              '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                              '∅': '', '³': '3', 'π': 'pi', '$': 'dollar'}

        self.puncts = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~""“”’∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&"

        # Misspelled words and slangs
        self.mispelled_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite',
                               'travelling': 'traveling',
                               'counselling': 'counseling',
                               'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                               'organisation': 'organization',
                               'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora',
                               'sallary': 'salary',
                               'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are',
                               'howcan': 'how can',
                               'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I',
                               'theBest': 'the best',
                               'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate',
                               "mastrubating": 'masturbating',
                               'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist',
                               'bigdata': 'big data',
                               '2k17': '2017', '2k18': '2018',
                               'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess',
                               "whst": 'what',
                               'watsapp': 'whatsapp',
                               'demonitisation': 'demonetization', 'demonitization': 'demonetization',
                               'demonetisation': 'demonetization', ' ur ': 'your', ' u r ': 'you are'}

    def build_vocab(self):
        sentences = self.text.apply(lambda x: x.split()).values
        for sentence in sentences:
            for word in sentence:
                try:
                    self.vocab[word] += 1
                except KeyError:
                    self.vocab[word] = 1

    def check_coverage(self, embeddings_index):
        for word in self.vocab.keys():
            try:
                self.known_words[word] = embeddings_index[word]
                self.nb_known_words += self.vocab[word]
            except:
                self.unknown_words[word] = self.vocab[word]
                self.nb_unknown_words += self.vocab[word]
                pass

        print(f'Found embeddings for {len(self.known_words) / len(self.vocab):.2} of vocab')
        print(
            f'Found embeddings for '
            f'{self.nb_known_words / (self.nb_known_words + self.nb_unknown_words):.2} '
            f'of all text')
        self.unknown_words = sorted(self.unknown_words.items(), key=operator.itemgetter(1))[::-1]

    # -------------------------------------------------------------------------
    #                           Processing Functions
    # -------------------------------------------------------------------------

    # Add words to embedding
    def add_lower_upper(self, embedding):
        count = 0
        for word in self.vocab:
            if word.lower() in embedding and word not in embedding:
                embedding[word] = embedding[word.lower()]
                count += 1
            if word in embedding and word.lower() not in embedding:
                embedding[word.lower()] = embedding[word]
                count += 1
        print(f"Added {count} words to embedding")

    def clean_contractions(self, text):
        specials = ["’", "‘", "´", "`"]
        for s in specials:
            text = text.replace(s, "'")
        text = ' '.join([self.contraction_mapping[t] if t in self.contraction_mapping else t for t in text.split(" ")])
        return text

    def clean_special_chars(self, text):
        for p in self.punct_mapping:
            text = text.replace(p, self.punct_mapping[p])

        for p in self.puncts:
            text = text.replace(p, '')

        specials = {
            '\u200b': ' ',
            '…': ' ... ',
            '\ufeff': '',
            'करना': '',
            'है': ''}  # Other special characters that I have to deal with in last
        for s in specials:
            text = text.replace(s, specials[s])

        return text

    def correct_spelling(self, text):
        for word in self.mispelled_dict.keys():
            text = text.replace(word, self.mispelled_dict[word])
        return text

    def clean_text(self, text):
        """
        Cleans the specified text

            * Lowers all text
            * Removes uncommon punctuations
            * Expands abbreviations
            * Corrects misspelled words
        """
        text = text.lower()
        text = self.clean_contractions(text)
        text = self.clean_special_chars(text)
        text = self.correct_spelling(text)
        return text
