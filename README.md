# Comments Toxicity Detection

This repository comprises the machine learning model to rate toxic comments from social networks. The data has been extracted from [Kaggle toxic comments classification challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

### The Model Idea

* Clean text: lower all text, remove uncommon signs, expand abbreviations, correct mispellt words
* Tokenizing text data
* Create embedding vector using [Glove.6B](https://nlp.stanford.edu/projects/glove/)
* Train a deeplearning network with a bidirectional LSTM layer followed by two fully connected layers.

### Installing Required Packages

```bash
pip install -r requirements.txt
```
### Building Application

```bash
python src/comments_toxicity.py
```

The app is built using Flask API.

Then go to your browser at http://0.0.0.0:80. The app should be up and running
