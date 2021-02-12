<p align="center">
  <img width="781" alt="comments-toxicity-detection-logo" src="https://user-images.githubusercontent.com/76659596/105877123-eb112280-5fff-11eb-9425-8432e693f92e.png">
</p>

This repository consists of the machine learning model to rate toxic comments from social networks. The data has been extracted from [Kaggle toxic comments classification challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

### The Model Idea

* <b>Clean text</b>: lower all text, remove uncommon signs, expand abbreviations and correct mispellt words
* <b>Tokenize text</b> data
* Create <b>Embedding Vector</b> using [Glove.6B](https://nlp.stanford.edu/projects/glove/)
* Train a deep learning network with a <b>bidirectional LSTM</b> layer followed by two fully connected layers

### Installing Required Packages

```bash
pip install -r requirements.txt
```
### Build Application

```bash
python src/comments_toxicity.py
```

The app is built using Flask Web Framework.

Then go to your browser at `http://localhost:80`. The app should be up and running
