# Traitement de données
# import pandas as pd
# from collections import defaultdict

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Traitement de données
import numpy as np

# Traitement de texte
import nltk
import bs4
# from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import time

# from scipy.sparse import csr_matrix


# Vectorisation de la target multilabel
# from sklearn.preprocessing import MultiLabelBinarizer

# Enregistrement de données
# from scipy.sparse import save_npz
# import pickle

# TFIDF Feature extraction
# from sklearn.feature_extraction.text import TfidfVectorizer

# Feature selection
# from skmultilearn.model_selection import iterative_train_test_split

# ML
# from sklearn.linear_model import SGDClassifier
# from sklearn.multiclass import OneVsRestClassifier

# Score
# from sklearn.metrics import jaccard_score
"""
# Word2 vec feature extraction
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import metrics as kmetrics

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# Génération de features
import gensim
"""
# BERT
import tensorflow_hub as hub

# Bert

# import transformers
# from transformers import AutoTokenizer, TFAutoModel

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

"""
os.environ["TF_KERAS"]='1'
print(tf.__version__)
print(tensorflow.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.test.is_built_with_cuda())
"""
# Tokenizer
tokenizr = nltk.RegexpTokenizer(r'[a-zA-Z]+')


def tokenizer_fct(sentence, html=False, tags=False):
    # print(sentence)
    if html:
        sentence_clean = bs4.BeautifulSoup(sentence, features="html.parser").text.lower()
    else:
        sentence_clean = sentence.lower()

    if tags:
        word_tokens = tokenizr.tokenize(sentence_clean)
    else:
        word_tokens = tokenizr.tokenize(sentence_clean)

    return word_tokens


# Stop words
stop_w = list(tuple(nltk.corpus.stopwords.words('english')))


def stop_word_filter_fct(list_words):
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2 and len(w) < 22]

    return filtered_w2


# Lemmatizer (base d'un mot)

def lemma_fct(list_words):
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]

    return lem_w


# Création de liste pour les tags multilabel
def spl(x):
    return x.split(' ')


#    if tags:
#       return lem_w
#  else:
#     transf_desc_text = ' '.join(lem_w)

# Fonction de préparation du texte pour le bag of words avec lemmatization (Countvectorizer et Tf_idf, Word2Vec)
def transform_bow_lem_fct(desc_text, html=False, tags=False):
    word_tokens = tokenizer_fct(desc_text, html, tags)
    sw = stop_word_filter_fct(word_tokens)
    lem_w = lemma_fct(sw)

    return lem_w


# Fonction de préparation du texte pour le bag of words avec lemmatization (Countvectorizer et Tf_idf, Word2Vec)
def transform_bow_lem_split_fct(desc_text, html=False, tags=False):
    word_tokens = tokenizer_fct(desc_text, html, tags)
    sw = stop_word_filter_fct(word_tokens)
    lem_w = lemma_fct(sw)

    return ' '.join(lem_w)


# Fonction de préparation du texte pour le Deep learning (USE et BERT)
def transform_dl_fct(desc_text, html=False, tags=False):
    word_tokens = tokenizer_fct(desc_text, html, tags)
    # sw = stop_word_filter_fct(word_tokens)
    # lw = lower_start_fct(word_tokens)
    # lem_w = lemma_fct(lw)
    transf_desc_text = ' '.join(word_tokens)
    return transf_desc_text

def feature_USE_fct(sentences, b_size):
    batch_size = b_size
    time1 = time.time()

    for step in range(len(sentences) // batch_size):
        idx = step * batch_size
        feat = embed(sentences[idx:idx + batch_size])

        if step == 0:
            features = feat
        else:
            features = np.concatenate((features, feat))

    time2 = np.round(time.time() - time1, 0)
    return features
