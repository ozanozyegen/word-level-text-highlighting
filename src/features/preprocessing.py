import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

import numpy as np
import gensim
from gensim.test.utils import common_texts, get_tmpfile, datapath
from gensim.models import Word2Vec, KeyedVectors
from configs.defaults import Globs

def get_embedding(config, data_tokenizer, EMBEDDING_DIM=200):
    if config.get('debug', False):
        word_limit = 1000
        nb_words = 1001
    else:
        word_limit = None
        nb_words = len(data_tokenizer.word_index) + 1

    w2v_model = KeyedVectors.load_word2vec_format(
        Globs.WORD2VEC_EMB_DIR, binary=True, limit=word_limit)

    def embedding_index(word):
        return w2v_model.word_vec(word)

    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    if word_limit:
        words_items = list(data_tokenizer.word_index.items())[:1000]
    else:
        words_items = data_tokenizer.word_index.items()
    for word, i in words_items:
        if word in w2v_model.vocab:
            embedding_matrix[i] = embedding_index(word)
    # NOTE: OOV tokens get 1 as the token value and [0,0,0,...] zero embedding
    print('Null word embeddings: %d' %
        np.sum(np.sum(embedding_matrix, axis=1) == 0)) # 0 is null word embedding

    return embedding_matrix, nb_words

def text_preprocessing(
    text,
    remove_stopwords=False,
    stem_words=False,
    stopwords_addition=[],
    stopwords_exclude=[],
    HYPHEN_HANDLE=2):

    """
    convert string text to lower and split into words.
    most punctuations are handled by replacing them with empty string.
    some punctuations are handled differently based on their occurences in the data.
    -  replaced with ' '
    few peculiar cases for better uniformity.
    'non*' replaced with 'non *'
    few acronyms identified
    SCID  Severe Combined ImmunoDeficiency
    ADA   Adenosine DeAminase
    PNP   Purine Nucleoside Phosphorylase
    LFA-1 Lymphocyte Function Antigen-1
    """
    text = text.lower().split()

    if remove_stopwords:
        stops = list(set(stopwords.words('english')) -
                     set(stopwords_exclude)) + stopwords_addition
        text = [w for w in text if w not in stops]

    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=_]", " ", text)
    text = re.sub(r"(that|what)(\'s)", r"\g<1> is", text)
    text = re.sub(r"i\.e\.", "that is", text)
    text = re.sub(r"(^non| non)", r"\g<1> ", text)
    text = re.sub(r"(^anti| anti)", r"\g<1> ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    #if HYPHEN_HANDLE == 1:
    #    text = re.sub(r"\-", "-", text)
    #elif HYPHEN_HANDLE == 2:
    text = text.replace("-", " ")
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r";", " ; ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.lower().split()

    if remove_stopwords:
        stops = list(set(stopwords.words('english')) -
                     set(stopwords_exclude)) + stopwords_addition
        text = [w for w in text if w not in stops]

    text = " ".join(text)

    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return(text)

def get_prediction(sentence, model, tokenizer, reversed_word_map,
    stopwords):
    org_words = sentence.split()
    sentence = org_words
    # sentence = text_preprocessing(sentence)
    seq = tokenizer.texts_to_sequences([sentence])

    seq = seq[0]
    result = []
    for idx in range(0, len(seq)):
        category = model.predict(np.atleast_2d([seq[idx]]))
        # print(category)
        if reversed_word_map[seq[idx]] in stopwords:
            cat = 0
        else:
            cat = category.argmax()
        result.append((reversed_word_map[seq[idx]], cat, org_words[idx]))
    assert len(org_words) == len(seq)
    return result

def create_annot_dataset(annot_seq, annot_data_hi, NGRAM):
    """ Creates a training dataset from YDO annotations for an NGRAM keras model
    Returns:
    X_train: (num_phrases, NGRAM) seq_ids
    y_train: (num_phrases) 0-1
    """
    X, y = [], []
    for para_seq, para_seq_hi in zip(annot_seq, annot_data_hi):
        if len(para_seq) < NGRAM:
            continue
        for i in range(len(para_seq)-NGRAM+1):
            ngram_seq, ngram_seq_hi = para_seq[i:i+NGRAM], para_seq_hi[i:i+NGRAM]
            X.append(ngram_seq)
            y.append( int(sum(ngram_seq_hi) == len(ngram_seq_hi)) ) # If all words are highlighted in the ngram
    return np.array(X), np.array(y)

def load_as_unigram(lst: list):
    return [[ele] for l in lst for ele in l]