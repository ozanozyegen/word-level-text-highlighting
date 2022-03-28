""" Code for loading data from external sources
    Datasets: UMLS, ICD, Non-medical, ydo_df
"""
from data.text_preprocessing import text_preprocessing

import numpy as np
import pandas as pd
import json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import itertools
from nltk import ngrams
from functools import partial
pre = partial(text_preprocessing, HYPHEN_HANDLE = 1)

def load_umls_data(config, DATA_DIR):
    UMLS_JSON = DATA_DIR + 'umls_updated.json'
    umls = json.load(open(UMLS_JSON)).values()
    print(f"#UMLS samples: {len(umls)}")
    if config.get('debug', False):
        umls = list(umls)[:100]
    # Take the samples with less than 30 characters
    umls_split = [words for words in umls if len(words) < 30]
    print(f"#UMLS samples after preprocessing: {len(umls_split)}")
    return umls_split

def load_icd_data(config, DATA_DIR):
    MEDICAL_CSV = DATA_DIR + 'icd_10_2017.csv'
    df_icd = pd.read_csv(MEDICAL_CSV, header=None, usecols=[3, 4])
    pos_cls = df_icd[4].tolist()
    if config.get('debug', False):
        pos_cls = pos_cls[:100]
    # pos_cls = [pre(_) for _ in pos_cls]
    # pos_cls = [x for x in pos_cls if len(x.split()) > 10]
    return pos_cls

def load_ydo_data(config, DATA_DIR, return_df=False):
    MEDICAL_CSV_2 = DATA_DIR + 'df_with_other.csv'
    df_other = pd.read_csv(MEDICAL_CSV_2)
    if return_df:
        return df_other
    return df_other['combined'].to_list()

def load_non_medical(config, DATA_DIR):
    NON_MEDICAL = DATA_DIR + 'big.txt'
    with open(NON_MEDICAL, encoding="utf-8") as file:
        neg_cls = [_.strip() for _ in " ".join([l.strip() for l in file]).split(".")]
    if config.get('debug', False):
        neg_cls = neg_cls[:100]
    neg_cls = [pre(_) for _ in neg_cls]
    # print("%d lines in neg_cls data." % len(neg_cls))
    return neg_cls


datasets = {'umls':load_umls_data, 'icd':load_icd_data, 'ydo':load_ydo_data, 'non_medical':load_non_medical}

def load_ext_data(config, medical_dataset_names:list,
                  non_medical_dataset_names:list,
                  DATA_DIR = 'data/raw/MCH/'):
    """ Loads the medical and non-medical data raw source
    Returns medical and non-medical texts is list of lists format
    """
    pos_cls = []
    for data_name in medical_dataset_names:
        data = datasets[data_name](config, DATA_DIR=DATA_DIR)
        print(f'Dataset: {data_name} \nNum docs: {len(data)}')
        pos_cls += data

    neg_cls = []
    for data_name in non_medical_dataset_names:
        data = datasets[data_name](config, DATA_DIR=DATA_DIR)
        print(f'Dataset: {data_name} \nNum docs: {len(data)}')
        neg_cls += data
    print(f'Total medical samples: {len(pos_cls)}')
    print(f'Total non-medical samples: {len(neg_cls)}')
    return pos_cls, neg_cls

def create_dataset(medical_seq, non_medical_seq, NGRAM=-1):
    """Create datasets for keras models
    Can generate datasets with
    - Only Unigrams
    - Unigrams and Bigrams and Trigrams
    - Varying input lengths

    Args:
        medical_seq (list): Medical input
        non_medical_seq (list): Non Medical input
        NGRAM (int, optional): Specifies the number of n-grams used

    Raises:
        ValueError: Unknown NGRAM value

    Returns:
        X_train ... y_test: Dataset splits
    """
    if NGRAM == 1:
        flat_medical_seq = list(itertools.chain.from_iterable(medical_seq))
        flat_non_medical_seq = list(itertools.chain.from_iterable(non_medical_seq))
        X = flat_medical_seq + flat_non_medical_seq
        y = [1] * len(flat_medical_seq) + [0] * len(flat_non_medical_seq)
        X, y = np.array(X).reshape(-1, 1), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0)

    elif NGRAM == 2 or NGRAM == 3:
        X, y = [], []
        for sequence in medical_seq:
            n_grams = ngrams(sequence, NGRAM)
            X += n_grams
        sz_medical = len(X)
        y += [1] * sz_medical
        for sequence in non_medical_seq:
            n_grams = ngrams(sequence, NGRAM)
            X += n_grams
        sz_nonmedical = len(X) - sz_medical
        y += [0] * sz_nonmedical
        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0)
    elif NGRAM == -1:
        raw_X = medical_seq + non_medical_seq
        raw_y = [np.array([1]*len(seq))  for seq in medical_seq] + [np.array([0]*len(seq)) for seq in non_medical_seq]

        # Varying length version
        X, y = raw_X, raw_y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    else:
        raise ValueError("Unknown NGRAM")

    return X_train, X_test, y_train, y_test
