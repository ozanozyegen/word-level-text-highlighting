from configs.defaults import Globs
from data.load_ext import load_ext_data
from data.annotations import Annotations
from tensorflow.keras.preprocessing.text import Tokenizer

def get_fit_keras_tokenizer(config):
    if config.get('debug', False):
        word_limit = 1000
    else:
        word_limit = None

    medical_data, non_medical_data = load_ext_data(config,
        medical_dataset_names=['icd', 'umls'],
        non_medical_dataset_names=['non_medical']
    )
    annotations = Annotations(config, Globs.ANNOT_TRAIN_PATH)
    annot_data, _ = annotations.get_documents_and_highlights()

    tokenizer = Tokenizer(oov_token=1, filters=[], num_words=word_limit)
    tokenizer.fit_on_texts(medical_data+non_medical_data+annot_data)
    reversed_word_map = dict(map(reversed, tokenizer.word_index.items()))
    return tokenizer, reversed_word_map