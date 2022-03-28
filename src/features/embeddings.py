from abc import ABC, abstractmethod
import os

from tqdm import tqdm
from deprecated import deprecated
import numpy as np
from tensorflow.keras.layers import Embedding
import emoji
from bert_serving.client import BertClient
from gensim.test.utils import common_texts, get_tmpfile, datapath
from gensim.models import Word2Vec, KeyedVectors
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

from configs.defaults import Globs
from helpers.timing import get_hr_min_sec_str

def embedding_selector(config):
    """ Pick up the embedding based on config """
    k = 'embedding_type'
    if config[k] == 'word2vec':
        return Word2VecEmbeddingGen
    elif config[k] == 'bert':
        return BERTEmbeddingGen

class EmbeddingGen(ABC):
    """ Embedding Matrix Generator """
    def __init__(self, config, data_tokenizer) -> None:
        self.config = config
        self.data_tokenizer = data_tokenizer
        if self.config.get('debug', False):
            self.word_limit = 1000
            self.nb_words = 1001
        else:
            self.word_limit = None
            self.nb_words = len(data_tokenizer.word_index) + 1
        if self.word_limit:
            self.words_items = list(self.data_tokenizer.word_index.items())[:1000]
            self.words_items = dict(self.words_items)
        else:
            self.words_items = self.data_tokenizer.word_index
        print('Generating the embedding matrix')
        self.cache_generate_embedding_matrix()
        print('Generated the embedding matrix')

    @abstractmethod
    def generate_embedding_matrix(self,):
        pass

    def generate_keras_embedding_layer(self,):
        return Embedding(self.nb_words,
            self.embedding_matrix.shape[1], # (nb_words, embedding_dim)
            weights = [self.embedding_matrix],
            trainable = False,
            mask_zero = False,
        )

    def cache_generate_embedding_matrix(self, dir='data/interim/embedding_cache/'):
        debug_mode = self.config.get('debug', False)
        if not os.path.exists(dir):
            os.makedirs(dir)
        embedding_type = self.config['embedding_type']
        cache_path = os.path.join(dir, embedding_type)
        full_cache_path = cache_path + '.npy' # np.save appends this auto
        if os.path.exists(full_cache_path) and not debug_mode:
            # Load from cache
            self.embedding_matrix = np.load(full_cache_path)
        else:
            # Generate and cache the embedding matrix
            self.generate_embedding_matrix()
            # Cache if we are not in debug mode
            if not debug_mode:
                np.save(cache_path, self.embedding_matrix)

class Word2VecEmbeddingGen(EmbeddingGen):
    def __init__(self, config, data_tokenizer) -> None:
        super().__init__(config, data_tokenizer)

    def generate_embedding_matrix(self, ):
        EMBEDDING_DIM = 200
        w2v_model = KeyedVectors.load_word2vec_format(
            Globs.WORD2VEC_EMB_DIR, binary=True, limit=self.word_limit)

        def embedding_index(word):
            return w2v_model.word_vec(word)

        self.embedding_matrix = np.zeros((self.nb_words, EMBEDDING_DIM))

        for word, i in self.words_items.items():
            if word in w2v_model.vocab:
                self.embedding_matrix[i] = embedding_index(word)

        # NOTE: OOV tokens get 1 as the token value and [0,0,0,...] zero embedding
        print('Null word embeddings: %d' %
            np.sum(np.sum(self.embedding_matrix, axis=1) == 0)) # 0 is null word embedding


class BERTEmbeddingGen(EmbeddingGen):
    def __init__(self, config, data_tokenizer) -> None:
        super().__init__(config, data_tokenizer)

    def generate_embedding_matrix(self, ):
        EMBEDDING_DIM = 768
        self.embedding_matrix = np.zeros((self.nb_words, EMBEDDING_DIM))
        words = list(self.words_items.keys())
        for idx, word in enumerate(words):
            if type(word) != str:
                print(f"Warning: Not str in tokenizer: {word}")
                words[idx] = str(word)

        token_ids = list(self.words_items.values())
        bc = BertClient(ip=Globs.BERTSERVER_IP)  # ip address of the server
        self.embed_sum(bc, words, token_ids)

        bc.close()

    def embed_sum(self, bc: BertClient, text, token_ids):
        print(f'{get_hr_min_sec_str()}-bertserver encoding started')
        result = bc.encode(text, show_tokens=True)
        print(f'{get_hr_min_sec_str()}-bertserver encoding ended')
        batch = []
        for sent, tensor, tokens, token_id in tqdm(zip(text, result[0], result[1], token_ids)):
            token_tensor = []
            sent_tensor = []
            tid = 0
            buffer = ''
            words = sent.lower().split()
            for i, t in enumerate(tokens):
                if t == '[CLS]' or t == '[SEP]':
                    continue
                else:
                    if t.startswith('##'):
                        t = t[2:]
                    buffer += t
                    token_tensor.append(tensor[i, :])
                    if buffer == words[tid]:
                        sent_tensor.append(np.stack(token_tensor).mean(axis=0))
                        token_tensor = []
                        buffer = ''
                        tid += 1
            if sent in emoji.UNICODE_EMOJI['en']: # Pass if emoji
                continue
            if len(sent_tensor) == 0:
                print(f'Embedding not found for:{sent}')
                continue
            self.embedding_matrix[token_id] = sent_tensor[0]

    @staticmethod
    @deprecated(""" Initially planned to load embeddings via async loading
                    No longer necessary since we are caching the embeddings.""")
    def embed_sum_async(bc: BertClient, text):
        # NOTE: Can implement BERT embeddings async for faster loading
        result = bc.encode_async(text, show_tokens=True)
        # print(result)
        batch = []
        for idx, a_result in enumerate(result):
            sent = text[idx]
            tensor, tokens = a_result[0], a_result[1]
            token_tensor = []
            sent_tensor = []
            tid = 0
            buffer = ''
            words = sent.lower().split()
            for i, t in enumerate(tokens):
                if t == '[CLS]' or t == '[SEP]':
                    continue
                else:
                    if t.startswith('##'):
                        t = t[2:]
                    buffer += t
                    token_tensor.append(tensor[i, :])
                    if buffer == words[tid]:
                        sent_tensor.append(np.stack(token_tensor).mean(axis=0))
                        token_tensor = []
                        buffer = ''
                        tid += 1
            # print(len(valid))
            # exit()
            # if tid != len(words) or len(sent_tensor) != len(words):
            #     print(sent.split())
            #     print(result[1])
            if sent in emoji.UNICODE_EMOJI['en']: # Pass if emoji
                continue
            if len(sent_tensor) == 0:
                print(f'Embedding not found for:{sent}')
                continue
            batch.append(np.stack(sent_tensor))
        yield batch


