from abc import ABC, abstractmethod
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

from data.tf_data_loader import FixedLengthBatchGenerator, VaryingLengthBatchGenerator
from features.embeddings import embedding_selector
from helpers.wandb_common import convert_wandb_config_to_dict
from models.loader import model_loader
from models.losses import binary_cross_entropy, acc_trim
from metrics.callbacks import Metrics

def model_selector(*args, **kwargs):
    config = args[0]
    model_key = 'model_name'
    if config[model_key] == 'unknown':
        raise ValueError('Unknown model_name')
    else:
        return KerasModelWrapper(*args, **kwargs)


class ModelWrapper(ABC):
    def __init__(self, config, wandb) -> None:
        self.config = config
        self.wandb = wandb

    @abstractmethod
    def _initialize_model(self, config):
        pass

    @abstractmethod
    def train(self, train_data, test_data):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class KerasModelWrapper(ModelWrapper):
    def __init__(self, config, wandb, keras_tokenizer) -> None:
        super().__init__(config, wandb)
        self.keras_tokenizer = keras_tokenizer
        if config is not None: # NOTE: If trained model will be loaded
            self._initialize_model(config)

            self.epochs = self.config.get('epochs', 100)
            if config.get('debug', False):
                self.epochs = 1

    def _initialize_model(self, config):
        embedding_generator = embedding_selector(config)(config,
                                                         self.keras_tokenizer)
        self.model = model_loader[config['model_name']](config,
                                                        embedding_generator)

    def train(self, train_data, test_data):
        print('Loading data')
        train_gen = FixedLengthBatchGenerator(train_data[0], train_data[1],
            batch_size=self.config.get('batch_size', 32),
            padding_y=2, max_len=self.config.get('max_len', 1000))
        test_gen = FixedLengthBatchGenerator(test_data[0], test_data[1],
            batch_size=self.config.get('batch_size', 32),
            padding_y=2, max_len=self.config.get('max_len', 1000))
        print('Loaded,Fitting model')
        self.model.fit(train_gen, validation_data=test_gen,
            epochs=self.epochs,
            callbacks=self._get_callbacks(test_gen)
            )

    def predict(self, X):
        if self.config.get('load_as_unigram', False):
            pred = self.model.predict(X)
            return pred[:,0]
        else:
            inp = np.array(X).T # (1, # tokens)
            assert inp.shape[0] == 1
            pred = self.model.predict(inp)
            if pred.ndim == 2:
                return pred[0,:]
            elif pred.ndim == 3:
                return pred[0,:,0]

    def _get_callbacks(self, validation_data):
        model_dir = self.wandb.run.dir
        callbacks = [
            Metrics(validation_data),
            EarlyStopping(monitor='val_acc',
                          patience=self.config.get('patience', 10),
                          restore_best_weights=True),
            CSVLogger(os.path.join(model_dir, 'log.csv'),
                      append=True, separator=';'),
            self.wandb.keras.WandbCallback(monitor='val_acc',
                                           save_model=False),
        ]
        return callbacks

    def save_config(self, SAVE_DIR):
        pickle.dump(convert_wandb_config_to_dict(self.config),
                    open(os.path.join(SAVE_DIR, 'config.pickle'), 'wb'))
        pickle.dump(self.keras_tokenizer, open(os.path.join(SAVE_DIR, 'keras_tokenizer.pickle'), 'wb'))

    def save_model(self, SAVE_DIR):
        self.model.save_weights(os.path.join(SAVE_DIR, 'model.h5'))

    def load(self, SAVE_DIR):
        self.config = pickle.load(open(os.path.join(SAVE_DIR, 'config.pickle'), 'rb'))
        self.keras_tokenizer = pickle.load(open(os.path.join(SAVE_DIR, 'keras_tokenizer.pickle'), 'rb'))
        self._initialize_model(self.config)
        self.model.load_weights(os.path.join(SAVE_DIR, 'model.h5'))
