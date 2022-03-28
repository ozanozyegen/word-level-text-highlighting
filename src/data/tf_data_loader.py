from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class FixedLengthBatchGenerator(Sequence):
    """Generates data for Tensorflow with fixed timesteps for
    many to many binary classification (like text highlighting)
    Each batch has timesteps max_len
    Arguments:
        X: List of lists, keras tokenizer sequences
        y: List of lists, highlights for each token
        padding_x: padding used to replace empty X values,
            masking applied to skip these elements in the model
        padding_y: padding used to replace empty y values,
            custom loss function applied to ignore these values in loss function
            this padding value must match with the custom loss function
        max_len: maximum timesteps (words) allowed
    """
    def __init__(self, X, y, batch_size=1, shuffle=True, padding_x=0, padding_y=-1, max_len = 1000):
        'Initialization'
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.padding_x, self.padding_y = padding_x, padding_y
        self.max_len = max_len
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y)/self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        Xb, yb = [], []
        for index in batch_indexes:
            Xb.append(self.X[index])
            yb.append(self.y[index])

        # NOTE: Truncation here causes data loss in long texts
        # some parts of the dataset might not be used
        Xb_padded = pad_sequences(Xb, padding='post',
                                  value=self.padding_x, maxlen=self.max_len)
        yb_padded = pad_sequences(yb, padding='post',
                                  value=self.padding_y, maxlen=self.max_len)

        assert Xb_padded.shape == yb_padded.shape # (batch_size, words)
        # return Xb_padded, np.expand_dims(yb_padded, axis=-1)
        return Xb_padded, yb_padded

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


class VaryingLengthBatchGenerator(Sequence):
    """Generates data for Tensorflow with varying timesteps for
    many to many binary classification (like text highlighting)
    Each batch has timesteps equal to the max timesteps length in the batch or to max_len
    Arguments:
        X: List of lists, keras tokenizer sequences
        y: List of lists, highlights for each token
        padding_x: padding used to replace empty X values,
            masking applied to skip these elements in the model
        padding_y: padding used to replace empty y values,
            custom loss function applied to ignore these values in loss function
            this padding value must match with the custom loss function
        max_len: Maximum timesteps allowed, longer samples are trimmed to max_len
    """
    def __init__(self, X, y, batch_size=1, shuffle=True, padding_x=0, padding_y=-1, max_len = 1000):
        'Initialization'
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.padding_x, self.padding_y = padding_x, padding_y
        self.max_len = max_len
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y)/self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        Xb, yb = [], []
        for index in batch_indexes:
            Xb.append(self.X[index])
            yb.append(self.y[index])

        Xb_padded = pad_sequences(Xb, padding='post', value=self.padding_x)
        yb_padded = pad_sequences(yb, padding='post', value=self.padding_y)
        if Xb_padded.shape[1] > self.max_len:
            Xb_padded = Xb_padded[:, :self.max_len]
            yb_padded = yb_padded[:, :self.max_len]

        assert Xb_padded.shape == yb_padded.shape
        return Xb_padded, yb_padded

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)