import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Bidirectional, \
    TimeDistributed, Masking, Reshape, Lambda, Conv1D, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.models import Model
from deprecated import deprecated

from features.embeddings import EmbeddingGen
from models.losses import acc_trim, binary_cross_entropy
from models.crf_model import ModelWithCRFLoss
from models.crf import CRF
from models.cnns import conv_layers

def lstm_tunable(config, embedding_gen: EmbeddingGen, add_cnn_feats=False):
    """ This will be a model that reads the full text and output a prediction for each word
    Tuning Params:
        num_layers (int): Number of Bidirectional LSTM layers
            default: 1
        num_units (int): number of units for the LSTM layers
            default: 100
        dropout_lstm (float 0-1): dropout rate for the LSTM layers
            default: 0
    """
    embedding_layer = embedding_gen.generate_keras_embedding_layer()
    embedding_dim = embedding_gen.embedding_matrix.shape[1]

    sequence_input = Input(shape=(None,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    if add_cnn_feats: # Turns the model to CNN-LSTM by adding extracted CNN feat
        conv_outs = conv_layers(embedded_sequences,
                               (None, embedding_dim), config)
        embedded_inp = Concatenate(axis=-1)([embedded_sequences, conv_outs])
    else:
        embedded_inp = embedded_sequences
    # Add LSTM layers over the input features
    num_layers = config.get('num_layers', 1)
    for lstm_layer_num in range(num_layers):
        if lstm_layer_num == 0:
            x = Bidirectional(LSTM(config.get('num_units', 100),
                                return_sequences=True,
                                activation='sigmoid',
                                dropout=config.get('dropout_lstm', 0)),
                                name=f'LSTM_{lstm_layer_num}')\
                                (embedded_inp)
        else:
            x = Bidirectional(LSTM(config.get('num_units', 100),
                                return_sequences=True,
                                activation='sigmoid',
                                dropout=config.get('dropout_lstm', 0)),
                                name=f'LSTM_{lstm_layer_num}')\
                                (x)

    out = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    out = Lambda(lambda x: K.squeeze(x, 2))(out)

    model = Model(inputs=[sequence_input], outputs=out)
    model.compile(loss=binary_cross_entropy, optimizer='adam', metrics=[acc_trim])
    print(model.summary())
    return model

def crf_many_many(config, embedding_gen: EmbeddingGen, add_cnn_feats=False):
    """
    Tuning Params:
        l1 (float 0-1): l1 regularization value
            default: 0
        l2 (float 0-1): l2 regularization value
            default: 0
    """
    regularization_param = l1_l2(config.get('l1', 0), config.get('l2', 0))
    embedding_layer = embedding_gen.generate_keras_embedding_layer()
    embedding_dim = embedding_gen.embedding_matrix.shape[1]

    sequence_input = Input(shape=(None,), dtype='int32')
    if add_cnn_feats: # Turns the model to CNN-LSTM by adding extracted CNN feat
        conv_outs = conv_layers(embedded_sequences,
                               (None, embedding_dim), config)
        embedded_inp = Concatenate(axis=-1)([embedded_sequences, conv_outs])
    else:
        embedded_inp = embedded_sequences
    out_lstm = Bidirectional(LSTM(64, return_sequences=True))(embedded_sequences)
    crf_layer = CRF(units=3, regularizer=regularization_param)
    out = crf_layer(out_lstm)

    base_model = Model(inputs=[sequence_input], outputs=out)
    crf_model = ModelWithCRFLoss(base_model, sparse_target=True)

    crf_model.compile(optimizer='adam', metrics=[acc_trim])

    return crf_model

def cnn_crf(config, embedding_gen: EmbeddingGen, add_cnn_feats=True):
    return crf_many_many(config, embedding_gen, add_cnn_feats)