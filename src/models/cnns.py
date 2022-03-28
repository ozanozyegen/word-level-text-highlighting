from ast import Lambda
from tensorflow.keras.layers import Input, Reshape, LSTM, Dense, Conv1D,\
    BatchNormalization, Dropout, Activation, GlobalAveragePooling1D,\
    Concatenate, Lambda, Conv2D
import tensorflow.keras.backend as K
from tensorflow.python.keras.models import Model

from features.embeddings import EmbeddingGen
from models.losses import acc_trim, binary_cross_entropy

def conv_layers(inp, inp_shape, config):
    conv_outs = []
    for kernel_size in config.get('kernel_sizes', [1,3,5]):
        conv_layer = Conv1D(
            filters=config.get('num_filters', 128),
            kernel_size=kernel_size,
            input_shape=inp_shape,
            strides=1,
            padding='same',
            activation='relu',
            name=f"Conv_1D_1x{kernel_size}"
        )
        conv_outs.append(conv_layer(inp))
    # Concatenate extracted features from convolutions with different kernel sizes
    if len(conv_outs) > 1:
        conv_outs = Concatenate(axis=-1)(conv_outs)
    else:
        conv_outs = conv_outs[0]
    return conv_outs


def cnn(config, embedding_gen: EmbeddingGen):
    """ Convolutional model that uses 1D convolutions to predict the highlights
    padding_type:
        causal: output[t] does not depend on output[t-1]
        same: with stride 1, even padding from left and right
              output size = input size
        using same padding is fine since highlighting is done in the full input

    Architecture:
        Since the highlighting task does not require complex features, a single
        layer of 1D Convs with different kernel sizes are used.
    Tuning Params:
        kernel_sizes (list): parallel convolution kernels, similar to resnet
            default: [1,3,5]
        num_filters (int): number of filters for the kernels
            default: 128
        dropout (float 0-1): dropout rate
            default: 0
    """
    max_len = config['max_len']
    embedding_layer = embedding_gen.generate_keras_embedding_layer()
    embedding_dim = embedding_gen.embedding_matrix.shape[1]

    sequence_input = Input(shape=(None,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    conv_outs = conv_layers(embedded_sequences, (None, embedding_dim), config)

    drop_out = Dropout(config.get('dropout', 0))(conv_outs)
    out_dense = Dense(1, activation='sigmoid')(drop_out)
    out = Lambda(lambda x: K.squeeze(x, 2))(out_dense)

    model = Model(inputs=[sequence_input], outputs=out)
    model.compile(loss=binary_cross_entropy, optimizer='adam', metrics=[acc_trim])
    print(model.summary())
    return model
