import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Bidirectional, \
    TimeDistributed, Masking, Reshape, Lambda, Conv1D
from tensorflow.keras.models import Model
from features.embeddings import EmbeddingGen

from models.losses import acc_trim, binary_cross_entropy

def embedding_dense(config, embedding_gen: EmbeddingGen):
    """ This will be a model that reads the full text and output a prediction for each word
        It only uses embeddings as a dense layer to make the predictions
        NOTE: Embeddings are not time distributed
    """
    max_len = config['max_len']
    embedding_layer = embedding_gen.generate_keras_embedding_layer()

    sequence_input = Input(shape=(None,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    out_dense = Dense(1, activation='sigmoid')(embedded_sequences)
    out = Lambda(lambda x: K.squeeze(x, 2))(out_dense)

    model = Model(inputs=[sequence_input], outputs=out)
    model.compile(loss=binary_cross_entropy, optimizer='adam', metrics=[acc_trim])
    print(model.summary())
    return model