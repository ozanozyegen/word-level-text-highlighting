import tensorflow as tf
import tensorflow.keras.backend as K

def acc_trim(y_true, y_pred):
    """ Binary Accuracy Calculation where paddings are ignored"""
    # assert y_true.shape == y_pred.shape
    y_true, y_pred = y_true[y_true != -1], y_pred[y_true != -1]
    y_true, y_pred = y_true[y_true != 2], y_pred[y_true != 2]
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def binary_cross_entropy(y_true, y_pred):
    y_true, y_pred = y_true[y_true != -1], y_pred[y_true != -1]
    y_true, y_pred = y_true[y_true != 2], y_pred[y_true != 2]
    return K.mean(
        K.binary_crossentropy(y_true, y_pred, from_logits=False), axis=-1
    )

def compute_dsc_loss(y_pred, y_true, alpha=0.6):
    """ CRF loss """
    y_pred = K.reshape(K.softmax(y_pred), (-1, y_pred.shape[2]))
    y = K.expand_dims(K.flatten(y_true), axis=1)
    probs = tf.gather_nd(y_pred, y, batch_dims=1)
    pos = K.pow(1 - probs, alpha) * probs
    dsc_loss = 1 - (2 * pos + 1) / (pos + 2)
    return dsc_loss