from models.model_gen import lstm_tunable, cnn_crf
from models.emb_dense import embedding_dense

model_loader = dict(
    lstm = lstm_tunable,
    cnn_crf = cnn_crf,
    embedding_dense = embedding_dense,
)