class Globs:
    PROJECT_NAME = 'text_highlight'
    ENTITY_NAME = 'ryerson_dsl'
    BERTSERVER_IP = '10.8.0.94'

    ANNOT_TRAIN_PATH = 'data/raw/chat_docx_annotated_train'
    ANNOT_TEST_PATH = 'data/raw/chat_docx_annotated_test'
    MTSAMPLES_TEST_PATH = 'data/raw/annotator_agreement/mtsamples/user1/'
    WORD2VEC_EMB_DIR = 'data/raw/embeddings/wikipedia-pubmed-and-PMC-w2v.bin'

    EMBEDDING_TYPES = ['word2vec', 'bert']
    TUNABLE_MODELS = ['lstm', 'cnn_crf']

_best_cnn_crf_conf = dict(
    kernel_sizes = [1],
    dropout = 0,
    l1 = 0.001,
    l2 = 0.01,
    num_filters = 64
)

_best_lstm_conf = dict(
    dropout_lstm = 0.1,
    num_layers = 2,
    num_units = 50
)

config_loader = dict(
    cnn_crf = _best_cnn_crf_conf,
    lstm = _best_lstm_conf,
    embedding_dense = dict(),
)