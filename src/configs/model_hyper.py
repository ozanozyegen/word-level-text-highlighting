import itertools

from models.lstm import cnn_crf

def _get_permutations(hyper_config_dict):
    configs = []
    keys, values = zip(*hyper_config_dict.items())
    for experiment in itertools.product(*values):
        configs.append({key:value for key, value in zip(keys, experiment)})
    return configs

_lstm_configs = dict(
    num_layers = [1, 2, 3],
    num_units = [50, 100, 150],
    dropout_lstm = [0, 0.01, 0.1],
)

_cnn_crf_configs = dict(
    kernel_sizes = [
        [1],
        [1, 3, 5]
    ],
    num_filters = [32, 64, 128],
    dropout = [0, 0.2, 0.5],
    l1 = [0, 0.001, 0.01],
    l2 = [0, 0.001, 0.01],
)

_cnn_configs = dict(
    kernel_sizes = [
        [1],
        [1, 3],
        [1, 3, 5]
    ],
    num_filters = [32, 64, 128],
    dropout = [0, 0.2, 0.5],
)

_crf_configs = dict(
    l1 = [0, 0.001, 0.01, 0.1],
    l2 = [0, 0.001, 0.01, 0.1],
)

hyper_configs = dict(
    lstm = _get_permutations(_lstm_configs),
    cnn = _get_permutations(_cnn_configs),
    crf = _get_permutations(_crf_configs),
    cnn_lstm = _get_permutations(_lstm_configs),
    cnn_crf = _get_permutations(_cnn_crf_configs),
)