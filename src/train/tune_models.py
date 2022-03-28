""" Run hyperparameter tuning on selected models
"""
import argparse
from configs.model_hyper import hyper_configs
from train.train_model import train

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--models', type=str,
    default=','.join(['lstm', 'cnn_crf']))
args = parser.parse_args()

models = args.models.split(',')

base_config = dict(
    debug=False,
    dataset='ydo_data',
    embedding_type='bert',
    save_model=False,
    max_len=1000,
    threshold=0.5,
    tuning_run=True
)

continue_flag = False

for model_name in models:
    all_permutations = hyper_configs[model_name]
    for permutation_config in all_permutations:
        config = base_config.copy()
        config['model_name'] = model_name
        config.update(permutation_config)
        train(config, wandb_tags=['tuning'])
