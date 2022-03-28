""" Pick the best parameters of hyperparameter tuning and
    re-train them and save models this time.
    We can run the models 10 times to account for the random initialization
"""
import argparse
from distutils.command.config import config
import os
import pickle
from configs.model_hyper import hyper_configs
from configs.defaults import Globs, config_loader
from train.train_model import train
from helpers.wandb_common import get_wandb_df, restore_wandb_online

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--models', type=str,
    default=','.join(Globs.TUNABLE_MODELS))
parser.add_argument('-t', '--tune_tag', type=str, default='tuning_v4')
parser.add_argument('-n', '--num_rounds', type=int, default=5)
parser.add_argument('-l', '--defaults', type=str, default='y')
args = parser.parse_args()
models = args.models.split(',')

df = get_wandb_df(Globs.PROJECT_NAME, wandb_user=Globs.ENTITY_NAME)
df_tuning_models = df[df.tags.apply(lambda x: args.tune_tag in x)]

for model_name in models:
    if args.defaults == 'y':
        best_model_config = config_loader[model_name]
        best_model_config['model_name'] = model_name
        best_model_config['dataset'] = 'ydo_data'
        best_model_config['max_len'] = 1000
    else:
        # Find id of the best model
        df_models = df_tuning_models.loc[(df['model_name'] == model_name)]
        df_models.sort_values('best_val_acc', ascending=False, inplace=True)
        best_model_id = df_models.iloc[0].id
        # Restore the config of the best model
        restore_wandb_online(Globs.PROJECT_NAME, best_model_id, Globs.ENTITY_NAME,
            restore_files=['config.pickle'])

        config_path = os.path.join('wandb', best_model_id, 'config.pickle')
        best_model_config = pickle.load(open(config_path, 'rb'))
        best_model_config['tuning_model_id'] = best_model_id
        print(model_name, ': ', best_model_id)

    print(best_model_config)
    # Retrain with the same config, this time saving the model
    new_config = best_model_config.copy()
    new_config['tuning_run'] = False # So we use the actual test data for eval
    new_config['save_model'] = True
    for num_round in range(args.num_rounds):
        new_config['num_round'] = num_round
        for embedding_type in ['bert', 'word2vec']:
            new_config['embedding_type'] = embedding_type
            train(new_config, wandb_tags=['tuning_best'])
