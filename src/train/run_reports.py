""" Pick the models trained in the tune_best_train.py and
    report the results on all datasets
"""
import argparse
import os
import pickle
from configs.defaults import Globs
from helpers.wandb_common import get_wandb_df, restore_wandb_online
from train.report import report_model


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', type=str,
    default=','.join(['ydo_data', 'mtsamples']))
parser.add_argument('-m', '--models', type=str,
    default=','.join(['lstm', 'cnn_crf', 'embedding_dense']))
parser.add_argument('-e', '--embedding_types', type=str,
    default=','.join(Globs.EMBEDDING_TYPES))
parser.add_argument('-t', '--tune_tag', type=str,
    default='tuning_best')
args = parser.parse_args()
datasets = args.datasets.split(',')
models = args.models.split(',')
embedding_types = args.embedding_types.split(',')

df = get_wandb_df(Globs.PROJECT_NAME, wandb_user=Globs.ENTITY_NAME)
df_tuning_models = df[df.tags.apply(lambda x: args.tune_tag in x)]

report_config = dict(
    threshold = 0.5,
    generate_plots = True,
    save_documents = False,
    report_debug = False
)

for model_name in models:
    for embedding_type in embedding_types:
        for dataset in datasets:
            report_config['embedding_type'] = embedding_type
            report_config['model_name'] = model_name
            # Find id of the best model
            df_models = df_tuning_models.loc[(df['model_name'] == model_name) &
                                    (df['embedding_type'] == embedding_type)]
            df_models.sort_values('best_val_acc', ascending=False, inplace=True)
            if len(df_models) == 0:
                print(f'Model not found for: {model_name}|{dataset}|{embedding_type}')
            else: # Generate the report
                report_config['dataset'] = dataset
                best_model_id = df_models.iloc[0].id
                report_config['model_id'] = best_model_id
                report_model(best_model_id,
                             report_config)