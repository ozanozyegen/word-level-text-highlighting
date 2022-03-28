import os, wandb
import pandas as pd
from typing import Union

def wandb_save(is_online, disable_code=False):
    if not is_online:
        os.environ['WANDB_SILENT']='true'
        os.environ['WANDB_MODE'] = 'dryrun'
    if disable_code:
        os.environ['WANDB_DISABLE_CODE'] = 'true'

def convert_wandb_config_to_dict(
    wandb_config: Union[wandb.sdk.wandb_config.Config, dict]):
    if type(wandb_config) is dict:
        return wandb_config
    return dict(wandb_config.items())

def get_wandb_df(project_name, wandb_user):
    """ Extracts all the exps under a project from wandb
    """
    # Change oreilly-class/cifar to <entity/project-name>
    api = wandb.Api()
    runs = api.runs(f"{wandb_user}/{project_name}")
    summary_list = []
    config_list = []
    name_list, tags_list = [], []
    for run in runs:
        # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # run.config is the input metrics.  We remove special values that start with _.
        config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')})

        name_list.append(run.id)  # run.name is the name of the run.
        tags_list.append(run.tags) # run.tags are the tags associated with the run

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({'id': name_list, 'tags': tags_list})
    all_df = pd.concat([name_df, config_df,summary_df], axis=1)
    return all_df

def restore_wandb_online(project_name, model_id, entity_name,
                         TEMP_DATA_PATH='wandb/',
                         restore_files=['model.h5', 'config.pickle']):
    """
    """
    TEMP_DATA_PATH = os.path.join(TEMP_DATA_PATH, model_id)
    for file_name in restore_files:
        _ = wandb.restore(file_name,
            run_path=f"{entity_name}/{project_name}/{model_id}", root=TEMP_DATA_PATH)