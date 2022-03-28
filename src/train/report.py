
import os
import wandb
from configs.defaults import Globs
from data.loader import data_loader
from helpers.wandb_common import restore_wandb_online, wandb_save
from metrics.report import KerasReporter, Reporter
from train.trainer import KerasModelWrapper
from helpers.gpu_selection import use_gpu

use_gpu(True)

def report_model(model_id, config: dict, wandb_tags=['report']):
    """ Creates a report of predictions using a pretrained model
    model_id: id of the pretrained model in Wandb
    config: configuration for reporting
        - dataset: name of test data
        - threshold: decision threshold for highlighting
        - generate_plots (bool): Plot ROC-AUC and PR-Recall Curves
        - report_debug (bool): highlights saved into docx files
    wandb_tags: special tags to seperate exps in wandb
    """
    wandb_save(is_online=True)
    run = wandb.init(project=Globs.PROJECT_NAME, entity=Globs.ENTITY_NAME,
        config=config,
        tags=wandb_tags, reinit=True)
    config = wandb.config
    restore_wandb_online(Globs.PROJECT_NAME, model_id, Globs.ENTITY_NAME,
        restore_files=['model.h5', 'config.pickle', 'keras_tokenizer.pickle'])

    _, test_data = data_loader(config)

    model = KerasModelWrapper(None, None, None)
    model.load(os.path.join('wandb', model_id))

    config['model'] = model.config.copy()
    reporter = KerasReporter(config, model, test_data, wandb)
    reporter.report_results()

    run.finish()

if __name__ == "__main__":
    # Example config
    config = dict(
        dataset = 'ydo_data',
        threshold = 0.2,

        generate_plots = False, # ROC-AUC and PR-Recall Plots
        save_documents = True, # Save preds as word files,
        report_debug = True, # Detailed report to files,
    )
    report_model(model_id='',
                 config=config,
                 wandb_tags=['report_extra'])