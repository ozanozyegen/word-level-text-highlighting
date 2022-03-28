from data.annotations import Annotations
from data.mtsamples import MtsamplesDataset
from configs.defaults import Globs


def data_loader(config):
    ''' Simple dataset loader
        Generates the correct dataset based on exp configuration
    '''
    dataset_name = config.get('dataset')
    if dataset_name == 'ydo_data':
        if config.get('tuning_run', False):
            return Annotations(config,
                               PATH=Globs.ANNOT_TRAIN_PATH,
                               is_tuning_train=True),\
                   Annotations(config,
                               PATH=Globs.ANNOT_TRAIN_PATH,
                               is_tuning_val=True)
        else: # Use actual train test data
            return Annotations(config, PATH=Globs.ANNOT_TRAIN_PATH),\
                   Annotations(config, PATH=Globs.ANNOT_TEST_PATH)
    elif dataset_name == 'mtsamples':
        return None, MtsamplesDataset(config, Globs.MTSAMPLES_TEST_PATH)
    else:
        raise ValueError('Unknown dataset')