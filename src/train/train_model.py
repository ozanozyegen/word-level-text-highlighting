import wandb

from configs.defaults import Globs
from data.annotations import Annotations
from data.loader import data_loader
from helpers.wandb_common import wandb_save
from helpers.gpu_selection import auto_gpu_selection, use_gpu
from models.loader import model_loader
from features.tokenizer import Tokenizer
from features.keras_tokenizer import get_fit_keras_tokenizer
from features.preprocessing import load_as_unigram
from train.trainer import model_selector

def train(config: dict, wandb_tags=[]):
    use_gpu(True)
    wandb_save(is_online=True)
    run = wandb.init(project=Globs.PROJECT_NAME, entity=str(Globs.ENTITY_NAME),
        config=config,
        tags=wandb_tags, reinit=True)
    config = wandb.config
    print(config)

    # Load datasets
    # NOTE: annotations_test is the validation data when config.tuning_run=True
    annotations_train, annotations_test = data_loader(config)

    annot_train, annot_train_hi = annotations_train.get_documents_and_highlights()
    annot_test, annot_test_hi = annotations_test.get_documents_and_highlights()

    keras_tokenizer, reversed_word_map = get_fit_keras_tokenizer(config)
    train_seq = keras_tokenizer.texts_to_sequences(annot_train)
    test_seq = keras_tokenizer.texts_to_sequences(annot_test)
    if config.get('load_as_unigram', False):
       train_seq = load_as_unigram(train_seq)
       annot_train_hi = load_as_unigram(annot_train_hi)
       test_seq = load_as_unigram(test_seq)
       annot_test_hi = load_as_unigram(annot_test_hi)

    model = model_selector(config, wandb, keras_tokenizer)
    model.train(train_data=(train_seq, annot_train_hi),
                  test_data=(test_seq, annot_test_hi))
    model.save_config(wandb.run.dir)
    if config.get('save_model'):
        model.save_model(wandb.run.dir)

    run.finish()

if __name__ == '__main__':
    train(dict(
        debug=False,
        dataset='ydo_data',
        load_as_unigram=True, # Loads the data as unigrams
        model_name='lstm',
        embedding_type='bert',
        save_model=True,
        max_len=10,
        threshold=0.5,
        tuning_run=False,
        # For the unigram lstm model
        num_units=50,
        num_layers=2,
        dropout_lstm=0.1,
    ))