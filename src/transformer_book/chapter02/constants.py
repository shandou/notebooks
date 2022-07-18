from types import SimpleNamespace

from transformer_book.chapter02 import config

CONFIG_KEYNAMES: SimpleNamespace = SimpleNamespace(
    BATCH_SIZE="batch_size",
    TRAINING_ARGUMENTS="training_arguments",
    PRETRAINED_MODEL_NAME="pretrained_model_name",
    TOKENIZE_PARAMS="tokenize_params",
)

DATASET_KEYNAMES: SimpleNamespace = SimpleNamespace(
    TRAIN="train", VALIDATION="validation", LABEL="label", TEXT="text"
)

TRAINING_ARGNAMES: SimpleNamespace = SimpleNamespace(
    **{
        key.upper(): key
        for key in config[CONFIG_KEYNAMES.TRAINING_ARGUMENTS].keys()
    }
)
