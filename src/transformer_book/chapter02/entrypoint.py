import datasets
from transformer_book.chapter02 import config
from transformer_book.chapter02.classifier_fine_tuning import (
    FineTuneClassifier,
)
from transformer_book.chapter02.constants import CONFIG_KEYNAMES
from transformer_book.chapter02.tokenize import tokenize, tokenizer

dataset = datasets.load_dataset("emotion")
PRETRAINED_MODEL_NAME: str = config[CONFIG_KEYNAMES.PRETRAINED_MODEL_NAME]

dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)


fine_tuner = FineTuneClassifier(data=dataset_encoded)
fine_tuner.fine_tune()
