from typing import Dict

from transformer_book.chapter02 import config
from transformer_book.chapter02.constants import (
    CONFIG_KEYNAMES,
    DATASET_KEYNAMES,
)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    config[CONFIG_KEYNAMES.PRETRAINED_MODEL_NAME]
)

tokenize_params: Dict[str, bool] = config[CONFIG_KEYNAMES.TOKENIZE_PARAMS]


def tokenize(batch):
    return tokenizer(batch[DATASET_KEYNAMES.TEXT], **tokenize_params)
