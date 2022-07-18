from typing import Any, Dict

import datasets
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformer_book.chapter02 import config
from transformer_book.chapter02.constants import (
    CONFIG_KEYNAMES,
    DATASET_KEYNAMES,
    TRAINING_ARGNAMES,
)
from transformer_book.chapter02.tokenize import tokenizer
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class Helper:
    @staticmethod
    def get_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def compute_metrics(pred) -> Dict[str, float]:
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}


class FineTuneClassifier:
    def __init__(self, data: datasets.DatasetDict):
        self.device: torch.device = Helper.get_device()
        self.data: datasets.DatasetDict = data
        self.pretrained_model_name: str = config[
            CONFIG_KEYNAMES.PRETRAINED_MODEL_NAME
        ]

    def _get_pretrained_classifier(self) -> AutoModelForSequenceClassification:
        num_labels: int = (
            self.data[DATASET_KEYNAMES.TRAIN]
            .features[DATASET_KEYNAMES.LABEL]
            .num_classes
        )
        return AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name, num_labels=num_labels
        ).to(self.device)

    def fine_tune(self):
        logging_steps: int = (
            len(self.data[DATASET_KEYNAMES.TRAIN])
            // config[CONFIG_KEYNAMES.BATCH_SIZE]
        )
        training_arguments: Dict[str, Any] = config[
            CONFIG_KEYNAMES.TRAINING_ARGUMENTS
        ]
        training_arguments[TRAINING_ARGNAMES.LOGGING_STEPS] = logging_steps
        training_args: TrainingArguments = TrainingArguments(
            **training_arguments
        )
        trainer = Trainer(
            model=self._get_pretrained_classifier(),
            args=training_args,
            compute_metrics=Helper.compute_metrics,
            train_dataset=self.data[DATASET_KEYNAMES.TRAIN],
            eval_dataset=self.data[DATASET_KEYNAMES.VALIDATION],
            tokenizer=tokenizer,
        )
        trainer.train()
