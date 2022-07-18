import datasets
import transformers
from transformer_book.chapter02.classifier_fine_tuning import (
    FineTuneClassifier,
)
from transformer_book.chapter02.tokenize import tokenize

dataset: datasets.DatasetDict = datasets.load_dataset("emotion")

dataset_encoded: datasets.DatasetDict = dataset.map(
    tokenize, batched=True, batch_size=None
)


fine_tuner: FineTuneClassifier = FineTuneClassifier(data=dataset_encoded)
fine_tuner.fine_tune()

preds_output: transformers.trainer_utils.PredictionOutput = (
    fine_tuner.predict()
)
print(preds_output.metrics)
