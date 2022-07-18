#%%
import json

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

#%%
text = "Tokenizing text is a core task of NLP"

#%%
tokenized_text = list(text)
print(tokenized_text)

#%%
# Numiericalzation:
#   Convert each token into an integer
token2idx = {ch: idx for idx, ch in enumerate(set(tokenized_text))}
print(token2idx)

print(f"Vocabolary size: {len(token2idx)}")
print(f"String length: {len(tokenized_text)}")

#%%
input_ids = []
for token in tokenized_text:
    print(token, token2idx[token])
    input_ids.append(token2idx[token])


#%%
input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
print(one_hot_encodings.shape)
# Dimension of one-hot encoding is (text_len, vocab_size)

## Subword tokenization
#%%
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
encoded_text = tokenizer(text)
print(encoded_text)

#%%
# Convert token ids back to tokens
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)

#%%
string = tokenizer.convert_tokens_to_string(tokens)
print(string)

#%%
print(tokenizer.vocab_size, tokenizer.model_max_length)
print(tokenizer.model_input_names)


## Tokenize by data batches
#%%
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


#%%
emotions = load_dataset("emotion")
# emotions.set_format(type='pandas')
print(tokenize(emotions["train"][:2]))


# %%
print(tokenizer.all_special_tokens, tokenizer.all_special_ids)
# %%
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
# %%
print(emotions_encoded["train"].column_names)

# %%
emotions_encoded["train"][0].keys()
# %%
print(len(emotions_encoded["train"][0]["attention_mask"]))
print(len(emotions["train"]), len(emotions_encoded["train"]))


## 1. Transformers as feature extractors:
#   Use pretrained model
# %%
from transformers import AutoModel

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
model = AutoModel.from_pretrained(model_ckpt).to(device)

# %%
text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
print(inputs)
#%%
tokenizer.convert_ids_to_tokens(
    inputs.input_ids.detach().cpu().numpy().squeeze()
)


# %%
# Convert inputs into dictionary
inputs = {k: v.to(device) for k, v in inputs.items()}
print(inputs)

# %%
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)
# %%
outputs.last_hidden_state.size()

# %%
def extract_hidden_states(batch):
    inputs = {
        k: v.to(device)
        for k, v in batch.items()
        if k in tokenizer.model_input_names
    }
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state

    return {"hidden_state": last_hidden_state[:, 0].detach().cpu().numpy()}
    # return {"hidden_state": last_hidden_state[:, 0]}


# %%
emotions_encoded.set_format(
    "torch", columns=["input_ids", "attention_mask", "label"]
)
# %%
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
# %%
print(emotions_hidden)
# %%
print(emotions_hidden["train"].column_names)


# %%
# Create features matrix
X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])

# %%
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
# %%

print(X_train.shape, X_valid.shape)
# %%

# Train a simple classifier
from sklearn.linear_model import LogisticRegression

# %%

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
# %%
lr_clf.score(X_valid, y_valid)

# %%
# Check against dummy classifier
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_valid, y_valid)


## 2. Fine tuning!
# %%
from transformers import AutoModelForSequenceClassification

# %%
num_labels = 6
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=num_labels
).to(device)
print(model)
# %%
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# %%
from transformers import Trainer, TrainingArguments

# %%
batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
# %%
model_name = f"{model_ckpt}-finetuned-emotion"
# %%
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=False,
    log_level="error",
)
# %%
from transformers import Trainer

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"],
    tokenizer=tokenizer,
)
# %%
trainer.train()
# %%
emotions_encoded
# %%
emotions_encoded["train"][0]
# %%
