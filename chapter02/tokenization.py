#%%
import json

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

#%%
for token in tokenized_text:
    print(token, token2idx[token])


#%%
import torch
import torch.nn.functional as F

input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
print(one_hot_encodings.shape)

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
emotions_encoded["train"][0]["text"]
# %%
len(emotions_encoded["train"][0]["attention_mask"])
# %%
tokenizer.vocab_size
# %%
# %%
