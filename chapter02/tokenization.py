from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


text = "Tokenizing text is a core task of NLP."
encoded_text = tokenizer(text)
# Words are mapped to unique integers in `input_ids` field.


tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)

string = tokenizer.convert_tokens_to_string(tokens)

tokenizer.vocab_size

tokenizer.model_max_length
