from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=False, use_fast=True,
                                          add_special_tokens=True)
sentence = ('Hello', 'Hello')
tokenised = tokenizer.encode_plus(sentence, truncation=True, max_length=32, padding='max_length',
                                  return_tensors='pt', add_special_tokens=True)['input_ids']
print(tokenised)
