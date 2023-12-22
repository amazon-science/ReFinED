#!/usr/bin/env python
# coding: utf-8
from transformers import AutoTokenizer
from datasets import load_dataset, ClassLabel, DatasetDict, concatenate_datasets
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
import evaluate
import numpy as np
from transformers import TrainingArguments
from transformers import Trainer


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
    
def replace_number(example):
    new_tag = []
    for x in example['ner_tags']:
        if x != 0:
            if x%2 == 0:
                new_tag.append(16)
            else:
                new_tag.append(15)
        else:
            new_tag.append(0)
    return {"new_tags":new_tag} 


datasets_subset = ['PAN-X.de', 'PAN-X.ar', 'PAN-X.en', 'PAN-X.es', 'PAN-X.fa', 'PAN-X.ta', 'PAN-X.tr', 'PAN-X.fr', 'PAN-X.it']
model_checkpoint ="bert-base-multilingual-cased"

all_class = {
    "O": 0,
    "B-DATE": 1,
    "I-DATE": 2,
    "B-CARDINAL": 3,
    "I-CARDINAL": 4,
    "B-MONEY": 5,
    "I-MONEY": 6,
    "B-PERCENT": 7,
    "I-PERCENT": 8,
    "B-TIME": 9,
    "I-TIME": 10,
    "B-ORDINAL": 11,
    "I-ORDINAL": 12,
    "B-QUANTITY": 13,
    "I-QUANTITY": 14,
    "B-MENTION": 15,
    "I-MENTION": 16
}
label_names = list(all_class.keys())

train, test, validation = [],[],[]
for subset in datasets_subset[:]:
    raw_datasets = load_dataset("xtreme",subset)
    
    raw_datasets["train"].features["ner_tags"].feature.names = list(all_class.keys())
    raw_datasets["validation"].features["ner_tags"].feature.names = list(all_class.keys())
    raw_datasets["test"].features["ner_tags"].feature.names = list(all_class.keys())
    
    raw_datasets["train"].features["ner_tags"].feature.num_classes = len(list(all_class.keys()))
    raw_datasets["validation"].features["ner_tags"].feature.num_classes = len(list(all_class.keys()))
    raw_datasets["test"].features["ner_tags"].feature.num_classes = len(list(all_class.keys()))
    
    raw_datasets = raw_datasets.map(replace_number)
    raw_datasets = raw_datasets.remove_columns(['ner_tags'])
    raw_datasets = raw_datasets.rename_column('new_tags', 'ner_tags')
    
     
    train.append(raw_datasets["train"])
    test.append(raw_datasets["test"])
    validation.append(raw_datasets["validation"])

raw_datasets_train = concatenate_datasets(train)
raw_datasets_test = concatenate_datasets(test)
raw_datasets_validation = concatenate_datasets(validation)
    
raw_datasets = DatasetDict({
    'train': raw_datasets_train,
    'test': raw_datasets_test,
    'validation': raw_datasets_validation})

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

metric = evaluate.load("seqeval")

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

args = TrainingArguments(
    f"ner_models/{model_checkpoint}-xtreme-9lans-md-only-ReFinED-Style-no-ja",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    num_train_epochs=7,
    weight_decay=0.01,
    save_total_limit = 1,
    load_best_model_at_end=True,
    per_device_train_batch_size = 16,
    fp16=True,
    per_device_eval_batch_size = 16,
    save_steps = 1200,
    logging_steps= 1200
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()

trainer.evaluate(tokenized_datasets['test'])
trainer.save_model(f"ner_models/{model_checkpoint}-xtreme-9lans-md-only-ReFinED-Style-no-ja-best")


