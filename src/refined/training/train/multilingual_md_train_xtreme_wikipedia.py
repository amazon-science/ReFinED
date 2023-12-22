from transformers import AutoTokenizer
from datasets import load_from_disk, ClassLabel, DatasetDict, concatenate_datasets
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
import evaluate
import numpy as np
from transformers import TrainingArguments
from transformers import Trainer
from glob import glob
from sklearn.model_selection import StratifiedKFold
import random
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("--train_size",
                    type=int,
                    default=200000,
                    required=False,
                    help="The size of training data")

parser.add_argument("--seed",
                    type=int,
                    default=0,
                    required=False,
                    help="The size of training data")


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

def label2idx_map(example):
    new_tag = []
    for tag in example["ner_tags"]:
        if tag == "O":
            new_tag.append(0)
        elif "B-" in tag:
            new_tag.append(15)
        elif "I-" in tag:
            new_tag.append(16)
        else:
            raise Exception(f"Cant find {tag}")
    example["ner_tags"] = new_tag
    return example

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
    example['ner_tags'] = new_tag
    return example

args = parser.parse_args()
datasets_subset = glob("wikipedia_extreme_tokenized_refined_style_10lan_no_ja/*")
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
model_checkpoint ="bert-base-multilingual-cased"
train_size = args.train_size
random_seed = args.seed

train, test, validation = [],[],[]
for subset in datasets_subset[:]:
    raw_datasets = load_from_disk(subset)
    
    random.seed(random_seed)
    try:
        rnd_num = random.sample(range(raw_datasets["train"].num_rows), train_size)
        train.append(raw_datasets["train"].select(rnd_num))
    except:
        train.append(raw_datasets["train"])

    validation.append(raw_datasets["validation"])
    
raw_datasets_train = concatenate_datasets(train)
raw_datasets_validation = concatenate_datasets(validation)
tokenized_datasets = DatasetDict({
    'train': raw_datasets_train,
    'validation': raw_datasets_validation})

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,add_prefix_space=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

metric = evaluate.load("seqeval")

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

args = TrainingArguments(
    f"ner_models/{model_checkpoint}-wikipedia-XTREME-Size{train_size}-seed{random_seed}-10lan-no-ja",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    num_train_epochs=5,
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

trainer.save_model(f"ner_models/{model_checkpoint}-bootstrapping-MD")