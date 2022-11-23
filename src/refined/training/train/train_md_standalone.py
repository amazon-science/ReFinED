import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Set

import torch
from sklearn.metrics import classification_report
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm, trange
from transformers import (
    AdamW,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from refined.dataset_reading.mention_detection.conll_reader import CoNLLNER
from refined.utilities.md_dataset_utils import (
    bio_to_offset_pairs,
    create_collate_fn,
)
from refined.dataset_reading.mention_detection.ontonotes_reader import OntoNotesNER
from refined.dataset_reading.mention_detection.webqsp_reader import WebQSPNER

sys.path.append("")


logger = logging.getLogger(__name__)

def train_md_model(
    resources_dir: str,
    datasets: List[str],
    transformer_name: str = "roberta-base",
    attention_probs_dropout_prob: float = 0.15,
    hidden_dropout_prob: float = 0.10,
    device: str = "cpu",
    max_seq: int = 512,
    batch_size: int = 32,
    fine_tune_all_layers: bool = True,
    num_epochs: int = 10,
    lr: float = 5e-6,
    max_articles: Optional[int] = None,
    bio_only: bool = True,
    ner_tag_to_num: Optional[Dict[str, int]] = None,
    additional_filenames: Optional[Dict[str, str]] = None,
    use_mention_tag: bool = False,
    filter_types: Optional[Set] = None,
    convert_types: Optional[Dict] = None
):

    if ner_tag_to_num is None:
        ner_tag_to_num = {"O": 0, "B": 1, "I": 2}

    hyperparams = locals()

    tokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True)

    model = AutoModelForTokenClassification.from_pretrained(
        transformer_name,
        num_labels=len(ner_tag_to_num),
        output_attentions=False,
        output_hidden_states=True,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        hidden_dropout_prob=hidden_dropout_prob,
    )

    datasets = list(map(str.lower, datasets))
    name_to_dataset = {
        "onto": OntoNotesNER,
        "conll": CoNLLNER,
        "webqsp": WebQSPNER,
        "conll-article": CoNLLNER,
        "onto-article": OntoNotesNER,
        "onto-lower": OntoNotesNER,
        "onto-article-lower": OntoNotesNER,
        "conll-lower": CoNLLNER,
        "conll-article-lower": CoNLLNER
    }

    train_datasets = {}
    dev_datasets = {}
    dataset_dir = os.path.join(resources_dir, "datasets")
    for dataset_name in datasets:
        dataset_class = name_to_dataset[dataset_name]
        if additional_filenames is not None:
            dataset_additional_filename = additional_filenames.get(dataset_name)
        else:
            dataset_additional_filename = None
        sentence_level = "article" not in dataset_name
        lower = "lower" in dataset_name

        if convert_types is not None:
            dset_convert_types = convert_types.get(dataset_name)
        else:
            dset_convert_types = None

        train_dataset = dataset_class(
            data_split="train",
            data_dir=dataset_dir,
            transformer_name=transformer_name,
            max_seq=max_seq,
            random_lower_case_prob=0.0,
            random_replace_question_mark=0.15,
            sentence_level=sentence_level,
            lower=lower,
            max_articles=max_articles,
            bio_only=bio_only,
            ner_tag_to_num=ner_tag_to_num,
            additional_filename=dataset_additional_filename,
            use_mention_tag=use_mention_tag,
            filter_types=filter_types,
            convert_types=dset_convert_types
        )
        dev_dataset = dataset_class(
            data_split="test",
            data_dir=dataset_dir,
            transformer_name=transformer_name,
            max_seq=max_seq,
            random_lower_case_prob=0.0,
            random_replace_question_mark=0.0,
            sentence_level=sentence_level,
            lower=lower,
            max_articles=max_articles,
            bio_only=bio_only,
            ner_tag_to_num=ner_tag_to_num,
            additional_filename=dataset_additional_filename,
            use_mention_tag=use_mention_tag,
            filter_types=filter_types,
            convert_types=dset_convert_types
        )
        train_datasets[dataset_name] = train_dataset
        dev_datasets[dataset_name] = dev_dataset
    train_dataset = ConcatDataset(train_datasets.values())
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=create_collate_fn(tokenizer.pad_token_id),
    )

    dev_dataloaders = {
        dataset_name: DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=create_collate_fn(tokenizer.pad_token_id),
        )
        for dataset_name, dataset in dev_datasets.items()
    }
    model.to(device)

    # consider freezing embedding layer
    if fine_tune_all_layers:
        optimizer_grouped_parameters = [
            {"params": [p for p in model.parameters() if p.requires_grad]}
        ]
    else:
        optimizer_grouped_parameters = [
            {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
        ]

    optimiser = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimiser, num_warmup_steps=0, num_training_steps=total_steps
    )
    train(
        model=model,
        train_dl=train_dataloader,
        dev_dls=dev_dataloaders,
        hyperparams=hyperparams,
        optimiser=optimiser,
        scheduler=scheduler,
        tokenizer=tokenizer,
        resources_dir=resources_dir,
        num_epochs=num_epochs,
        device=device,
        ner_tag_to_num=ner_tag_to_num
    )


def main():
    parser = argparse.ArgumentParser(description="Train MD model.")
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="device id",
    )
    parser.add_argument(
        "--dataset",
        default="onto",
        type=str,
        help="onto, conll, conll-article, webqsp, all (multi-task), or a json list (e.g. [onto, webqsp])",
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="Epochs",
    )
    parser.add_argument(
        "--lr",
        default=5e-5,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--bs",
        default=32,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=False,
        help="Data dir",
    )
    parser.add_argument(
        "--max_seq",
        default=510,
        type=int,
        required=False,
        help="max_seq",
    )
    parser.add_argument(
        "--fine_tune", default=True, type=lambda x: (str(x).lower() in {"true", "1", "yes", "y"})
    )
    parser.add_argument(
        "--cased", default=False, type=lambda x: (str(x).lower() in {"true", "1", "yes", "y"})
    )
    parser.add_argument(
        "--large", default=False, type=lambda x: (str(x).lower() in {"true", "1", "yes", "y"})
    )
    parser.add_argument(
        "--distilled", default=False, type=lambda x: (str(x).lower() in {"true", "1", "yes", "y"})
    )
    parser.add_argument(
        "--n_gpu",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_output",
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--restore_model_path",
        default=None,
        type=str,
        help="The path to the model checkpoint to restore.",
    )
    parser.add_argument(
        "--attention_probs_dropout_prob",
        default=0.1,
        type=float,
        help="attention_probs_dropout_prob.",
    )
    parser.add_argument(
        "--hidden_dropout_prob",
        default=0.1,
        type=float,
        help="hidden_dropout_prob.",
    )
    parser.add_argument(
        "--transformer_name",
        default="roberta-base",
        type=str,
        help="transformer_name.",
    )

    args = parser.parse_args()


    # if datasets[0] == '[':
    #     print(datasets)
    #     args.dataset = json.loads(args.dataset)
    # else:
    #     print(args.dataset)
    #     args.dataset = json.loads('[' + args.dataset + ']')


def train(
    model,
    train_dl,
    dev_dls,
    optimiser,
    scheduler,
    tokenizer,
    num_epochs: int,
    device: str,
    hyperparams: Dict[str, Any],
    resources_dir: str,
    ner_tag_to_num: Dict[str, int]
):
    max_grad_norm = 1.0
    max_n = 10e10
    scaler = GradScaler()
    for epoch_num in trange(num_epochs):
        logger.info(f"Epoch num: {epoch_num}")
        model.train()
        for step_num, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
            if step_num > max_n:
                break
            batch = (tns.to(device) for tns in batch)
            tokens, attention_mask, labels = batch
            optimiser.zero_grad()
            with autocast():
                output = model(input_ids=tokens, labels=labels, attention_mask=attention_mask)
            loss = output[0]
            if step_num % 25 == 0:
                print(f"loss = {loss.item()}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()

        macro_f1 = evaluate(
            model=model,
            dev_dls=dev_dls,
            tokenizer=tokenizer,
            device=device,
            ner_tag_to_num=ner_tag_to_num
        )

        save(
            model=model,
            dev_dls=dev_dls,
            epoch=epoch_num,
            hyperparams=hyperparams,
            resources_dir=resources_dir,
            macro_f1=macro_f1
        )


def save(macro_f1, model, dev_dls, resources_dir: str, hyperparams: Dict[str, Any], epoch: int):
    output_dir = os.path.join(
        resources_dir, f"{'-'.join(dev_dls.keys())}-epoch-{epoch}-mf1-{macro_f1}"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info("Saving model checkpoint to %s", resources_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.pt"))

    # saves your training arguments together with the trained model
    with open(os.path.join(output_dir, "training_args.json"), "wb") as f:
        for k, v in hyperparams.items():
            if type(v) == set:
                hyperparams[k] = list(v)
        f.write(json.dumps(hyperparams, indent=4).encode("utf-8"))


def evaluate(
    model, dev_dls, tokenizer, device: str, ner_tag_to_num: Dict[str, int]
):
    model.eval()
    f1s = []
    for dl_name, dl in dev_dls.items():
        all_preds = []
        all_trues = []
        tp_total = 0
        fp_total = 0
        total_preds = 0
        total_actual = 0
        logger.info(f"data = {dl_name}")
        for batch in tqdm(dl, total=len(dl)):
            batch = (tns.to(device) for tns in batch)
            tokens, attention_mask, labels = batch

            with torch.no_grad():
                output = model(input_ids=tokens, attention_mask=attention_mask)
                preds = output[0]
                mask = attention_mask.flatten() == 1.0
                gold_labels = labels.flatten()[mask].detach().cpu().numpy().tolist()
                label_preds = preds.argmax(dim=2).flatten()[mask].detach().cpu().numpy().tolist()
                all_trues.extend(gold_labels)
                all_preds.extend(label_preds)

                # error analysis
                bs_preds = preds.argmax(dim=2)
                for bs_idx in range(tokens.size(0)):
                    tokens_sent = tokens[bs_idx]
                    mask = tokens_sent != tokenizer.pad_token_id
                    preds_sent = bs_preds[bs_idx][mask].tolist()
                    labels_sent = labels[bs_idx][mask].tolist()

                    actual_spans = bio_to_offset_pairs(labels_sent)
                    pred_spans = bio_to_offset_pairs(preds_sent)

                    total_preds += len(pred_spans)
                    tp_total += len(pred_spans & actual_spans)
                    fp_total += len(pred_spans - actual_spans)
                    total_actual += len(actual_spans)

        p = tp_total / (total_preds + 1e-10)
        r = tp_total / (total_actual + 1e-10)
        f1 = 2 * (p * r) / (p + r + 1e-10)
        f1s.append(f1)
        print(f"Span level metrics: p = {p:.3f}, r = {r:.3f}, f1 = {f1:.3f}")
        print(classification_report(all_trues,
                                    all_preds,
                                    target_names=list(ner_tag_to_num.keys()),
                                    labels=list(ner_tag_to_num.values())))

    # saving
    macro_f1 = sum(f1s) / len(f1s)

    return macro_f1


if __name__ == "__main__":
    main()
