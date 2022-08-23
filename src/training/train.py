import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Tuple

import model_components.config
import numpy as np
import torch
from dataset_reading.mention_detection.conll_reader import CoNLLNER
from dataset_reading.mention_detection.md_dataset_utils import (
    bio_to_offset_pairs,
    create_collate_fn,
)
from dataset_reading.mention_detection.ontonotes_reader import OntoNotesNER
from dataset_reading.mention_detection.webqsp_reader import WebQSPNER
from dataset_reading.wikipedia_dataset import WikipediaDataset
from doc_preprocessing.dataclasses import BatchedElementsTns, get_tokenizer_cached
from doc_preprocessing.doc_preprocessor import DocumentPreprocessorMemoryBased
from model_components.config import ModelConfig
from model_components.refined_model import RefinedModel
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from utilities.general_utils import cycle

sys.path.append(".")


def wc(filename: str) -> int:
    return int(subprocess.check_output(["wc", "-l", filename]).split()[0])


# multi-task loss weighting (selected based on single-task converging order of magnitude)
MD_scale = 0.01
ET_scale = 1
# ED = 0.01  # if not detached
ED_scale = 1
Description_scale = 0.01
KG_scale = 0.01


TB_WRITER = None

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # ensures first 4 GPUs are visible
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # disable synchronizing, can enable for debugging
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # disable synchronizing, can enable for debugging (breaks multi-gpu)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    global TB_WRITER
    # model_config = ModelConfig(data_dir='/tom_data/data_2020_feb')
    # model_config = ModelConfig(data_dir='/big_data/tmp/data')
    # /data/tayoola/2021_data
    # /tom_data/kge_exp/data
    # /big_data/2021_data'
    # /tom_data/2021_data
    model_config = ModelConfig(
        data_dir="/big_data/2021_data", freeze_embedding_layers=False, freeze_layers=[],
        transformer_name="roberta-base"
    )
    parser = argparse.ArgumentParser(description="Train NER/ED model on Wikipedia links.")
    parser.add_argument(
        "--zero_shot_mode",
        action="store_true",
        help="When enabled this will filter entities that appear in ED datasets.",
    )
    parser.add_argument(
        "--ignore_pem_values",
        action="store_true",
        help="Set all pem (priors) values to 0.",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name the training run.",
    )
    parser.add_argument(
        "--relation_method",
        type=str,
        help="Simple, attention, learnable",
    )
    parser.add_argument(
        "--self_attention_dropout",
        type=float,
        default=0.0,
        help="Self attention dropout",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="cpu or cuda device",
    )
    parser.add_argument(
        "--lr_ner_scale",
        default=model_config.lr_ner_scale,
        type=int,
        help="Scale learning rate for NER layer to compensate for large number of targets and BCE.",
    )
    parser.add_argument(
        "--freeze_layers",  # json list
        default=model_config.freeze_layers,
        type=str,
        help="Freezes certain layers of Bert (e.g. [0, 1, 2] freezes first 3).",
    )
    parser.add_argument(
        "--freeze_embedding_layers",
        default=model_config.freeze_embedding_layers,
        type=bool,
        help="Freezes all embedding layers.",
    )
    parser.add_argument(
        "--freeze_all_bert_layers",
        default=model_config.freeze_all_bert_layers,
        type=str,
        help="Freezes all bert layers.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the files for the task.",
    )
    parser.add_argument(
        "--restore_model_path",
        default=None,
        type=str,
        help="The path to the model checkpoint to restore.",
    )
    parser.add_argument(
        "--dev_split_percent",
        default=0.3,
        type=float,
        help="(>0.3%)Percentage (%) of the training data that will be set aside for evaluation (dev/validation set).",
    )
    parser.add_argument(
        "--ed_layer_dropout",
        default=model_config.ed_layer_dropout,
        type=float,
        help="Dropout in ED layer.",
    )
    parser.add_argument(
        "--max_candidates",
        default=model_config.max_candidates,
        type=int,
        help="Max candidates from pem lookup.",
    )
    parser.add_argument(
        "--ner_layer_dropout",
        default=model_config.ner_layer_dropout,
        type=float,
        help="Dropout in NER layer.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set this flag if you are debugging (if enabled data files will not be loaded).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Restores optimiser and scheduler when restoring model.",
    )
    parser.add_argument(
        "--steps_per_eval",
        default=25000,
        type=int,
        help="Number of steps before evaluating on dev set.",
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        default=model_config.per_gpu_batch_size,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--n_gpu",
        default=model_config.n_gpu,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=model_config.gradient_accumulation_steps,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=model_config.learning_rate,
        type=float,
        help="The initial learning rate for Adam.",
    )

    parser.add_argument(
        "--num_train_epochs",
        default=model_config.num_train_epochs,
        type=int,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--warmup_steps",
        default=model_config.warmup_steps,
        type=int,
        help="Linear warmup over warm up steps.",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save checkpoint every X updates steps."
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
        "--use_cpu",
        action="store_true",
        help="Use CPU only",
    )
    parser.add_argument(
        "--only_ner",
        action="store_true",
        help="Only train NER",
    )
    parser.add_argument(
        "--only_ed",
        action="store_true",
        help="Only train ED",
    )
    parser.add_argument(
        "--train_md",
        action="store_true",
        help="Train MD",
    )
    parser.add_argument(
        "--use_md_dataset",
        action="store_true",
        help="Use MD dataset (multi-task)",
    )
    parser.add_argument(
        "--mask_mentions",
        action="store_true",
        help="Mask mentions 80% of time 5% random token",
    )
    parser.add_argument(
        "--mask_prob",
        default=0.70,
        type=float,
        help="prob that mention will be masked",
    )
    parser.add_argument(
        "--mask_random_prob",
        default=0.05,
        type=float,
        help="prob that masked mention will be replaced by random token in vocab",
    )
    parser.add_argument(
        "--transformer_name",
        default="roberta-base",
        type=str,
        help="transformer name",
    )
    parser.add_argument(
        "--datasets_md",
        default='["conll"]',
        type=str,
        help='onto, conll, conll-article, webqsp, all (multi-task), or a json list (e.g. ["onto", "webqsp"])',
    )
    parser.add_argument(
        "--max_seq",
        default=300,
        type=int,
        help="max seq length",
    )
    parser.add_argument(
        "--candidate_dropout",
        default=0.0,
        type=float,
        help="probability that candidates will be removed from pem (0%) during training (10-20% could be tried)",
    )
    parser.add_argument(
        "--max_mentions",
        default=40,  # sub-sample to save GPU memory
        type=int,
        help="maximum number of mentions per batch (will randomly sample if needed)",
    )
    parser.add_argument(
        "--sample_candidates",
        default=5,  # sub-sample to save GPU memory
        type=int,
        help="number of candidates to sample from the max_candidates candidates (-1) means no sampling",
    )
    parser.add_argument(
        "--ignore_descriptions",
        action="store_true",
        help="ignore descriptions in the entity scoring function",
    )
    parser.add_argument(
        "--ignore_types",
        action="store_true",
        help="ignore types in the entity scoring function",
    )

    parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")

    args = parser.parse_args()
    if args.max_candidates < 30:
        logger.warning("args.max_candidates" + str(args.max_candidates))
        logger.warning(
            "consider using args.sample_candidates instead of decreasing number of candidates (30, 5)"
            "is an example of values to set to save memory and train for all 30 candidates."
        )

    TB_WRITER = SummaryWriter(comment=args.name)
    model_components.config.MAX_SEQ = args.max_seq

    if isinstance(args.freeze_layers, str):
        args.freeze_layers = json.loads(args.freeze_layers)

    if args.only_ner:
        # saves memory and speeds up data loading
        args.max_candidates = 1
        logger.info(f"Only NER = {args.only_ner}")

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use "
            f"--overwrite_output_dir to overwrite."
        )

    logger.info(f"Using data directory {args.data_dir}")
    data_dir = args.data_dir
    data_preprocessor = DocumentPreprocessorMemoryBased(
        data_dir=data_dir,
        debug=args.debug,
        max_candidates=args.max_candidates,
        transformer_name=args.transformer_name,
    )
    data_preprocessor.zero_string_features = True
    if args.ignore_pem_values:
        logger.info("Setting pem values to 0")
        data_preprocessor.zero_features = True

    class_to_label = data_preprocessor.class_to_label
    # device = torch.device("cuda:0" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    device = args.device
    model = RefinedModel(
        ModelConfig(data_dir=args.data_dir, transformer_name=args.transformer_name),
        preprocessor=data_preprocessor,
        ignore_descriptions=args.ignore_descriptions,
        ignore_types=args.ignore_types,
    )

    if args.restore_model_path is not None:
        logger.info(f"Restoring model checkpoint from {args.restore_model_path}.")
        checkpoint = torch.load(args.restore_model_path, map_location="cpu")

        # Get rid of params which were in pretrained model but are not in new model
        checkpoint = {name: param for name, param in checkpoint.items() if name in model.state_dict()}

        # Get rid of params which have different size in pretrained model to new model
        new_checkpoint = {}
        for name, param in checkpoint.items():
            if param.size() == model.state_dict()[name].size():
                new_checkpoint[name] = param
            else:
                logger.warning(f"Removing {name} from checkpoint as has different size in current model")

        model.load_state_dict(new_checkpoint, strict=False)

    for params in model.parameters():
        params.requires_grad = True

    if model_config.freeze_all_bert_layers:
        for param in model.transformer.parameters():
            param.requires_grad = False
        logger.info(f"Froze all transformer layers")

    if model_config.freeze_embedding_layers:
        for param in list(model.transformer.embeddings.parameters()):
            param.requires_grad = False
        logger.info(
            f"Froze transformer embedding layers (weights will be fixed during training) main"
        )

        for param in list(model.ed_2.description_encoder.transformer.embeddings.parameters()):
            param.requires_grad = False
        logger.info(
            f"Froze transformer embedding layers (weights will be fixed during training) description"
        )

    # freeze_layers is list [0, 1, 2, 3] representing layer number
    logger.info(f"Freeezing layers: {args.freeze_layers}")
    for layer_idx in args.freeze_layers:
        for param in list(model.transformer.encoder.layer[layer_idx].parameters()):
            param.requires_grad = False
        logger.info(f"Froze encoder layer {layer_idx} (weights will be fixed during training)")

    if args.only_ed:
        # freeze all parameters except main ED scoring function
        for params in model.parameters():
            params.requires_grad = False
        for params in model.entity_disambiguation.parameters():
            params.requires_grad = True

    trainable_params = 0
    total_params = 0
    for params in model.parameters():
        if params.requires_grad:
            trainable_params += params.numel()
        total_params += params.numel()
    logger.info(f"Total parameters {total_params}")
    logger.info(f"Trainable parameters {trainable_params}")

    if args.n_gpu > 1:
        logger.info("Using multiple GPUs for model")
        model = torch.nn.DataParallel(
            model, device_ids=list(range(args.n_gpu)), output_device=device
        ).to(device)
    else:
        model = model.to(device)
    # num_data_workers = 16 if args.relation_method == 'simple' else 4
    num_data_workers = 16
    # num_data_dev_workers = 8
    num_data_dev_workers = 4

    train_dataloader_md = None
    dev_dataloaders_md = None
    tokenizer = None
    if args.use_md_dataset:
        logger.info(f"Using f{args.datasets_md} for mention detection.")
        args.datasets_md = json.loads(
            args.datasets_md if args.datasets_md[0] == "[" else "[" + args.datasets_md + "]"
        )

        # MD datasets
        name_to_dataset_md = {
            "onto": OntoNotesNER,
            "conll": CoNLLNER,
            "webqsp": WebQSPNER,
            "conll-article": CoNLLNER,
            "onto-article": OntoNotesNER,
        }
        train_datasets_md = {}
        dev_datasets_md = {}
        for dataset_name in args.datasets_md:
            dataset_class = name_to_dataset_md[dataset_name]
            sentence_level = "article" not in dataset_name
            train_dataset_md = dataset_class(
                data_split="train",
                data_dir=args.data_dir,
                transformer_name=args.transformer_name,
                max_seq=args.max_seq,
                random_lower_case_prob=0.15,
                random_replace_question_mark=0.15,
                sentence_level=sentence_level,
            )
            dev_dataset_md = dataset_class(
                data_split="test",
                data_dir=args.data_dir,
                transformer_name=args.transformer_name,
                max_seq=args.max_seq,
                random_lower_case_prob=0.0,
                random_replace_question_mark=0.0,
                sentence_level=sentence_level,
            )
            train_datasets_md[dataset_name] = train_dataset_md
            dev_datasets_md[dataset_name] = dev_dataset_md
        train_dataset_md = ConcatDataset(train_datasets_md.values())

        pad_token_value = data_preprocessor.pad_id

        assert "bert-" in args.transformer_name or "roberta-" in args.transformer_name, (
            "Currently only BERT and Roberta " "are supported."
        )
        tokenizer = get_tokenizer_cached(args.transformer_name, data_dir=data_preprocessor.data_dir)

        train_dataloader_md = cycle(
            DataLoader(
                dataset=train_dataset_md,
                batch_size=args.per_gpu_batch_size * args.n_gpu // 2,
                shuffle=True,
                num_workers=1,
                collate_fn=create_collate_fn(pad_token_value),
            )
        )
        dev_dataloaders_md = {
            dataset_name: DataLoader(
                dataset=dataset,
                batch_size=args.per_gpu_batch_size * args.n_gpu,
                shuffle=False,
                num_workers=1,
                collate_fn=create_collate_fn(pad_token_value),
            )
            for dataset_name, dataset in dev_datasets_md.items()
        }

    dataset_size = wc(os.path.abspath(os.path.join(data_dir, "wikipedia_links_aligned_spans.json")))
    logger.info(f"Dataset size {dataset_size}")
    wikipedia_dataset = WikipediaDataset(
        data_preprocessor,
        0,
        int(dataset_size * 0.95),
        batch_size=args.per_gpu_batch_size * args.n_gpu,
        num_workers=num_data_workers,
        # num_workers=1,
        prefetch=1000,
        mask=args.mask_prob,
        random_mask=args.mask_random_prob,
        lower_case_prob=0.05,
        candidate_dropout=args.candidate_dropout,
        max_mentions=args.max_mentions,
        sample_k_candidates=5,
        dataset_path=os.path.join(data_dir, "wikipedia_links_aligned_spans.json"),
        qcodes_to_filter=None,
    )

    wikipedia_dataset_dev = WikipediaDataset(
        data_preprocessor,
        int(dataset_size * 0.95),
        dataset_size,
        batch_size=args.per_gpu_batch_size * args.n_gpu,
        num_workers=num_data_dev_workers if args.relation_method == "simple" else 1,
        # num_workers=1,
        dataset_path=os.path.join(data_dir, "wikipedia_links_aligned_spans.json"),
        prefetch=1000,
        lower_case_prob=0.00,
        candidate_dropout=0.0,
        max_mentions=None if args.relation_method == "simple" else args.max_mentions,
        mask=0.0,
        random_mask=0.0,
        qcodes_to_filter=None,
    )
    # TODO enable pin_memory again (this is test to see if it stops freezing at 10058)
    wikipedia_dataloader = DataLoader(
        wikipedia_dataset, batch_size=None, num_workers=num_data_workers, pin_memory=False
    )
    # wikipedia_dataloader = DataLoader(wikipedia_dataset, batch_size=None, num_workers=0, pin_memory=True)
    wikipedia_dataloader_dev = DataLoader(
        wikipedia_dataset_dev,
        batch_size=None,
        num_workers=num_data_dev_workers if args.relation_method == "simple" else 1,
        # num_workers=0,
        pin_memory=False,
    )

    _model = model.module if hasattr(model, "module") else model
    optimizer = AdamW(
        [
            {"params": _model.get_md_params(), "lr": args.learning_rate / MD_scale},
            {"params": _model.get_et_params(), "lr": (args.learning_rate / ET_scale) * 100},
            {"params": _model.get_desc_params(), "lr": args.learning_rate},
            {"params": _model.get_ed_params(), "lr": args.learning_rate / ED_scale},
            # {
            #     "params": _model.get_kg_params(),
            #     "lr": args.learning_rate,
            # },  # consider add KG_scale here
            # {
            #     "params": _model.get_final_ed_params(),
            #     "lr": args.learning_rate / ED_scale,
            # },  # consider scale
            {"params": _model.get_parameters_not_to_scale(), "lr": args.learning_rate},
        ],
        lr=args.learning_rate,
        eps=1e-8,
    )

    epochs = args.num_train_epochs

    # Total number of training steps is number of batches * number of epochs.
    # wikipedia_dataset len returns number of batches
    total_steps = len(wikipedia_dataset) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps / args.gradient_accumulation_steps,
    )

    if args.restore_model_path is not None and args.resume:
        logger.info("Restoring optimizer and scheduler")
        optimizer_checkpoint = torch.load(
            os.path.join(args.restore_model_path.replace("model.pt", ""), "optimizer.pt"),
            map_location="cpu",
        )
        scheduler_checkpoint = torch.load(
            os.path.join(args.restore_model_path.replace("model.pt", ""), "scheduler.pt"),
            map_location="cpu",
        )
        optimizer.load_state_dict(optimizer_checkpoint)
        scheduler.load_state_dict(scheduler_checkpoint)

    logger.info("***** Running training *****")
    logger.info(
        "  Num examples = %d", len(wikipedia_dataset) * args.per_gpu_batch_size * args.n_gpu
    )
    logger.info("  Num Steps = %d", len(wikipedia_dataset) / args.gradient_accumulation_steps)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_batch_size)
    logger.info("  Num GPUs = %d", args.n_gpu)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    logger.info("***** ************** *****")
    train(
        dataloader=wikipedia_dataloader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
        dataloader_dev=wikipedia_dataloader_dev,
        class_to_idx=data_preprocessor.class_to_idx,
        class_to_label=class_to_label,
        args=args,
        md_dl_train=train_dataloader_md,
        md_dl_dev=dev_dataloaders_md,
        tokenizer=tokenizer,
        model_config=model_config,
        data_preprocessor=data_preprocessor,
    )


def print_lr(optimizer):
    logger.info("Learning rates:")
    for param_group in optimizer.param_groups:
        logger.info(param_group["lr"])


def train(
    dataloader,
    model,
    optimizer,
    scheduler,
    epochs,
    device,
    dataloader_dev,
    class_to_idx,
    class_to_label,
    args,
    md_dl_train=None,
    md_dl_dev=None,
    tokenizer=None,
    model_config=None,
    data_preprocessor=None,
):
    md_every_n_steps = 10
    # loss variables added here because del is called in the training for-loop to enable better gc
    loss = None
    ed_loss = None
    et_loss = None
    md_loss_md = None
    last_time = time.time()
    model.train()
    global_step_num = 0
    mentions_seen = 0
    for epoch_num in tqdm(range(epochs), desc="Epoch"):
        logger.info(f"epoch: {epoch_num}")
        logger.info(f"lr:")
        print_lr(optimizer)
        model.train()
        for step, batch in tqdm(enumerate(dataloader), desc="Iteration"):  # 135,297it
            global_step_num += 1

            start_time = time.time()
            delta_time = start_time - last_time
            if (step + 1) % 50 == 0:
                logger.info(f"dataloader time = {delta_time}")

            if args.n_gpu == 1:
                batch = tuple(
                    x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
                    for x in batch
                )

            if md_dl_train is not None and step % md_every_n_steps == 0 and args.use_md_dataset:
                batch_md = next(md_dl_train)
                if args.n_gpu == 1:
                    batch_md = tuple(
                        x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
                        for x in batch_md
                    )
                b_token_id_values_md, b_attention_mask_values_md, b_ner_labels_md = batch_md

            batch = BatchedElementsTns(*[x if isinstance(x, torch.Tensor) else x for x in batch])
            (
                md_loss,
                _,
                et_loss,
                _,
                ed_loss,
                ed_logits,
                _,
                _,
                _,
                description_loss,
                _
            ) = model(
                token_ids=batch.token_id_values,
                token_acc_sums=batch.token_acc_sum_values,
                entity_mask=batch.entity_mask_values,
                class_targets=batch.class_target_values,
                attention_mask=batch.attention_mask_values,
                token_type_ids=batch.token_type_values,
                candidate_entity_targets=batch.candidate_target_values,
                candidate_pem_values=batch.pem_values,
                candidate_classes=batch.candidate_class_values,
                candidate_pme_values=batch.pme_values,
                entity_index_mask_values=batch.entity_index_mask_values,
                spans=None,
                batch_elements=batch.batch_elements,
                cand_ids=batch.candidate_qcode_values,
                ner_labels=batch.ner_labels,
                cand_desc=batch.candidate_desc,
                candidate_features=batch.candidate_features,
                batch_elements_included=torch.arange(
                    batch.entity_index_mask_values.size(0)
                ).unsqueeze(-1),
            )
            if md_dl_train is not None and step % md_every_n_steps == 0 and args.use_md_dataset:
                md_loss_md, _, _, _, _, _, _, _, _, _ = model(
                    token_ids=b_token_id_values_md,
                    attention_mask=b_attention_mask_values_md,
                    ner_labels=b_ner_labels_md,
                )
                raise Exception

            if step % 50 == 0:
                if args.only_ner:
                    logger.info(
                        f"et_loss={et_loss.mean().item()}, md_loss={md_loss.mean().item()},"
                        f"md_loss_md={md_loss_md.mean().item() if md_loss_md is not None else None}"
                    )
                elif args.only_ed:
                    logging.info(f"ed_loss={ed_loss.mean().item()}")
                else:
                    logger.info(
                        f"et_loss={et_loss.mean().item()}, ed_loss={ed_loss.mean().item()}, "
                        f"md_loss={md_loss.mean().item() if md_loss is not None else None}, "
                        f"md_loss_md={md_loss_md.mean().item() if md_loss_md is not None else None}, "
                        f"desc_loss={description_loss.mean().item() if description_loss is not None else None},"
                        # f'kg_loss={kg_loss.mean().item() if kg_loss is not None else None}'
                        # f'final_ed_loss={final_loss.mean().item() if kg_loss is not None else None}'
                    )

            TB_WRITER.add_scalar(
                "Mention_Detection_Loss/train", md_loss.mean().item(), global_step_num
            )
            TB_WRITER.add_scalar("Entity_Typing_Loss/train", et_loss.mean().item(), global_step_num)
            TB_WRITER.add_scalar(
                "Entity_Description_Loss/train", description_loss.mean().item(), global_step_num
            )
            TB_WRITER.add_scalar(
                "Entity_Disambiguation_Loss/train", ed_loss.mean().item(), global_step_num
            )

            TB_WRITER.add_scalar(
                "Total_Loss/train",
                md_loss.mean().item()
                + et_loss.mean().item()
                + ed_loss.mean().item()
                + description_loss.mean().item(),
                global_step_num,
            )
            mentions_seen += batch.candidate_target_values[batch.entity_index_mask_values].size(0)
            TB_WRITER.add_scalar("Mentions_Seen/train", mentions_seen, global_step_num)

            if args.only_ner:
                loss = et_loss + md_loss
            elif args.only_ed:
                loss = ed_loss
            else:
                loss = (
                    (et_loss * ET_scale)
                    + (ed_loss * ED_scale)
                    + (md_loss * MD_scale)
                    + (description_loss * Description_scale)
                )

            if md_loss_md is not None and step % md_every_n_steps == 0 and False:
                loss = loss.mean()
                loss += md_loss_md.mean() * 0.01

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps >= 1:
                loss = loss / args.gradient_accumulation_steps

            loss = loss.to(device, non_blocking=False)
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if step % 50 == 0:
                logger.info(loss.item() * args.gradient_accumulation_steps)
                logger.info(f"total loss {loss.mean().item() * args.gradient_accumulation_steps}")

            del loss, et_loss, ed_loss, batch
            last_time = time.time()

            if (step + 1) % args.steps_per_eval == 0:
                model.eval()
                sample_size = 10 if args.debug else 250
                ner_f1, ed_acc = evaluate(
                    dataloader_dev=dataloader_dev,
                    model=model,
                    args=args,
                    device=device,
                    class_to_label=class_to_label,
                    class_to_idx=class_to_idx,
                    sample_size=sample_size,
                    print_report=False,
                    benchmark=False,
                    epoch_num=global_step_num,
                    data_preprocessor=data_preprocessor,
                )
                if args.use_md_dataset:
                    evaluate_md(
                        model=model,
                        dev_dls=md_dl_dev,
                        pad_token_id=tokenizer.pad_token_id,
                        device=device,
                    )

                # saving
                output_dir = os.path.join(
                    args.output_dir, f"checkpoint-{step}-ner_f1-{ner_f1}-{ed_acc}"
                )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                logger.info("Saving model checkpoint to %s", args.output_dir)

                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training

                torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.pt"))
                # saves your training arguments together with the trained model
                with open(os.path.join(output_dir, "training_args.json"), "wb") as f:
                    f.write(json.dumps(args.__dict__, indent=4).encode("utf-8"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                model_config.to_file(os.path.join(output_dir, "config.json"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

                model.train()
                print_lr(optimizer)


def evaluate(
    dataloader_dev,
    model,
    args,
    device,
    class_to_label,
    class_to_idx,
    epoch_num,
    print_report=False,
    sample_size=200,
    benchmark=False,
    data_preprocessor=None,
) -> Tuple[float, float]:
    data_preprocessor.max_candidates = 30
    correct_ed = 0
    correct_ed_final = 0
    correct_ed_final_no_ent = 0
    correct_ed_with_no_ent = 0
    total_ed = 0

    preds_ner = []
    actuals_ner = []

    total_md_loss = 0
    total_et_loss = 0
    total_ed_loss = 0
    total_desc_loss = 0
    tp, fp, tn, fn, fn_negative = 0, 0, 0, 0, 0
    total_positives, total_negatives = 0, 0
    i = 0
    for step, batch in tqdm(enumerate(dataloader_dev)):
        i += 1
        if i > sample_size:
            break
        model.eval()
        if args.n_gpu == 1:
            batch = BatchedElementsTns(
                *[
                    x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
                    for x in batch
                ]
            )

        with torch.no_grad():
            batch = BatchedElementsTns(*[x if isinstance(x, torch.Tensor) else x for x in batch])
            (
                md_loss,
                md_activations,
                et_loss,
                et_activations,
                ed_loss,
                ed_activations,
                spans,
                other_spans,
                cand_ids,
                description_loss,
                _
            ) = model(
                token_ids=batch.token_id_values,
                token_acc_sums=batch.token_acc_sum_values,
                entity_mask=batch.entity_mask_values,
                class_targets=batch.class_target_values,
                attention_mask=batch.attention_mask_values,
                token_type_ids=batch.token_type_values,
                candidate_entity_targets=batch.candidate_target_values,
                candidate_pem_values=batch.pem_values,
                candidate_classes=batch.candidate_class_values,
                candidate_pme_values=batch.pme_values,
                entity_index_mask_values=batch.entity_index_mask_values,
                spans=None,
                batch_elements=batch.batch_elements,
                cand_ids=batch.candidate_qcode_values,
                ner_labels=batch.ner_labels,
                cand_desc=batch.candidate_desc,
                candidate_features=batch.candidate_features,
                batch_elements_included=torch.arange(batch.token_id_values.size(0)).unsqueeze(-1),
            )

            total_md_loss += md_loss.mean().item()
            total_et_loss += et_loss.mean().item()
            total_ed_loss += ed_loss.mean().item()
            total_desc_loss += description_loss.mean().item()

            if benchmark:
                continue

        # ner - expand tensors
        if et_activations is not None:
            span_classes = batch.class_target_values[batch.entity_index_mask_values]
            span_classes_exp = torch.zeros(
                size=(span_classes.size(0), len(class_to_idx) + 1),
                dtype=torch.float32,
                device=span_classes.device,
            )
            span_classes_exp[
                torch.arange(span_classes.size(0)).unsqueeze(1).expand(span_classes.size()),
                span_classes,
            ] = 1
            actual_ner = span_classes_exp.detach().int().cpu().numpy()
            pred_ner = torch.round(et_activations).detach().int().cpu().numpy()
            preds_ner += pred_ner.tolist()
            actuals_ner += actual_ner.tolist()

            del et_activations
        if not args.only_ner:
            # ed
            # add to function and call with different scores
            pred_ed = ed_activations[:, :-1].argmax(dim=1).to(device, non_blocking=True)
            pred_ed_with_no_ent = ed_activations.argmax(dim=1).to(device, non_blocking=True)
            del ed_activations
            actual_ed = (
                batch.candidate_target_values[batch.entity_index_mask_values]
                .argmax(dim=1)
                .to(device, non_blocking=True)
            )

            comparison_ed = (actual_ed == pred_ed).int()
            comparison_ed_with_no_ent = (actual_ed == pred_ed_with_no_ent).int()

            has_gold = actual_ed != 30
            pred_an_entity = pred_ed != 30
            tp += int((pred_ed_with_no_ent[has_gold] == actual_ed[has_gold]).sum().cpu())
            tn += int((pred_ed_with_no_ent[~has_gold] == actual_ed[~has_gold]).sum().cpu())

            # predicted no entity when should not have
            fn_negative += int(
                (pred_ed_with_no_ent[~pred_an_entity] != actual_ed[~pred_an_entity]).sum().cpu()
            )

            # predicted an entity and it is wrong
            fp += int(
                (pred_ed_with_no_ent[pred_an_entity] != actual_ed[pred_an_entity]).sum().cpu()
            )

            # missed the correct true entity (fn_negative + fp)
            fn += int(
                (pred_ed_with_no_ent[~pred_an_entity] != actual_ed[~pred_an_entity]).sum().cpu()
            ) + int((pred_ed_with_no_ent[pred_an_entity] != actual_ed[pred_an_entity]).sum().cpu())

            total_positives += int(has_gold.sum())
            total_negatives += int((~has_gold).sum())
            total_entities = comparison_ed.size(0)
            true_preds_ed = int(comparison_ed.sum().detach().cpu())

            true_preds_ed_with_no_ent = int(comparison_ed_with_no_ent.sum().detach().cpu())
            correct_ed += true_preds_ed

            correct_ed_with_no_ent += true_preds_ed_with_no_ent
            total_ed += total_entities
        del batch
    try:
        ed_acc = None
        ed_acc_no_ent = None
        ed_acc_final = None
        ed_acc_final_no_ent = None
        if not args.only_ner:
            ed_acc = float(correct_ed / total_ed * 100.0)
            ed_acc_no_ent = float(correct_ed_with_no_ent / total_ed * 100.0)
            ed_acc_final = float(correct_ed_final / total_ed * 100.0)
            ed_acc_final_no_ent = float(correct_ed_final_no_ent / total_ed * 100.0)
            logger.info("ED Accuracy {:.3f}%".format(ed_acc))
            logger.info("ED Accuracy (with no ent) {:.3f}%".format(ed_acc_no_ent))
            logger.info("ED Accuracy final {:.3f}%".format(ed_acc_final))
            logger.info("ED Accuracy final (with no ent) {:.3f}%".format(ed_acc_final_no_ent))
        ner_f1 = float(f1_score(np.array(actuals_ner), np.array(preds_ner), average="micro"))
        ner_f1_macro = float(f1_score(np.array(actuals_ner), np.array(preds_ner), average="macro"))
        logger.info("NER micro f1 score = {:.3f}".format(ner_f1))
        logger.info("NER macro f1 score = {:.3f}".format(ner_f1_macro))
        if print_report:
            logger.info(
                classification_report(
                    np.array(actuals_ner),
                    np.array(preds_ner),
                    output_dict=False,
                    target_names=["PAD"]
                    + [
                        class_to_label[str(x)] if str(x) in class_to_label else str(x)
                        for x in class_to_idx.keys()
                    ],
                )
            )
        del preds_ner, actuals_ner
        TB_WRITER.add_scalar("Entity_Typing_Micro_F1/test", ner_f1, epoch_num)
        TB_WRITER.add_scalar("Entity_Typing_Macro_F1/test", ner_f1_macro, epoch_num)
        TB_WRITER.add_scalar("Entity_Disambiguation_Accuracy/test", ed_acc, epoch_num)
        TB_WRITER.add_scalar(
            "Entity_Disambiguation_Accuracy_With_No_Entity/test", ed_acc_no_ent, epoch_num
        )
        TB_WRITER.add_scalar("Entity_Disambiguation_Accuracy_Final/test", ed_acc_final, epoch_num)
        TB_WRITER.add_scalar(
            "Entity_Disambiguation_Accuracy_With_No_Entity_Final/test",
            ed_acc_final_no_ent,
            epoch_num,
        )
        TB_WRITER.add_scalar("ED/test/breakdown/tp", tp, epoch_num)
        TB_WRITER.add_scalar("ED/test/breakdown/fp", fp, epoch_num)
        TB_WRITER.add_scalar("ED/test/breakdown/tn", tn, epoch_num)
        TB_WRITER.add_scalar("ED/test/breakdown/fn", fn, epoch_num)
        TB_WRITER.add_scalar("ED/test/breakdown/fn_negative", fn_negative, epoch_num)
        TB_WRITER.add_scalar("ED/test/breakdown/total_positives", total_positives, epoch_num)
        TB_WRITER.add_scalar("ED/test/breakdown/total_negatives", total_negatives, epoch_num)
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        TB_WRITER.add_scalar("ED/test/breakdown/p", p, epoch_num)
        TB_WRITER.add_scalar("ED/test/breakdown/r", r, epoch_num)
        TB_WRITER.add_scalar("ED/test/breakdown/f1", 2 * p * r / (p + r + 1e-8), epoch_num)
        TB_WRITER.add_scalar(
            "ED/test/breakdown/no_entity_p", tn / (tn + fn_negative + 1e-8), epoch_num
        )
        TB_WRITER.add_scalar(
            "ED/test/breakdown/no_entity_r", tn / (total_negatives + 1e-8), epoch_num
        )

        TB_WRITER.add_scalar(
            "Mention_Detection_Total_Loss/test", total_md_loss / sample_size, epoch_num
        )
        TB_WRITER.add_scalar(
            "Entity_Typing_Total_Loss/test", total_et_loss / sample_size, epoch_num
        )
        TB_WRITER.add_scalar(
            "Entity_Description_Total_Loss/test", total_desc_loss / sample_size, epoch_num
        )
        TB_WRITER.add_scalar(
            "Entity_Disambiguation_Total_Loss/test", total_ed_loss / sample_size, epoch_num
        )
        TB_WRITER.add_scalar(
            "Total_Loss/test",
            (total_md_loss + total_et_loss + total_et_loss + total_desc_loss) / sample_size,
            epoch_num,
        )
        data_preprocessor.max_candidates = args.max_candidates
        return ner_f1, ed_acc
    except Exception as err:
        data_preprocessor.max_candidates = args.max_candidates
        logger.error(err)
        return 0, 0


def evaluate_md(model, dev_dls, pad_token_id, device, return_predictions: bool = False):
    model.eval()
    f1s = []
    predictions = {}
    for dl_name, dl in dev_dls.items():
        predictions[dl_name] = []
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
                _, md_activations, _, _, _, _, _, _, _, _, _ = model(
                    token_ids=tokens, attention_mask=attention_mask, md_only=True
                )
                preds = md_activations
                mask = attention_mask.flatten() == 1.0
                gold_labels = labels.flatten()[mask].detach().cpu().numpy().tolist()
                label_preds = preds.argmax(dim=2).flatten()[mask].detach().cpu().numpy().tolist()
                all_trues.extend(gold_labels)
                all_preds.extend(label_preds)

                # error analysis
                bs_preds = preds.argmax(dim=2)
                for bs_idx in range(tokens.size(0)):
                    tokens_sent = tokens[bs_idx]
                    mask = tokens_sent != pad_token_id
                    preds_sent = bs_preds[bs_idx][mask].tolist()
                    labels_sent = labels[bs_idx][mask].tolist()

                    actual_spans = bio_to_offset_pairs(labels_sent)
                    pred_spans = bio_to_offset_pairs(preds_sent)

                    total_preds += len(pred_spans)
                    tp_total += len(pred_spans & actual_spans)
                    fp_total += len(pred_spans - actual_spans)
                    total_actual += len(actual_spans)

                    predictions[dl_name].append({"tokens": tokens_sent, "labels": labels_sent, "preds": preds_sent})

        p = tp_total / (total_preds + 1e-10)
        r = tp_total / (total_actual + 1e-10)
        f1 = 2 * (p * r) / (p + r + 1e-10)
        f1s.append(f1)
        logger.info(f"Span level metrics: p = {p:.3f}, r = {r:.3f}, f1 = {f1:.3f}")
        logger.info(classification_report(all_trues, all_preds, target_names=list(model.ner_tag_to_ix.keys()), digits=3))

    macro_f1 = sum(f1s) / len(f1s)
    logger.info(f"Macro (MD on manual datasets) F1 span level = {macro_f1}")

    if return_predictions:
        return predictions
    else:
        return None


if __name__ == "__main__":
    main()
