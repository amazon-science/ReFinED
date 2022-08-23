# use iterable datalaoder instead of loading into memory all documents at once
# add more evaluation steps (BLINK only uses 100k for fine-tuning and lower lr)
# train less data and see results show results every 50k steps or so

import argparse
import os
import sys
import time

import torch
from dataset_reading.dataset_factory import Datasets
from dataset_reading.document_dataset import DocIterDataset
from doc_preprocessing.dataclasses import BatchedElementsTns
from evaluation.evaluation import evaluate_on_docs
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from refined.processor import Refined

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # ensures first 4 GPUs are visible


code_dir = "/data/tayoola/2021_code/"

sys.path.append(code_dir)


def main():
    parser = argparse.ArgumentParser(description="Train ED model on WikiLinksNED dataset.")
    parser.add_argument(
        "--debug",
        default=False,
        type=bool,
        help="debug mode.",
    )
    parser.add_argument(
        "--restore_model_checkpoint",
        default=False,
        type=bool,
        help="restore (Wikipedia dataset pretrained) model checkpoint when True (will be overwriten by "
        "restore_model_path).",
    )
    parser.add_argument(
        "--restore_model_path",
        default=None,
        type=str,
        help="The path to the model checkpoint to restore.",
    )
    parser.add_argument(
        "--batch_size",
        default=6,
        type=int,
        help="batch size (1 mention per batch)",
    )
    parser.add_argument(
        "--steps_per_eval",
        default=4000,
        type=int,
        help="Number of steps before evaluating on dev set.",
    )
    parser.add_argument(
        "--epochs",
        default=4,
        type=int,
        help="batch size (1 mention per batch)",
    )
    parser.add_argument(
        "--n_gpu",
        default=4,
        type=int,
        help="batch size (1 mention per batch)",
    )
    parser.add_argument(
        "--mask_prob",
        default=0.80,
        type=float,
        help="batch size (1 mention per batch)",
    )
    parser.add_argument(
        "--random_mask_prob",
        default=0.05,
        type=float,
        help="batch size (1 mention per batch)",
    )
    parser.add_argument(
        "--results_file",
        default="ed_results.json",
        type=str,
        help="file to write results",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="device",
    )
    args = parser.parse_args()

    el = False
    # do not use precomputed descriptions or string features
    data_dir = "/data/tayoola/2021_data"
    # model_dir = '/data/tayoola/data/trained_models/21_march'
    model_dir = "/data/tayoola/2021_code/TomED/best_model_2"
    datasets_dir = "/data/tayoola/2021_data/datasets"
    device = args.device
    if args.restore_model_path is not None:
        print(f"Restoring model checkpoint from {args.restore_model_path}.")
        model_dir = args.restore_model_path.replace("/model.pt", "")
        restore_model_checkpoint = True
    else:
        restore_model_checkpoint = args.restore_model_checkpoint
        if restore_model_checkpoint:
            print("Restoring Wikipeida weights")
        else:
            print("Using inital weights (not restoring from EL model)")

    refined = Refined(
        model_dir=model_dir,
        data_dir=data_dir,
        debug=args.debug,
        requires_redirects_and_disambig=True,
        backward_coref=True,
        device=device,
        use_cpu=False,
        n_gpu=args.n_gpu,
        restore_model_checkpoint=restore_model_checkpoint,
    )

    refined.preprocessor.max_candidates = 30
    refined.preprocessor.zero_string_features = True
    refined.preprocessor.precomputed_descriptions = None

    # refined.model = RefinedModel.from_pretrained('/data/tayoola/2021_code/TomED/best_model_2', refined.preprocessor)
    # refined.model.to(refined.device)
    # _ = refined.model.eval()
    # try higher learning rate

    refined.preprocessor.max_candidates = 30
    refined.preprocessor.zero_string_features = True
    refined.preprocessor.precomputed_descriptions = None
    datasets = Datasets(preprocessor=refined.preprocessor, datasets_path=datasets_dir)

    if args.debug:
        docs = []
        i = 0
        for doc in datasets.get_wikilinks_ned_docs(
            split="train", include_gold_label=True, sample_k_candidates=5
        ):
            docs.append(doc)
            i += 1
            if i > 1000:
                break
    else:
        docs = []
        i = 0
        for doc in datasets.get_wikilinks_ned_docs(
            split="train", include_gold_label=True, sample_k_candidates=5
        ):
            docs.append(doc)
            i += 1
            if i > 1000000:
                break

    dataset = DocIterDataset(
        docs=docs,
        preprocessor=refined.preprocessor,
        mask=True,
        mask_prob=args.mask_prob,
        random_mask_prob=args.random_mask_prob,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size * args.n_gpu,
        shuffle=False,
        num_workers=8,
        collate_fn=dataset.collate,
    )

    _ = refined.model.train()
    model = refined.model

    for params in model.parameters():
        params.requires_grad = True

    ed_dropout = 0.05
    et_dropout = 0.10
    # check this is ok to do (does it copy across to all gpus)
    _model = model.module if hasattr(model, "module") else model
    _model.entity_disambiguation.dropout.p = ed_dropout
    _model.entity_typing.dropout.p = et_dropout

    gradient_accumulation_steps = 5

    epochs = args.epochs

    if el:
        optimizer = AdamW(
            [
                {"params": _model.get_md_params(), "lr": 5e-6},
                {"params": _model.get_et_params(), "lr": 5e-4},
                {"params": _model.get_desc_params(), "lr": 5e-6},
                {"params": _model.get_ed_params(), "lr": 5e-4},
                {"params": _model.get_parameters_not_to_scale(), "lr": 5e-6},
            ],
            lr=5e-3,  # not used I think
            eps=1e-8,
        )
    else:
        optimizer = AdamW(
            [
                {"params": _model.get_et_params(), "lr": 5e-4},
                {"params": _model.get_desc_params(), "lr": 5e-6},
                {"params": _model.get_ed_params(), "lr": 5e-4},
                {"params": _model.get_parameters_not_to_scale(), "lr": 5e-6},
            ],
            lr=5e-6,  # not used I think
            eps=1e-8,
        )

    total_steps = 2000000 * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=total_steps / gradient_accumulation_steps
    )

    for epoch_num in trange(epochs):

        evaluate(refined, model, epoch_num, datasets)
        total_loss = 0
        it_start_time = time.time()
        for step, batch in tqdm(enumerate(dataloader)):
            # print(f'{round(time.time() - it_start_time, 5)}s to load batch')
            if args.n_gpu == 1:
                batch = BatchedElementsTns(
                    *[x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
                )
            (
                md_loss,
                _,
                et_loss,
                _,
                ed_loss,
                ed_logits,
                _,
                _,
                description_loss,
                _,
                kg_loss,
                kg_scores,
                final_loss,
                final_score,
                _,
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

            loss = ed_loss + et_loss + (description_loss * 0.01)
            if el:
                loss += md_loss * 0.01
            if gradient_accumulation_steps >= 1:
                loss = loss / gradient_accumulation_steps

            if args.n_gpu > 1:
                loss = loss.mean()

            total_loss += loss.item()
            if step % 500 == 0:
                print(f"total_loss: {total_loss/(step+1e-8)}")

            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            # del tensors to save memory
            # del batch, loss
            if (step + 1) % args.steps_per_eval == 0:
                evaluate(refined, model, epoch_num, datasets)
            it_start_time = time.time()


def evaluate(refined, model, epoch_num, datasets):
    model.eval()
    refined.preprocessor.max_candidates = 30
    metrics = evaluate_on_docs(
        refined=refined,
        docs=datasets.get_wikilinks_ned_docs(split="test", include_gold_label=True),
        progress_bar=True,
        sample_size=500,
    )
    model.train()
    print("Metrics: ", metrics.get_summary(), "\n\n")
    print(f"epoch_num: {epoch_num}", "\n\n")
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    torch.save(model_to_save.state_dict(), f"model_{epoch_num}-{metrics.get_f1()}.pt")


if __name__ == "__main__":
    main()
