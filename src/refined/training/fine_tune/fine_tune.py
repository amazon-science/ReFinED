import os
import shutil
import time
from statistics import mean
from typing import Dict, Iterable, Optional

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm
from transformers import get_linear_schedule_with_warmup

from refined.data_types.doc_types import Doc
from refined.dataset_reading.entity_linking.document_dataset import DocDataset
from refined.evaluation.evaluation import get_datasets_obj, evaluate
from refined.inference.processor import Refined
from refined.training.fine_tune.fine_tune_args import FineTuningArgs, parse_fine_tuning_args
from refined.training.train.training_args import TrainingArgs
from refined.utilities.general_utils import get_logger
LOG = get_logger(name=__name__)

def main():
    fine_tuning_args = parse_fine_tuning_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    LOG.info("Fine-tuning end-to-end EL" if fine_tuning_args.el else "Fine-tuning ED only.")
    refined = Refined.from_pretrained(model_name=fine_tuning_args.model_name,
                                      entity_set=fine_tuning_args.entity_set,
                                      use_precomputed_descriptions=fine_tuning_args.use_precomputed_descriptions,
                                      device=fine_tuning_args.device)

    datasets = get_datasets_obj(preprocessor=refined.preprocessor)

    evaluation_dataset_name_to_docs = {
        "AIDA": list(datasets.get_aida_docs(
            split="dev",
            include_gold_label=True,
            filter_not_in_kb=True,
            include_spans=True,
        ))
    }
    start_fine_tuning_task(refined=refined,
                           fine_tuning_args=fine_tuning_args,
                           train_docs=list(datasets.get_aida_docs(split="train", include_gold_label=True)),
                           evaluation_dataset_name_to_docs=evaluation_dataset_name_to_docs)


def start_fine_tuning_task(refined: 'Refined', train_docs: Iterable[Doc],
                           evaluation_dataset_name_to_docs: Dict[str, Iterable[Doc]],
                           fine_tuning_args: FineTuningArgs):
    LOG.info("Fine-tuning end-to-end EL" if fine_tuning_args.el else "Fine-tuning ED only.")
    train_docs = list(train_docs)
    training_dataset = DocDataset(
        docs=train_docs,
        preprocessor=refined.preprocessor
    )
    training_dataloader = DataLoader(
        dataset=training_dataset, batch_size=fine_tuning_args.batch_size, shuffle=True, num_workers=1,
        collate_fn=training_dataset.collate
    )

    model = refined.model

    if fine_tuning_args.restore_model_path is not None:
        LOG.info(f'Restored model from {fine_tuning_args.restore_model_path}')
        checkpoint = torch.load(fine_tuning_args.restore_model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

    for params in model.parameters():
        params.requires_grad = True

    model.entity_disambiguation.dropout.p = fine_tuning_args.ed_dropout
    model.entity_typing.dropout.p = fine_tuning_args.et_dropout

    param_groups = [
        {"params": model.get_et_params(), "lr": fine_tuning_args.lr * 100},
        {"params": model.get_desc_params(), "lr": fine_tuning_args.lr},
        {"params": model.get_ed_params(), "lr": fine_tuning_args.lr * 100},
        {"params": model.get_parameters_not_to_scale(), "lr": fine_tuning_args.lr}
    ]
    if fine_tuning_args.el:
        param_groups.append({"params": model.get_md_params(), "lr": fine_tuning_args.lr})

    optimizer = AdamW(param_groups, lr=fine_tuning_args.lr, eps=1e-8)

    total_steps = len(training_dataloader) * fine_tuning_args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=fine_tuning_args.num_warmup_steps,
        num_training_steps=total_steps / fine_tuning_args.gradient_accumulation_steps
    )

    run_fine_tuning_loops(refined=refined, fine_tuning_args=fine_tuning_args,
                          training_dataloader=training_dataloader, optimizer=optimizer,
                          scheduler=scheduler, evaluation_dataset_name_to_docs=evaluation_dataset_name_to_docs,
                          checkpoint_every_n_steps=fine_tuning_args.checkpoint_every_n_steps)


def run_fine_tuning_loops(refined: Refined, fine_tuning_args: TrainingArgs, training_dataloader: DataLoader,
                          optimizer: AdamW, scheduler, evaluation_dataset_name_to_docs: Dict[str, Iterable[Doc]],
                          checkpoint_every_n_steps: int = 1000000, scaler: GradScaler = GradScaler()):
    model = refined.model
    best_recall = 0.0
    for epoch_num in trange(fine_tuning_args.epochs):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        model.train()
        LOG.info(f"Starting epoch number {epoch_num}")
        for param_group in optimizer.param_groups:
            LOG.info(f"lr: {param_group['lr']}")
        total_loss = 0.0
        for step, batch in tqdm(enumerate(training_dataloader), total=len(training_dataloader)):
            batch = batch.to(fine_tuning_args.device) 
            with autocast():
                output = model(batch=batch)
                loss = output.ed_loss + output.et_loss + (output.description_loss * 0.01)
                if fine_tuning_args.el:
                    loss += output.md_loss * 0.01
                if fine_tuning_args.gradient_accumulation_steps >= 1:
                    loss = loss / fine_tuning_args.gradient_accumulation_steps

            loss = loss.mean()
            total_loss += loss.item()

            if step % 100 == 99: 
                LOG.info(f"Loss: {total_loss / step}")

            scaler.scale(loss).backward()

            if (step + 1) % fine_tuning_args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % checkpoint_every_n_steps == 0:
                best_recall = run_checkpoint_eval_and_save(best_recall, evaluation_dataset_name_to_docs, fine_tuning_args,
                                                       refined, optimizer=optimizer, scaler=scaler,
                                                       scheduler=scheduler)

        best_recall = run_checkpoint_eval_and_save(best_recall, evaluation_dataset_name_to_docs, fine_tuning_args,
                                               refined, optimizer=optimizer, scaler=scaler,
                                               scheduler=scheduler)
def run_checkpoint_eval_and_save(best_recall: float, evaluation_dataset_name_to_docs: Dict[str, Iterable[Doc]],
                                 fine_tuning_args: TrainingArgs, refined: Refined, optimizer: AdamW,
                                 scaler: GradScaler,
                                 scheduler):
    torch.cuda.empty_cache()
    evaluation_metrics = evaluate(refined=refined,
                                  evaluation_dataset_name_to_docs=evaluation_dataset_name_to_docs,
                                  el=fine_tuning_args.el,  # only evaluate EL when training EL
                                  ed=True,  # always evaluate standalone ED
                                  ed_threshold=fine_tuning_args.ed_threshold)
    if fine_tuning_args.checkpoint_metric == 'el':
        LOG.info("Using EL performance for checkpoint metric")
        average_recall = mean([metrics.get_recall() for metrics in evaluation_metrics.values() if metrics.el])
    elif fine_tuning_args.checkpoint_metric == 'ed':
        LOG.info("Using ED performance for checkpoint metric")
        average_recall = mean([metrics.get_recall() for metrics in evaluation_metrics.values() if not metrics.el])
    else:
        raise Exception("--checkpoint_metric (`checkpoint_metric`) needs to be set to el or ed,")

    if average_recall > best_recall:
        LOG.info(f"Obtained best recall so far of {average_recall:.3f} (previous best {best_recall:.3f})")
        best_recall = average_recall
        model_output_dir = os.path.join(fine_tuning_args.output_dir, fine_tuning_args.experiment_name)
        if os.path.exists(model_output_dir):
            shutil.rmtree(model_output_dir)
        model_output_dir = os.path.join(model_output_dir, f"Recall_{average_recall:.4f}")
        os.makedirs(model_output_dir, exist_ok=True)
        LOG.info(f"Storing model at {model_output_dir} along with optimizer, scheduler, and scaler")
        model_to_save = (
            refined.model.module if hasattr(refined.model, "module") else refined.model
        )
        torch.save(model_to_save.state_dict(), os.path.join(model_output_dir, "model.pt"))
        fine_tuning_args.to_file(os.path.join(model_output_dir, "fine_tuning_args.json"))
        model_to_save.config.to_file(os.path.join(model_output_dir, "config.json"))

        #save optimiser, scheduler, and scaler so training can be resumed if it crashes
        torch.save(optimizer.state_dict(), os.path.join(model_output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(model_output_dir, "scheduler.pt"))
        torch.save(scaler.state_dict(), os.path.join(model_output_dir, "scaler.pt"))

    torch.cuda.empty_cache()
    return best_recall



def fine_tune_on_docs(refined: Refined, train_docs: Iterable[Doc], eval_docs: Iterable[Doc],
                      fine_tuning_args: Optional[FineTuningArgs] = None):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if fine_tuning_args is None:
        fine_tuning_args = FineTuningArgs(experiment_name=f'{int(time.time())}')
    evaluation_dataset_name_to_docs = {
        "custom_eval_dataset": list(eval_docs)
    }
    start_fine_tuning_task(refined=refined, train_docs=train_docs, fine_tuning_args=fine_tuning_args,
                           evaluation_dataset_name_to_docs=evaluation_dataset_name_to_docs)


if __name__ == "__main__":
    main()
