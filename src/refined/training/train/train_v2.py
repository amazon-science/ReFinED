import os
from typing import List

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from refined.data_types.doc_types import Doc
from refined.dataset_reading.entity_linking.wikipedia_dataset import WikipediaDataset
from refined.doc_preprocessing.preprocessor import PreprocessorInferenceOnly
from refined.doc_preprocessing.wikidata_mapper import WikidataMapper
from refined.inference.processor import Refined
from refined.model_components.config import NER_TAG_TO_IX, ModelConfig
from refined.model_components.refined_model import RefinedModel
from refined.resource_management.aws import S3Manager
from refined.resource_management.resource_manager import ResourceManager
from refined.torch_overrides.data_parallel_refined import DataParallelReFinED
from refined.training.fine_tune.fine_tune import run_fine_tuning_loops
from refined.training.train.training_args import parse_training_args
from refined.utilities.general_utils import get_logger

LOG = get_logger(name=__name__)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # DDP (ensure batch_elements_included is used)

    training_args = parse_training_args()

    resource_manager = ResourceManager(S3Manager(),
                                       data_dir=training_args.data_dir,
                                       entity_set=training_args.entity_set,
                                       load_qcode_to_title=True,
                                       load_descriptions_tns=True,
                                       model_name=None
                                       )
    if training_args.download_files:
        resource_manager.download_data_if_needed()
        resource_manager.download_additional_files_if_needed()
        resource_manager.download_training_files_if_needed()

    preprocessor = PreprocessorInferenceOnly(
        data_dir=training_args.data_dir,
        debug=training_args.debug,
        max_candidates=training_args.num_candidates_train,
        transformer_name=training_args.transformer_name,
        ner_tag_to_ix=NER_TAG_TO_IX,  # for now include default ner_to_tag_ix can make configurable in future
        entity_set=training_args.entity_set,
        use_precomputed_description_embeddings=False
    )

    wikidata_mapper = WikidataMapper(resource_manager=resource_manager)

    wikipedia_dataset_file_path = resource_manager.get_training_data_files()['wikipedia_training_dataset']
    training_dataset = WikipediaDataset(
        # start=100,
        start=100,
        end=100000000,  # large number means every line will be read until the end of the file
        preprocessor=preprocessor,
        resource_manager=resource_manager,
        wikidata_mapper=wikidata_mapper,
        dataset_path=wikipedia_dataset_file_path,
        batch_size=training_args.batch_size,
        num_workers=8 * training_args.n_gpu,
        prefetch=100,  # add random number for each worker and have more than 2 workers to remove waiting
        mask=training_args.mask_prob,
        random_mask=training_args.mask_random_prob,
        lower_case_prob=0.05,
        candidate_dropout=training_args.candidate_dropout,
        max_mentions=training_args.max_mentions,
        sample_k_candidates=5,
        add_main_entity=True
    )
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=None, num_workers=8 * training_args.n_gpu,
                                     # pin_memory=True if training_args.n_gpu == 1 else False,
                                     pin_memory=True,  # may break ddp and dp training
                                     prefetch_factor=5,  # num_workers * prefetch_factor
                                     persistent_workers=True  # persistent_workers means memory is stable across epochs
                                     )
    eval_docs: List[Doc] = list(iter(WikipediaDataset(
        start=0,
        end=100,  # first 100 docs are used for eval
        preprocessor=preprocessor,
        resource_manager=resource_manager,
        wikidata_mapper=wikidata_mapper,
        dataset_path=wikipedia_dataset_file_path,
        return_docs=True,  # this means the dataset will return `Doc` objects instead of BatchedElementsTns
        batch_size=1 * training_args.n_gpu,
        num_workers=1,
        prefetch=1,
        mask=0.0,
        random_mask=0.0,
        lower_case_prob=0.0,
        candidate_dropout=0.0,
        max_mentions=25,  # prevents memory issues
        add_main_entity=True  # add weak labels,
    )))

    model = RefinedModel(
        ModelConfig(data_dir=preprocessor.data_dir,
                    transformer_name=preprocessor.transformer_name,
                    ner_tag_to_ix=preprocessor.ner_tag_to_ix
                    ),
        preprocessor=preprocessor
    )

    if training_args.restore_model_path is not None:
        # TODO load `ModelConfig` file (from the directory) and initialise RefinedModel from that
        # to avoid issues when model config differs
        LOG.info(f'Restored model from {training_args.restore_model_path}')
        checkpoint = torch.load(training_args.restore_model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

    if training_args.n_gpu > 1:
        model = DataParallelReFinED(model, device_ids=list(range(training_args.n_gpu)), output_device=training_args.device)
    model = model.to(training_args.device)

    # wrap a ReFinED processor around the model so evaluation methods can be run easily
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    refined = Refined(
        model_file_or_model=model,
        model_config_file_or_model_config=model_to_save.config,
        preprocessor=preprocessor,
        device=training_args.device
    )

    param_groups = [
        {"params": model_to_save.get_et_params(), "lr": training_args.lr * 100},
        {"params": model_to_save.get_desc_params(), "lr": training_args.lr},
        {"params": model_to_save.get_ed_params(), "lr": training_args.lr * 100},
        {"params": model_to_save.get_parameters_not_to_scale(), "lr": training_args.lr}
    ]
    if training_args.el:
        param_groups.append({"params": model_to_save.get_md_params(), "lr": training_args.lr})

    optimizer = AdamW(param_groups, lr=training_args.lr, eps=1e-8)

    total_steps = len(training_dataloader) * training_args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=total_steps / training_args.gradient_accumulation_steps
    )

    scaler = GradScaler()

    if training_args.restore_model_path is not None and training_args.resume:
        LOG.info("Restoring optimizer and scheduler")
        optimizer_checkpoint = torch.load(
            os.path.join(os.path.dirname(training_args.restore_model_path), "optimizer.pt"),
            map_location="cpu",
        )
        scheduler_checkpoint = torch.load(
            os.path.join(os.path.dirname(training_args.restore_model_path), "scheduler.pt"),
            map_location="cpu",
        )
        scaler_checkpoint = torch.load(
            os.path.join(os.path.dirname(training_args.restore_model_path), "scaler.pt"),
            map_location="cpu",
        )
        optimizer.load_state_dict(optimizer_checkpoint)
        scheduler.load_state_dict(scheduler_checkpoint)
        scaler.load_state_dict(scaler_checkpoint)

    run_fine_tuning_loops(
        refined=refined,
        fine_tuning_args=training_args,
        training_dataloader=training_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluation_dataset_name_to_docs={'WIKI_DEV': eval_docs},
        checkpoint_every_n_steps=training_args.checkpoint_every_n_steps
    )


if __name__ == "__main__":
    main()
