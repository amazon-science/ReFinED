import json
import os
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional

import torch

from refined.offline_data_generation.clean_wikipedia import str2bool
from refined.utilities.general_utils import get_logger

LOG = get_logger(__name__)


@dataclass
class TrainingArgs:
    # TrainingArgs is used to store sensible defaults for training
    class_name: str = 'TrainingArgs'
    experiment_name: str = f'{int(time.time())}'
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    el: bool = True  # end-to-end entity linking (MD + ED + ET) when True else
    # it will train entity disambiguation (ED) and entity typing (ET)
    ed_dropout: float = 0.05
    et_dropout: float = 0.10
    temperature_scaling: float = 0.02
    gradient_accumulation_steps: int = 1
    epochs: int = 2
    lr: float = 5e-5
    batch_size: int = 32  # 64 uses around 40 GB++, 32 uses 22 GB
    ed_threshold: float = 0.15
    num_warmup_steps: int = 10
    num_candidates_train: int = 30
    num_candidates_eval: int = 30
    use_precomputed_descriptions: bool = False
    output_dir: str = 'fine_tuned_models'
    restore_model_path: Optional[str] = None
    # This can be either 'wikipedia' or 'wikidata'. It is the entity set that model is considering when performing
    # entity linking.
    entity_set: str = 'wikidata'
        
    languages: str = 'ar_de_en_es_fa_sr_ta_tr_fr_it'
    number_of_language: int = 11

    data_dir: str = 'data' 
    debug: bool = False
    transformer_name: str = "bert-base-multilingual-cased" 
    n_gpu: int = 8
    mask_prob: float = 0.70
    mask_random_prob: float = 0.05
    candidate_dropout: float = 0.0
    max_mentions: int = 25
    download_files: bool = False
    checkpoint_every_n_steps: int = 2000
    resume: bool = False  # Resume training with same optimizer, scheduler, scaler (useful if previously crashed).

    checkpoint_metric: Optional[str] = None  # Needs to be "el" or "ed". By default it will be "el" if el is True
    # and "ed" if el is False.

    def __post_init__(self):
        if self.checkpoint_metric is None:
            self.checkpoint_metric = "el" if self.el else "ed"
        if self.checkpoint_metric is not None:
            # check input value
            self.checkpoint_metric = self.checkpoint_metric.lower()
            assert self.checkpoint_metric in {"el", "ed"}, "--checkpoint_metric must be 'el' or 'ed'."
        

    def post_add_command_line_args(self):
        self.__post_init__()
        LOG.info(f"Using batch size {self.batch_size} as n_gpu is {self.n_gpu}.")

    def add_command_line_args(self, args) -> None:
        for arg in vars(args):
            if arg in self.__dict__:
                setattr(self, arg, getattr(args, arg))
            else:
                raise Exception(f"Unrecognized argument {arg}")
        self.post_add_command_line_args()

    @classmethod
    def from_file(cls, filepath: str):
        with open(filepath, "r") as f:
            cfg = json.load(f)
        return cls(**cfg)

    def to_file(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.__dict__, f)


def parse_training_args() -> TrainingArgs:
    training_args = TrainingArgs()
    parser = ArgumentParser("This script is used to train the model for end-to-end EL or ED.")
    parser.add_argument(
        "--languages",
        default=training_args.languages,
        type=str,
        required=False,
        help="Languages that you want to train a model, e.g., en_es_de",
    )
    parser.add_argument(
        "--experiment_name",
        default=training_args.experiment_name,
        type=str,
        required=False,
        help="experiment name, determines file_path to store saved model. "
             "Ensure it is unique to avoid overwriting saved models.",
    )
    parser.add_argument(
        "--device",
        default=training_args.device,
        type=str,
        help="device id",
    )
    parser.add_argument(
        "--el",
        default=training_args.el,
        type=str2bool,
        help="trains end-to-end EL (MD + ET + ED), if False will train (ET + ED) only",
    )
    parser.add_argument(
        "--epochs",
        default=training_args.epochs,
        type=int,
        help="Epochs",
    )
    parser.add_argument(
        "--batch_size",
        default=training_args.batch_size,
        type=int,
        help="batch size per GPU",
    )
    parser.add_argument(
        "--num_candidates_train",
        default=training_args.num_candidates_train,
        type=int,
        help="max_candidates_train number of candidate entities to use during training.",
    )
    parser.add_argument(
        "--num_candidates_eval",
        default=training_args.num_candidates_eval,
        type=int,
        help="max_candidates_eval number of candidate entities to use during evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=training_args.gradient_accumulation_steps,
        type=int,
        help="gradient_accumulation_steps",
    )
    parser.add_argument(
        "--lr",
        default=training_args.lr,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--temperature_scaling",
        default=training_args.temperature_scaling,
        type=float,
        help="temperature_scaling",
    )
    parser.add_argument(
        "--ed_dropout",
        default=training_args.ed_dropout,
        type=float,
        help="ed_droput",
    )
    parser.add_argument(
        "--et_dropout",
        default=training_args.et_dropout,
        type=float,
        help="et_droput",
    )
    parser.add_argument(
        "--ed_threshold",
        default=training_args.ed_threshold,
        type=float,
        help="ed_threshold is the model softmax confidence score threshold to use as a cutoff for evaluation.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        default=training_args.num_warmup_steps,
        type=int,
        help="num_warmup_steps",
    )
    parser.add_argument(
        "--use_precomputed_descriptions",
        default=training_args.use_precomputed_descriptions,
        type=str2bool,
        help="""use_precomputed_descriptions should typically be False. If precomputed_descriptions are used it
                will mean that the model does not update the entity description embeddings, which will limit
                the benefit of fine-tuning. Only use `precomputed_descriptions` when you believe the current
                description embeddings are expressive enough to obtain strong performance and you want to speed
                up the fine-tuning by not updating them.
        """,
    )
    parser.add_argument(
        "--output_dir",
        default=training_args.output_dir,
        type=str,
        help="output_dir this is the relative or absolute file path where the fine-tuned model will be saved.",
    )
    parser.add_argument(
        "--entity_set",
        default=training_args.entity_set,
        type=str,
        help="""This can be either 'wikipedia' or 'wikidata'. It is the entity set that model is
        considering when performing entity linking. Note that once the model is trained the entity set can be changed
        but performance may be degraded.""",
    )

    parser.add_argument(
        "--data_dir",
        default=training_args.data_dir,
        type=str,
        help="file path to directory containing lookup files and wikipedia_data.",
    )
    parser.add_argument(
        "--transformer_name",
        default=training_args.transformer_name,
        type=str,
        help="transformer_name",
    )
    parser.add_argument(
        "--n_gpu",
        default=training_args.n_gpu,
        type=int,
        help="n_gpu",
    )
    parser.add_argument(
        "--mask_prob",
        default=training_args.mask_prob,
        type=float,
        help="Probability of replacing a mention text with [mask] tokens.",
    )
    parser.add_argument(
        "--mask_random_prob",
        default=training_args.mask_random_prob,
        type=float,
        help="Probability of replacing a masked mention text with random tokens.",
    )
    parser.add_argument(
        "--candidate_dropout",
        default=training_args.candidate_dropout,
        type=float,
        help="Probability of removing correct entity from candidate list (making NOTA correct target).",
    )
    parser.add_argument(
        "--max_mentions",
        default=training_args.max_mentions,
        type=int,
        help="Max mentions per chunk (limits memory usage).",
    )
    parser.add_argument(
        "--download_files",
        default=training_args.download_files,
        type=str2bool,
        help="Download files.",
    )
    parser.add_argument(
        "--checkpoint_every_n_steps",
        default=training_args.checkpoint_every_n_steps,
        type=int,
        help="""checkpoint_every_n_steps.""",
    )
    parser.add_argument(
        "--resume",  # only valid for training_script because fine-tuning is usually quick to restart
        default=training_args.resume,
        type=str2bool,
        help="""resume training with same optimizer, scheduler, scaler (useful if previously crashed).""",
    )
    parser.add_argument(
        "--restore_model_path",
        default=training_args.restore_model_path,
        type=str,
        help="""File path to model.pt file. Folder should contain optimizer, scheudler, scaler, config if resume is 
        set to yes""",
    )
    parser.add_argument(
        "--checkpoint_metric",
        default="ed",
        type=str,
        help="""Needs to be "el" or "ed". Determines whether EL or ED F1 score will be used for checkpoint evaluation.
        By default "ed" will be used.
        When training EL on Wikipedia hyperlinks this should be set to "ed" because "el" evaluation is not reliable
        because Wikipedia hyperlinks are partial EL annotations.""",
    )

    args = parser.parse_args()
    training_args.add_command_line_args(args=args)
    return training_args
