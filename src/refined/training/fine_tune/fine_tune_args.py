import json
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Optional

import torch.cuda

from refined.offline_data_generation.clean_wikipedia import str2bool
from refined.training.train.training_args import TrainingArgs


@dataclass
class FineTuningArgs(TrainingArgs):
    # FineTuningArgs is used to store sensible defaults for fine-tuning
    experiment_name: str = field(default_factory=str)
    class_name: str = 'FineTuningArgs'
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    el: bool = True
    ed_dropout: float = 0.05
    et_dropout: float = 0.10
    gradient_accumulation_steps: int = 4
    epochs: int = 10
    lr: float = 5e-5
    batch_size: int = 1
    ed_threshold: float = 0.15
    num_warmup_steps: int = 10
    num_candidates_train: int = 30
    num_candidates_eval: int = 30
    use_precomputed_descriptions: bool = False
    output_dir: str = 'fine_tuned_models'
    restore_model_path: Optional[str] = None

    # This can be either wikipedia_model or wikipedia_model_with_numbers
    # `wikipedia_model` only detects named entities ensure training `span` have `coarse_type=MENTION`
    # `wikipedia_model` detects named entities, dates, and numeric values. `span` has one of the
    # following `coarse_type` values:
    #  - DATE
    #  - CARDINAL
    #  - MONEY
    #  - PERCENT
    #  - TIME
    #  - ORDINAL
    #  - QUANTITY
    #  - MENTION
    model_name: str = 'wikipedia_model'  # or wikipedia_model_with_numbers

    # This can be either 'wikipedia' or 'wikidata'. It is the entity set that model is considering when performing
    # entity linking.
    entity_set: str = 'wikipedia'

    # set to high value because it evaluates after each epoch and each epoch is often short for fine-tuning
    checkpoint_every_n_steps: int = 1000000

    def add_command_line_args(self, args) -> None:
        for arg in vars(args):
            if arg in self.__dict__:
                setattr(self, arg, getattr(args, arg))
            else:
                raise Exception(f"Unrecognized argument {arg}")

    @classmethod
    def from_file(cls, filepath: str):
        with open(filepath, "r") as f:
            cfg = json.load(f)
        return cls(**cfg)

    def to_file(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.__dict__, f)


def parse_fine_tuning_args() -> FineTuningArgs:
    fine_tuning_args = FineTuningArgs()
    parser = ArgumentParser("This script is used to fine-tune the model for end-to-end EL or ED.")
    parser.add_argument(
        "--experiment_name",
        default=fine_tuning_args.experiment_name,
        type=str,
        required=True,
        help="experiment name, determines file_path to store saved model. "
             "Ensure it is unique to avoid overwriting saved models.",
    )
    parser.add_argument(
        "--device",
        default=fine_tuning_args.device,
        type=str,
        help="device id",
    )
    parser.add_argument(
        "--el",
        default=fine_tuning_args.el,
        type=str2bool,
        help="device id",
    )
    parser.add_argument(
        "--epochs",
        default=fine_tuning_args.epochs,
        type=int,
        help="Epochs",
    )
    parser.add_argument(
        "--batch_size",
        default=fine_tuning_args.batch_size,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--num_candidates_train",
        default=fine_tuning_args.num_candidates_train,
        type=int,
        help="max_candidates_train number of candidate entities to use during training.",
    )
    parser.add_argument(
        "--num_candidates_eval",
        default=fine_tuning_args.num_candidates_eval,
        type=int,
        help="max_candidates_eval number of candidate entities to use during evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=fine_tuning_args.gradient_accumulation_steps,
        type=int,
        help="gradient_accumulation_steps",
    )
    parser.add_argument(
        "--lr",
        default=fine_tuning_args.lr,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--ed_dropout",
        default=fine_tuning_args.ed_dropout,
        type=float,
        help="ed_droput",
    )
    parser.add_argument(
        "--et_dropout",
        default=fine_tuning_args.et_dropout,
        type=float,
        help="et_droput",
    )
    parser.add_argument(
        "--ed_threshold",
        default=fine_tuning_args.ed_threshold,
        type=float,
        help="ed_threshold is the model softmax confidence score threshold to use as a cutoff for evaluation.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        default=fine_tuning_args.num_warmup_steps,
        type=int,
        help="num_warmup_steps",
    )
    parser.add_argument(
        "--use_precomputed_descriptions",
        default=fine_tuning_args.use_precomputed_descriptions,
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
        default=fine_tuning_args.output_dir,
        type=str,
        help="output_dir this is the relative or absolute file path where the fine-tuned model will be saved.",
    )
    parser.add_argument(
        "--model_name",
        default=fine_tuning_args.model_name,
        type=str,
        help="""This can be either wikipedia_model or wikipedia_model_with_numbers
    `wikipedia_model` only detects named entities. Ensure the training dataset returns a list of `span` with
     `coarse_type="MENTION"`
    `wikipedia_model` detects named entities, dates, and numeric values. `span` has one of the
    following `coarse_type` values:
      - "DATE"
      - "CARDINAL"
      - "MONEY"
      - "PERCENT"
      - "TIME"
      - "ORDINAL"
      - "QUANTITY"
      - "MENTION"
      Note that datasets should include non-MENTION spans in the `md_spans` list because this informs the model
      not to attempt to link these entities to a knowledge base. Example:
          Doc.from_text_with_spans(text='...', spans=[...], md_spans=[Span(..., coarse_type="DATE"),...])
      """,
    )
    parser.add_argument(
        "--entity_set",
        default=fine_tuning_args.entity_set,
        type=str,
        help="""This can be either 'wikipedia' or 'wikidata'. It is the entity set that model is
        considering when performing entity linking. Note that once the model is trained the entity set can be changed
        but performance may be degraded.""",
    )
    parser.add_argument(
        "--checkpoint_every_n_steps",
        default=fine_tuning_args.checkpoint_every_n_steps,
        type=int,
        help="""checkpoint_every_n_steps.""",
    )

    args = parser.parse_args()
    fine_tuning_args.add_command_line_args(args=args)
    return fine_tuning_args
