import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from refined.utilities.general_utils import get_tokenizer

NER_TAG_TO_IX = {
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

@dataclass
class ModelConfig:
    data_dir: str
    transformer_name: str
    max_seq: int = 510
    learning_rate: float = 5e-5
    num_train_epochs: int = 2
    freeze_all_bert_layers: bool = False
    gradient_accumulation_steps: int = 1
    per_gpu_batch_size: int = 12
    freeze_embedding_layers: bool = False
    freeze_layers: List[str] = field(default_factory=lambda: [])
    n_gpu: int = 4
    lr_ner_scale: int = 100
    ner_layer_dropout: float = 0.10  # was 0.15
    ed_layer_dropout: float = 0.05  # should add pem specific dropout
    max_candidates: int = 30
    warmup_steps: int = 10000  # 5000 could be used when restoring model
    logging_steps: int = 500
    save_steps: int = 500
    detach_ed_layer: bool = True
    only_ner: bool = False
    only_ed: bool = False
    md_layer_dropout: float = 0.1
    debug: bool = False

    sep_token_id: Optional[int] = None
    cls_token_id: Optional[int] = None
    mask_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    vocab_size: Optional[int] = None

    ner_tag_to_ix: Dict[str, int] = field(default_factory=lambda: NER_TAG_TO_IX)

    def __post_init__(self):
        tokenizer = get_tokenizer(transformer_name=self.transformer_name, data_dir=self.data_dir)
        self.sep_token_id = tokenizer.sep_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size

    @classmethod
    def from_file(cls, filename: str, data_dir: str):
        with open(filename, "r") as f:
            cfg = json.load(f)
            cfg["data_dir"] = data_dir
            return cls(**cfg)

    def to_file(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.__dict__, f)
