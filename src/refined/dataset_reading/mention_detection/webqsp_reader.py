import json
from random import random
from typing import List, Set, Tuple, Dict, Optional

import torch
from refined.utilities.md_dataset_utils import (
    create_collate_fn,
    tokenize_and_preserve_labels,
)
from torch import Tensor
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer
from refined.utilities.general_utils import batch_items


class WebQSPNER(Dataset):
    def __init__(
        self,
        data_dir: str,
        ner_tag_to_num: Dict[str, int],
        data_split: str = "train",
        bio_only: bool = True,
        sentence_level: bool = True,
        transformer_name: str = "roberta-base",
        max_seq: int = 100,
        random_lower_case_prob: float = 0.15,
        filter_types: Set[str] = None,
        random_replace_question_mark: float = 0.15,
        lower: bool = False,
        use_mention_tag: bool = False,
        convert_types: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        max_seq -= 2
        assert data_split in ["train", "dev", "test"]
        assert transformer_name in [
            "bert-base-uncased",
            "bert-base-cased",
            "bert-large-uncased",
            "bert-large-cased",
            "distilbert-base-cased",
            "distilbert-base-uncased",
            "roberta-base",
        ]
        assert max_seq <= 510, "max seq must be below 512"
        self.lower = lower
        self.random_lower_case_prob = random_lower_case_prob
        self.random_replace_question_mark = random_replace_question_mark
        # if filter_types is None:
        #     self.filter_types = {"TIME", "DURATION", "NUMBER", "ORDINAL", "DATE"}
        # else:
        self.filter_types = filter_types
        self.max_seq = max_seq
        data_split_to_filename = {
            "train": "webqsp_training_data_ner.json",
            "test": "webqsp_test_data_ner.json",
        }
        self.special_tag_to_text = {
            "-LRB-": "(",
            "-RRB-": ")",
            "-LCB-": "{",
            "-RCB-": "}",
            "-LSB-": "[",
            "-RSB-": "]",
            "``": '"',
            "''": '"',
        }
        self.file_path = f'{data_dir.rstrip("/")}/{data_split_to_filename[data_split]}'
        self.ner_tag_to_num = ner_tag_to_num
        self.use_mention_tag = use_mention_tag
        self.convert_types = convert_types
        self.num_labels = len(self.ner_tag_to_num)
        self.bio_only = bio_only
        self.sentence_level = sentence_level
        self.transformer_name = transformer_name
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True)
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id
        # [token, ner_tag]
        self.batch_elements: List[List[Tuple[str, str]]] = self.get_batch_elements()
        self.collate = create_collate_fn(self.pad_id)

    def get_batch_elements(self) -> List[List[Tuple[str, str]]]:

        batch_elements: List[List[Tuple[str, str]]] = []

        for line in self.read_files():

            words_t, ner_t, _ = tokenize_and_preserve_labels(line["words"], line["ners"], self.tokenizer)

            for batch_element in batch_items(zip(words_t, ner_t), n=self.max_seq):
                batch_elements.append(batch_element)

        return batch_elements

    def read_files(self):

        with open(self.file_path, "r") as f:
            for line in f:
                sent = json.loads(line)
                words = sent["words"]
                if self.lower:
                    words = [word.lower() for word in words]
                words = [
                    self.special_tag_to_text[word] if word in self.special_tag_to_text else word
                    for word in words
                ]
                ners = []
                in_ent = False
                for ner in sent["ners"]:
                    if ner == "O" or (self.filter_types is not None and ner in self.filter_types):
                        ners.append("O")
                        in_ent = False
                    else:
                        ners.append(self.get_ner_tag(ner, in_ent=in_ent))
                        in_ent = True
                if all(True if ner == "O" else False for ner in ners):
                    continue

                yield {"words": words, "ners": ners, "dataset_id": sent["dataset_id"], "text": sent["text"]}

    def get_ner_tag(self, ner: str, in_ent: bool):

        if self.convert_types is not None and ner in self.convert_types:
            ner = self.convert_types[ner]

        start_tag = "I" if in_ent else "B"

        if self.bio_only:
            return start_tag

        if self.use_mention_tag:
            if "B-" + ner in self.ner_tag_to_num:
                new_ner = start_tag + "-" + ner
            else:
                new_ner = start_tag + "-MENTION"

        else:
            if "B-" + ner in self.ner_tag_to_num:
                new_ner = start_tag + "-" + ner
            else:
                new_ner = "O"

        return new_ner

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item: List[Tuple[str, str]] = self.batch_elements[index]
        tokens, ners = zip(*item)

        if random() < self.random_replace_question_mark:
            if random() < 0.5:
                tokens = [token for token in tokens][:-1] + ["."]
            else:
                tokens = [token for token in tokens][:-1]
                ners = [ner for ner in ners][:-1]

        token_ids = torch.tensor(
            [self.cls_id] + self.tokenizer.convert_tokens_to_ids(tokens) + [self.sep_id],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [self.ner_tag_to_num["O"]]
            + list(map(lambda x: self.ner_tag_to_num[x], ners))
            + [self.ner_tag_to_num["O"]],
            dtype=torch.long,
        )
        return token_ids, labels

    def __len__(self) -> int:
        return len(self.batch_elements)
