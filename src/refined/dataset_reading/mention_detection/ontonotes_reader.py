import json
from typing import List, Set, Tuple, Optional, Dict

import torch
from refined.utilities.md_dataset_utils import (
    create_collate_fn,
    tokenize_and_preserve_labels,
)
from torch import Tensor
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer
from refined.utilities.general_utils import batch_items


class OntoNotesNER(Dataset):
    def __init__(
        self,
        data_dir: str,
        ner_tag_to_num: Dict[str, int],
        data_split: str = "train",
        bio_only: bool = True,
        sentence_level: bool = True,
        transformer_name: str = "roberta-base",
        max_seq: int = 510,
        filter_types: Set[str] = None,
        random_lower_case_prob: float = 0.15,
        lower: bool = False,
        max_articles: Optional[int] = None,
        use_mention_tag: bool = False,
        preload_file_contents: bool = True,
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
        assert 0.0 <= random_lower_case_prob <= 1.0, "random_lower_case_prob must be in [0,1]"
        self.random_lower_case_prob = random_lower_case_prob
        # if filter_types is None:
        #     self.filter_types = {
        #         "B-CARDINAL",
        #         "I-CARDINAL",
        #         "B-MONEY",
        #         "I-MONEY",
        #         "B-PERCENT",
        #         "I-PERCENT",
        #         "B-TIME",
        #         "I-TIME",
        #         "B-ORDINAL",
        #         "I-ORDINAL",
        #         "B-QUANTITY",
        #         "I-QUANTITY",
        #     }
        # else:
        self.filter_types = filter_types

        self.lower = lower
        self.max_seq = max_seq

        data_split_to_filename = {
            "train": "ontonotes_train_articles.json",
            "dev": "ontonotes_development_articles.json",
            "test": "ontonotes_test_articles.json",
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
            "' '": '"',
        }
        self.file_path = f'{data_dir.rstrip("/")}/{data_split_to_filename[data_split]}'
        self.ner_tag_to_num = ner_tag_to_num
        self.use_mention_tag = use_mention_tag

        assert max(ner_tag_to_num.values()) == len(ner_tag_to_num) - 1, "Class numbers should increase incrementally" \
                                                                        " from 0"

        self.num_labels = len(self.ner_tag_to_num)
        self.bio_only = bio_only
        self.sentence_level = sentence_level
        self.transformer_name = transformer_name
        self.max_articles = max_articles
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True)
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id
        # [token, ner_tag]
        if preload_file_contents:
            self.batch_elements: List[List[Tuple[str, str]]] = self.load_file_contents()
        self.collate = create_collate_fn(pad_id=self.pad_id)

    def load_raw_file_contents(self) -> List[List[Tuple[str, str]]]:
        with open(self.file_path, "r") as f:
            for ix, line in enumerate(f):

                if self.max_articles and ix == self.max_articles:
                    break

                sents = json.loads(line)
                if self.lower:
                    for sent in sents:
                        sent["words"] = [word.lower() for word in sent["words"]]
                if not self.sentence_level:
                    doc_words = [word for sent in sents for word in sent["words"]]
                    doc_ners = [ner for sent in sents for ner in sent["ners"]]
                    sents = [{"words": doc_words, "ners": doc_ners}]
                for sent in sents:
                    words = sent["words"]
                    words = [
                        self.special_tag_to_text[word] if word in self.special_tag_to_text else word
                        for word in words
                    ]

                    if self.bio_only:
                        ners = [ner[0] if ner not in self.filter_types else "O" for ner in sent["ners"]]
                    else:
                        ners = []
                        for ner in sent["ners"]:

                            if self.filter_types is not None and ner in self.filter_types:
                                ners.append("O")
                                continue

                            if ner not in self.ner_tag_to_num:
                                if self.use_mention_tag:
                                    ner = ner.split("-")[0] + "-MENTION"
                                else:
                                    ner = "O"

                            ners.append(ner)

                    yield words, ners

    def load_file_contents(self):

        batch_elements: List[List[Tuple[str, str]]] = []

        for words, ners in self.load_raw_file_contents():

            words_t, ner_t, _ = tokenize_and_preserve_labels(words, ners, self.tokenizer)
            for batch_element in batch_items(zip(words_t, ner_t), n=self.max_seq):
                batch_elements.append(batch_element)

        return batch_elements

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item: List[Tuple[str, str]] = self.batch_elements[index]
        tokens, ners = zip(*item)
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
