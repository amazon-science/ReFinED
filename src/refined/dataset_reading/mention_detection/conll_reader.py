from typing import List, Tuple, Dict, Set, Optional
from tqdm import tqdm

import torch
from refined.utilities.md_dataset_utils import (
    create_collate_fn,
    tokenize_and_preserve_labels,
    bio_to_offset_pairs
)
from torch import Tensor
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer
from refined.utilities.general_utils import batch_items
from refined.inference.standalone_md import MentionDetector


CONLL_NER_TAG_TO_NUM = {
    'O': 0,
    'B-LOC': 1,
    'B-MISC': 2,
    'B-ORG': 3,
    'B-PER': 4,
    'I-LOC': 5,
    'I-MISC': 6,
    'I-ORG': 7,
    'I-PER': 8
}


class CoNLLNER(Dataset):
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
        lower: bool = False,
        additional_filename: Optional[str] = None,
        use_mention_tag: bool = False,
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
        self.random_lower_case_prob = random_lower_case_prob
        self.max_seq = max_seq
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

        self.data_split = data_split
        self.data_dir = data_dir
        self.file_path = self.get_filepath(data_split, additional_filename)
        self.ner_tag_to_num = ner_tag_to_num
        self.use_mention_tag = use_mention_tag
        self.num_labels = len(self.ner_tag_to_num)
        self.bio_only = bio_only
        self.sentence_level = sentence_level
        self.transformer_name = transformer_name
        self.lower = lower
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True)
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id

        # [token, ner_tag]
        self.batch_elements: List[List[Tuple[str, str]]] = self.load_file_contents1()
        self.collate = create_collate_fn(pad_id=self.pad_id)

    def get_filepath(self, split: str, additional_filename: Optional[str]) -> str:
        data_split_to_filename = {
            "train": "conll_train.txt",
            "dev": "conll_dev.txt",
            "test": "conll_test.txt",
        }
        file_path = data_split_to_filename[split]

        if additional_filename is not None:
            file_path = file_path[:-4] + additional_filename + '.txt'

        return f'{self.data_dir.rstrip("/")}/{file_path}'

    def load_file_contents1(self) -> List[List[Tuple[str, str]]]:
        # when self.sentence_level=True each doc will be a sentence
        batch_elements = []
        docs: List[List[List[Tuple[str, str, str, str]]]] = self.read_file_as_docs(
            self.file_path, self.bio_only, self.sentence_level, lower=self.lower
        )
        for doc in docs:
            onto_notes_tokens = [token for sent in doc for token in sent]
            tokens, _, _, ners = zip(*onto_notes_tokens)
            # can make text noisy here
            tokens_t, ner_t, _ = tokenize_and_preserve_labels(tokens, ners, self.tokenizer)
            for batch_element in batch_items(zip(tokens_t, ner_t), n=self.max_seq):
                batch_elements.append(batch_element)
        return batch_elements

    def read_file_as_docs(self,
        file_path, bio_only=True, sentence_level=True, lower: bool = False, leave_all_mentions: bool = False
    ) -> List[List[List[Tuple[str, str, str, str]]]]:
        """
        Each doc is a list of sentences, each sentence is a list of tokens (tuples)
        :param file_path: file path
        :param bio_only: bio only means ignore type
        :param sentence_level: sentence level or article level
        :param lower: lower means lower case the text
        :return: return list of docs (each doc is list of sentence, each sentence is list of tokens (tuples))
        """
        with open(file_path, "r") as f:
            docs: List[List[List[Tuple[str, str, str, str]]]] = []
            current_doc_sents: List[List[Tuple[str, str, str, str]]] = []
            current_sent: List[Tuple[str, str, str, str]] = []
            for line in f:
                parts = line.split(" ")
                if len(parts) == 4 and "-DOCSTART-" not in line:
                    text, pos, dep, ner = parts
                    text = self.special_tag_to_text[text] if text in self.special_tag_to_text else text
                    ner = ner.rstrip("\n")

                    if not leave_all_mentions:
                        if bio_only:
                            ner = ner[0]
                        else:
                            if ner not in self.ner_tag_to_num:
                                ner = ner.split("-")[0] + "-MENTION" if self.use_mention_tag else "O"
                    if lower:
                        text = text.lower()
                    current_sent.append((text, pos, dep, ner))
                else:
                    if len(current_sent) > 0:
                        current_doc_sents.append(current_sent)
                        if sentence_level:
                            docs.append(current_doc_sents)
                            current_doc_sents = []
                        current_sent = []
                    if "-DOCSTART-" in line and len(current_doc_sents) > 0:
                        # start new doc
                        docs.append(current_doc_sents)
                        current_doc_sents = []

            return docs

    @staticmethod
    def relabel_doc(doc: List[List[Tuple[str, str, str, str]]], mention_detector: MentionDetector,
                    ner_types_to_add: Set[str]) -> List[List[Tuple[str, str, str, str]]]:
        """
        Add new NER labels to a doc, using a trained MentionDetector
        """
        onto_notes_tokens = [token for sent in doc for token in sent]
        words, _, _, ners = zip(*onto_notes_tokens)
        ners = list(ners)

        ner_preds = mention_detector.process_words(words)

        # Get (start_ix, end_ix, label) for each detected mention
        ner_pred_tuples = bio_to_offset_pairs(ner_preds, use_labels=True)

        # Add new predicted labels for all types in ner_types_to_add
        for start_ix, end_ix, ner_type in ner_pred_tuples:

            if ner_type not in ner_types_to_add:
                continue

            # If there is already a ner label for any of the tokens, then don't overwrite it
            current_labels = ners[start_ix:end_ix]
            if set(current_labels) != {'O'}:
                continue

            new_labels = ['B-' + ner_type] + ['I-' + ner_type for _ in range(end_ix - start_ix - 1)]
            ners[start_ix:end_ix] = new_labels

        start_ix = 0
        relabelled_doc = []

        for sentence in doc:
            sentence = [list(labels) for labels in sentence]
            new_ners = ners[start_ix: start_ix + len(sentence)]
            for ix, new_token_ner in enumerate(new_ners):
                sentence[ix][3] = new_token_ner
            sentence = [tuple(labels) for labels in sentence]
            relabelled_doc.append(sentence)
            start_ix += len(sentence)

        return relabelled_doc

    def relabel_dataset(self, additional_filename: str, ner_types_to_add: Set[str],
                        mention_detector: MentionDetector) -> None:
        """
        Create a new version of a dataset using MentionDetector to add additional NER labels (e.g. for DATE spans)
        """
        assert not self.lower, "Lowercasing text whilst re-labelling will lose casing information when file is written" \
                               "back to disk"
        assert not self.sentence_level, "Should relabel at article level to preserve article structure of dataset"

        new_file_path = self.get_filepath(self.data_split, additional_filename=additional_filename)

        with open(new_file_path, "w") as write_file:

            for doc in tqdm(
                    self.read_file_as_docs(self.file_path, bio_only=False, sentence_level=self.sentence_level,
                                           lower=self.lower, leave_all_mentions=True)):

                # Relabel the doc using the trained NER model
                new_doc = self.relabel_doc(doc=doc, mention_detector=mention_detector,
                                           ner_types_to_add=ner_types_to_add)

                # Write the new doc to file in original format
                write_file.write("-DOCSTART- -X- -X- O\n\n")

                for sentence in new_doc:
                    for labels in sentence:
                        write_file.write(" ".join(labels) + "\n")
                    write_file.write("\n")

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
