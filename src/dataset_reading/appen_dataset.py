import json
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from dataset_reading.dataset_utils import mask_mentions
from doc_preprocessing.dataclasses import (
    BatchedElementsTns,
    BatchElement,
    BatchElementTns,
    Doc,
    Span,
    tokenize,
)
from doc_preprocessing.doc_preprocessor import DocumentPreprocessorMemoryBased
from doc_preprocessing.preprocessing_utils import (
    collate_batch_elements_tns,
    convert_batch_element_to_tensors,
)
from torch.utils.data.dataset import Dataset


class AppenDataset(Dataset):
    """
    This is a dataset reader for CoNLL-AIDA dataset.
    For consistency, the files used were generated from GERBIL benchmarking framework:
    https://github.com/dice-group/gerbil/blob/master/src/main/java/org/aksw/gerbil/dataset/impl/aida/AIDACoNLLDataset.java
    """

    def __init__(
        self,
        preprocessor: DocumentPreprocessorMemoryBased,
        seed=0,
        dataset_split: str = "train",
        data_dir: str = "/big_data",
        filter_disambiguation: bool = True,
        mask: bool = False,
        max_num_classes: int = 150,
    ):
        """
        Constructs dataset object.
        :param preprocessor: utility class that provides methods for WikiData lookups and general pre-processing
        :param max_candidates: maximum number of candidate entities per entity (top k according to P(e|m))
        :param seed: for reproducibility of masking
        :param dataset_split: dataset split to use ('train', 'dev', 'test')
        :param data_dir: path directory containing the dataset files
        :param filter_disambiguation: determines whether to filter annotations that refer to a disambiguation page
        :param mask: determines whether to mask the entity mentions (may be useful for training)
        :param max_num_classes: maximum number of classes an entity can belong to
        """
        assert dataset_split in ["train", "dev", "test"]
        random.seed(seed)
        split_to_filename = {"train": "appen_er_docs.jsonl"}
        file_path = f"{data_dir}/{split_to_filename[dataset_split]}"
        if filter_disambiguation:
            self.disambiguation_qcodes: Set[str] = preprocessor.disambiguation_qcodes
        else:
            self.disambiguation_qcodes: Set[str] = set()
        self.max_candidates = preprocessor.max_candidates
        self.num_classes = len(preprocessor.classes)
        self.max_num_classes = max_num_classes
        self.preprocessor = preprocessor

        self.tkid_to_qcode = dict()
        for wiki_title, tkid in self.preprocessor.wiki_to_tkid.items():
            qcode = preprocessor.map_title_to_qcode(wiki_title)
            if qcode is None:
                continue
            self.tkid_to_qcode[tkid] = qcode
        self.batch_elements: List[BatchElement] = []
        self._load_aida_file_contents(file_path)
        self.mask = mask

    def _load_aida_file_contents(self, filename: str):
        """
        Loads AIDA file into self.batch_elements list.
        :param filename: file path for AIDA dataset file
        """
        with open(filename, "r") as f:
            for idx, line in enumerate(f):
                if idx >= 259:
                    break
                parsed_line = json.loads(line)
                self.batch_elements.extend(
                    [
                        b
                        for b in self.from_dataset_line(parsed_line).to_batch_elements(
                            self.preprocessor.data_dir
                        )
                        if len(b.spans) > 0
                    ]
                )

    def from_dataset_mention(
        self, dataset_mention: Dict[Any, Any], doc_text: str, person_coref: Dict
    ) -> Tuple[Optional[Span], Dict[str, Any]]:
        tkid = dataset_mention["ent_id"]
        if tkid not in self.tkid_to_qcode:
            return None, person_coref
        gold_qcode = self.tkid_to_qcode[tkid]
        if gold_qcode is None or gold_qcode in self.preprocessor.disambiguation_qcodes:
            return None, person_coref
        start = dataset_mention["char_start"]
        ln = dataset_mention["length"]
        text = doc_text[start : start + ln]
        candidate_qcodes, person_coref = self.preprocessor.get_candidates(
            surface_form=text, person_coref_ref=person_coref
        )

        return (
            Span(
                text=text,
                start=start,
                ln=ln,
                gold_entity_id=gold_qcode,
                candidate_entity_ids=candidate_qcodes,
            ),
            person_coref,
        )

    def from_dataset_line(self, dataset_line: Dict[str, Any]) -> "Doc":
        """
        Constructs `Doc` object from gerbil dataset line.
        e.g. code used in Gerbil benchmarking framework to read dataset:
        https://github.com/dice-group/gerbil/blob/master/src/main/java/org/aksw/gerbil/dataset/impl/aida/AIDACoNLLDataset.java
        :param dataset_line: python dict for output of a standard gerbil dataset reader
        :return: `Doc`, spans contains candidates and gold entities so can be used for training and evaluation
        """
        dataset_mentions = dataset_line["entities"]
        dataset_mentions.sort(key=lambda x: x["char_start"])
        text = " ".join([token["text"] for token in dataset_line["tokens"]])
        tokens = tokenize(
            text,
            transformer_name=self.preprocessor.transformer_name,
            data_dir=self.preprocessor.data_dir,
        )
        spans: List[Span] = []
        md_spans: List[Span] = []
        line_persons_coref: Dict = dict()
        for dataset_mention in dataset_mentions:
            start = dataset_mention["char_start"]
            ln = dataset_mention["length"]
            md_spans.append(Span(start=start, ln=ln, text=text[start : start + ln]))
            # populate persons_coref. This enables backward person co-reference.
            _, line_persons_coref = self.from_dataset_mention(
                dataset_mention=dataset_mention, doc_text=text, person_coref=line_persons_coref
            )
        for dataset_mention in dataset_mentions:
            span, line_persons_coref = self.from_dataset_mention(
                dataset_mention=dataset_mention, doc_text=text, person_coref=line_persons_coref
            )
            if span is not None:
                spans.append(span)
        return Doc(text=text, spans=spans, tokens=tokens, md_spans=md_spans)

    def __getitem__(self, index: int) -> BatchElementTns:
        """
        Returns the tensors for a single batch element.
        :param index: the index of the batch element to return
        :return: batch element converted to a tuple of tensors
        """
        if self.mask:
            batch_elem = mask_mentions(
                self.batch_elements[index],
                mask_token_id=self.preprocessor.mask_id,
                vocab_size=self.preprocessor.vocab_size,
            )
        else:
            batch_elem = self.batch_elements[index]
        return convert_batch_element_to_tensors(
            batch_element=batch_elem, processor=self.preprocessor
        )

    def __len__(self) -> int:
        """
        Returns number of batch elements.
        :return: Number of batch elements.
        """
        return len(self.batch_elements)

    def collate(self, batch_elements_tns: List[BatchElementTns]) -> BatchedElementsTns:
        """
        Collates a list of tensors for batch elements into a single batch.
        - Pads tensors to longest length in batch
        - Corrects masks to prevent information loss
        - Stacks tensors
        :param batch_elements_tns: list of tensors for batch elements
        :return: a tuple of tensors where all tensors in the list are vertically stacked
        """
        return collate_batch_elements_tns(
            batch_elements_tns=batch_elements_tns, token_pad_value=self.preprocessor.pad_id
        )
